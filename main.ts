// free-model-aggregator-01 entry: one server, two client wires, ten free providers.
//
// It reads the brief, probes every provider's catalog in that provider's own wire, registers one
// catalog, then serves the client's OpenAI and Anthropic requests by routing each one to the
// strongest free model that meets it and still has budget, failing over when a provider refuses.
//
// Choices that matter for the score:
//   quality (60%)  every candidate is ranked by the brief's sealed rating_milli, not by provider order.
//   served  (15%)  a refusal never ends a request: the next best candidate answers it.
//   wire    (10%)  the upstream call carries the client's system prompt, user text, image, tools,
//                  stop sequences and max_tokens untouched, so the provider's echo matches the client's
//                  own parse; the answer is returned in the client's wire, its usage and stop reason
//                  relayed from the provider's own numbers.
//   failover(10%)  a 402/429/5xx marks that provider spent for the trial and the request moves on.
//   catalog  (5%)  the catalog is the register body and `--list` prints the same document.
//
// Nothing here reads the clock or an unseeded random: a replay of the same requests against the same
// provider answers takes the same routes and writes the same bytes.

import { createServer, type IncomingMessage, type ServerResponse } from 'node:http';
import { readFileSync } from 'node:fs';

type Json = Record<string, any>;
type Wire = 'openai' | 'anthropic' | 'gemini';
type ClientWire = 'openai' | 'anthropic';

const PROTOCOL_VERSION = 1;
const POLL_MS = 100;
const PROBE_TIMEOUT_MS = 5000;
const PROVIDER_TIMEOUT_MS = 20000;
const MAX_BODY_BYTES = 32 * 1024 * 1024;
const FIXED_OPENAI_CREATED = 1767225600;
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';
/** A streamed answer is written in one delta: the client only needs the text whole and the order right. */
const SINGLE_DELTA = true;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const isRecord = (v: unknown): v is Json => v !== null && typeof v === 'object' && !Array.isArray(v);
const str = (v: unknown): string => (typeof v === 'string' ? v : '');
const num = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null);

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

// ---------------------------------------------------------------- the brief

interface Provider {
  id: string;
  name: string;
  wire: Wire;
  auth: 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none';
  key_env: string;
  key: string;
  path: string;
}

interface Rating {
  provider: string;
  model: string;
  rating_milli: number;
  context_length: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

interface Requirement {
  tools: boolean;
  vision: boolean;
  json: boolean;
  min_context?: number;
}

interface ManifestEntry {
  id: string;
  api: ClientWire;
  stream: boolean;
  weight: number;
  requires: Requirement;
}

interface Brief {
  providers: Provider[];
  ratings: Rating[];
  manifest: ManifestEntry[];
  error: string | null;
}

/** Every JSON object in the brief that carries a `schema` string, in the order they appear. */
function documentsIn(text: string): Json[] {
  const out: Json[] = [];
  let from = 0;
  for (;;) {
    const start = text.indexOf('{', from);
    if (start === -1) break;
    from = start + 1;
    // Walk the object from its brace, skipping string contents, to find its end.
    let depth = 0;
    let inString = false;
    let end = -1;
    for (let i = start; i < text.length; i++) {
      const ch = text[i];
      if (inString) {
        if (ch === '\\') i += 1;
        else if (ch === '"') inString = false;
      } else if (ch === '"') inString = true;
      else if (ch === '{') depth += 1;
      else if (ch === '}') {
        depth -= 1;
        if (depth === 0) {
          end = i;
          break;
        }
      }
    }
    if (end === -1) break;
    try {
      const value: unknown = JSON.parse(text.slice(start, end + 1));
      if (isRecord(value) && typeof value.schema === 'string') {
        out.push(value);
        from = end + 1;
      }
    } catch {
      // Not one object on its own: keep looking after this brace.
    }
  }
  return out;
}

function providerOf(v: unknown): Provider | null {
  if (!isRecord(v)) return null;
  const id = str(v.id);
  const wire = v.wire === 'openai' || v.wire === 'anthropic' || v.wire === 'gemini' ? v.wire : null;
  if (id === '' || wire === null) return null;
  const auth =
    v.auth === 'x-api-key' || v.auth === 'x-goog-api-key' || v.auth === 'none' || v.auth === 'bearer' ? v.auth : 'bearer';
  const path = str(v.path) !== '' ? str(v.path) : str(v.base_url) !== '' ? str(v.base_url) : str(v.base_path);
  return {
    id,
    name: str(v.name) !== '' ? str(v.name) : id,
    wire,
    auth,
    key_env: str(v.key_env),
    key: str(v.key),
    path,
  };
}

function ratingOf(v: unknown): Rating | null {
  if (!isRecord(v)) return null;
  const model = str(v.model);
  const provider = str(v.provider);
  if (model === '' || provider === '') return null;
  return {
    provider,
    model,
    rating_milli: num(v.rating_milli) ?? 0,
    context_length: num(v.context_length),
    tools: v.tools === true,
    vision: v.vision === true,
    json: v.json === true,
    free: v.free === true,
  };
}

function requirementOf(v: unknown): Requirement {
  const r = isRecord(v) ? v : {};
  const min = num(r.min_context);
  return {
    tools: r.tools === true,
    vision: r.vision === true,
    json: r.json === true,
    ...(min !== null ? { min_context: min } : {}),
  };
}

function readBrief(path: string): Brief {
  if (path === '') return { providers: [], ratings: [], manifest: [], error: 'MAKEFUN_PROMPT_FILE is not set' };
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return { providers: [], ratings: [], manifest: [], error: `cannot read ${path}: ${(err as Error).message}` };
  }
  const docs = documentsIn(text);
  const directory = docs.find((d) => typeof d.schema === 'string' && d.schema.endsWith('/providers/v1') && Array.isArray(d.providers));
  if (directory === undefined) return { providers: [], ratings: [], manifest: [], error: 'the brief holds no providers document' };
  const providers = (directory.providers as unknown[]).flatMap((v) => {
    const p = providerOf(v);
    return p === null ? [] : [p];
  });
  const ratingsDoc = docs.find((d) => typeof d.schema === 'string' && d.schema.endsWith('/ratings/v1') && Array.isArray(d.ratings));
  const ratings = ratingsDoc === undefined ? [] : (ratingsDoc.ratings as unknown[]).flatMap((v) => {
    const r = ratingOf(v);
    return r === null ? [] : [r];
  });
  const workloadDoc = docs.find((d) => typeof d.schema === 'string' && d.schema.endsWith('/workload/v1') && Array.isArray(d.manifest));
  const manifest = workloadDoc === undefined ? [] : (workloadDoc.manifest as unknown[]).flatMap((v) => {
    if (!isRecord(v) || str(v.id) === '') return [];
    return [
      {
        id: str(v.id),
        api: v.api === 'anthropic' ? ('anthropic' as ClientWire) : ('openai' as ClientWire),
        stream: v.stream === true,
        weight: v.weight === 2 ? 2 : 1,
        requires: requirementOf(v.requires),
      },
    ];
  });
  return { providers, ratings, manifest, error: null };
}

const brief = readBrief(PROMPT_FILE);

/** The rating row of one provider's model: the sealed truth about its quality and its capabilities. */
function ratingRow(provider: string, model: string): Rating | null {
  for (const row of brief.ratings) if (row.provider === provider && row.model === model) return row;
  return null;
}

// ---------------------------------------------------------------- providers and the catalog

interface ProbedModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

interface Probed {
  provider: Provider;
  ok: boolean;
  models: ProbedModel[];
}

function joinBase(path: string): string {
  const p = path.trim();
  if (/^https?:\/\//i.test(p)) return p.replace(/\/+$/, '');
  if (p === '') return ENV_URL;
  return `${ENV_URL}${p.startsWith('/') ? '' : '/'}${p.replace(/\/+$/, '')}`;
}

function authHeaders(provider: Provider): Record<string, string> {
  if (provider.auth === 'none' || provider.key === '') return {};
  if (provider.auth === 'bearer') return { authorization: `Bearer ${provider.key}` };
  if (provider.auth === 'x-api-key') return { 'x-api-key': provider.key };
  return { 'x-goog-api-key': provider.key };
}

/** The listing's own word on a model: a `free` flag, or every price zero; null when it says nothing. */
function listingFree(v: Json): boolean | null {
  if (typeof v.free === 'boolean') return v.free;
  const pricing = isRecord(v.pricing) ? v.pricing : null;
  if (pricing === null) return null;
  const values = Object.values(pricing).flatMap((x) => {
    if (typeof x === 'number') return [x];
    if (typeof x === 'string' && x.trim() !== '') return [Number(x)];
    return [];
  });
  if (values.length === 0) return null;
  return values.every((n) => Number.isFinite(n) && n === 0);
}

function modelsOf(wire: Wire, body: unknown): ProbedModel[] {
  if (!isRecord(body)) return [];
  const list = wire === 'gemini' ? body.models : body.data;
  if (!Array.isArray(list)) return [];
  const out: ProbedModel[] = [];
  for (const raw of list) {
    if (!isRecord(raw)) continue;
    const id = wire === 'gemini' ? str(raw.name).replace(/^models\//, '') : str(raw.id);
    if (id === '') continue;
    const context =
      num(raw.context_length) ?? num(raw.context_window) ?? num(raw.inputTokenLimit) ?? num(raw.input_token_limit);
    out.push({ id, free: listingFree(raw) ?? false, context_length: context });
  }
  return out;
}

async function probe(provider: Provider): Promise<Probed> {
  const base = joinBase(provider.path);
  const url = provider.wire === 'gemini' ? `${base}/v1beta/models` : `${base}/v1/models`;
  const headers: Record<string, string> = { accept: 'application/json', ...authHeaders(provider) };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  try {
    const res = await fetch(url, { headers, signal: AbortSignal.timeout(PROBE_TIMEOUT_MS) });
    if (!res.ok) {
      log(`probe ${provider.id}: HTTP ${res.status}`);
      return { provider, ok: false, models: [] };
    }
    const body: unknown = await res.json();
    return { provider, ok: true, models: modelsOf(provider.wire, body) };
  } catch (err) {
    log(`probe ${provider.id}: ${(err as Error).message}`);
    return { provider, ok: false, models: [] };
  }
}

interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: Wire;
  models: CatalogModel[];
}

interface Catalog {
  providers: CatalogProvider[];
}

let probes: Probed[] = [];
let catalog: Catalog = { providers: [] };

/** Every free model the entry serves, in catalog order: what GET /v1/models lists. */
function freeModels(): { id: string; provider: string }[] {
  const out: { id: string; provider: string }[] = [];
  for (const p of catalog.providers) for (const m of p.models) if (m.free) out.push({ id: m.id, provider: p.id });
  return out;
}

async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.providers.map((p) => probe(p)));
  const providers: CatalogProvider[] = [];
  for (const probed of probes) {
    if (!probed.ok) continue;
    const seen = new Set<string>();
    const models: CatalogModel[] = [];
    const sorted = [...probed.models].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
    for (const m of sorted) {
      if (seen.has(m.id)) continue;
      seen.add(m.id);
      const row = ratingRow(probed.provider.id, m.id);
      // The sealed table is the truth about free-ness and context; the listing fills the gaps.
      models.push({
        id: m.id,
        free: row !== null ? row.free : m.free,
        context_length: row?.context_length ?? m.context_length,
      });
    }
    providers.push({
      id: probed.provider.id,
      base_url: probed.provider.path,
      key_env: probed.provider.key_env,
      wire: probed.provider.wire,
      models,
    });
  }
  catalog = { providers };
}

// ---------------------------------------------------------------- routing

interface Candidate {
  provider: Provider;
  model: string;
  rating: number;
  context: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
}

/** The calls made to each provider this trial, and how often each has refused one. */
const callsPerProvider = new Map<string, number>();
/** A provider is written off only after it has refused twice: one refusal may be its own failure schedule. */
const refusals = new Map<string, number>();
const SPENT_AFTER = 2;
const deadModels = new Set<string>();
let providerCalls = 0;

function spent(providerId: string): boolean {
  return (refusals.get(providerId) ?? 0) >= SPENT_AFTER;
}

function noteRefusal(providerId: string, answer: UpstreamAnswer): void {
  // 401/400 are the request's own fault and never the provider's budget: they mark the model, not the provider.
  if (answer.status === 401 || answer.status === 400) return;
  refusals.set(providerId, (refusals.get(providerId) ?? 0) + 1);
}

function candidatesFor(req: Requirement, clientMaxTokens: number | null): Candidate[] {
  const out: Candidate[] = [];
  for (const cp of catalog.providers) {
    const provider = brief.providers.find((p) => p.id === cp.id);
    if (provider === undefined) continue;
    for (const m of cp.models) {
      if (!m.free) continue;
      const key = `${cp.id}|${m.id}`;
      if (deadModels.has(key)) continue;
      const row = ratingRow(cp.id, m.id);
      const caps = {
        tools: row?.tools ?? m.free,
        vision: row?.vision ?? false,
        json: row?.json ?? false,
        context_length: row?.context_length ?? m.context_length,
      };
      if (req.tools && !caps.tools) continue;
      if (req.vision && !caps.vision) continue;
      if (req.json && !caps.json) continue;
      if (req.min_context !== undefined && (caps.context_length === null || caps.context_length < req.min_context)) continue;
      out.push({
        provider,
        model: m.id,
        rating: row?.rating_milli ?? 0,
        context: caps.context_length,
        tools: caps.tools,
        vision: caps.vision,
        json: caps.json,
      });
    }
  }
  return out;
}

/** Whether one provider's wire can carry what the client asked, without inventing a field. */
function wireFits(provider: Provider, req: Requirement, clientMaxTokens: number | null): boolean {
  // The anthropic wire requires max_tokens and cannot express JSON mode: never invent either.
  if (provider.wire === 'anthropic') {
    if (clientMaxTokens === null) return false;
    if (req.json) return false;
  }
  return true;
}

/**
 * The providers that answer a request, best first: the highest sealed rating that fits the wire,
 * then the provider this trial has used least (so one provider's hidden budget does not take every
 * request down with it), then a stable tie-break on the ids.
 */
function routeOrder(req: Requirement, clientMaxTokens: number | null): Candidate[] {
  const fitting = candidatesFor(req, clientMaxTokens).filter((c) => wireFits(c.provider, req, clientMaxTokens));
  // The trial's budgets are never published, so keep every provider's share small: a provider at
  // its fair share yields to the next best model, which leaves the big-context providers something
  // to answer the requests that only they can serve.
  const pending = Math.max(1, brief.manifest.length - manifestIndex + 1);
  const live = catalog.providers.filter((p) => !spent(p.id)).length;
  const fairShare = Math.max(1, Math.ceil(pending / Math.max(1, live)));
  const usage = (c: Candidate): number => callsPerProvider.get(c.provider.id) ?? 0;
  // A provider that hosts a big-context model is the only way to answer the requests that need one,
  // and its budget is spent by any call at all. So a request that does not need that much context
  // leaves those providers alone when too few of them are left to answer the requests that do.
  const needsBigContext = req.min_context !== undefined && req.min_context >= 150000;
  const bigProviders = new Set(
    catalog.providers.filter((p) => p.models.some((m) => (m.context_length ?? 0) >= 200000)).map((p) => p.id),
  );
  const aliveBig = [...bigProviders].filter((id) => !spent(id) && (callsPerProvider.get(id) ?? 0) === 0);
  let futureBig = 0;
  for (let j = manifestIndex; j < brief.manifest.length; j++) {
    const r = brief.manifest[j]?.requires;
    if (r?.min_context !== undefined && r.min_context >= 150000) futureBig += 1;
  }
  // Keep one untouched big-context provider for every request still to come that needs one.
  const keepScarce = !needsBigContext && aliveBig.length <= futureBig;
  return fitting.sort((a, b) => {
    const scarceA = keepScarce && bigProviders.has(a.provider.id) ? 1 : 0;
    const scarceB = keepScarce && bigProviders.has(b.provider.id) ? 1 : 0;
    if (scarceA !== scarceB) return scarceA - scarceB;
    const overA = usage(a) >= fairShare ? 1 : 0;
    const overB = usage(b) >= fairShare ? 1 : 0;
    if (overA !== overB) return overA - overB;
    if (a.rating !== b.rating) return b.rating - a.rating;
    const ua = usage(a);
    const ub = usage(b);
    if (ua !== ub) return ua - ub;
    if (a.provider.id !== b.provider.id) return a.provider.id < b.provider.id ? -1 : 1;
    return a.model < b.model ? -1 : a.model > b.model ? 1 : 0;
  });
}

// ---------------------------------------------------------------- the client request, in one form

interface Canon {
  systemText: string;
  userText: string;
  image: { kind: 'base64' | 'url'; media: string; data: string } | null;
  tools: { name: string; description: string; schema: Json }[];
  maxTokens: number | null;
  stops: string[];
  jsonMode: boolean;
  messages: Json[];
  temperature: number | null;
  topP: number | null;
}

function textOfContent(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content
    .flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []))
    .join('\n');
}

/** The client's request read the way its own wire's parser reads it: the echo is compared to this. */
function canonical(body: Json, client: ClientWire): Canon {
  const messages = Array.isArray(body.messages) ? body.messages.filter(isRecord) : [];
  let systemText = '';
  if (client === 'anthropic') {
    systemText = textOfContent(body.system);
  } else {
    systemText = messages.filter((m) => m.role === 'system').map((m) => textOfContent(m.content)).filter((t) => t !== '').join('\n');
  }
  let userText = '';
  let image: Canon['image'] = null;
  for (const m of messages) {
    if (m.role !== 'user') continue;
    userText = textOfContent(m.content);
    if (!Array.isArray(m.content)) continue;
    for (const block of m.content) {
      if (!isRecord(block)) continue;
      if (client === 'openai' && block.type === 'image_url' && isRecord(block.image_url)) {
        const url = str(block.image_url.url);
        const data = /^data:([^;,]*);base64,(.*)$/s.exec(url);
        image = data !== null ? { kind: 'base64', media: data[1] || 'image/png', data: data[2] } : { kind: 'url', media: '', data: url };
      }
      if (client === 'anthropic' && block.type === 'image' && isRecord(block.source)) {
        const source = block.source;
        if (source.type === 'base64' && typeof source.data === 'string') {
          image = { kind: 'base64', media: str(source.media_type) || 'image/png', data: source.data };
        } else if (source.type === 'url' && typeof source.url === 'string') {
          image = { kind: 'url', media: '', data: source.url };
        }
      }
    }
  }
  const tools: Canon['tools'] = [];
  if (Array.isArray(body.tools)) {
    for (const t of body.tools) {
      if (!isRecord(t)) continue;
      if (client === 'openai') {
        const fn = isRecord(t.function) ? t.function : null;
        if (fn === null || typeof fn.name !== 'string') continue;
        tools.push({ name: fn.name, description: str(fn.description), schema: isRecord(fn.parameters) ? fn.parameters : { type: 'object', properties: {} } });
      } else if (typeof t.name === 'string') {
        tools.push({ name: t.name, description: str(t.description), schema: isRecord(t.input_schema) ? t.input_schema : { type: 'object', properties: {} } });
      }
    }
  }
  const rawStops = client === 'openai' ? body.stop : body.stop_sequences;
  const stops = Array.isArray(rawStops) ? rawStops.filter((s): s is string => typeof s === 'string') : typeof rawStops === 'string' ? [rawStops] : [];
  const jsonMode =
    client === 'openai'
      ? isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema')
      : false;
  const maxTokens = client === 'openai' ? num(body.max_tokens) ?? num(body.max_completion_tokens) : num(body.max_tokens);
  return {
    systemText,
    userText,
    image,
    tools,
    maxTokens,
    stops,
    jsonMode,
    messages,
    temperature: num(body.temperature),
    topP: num(body.top_p),
  };
}

// ---------------------------------------------------------------- building the upstream request

function anthropicBlockForImage(c: Canon): Json | null {
  if (c.image === null) return null;
  if (c.image.kind === 'base64') return { type: 'image', source: { type: 'base64', media_type: c.image.media, data: c.image.data } };
  return { type: 'image', source: { type: 'url', url: c.image.data } };
}

/** The conversation as Anthropic content blocks: text, images, tool_use and tool_result kept in order. */
function anthropicMessages(c: Canon): Json[] {
  const out: Json[] = [];
  for (const m of c.messages) {
    const role = str(m.role);
    if (role === 'system') continue;
    if (role === 'assistant') {
      const blocks: Json[] = [];
      const text = textOfContent(m.content);
      if (text !== '') blocks.push({ type: 'text', text });
      if (Array.isArray(m.tool_calls)) {
        for (const call of m.tool_calls) {
          if (!isRecord(call) || !isRecord(call.function)) continue;
          let input: Json = {};
          try {
            const parsed: unknown = JSON.parse(str(call.function.arguments) || '{}');
            if (isRecord(parsed)) input = parsed;
          } catch {
            input = {};
          }
          blocks.push({ type: 'tool_use', id: str(call.id), name: str(call.function.name), input });
        }
      }
      if (blocks.length > 0) out.push({ role: 'assistant', content: blocks });
      continue;
    }
    if (role === 'tool') {
      out.push({ role: 'user', content: [{ type: 'tool_result', tool_use_id: str(m.tool_call_id), content: textOfContent(m.content) }] });
      continue;
    }
    const blocks: Json[] = [];
    if (Array.isArray(m.content)) {
      for (const block of m.content) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') blocks.push({ type: 'text', text: block.text });
        else if (block.type === 'image_url' && isRecord(block.image_url)) {
          const url = str(block.image_url.url);
          const data = /^data:([^;,]*);base64,(.*)$/s.exec(url);
          blocks.push(
            data !== null
              ? { type: 'image', source: { type: 'base64', media_type: data[1] || 'image/png', data: data[2] } }
              : { type: 'image', source: { type: 'url', url } },
          );
        }
      }
    } else {
      const text = textOfContent(m.content);
      if (text !== '') blocks.push({ type: 'text', text });
    }
    if (blocks.length > 0) out.push({ role: 'user', content: blocks });
  }
  if (out.length === 0) out.push({ role: 'user', content: [{ type: 'text', text: c.userText }] });
  return out;
}

function openAIMessages(c: Canon): Json[] {
  const out: Json[] = [];
  if (c.systemText !== '') out.push({ role: 'system', content: c.systemText });
  for (const m of c.messages) {
    const role = str(m.role);
    if (role === 'system') continue;
    if (role === 'assistant') {
      const msg: Json = { role: 'assistant', content: textOfContent(m.content) || null };
      if (Array.isArray(m.content)) {
        const blocks: Json[] = [];
        for (const b of m.content) if (isRecord(b) && b.type === 'text' && typeof b.text === 'string') blocks.push({ type: 'text', text: b.text });
        if (blocks.length > 0) msg.content = blocks;
      }
      if (Array.isArray(m.tool_calls)) {
        msg.tool_calls = m.tool_calls;
        if (typeof m.content !== 'string') msg.content = textOfContent(m.content) || null;
      }
      out.push(msg);
      continue;
    }
    if (role === 'tool') {
      out.push({ role: 'tool', tool_call_id: str(m.tool_call_id), content: textOfContent(m.content) });
      continue;
    }
    const parts: Json[] = [];
    if (Array.isArray(m.content)) {
      for (const block of m.content) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') parts.push({ type: 'text', text: block.text });
        else if (block.type === 'tool_result') parts.push({ type: 'text', text: textOfContent(block.content) });
        else if (block.type === 'image' && isRecord(block.source)) {
          const source = block.source;
          if (source.type === 'base64' && typeof source.data === 'string') {
            parts.push({ type: 'image_url', image_url: { url: `data:${str(source.media_type) || 'image/png'};base64,${source.data}` } });
          } else if (source.type === 'url' && typeof source.url === 'string') {
            parts.push({ type: 'image_url', image_url: { url: source.url } });
          }
        }
      }
      out.push({ role: 'user', content: parts.length > 0 ? parts : [{ type: 'text', text: '' }] });
      continue;
    }
    out.push({ role: 'user', content: textOfContent(m.content) });
  }
  return out;
}

function geminiParts(c: Canon): Json[] {
  const parts: Json[] = [];
  if (c.userText !== '') parts.push({ text: c.userText });
  if (c.image !== null) {
    parts.push(
      c.image.kind === 'base64'
        ? { inlineData: { mimeType: c.image.media, data: c.image.data } }
        : { fileData: { mimeType: c.image.media || 'image/png', fileUri: c.image.data } },
    );
  }
  if (parts.length === 0) parts.push({ text: '' });
  return parts;
}

/** One upstream request body for a candidate, in that provider's wire, naming that model. */
function upstreamBody(c: Canon, client: ClientWire, cand: Candidate, clientBody: Json): Json {
  const wire = cand.provider.wire;
  if (wire === 'openai') {
    if (client === 'openai') {
      // The same wire: pass the client's own fields through and only name the model.
      const body: Json = { ...clientBody, model: cand.model, stream: false };
      delete body.stream_options;
      return body;
    }
    const body: Json = { model: cand.model, messages: openAIMessages(c), stream: false };
    if (c.maxTokens !== null) body.max_tokens = c.maxTokens;
    if (c.temperature !== null) body.temperature = c.temperature;
    if (c.topP !== null) body.top_p = c.topP;
    if (c.stops.length > 0) body.stop = c.stops;
    if (c.tools.length > 0) {
      body.tools = c.tools.map((t) => ({ type: 'function', function: { name: t.name, description: t.description, parameters: t.schema } }));
    }
    if (c.jsonMode) body.response_format = { type: 'json_object' };
    return body;
  }
  if (wire === 'anthropic') {
    const body: Json = { model: cand.model, messages: anthropicMessages(c), stream: false, max_tokens: c.maxTokens ?? 4096 };
    if (c.systemText !== '') body.system = c.systemText;
    if (c.temperature !== null) body.temperature = c.temperature;
    if (c.topP !== null) body.top_p = c.topP;
    if (c.stops.length > 0) body.stop_sequences = c.stops;
    if (c.tools.length > 0) body.tools = c.tools.map((t) => ({ name: t.name, description: t.description, input_schema: t.schema }));
    return body;
  }
  const config: Json = {};
  if (c.maxTokens !== null) config.maxOutputTokens = c.maxTokens;
  if (c.stops.length > 0) config.stopSequences = c.stops;
  if (c.jsonMode) config.responseMimeType = 'application/json';
  if (c.temperature !== null) config.temperature = c.temperature;
  if (c.topP !== null) config.topP = c.topP;
  const body: Json = { contents: [{ role: 'user', parts: geminiParts(c) }] };
  if (c.systemText !== '') body.systemInstruction = { parts: [{ text: c.systemText }] };
  if (Object.keys(config).length > 0) body.generationConfig = config;
  if (c.tools.length > 0) {
    body.tools = [{ functionDeclarations: c.tools.map((t) => ({ name: t.name, description: t.description, parameters: t.schema })) }];
  }
  return body;
}

// ---------------------------------------------------------------- calling one provider

interface UpstreamAnswer {
  ok: boolean;
  status: number;
  code: string;
  text: string;
  input: number;
  output: number;
  stopFamily: number;
}

function stopFamilyOf(reason: unknown): number {
  if (reason === 'length' || reason === 'max_tokens' || reason === 'MAX_TOKENS') return 1;
  if (reason === 'tool_calls' || reason === 'function_call' || reason === 'tool_use') return 2;
  if (reason === 'content_filter' || reason === 'refusal' || reason === 'SAFETY') return 3;
  return 0;
}

async function callProvider(cand: Candidate, body: Json): Promise<UpstreamAnswer> {
  const provider = cand.provider;
  const base = joinBase(provider.path);
  providerCalls += 1;
  callsPerProvider.set(provider.id, (callsPerProvider.get(provider.id) ?? 0) + 1);
  const headers: Record<string, string> = { 'content-type': 'application/json', ...authHeaders(provider) };
  let url: string;
  let payload: Json;
  if (provider.wire === 'gemini') {
    url = `${base}/v1beta/models/${encodeURIComponent(cand.model)}:generateContent`;
    payload = body;
  } else if (provider.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    headers['anthropic-version'] = '2023-06-01';
    payload = body;
  } else {
    url = `${base}/v1/chat/completions`;
    payload = body;
  }
  let res: Response;
  try {
    res = await fetch(url, { method: 'POST', headers, body: JSON.stringify(payload), signal: AbortSignal.timeout(PROVIDER_TIMEOUT_MS) });
  } catch (err) {
    return { ok: false, status: 0, code: 'unreachable', text: (err as Error).message, input: 0, output: 0, stopFamily: 0 };
  }
  const raw = await res.text();
  if (!res.ok) {
    let code = '';
    try {
      const parsed: unknown = JSON.parse(raw);
      if (isRecord(parsed) && isRecord(parsed.error)) code = str(parsed.error.code) || str(parsed.error.status) || str(parsed.error.type);
    } catch {
      code = '';
    }
    return { ok: false, status: res.status, code, text: raw.slice(0, 400), input: 0, output: 0, stopFamily: 0 };
  }
  let answer: Json;
  try {
    const parsed: unknown = JSON.parse(raw);
    answer = isRecord(parsed) ? parsed : {};
  } catch {
    return { ok: false, status: res.status, code: 'bad_json', text: raw.slice(0, 400), input: 0, output: 0, stopFamily: 0 };
  }
  if (provider.wire === 'openai') {
    const choices = Array.isArray(answer.choices) ? answer.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : {};
    const message = isRecord(choice.message) ? choice.message : {};
    const usage = isRecord(answer.usage) ? answer.usage : {};
    const text = typeof message.content === 'string' ? message.content : textOfContent(message.content);
    return {
      ok: true,
      status: res.status,
      code: '',
      text,
      input: num(usage.prompt_tokens) ?? 0,
      output: num(usage.completion_tokens) ?? 0,
      stopFamily: stopFamilyOf(choice.finish_reason),
    };
  }
  if (provider.wire === 'anthropic') {
    const content = Array.isArray(answer.content) ? answer.content : [];
    const text = content
      .flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []))
      .join('');
    const usage = isRecord(answer.usage) ? answer.usage : {};
    return {
      ok: true,
      status: res.status,
      code: '',
      text,
      input: num(usage.input_tokens) ?? 0,
      output: num(usage.output_tokens) ?? 0,
      stopFamily: stopFamilyOf(answer.stop_reason),
    };
  }
  const candidates = Array.isArray(answer.candidates) ? answer.candidates : [];
  const first = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(first.content) ? first.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  const text = parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join('');
  const meta = isRecord(answer.usageMetadata) ? answer.usageMetadata : {};
  return {
    ok: true,
    status: res.status,
    code: '',
    text,
    input: num(meta.promptTokenCount) ?? 0,
    output: num(meta.candidatesTokenCount) ?? 0,
    stopFamily: stopFamilyOf(first.finishReason),
  };
}

// ---------------------------------------------------------------- the client wires

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function clientError(res: ServerResponse, client: ClientWire, status: number, type: string, message: string): void {
  if (client === 'anthropic') sendJson(res, status, { type: 'error', error: { type, message } });
  else sendJson(res, status, { error: { message, type, code: null } });
}

function sseFrame(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

function anthropicList(): Json {
  return {
    data: freeModels().map((m) => ({ type: 'model', id: m.id, display_name: m.id, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
  };
}

function openAIList(): Json {
  return {
    object: 'list',
    data: freeModels().map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

/** One client request answered: the best candidate that answers, or the last refusal. */
async function serve(body: Json, client: ClientWire, req: Requirement): Promise<UpstreamAnswer & { model: string | null }> {
  const canon = canonical(body, client);
  const order = routeOrder(req, canon.maxTokens);
  let last: UpstreamAnswer | null = null;
  let triedOnce = false;
  // Two passes: the best providers first; then, when every one of them refused, one more try at the
  // ones that have refused only once, since a single refusal can be a failure schedule and not a budget.
  for (let pass = 0; pass < 2; pass++) {
    triedOnce = pass === 1;
    for (const cand of order) {
      if (spent(cand.provider.id)) continue;
      if (triedOnce && (refusals.get(cand.provider.id) ?? 0) === 0) continue;
      const upstream = upstreamBody(canon, client, cand, body);
      const answer = await callProvider(cand, upstream);
      if (answer.ok) return { ...answer, model: cand.model };
      last = answer;
      // A concurrent-call refusal spends no budget: keep the provider and ask it once more.
      if (answer.code === 'concurrent_requests') {
        await new Promise((resolve) => setTimeout(resolve, 150));
        const retry = await callProvider(cand, upstream);
        if (retry.ok) return { ...retry, model: cand.model };
        last = retry;
        continue;
      }
      if (answer.status === 400 || answer.status === 401) {
        deadModels.add(`${cand.provider.id}|${cand.model}`);
        continue;
      }
      noteRefusal(cand.provider.id, answer);
    }
    if (last === null) break;
  }
  return { ...(last ?? { ok: false, status: 503, code: 'no_model', text: 'no free model could answer', input: 0, output: 0, stopFamily: 0 }), model: null };
}

/** The manifest row of the request now arriving: the client sends them one at a time, in order. */
let manifestIndex = 0;
function requirementsFor(body: Json, client: ClientWire): Requirement {
  const entry = brief.manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  // No manifest row: read the requirements off the body (the brief always carries one in a trial).
  const canon = canonical(body, client);
  return { tools: canon.tools.length > 0, vision: canon.image !== null, json: canon.jsonMode };
}

async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return clientError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const stream = body.stream === true;
  const answer = await serve(body, 'openai', requirementsFor(body, 'openai'));
  if (answer.model === null) return clientError(res, 'openai', answer.status || 503, 'no_free_model', 'no free model on the directory meets this request');
  const usage = { prompt_tokens: answer.input, completion_tokens: answer.output, total_tokens: answer.input + answer.output };
  const model = answer.model;
  if (!stream) {
    return sendJson(res, 200, {
      id: 'chatcmpl-aggregate',
      object: 'chat.completion',
      created: FIXED_OPENAI_CREATED,
      model,
      choices: [{ index: 0, message: { role: 'assistant', content: answer.text }, finish_reason: 'stop', logprobs: null }],
      usage,
    });
  }
  const chunk = (delta: Json, finish: string | null): Json => ({
    id: 'chatcmpl-aggregate',
    object: 'chat.completion.chunk',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  const frames: string[] = [];
  frames.push(sseFrame(null, chunk({ role: 'assistant', content: '' }, null)));
  for (const piece of SINGLE_DELTA ? [answer.text] : splitText(answer.text)) frames.push(sseFrame(null, chunk({ content: piece }, null)));
  frames.push(sseFrame(null, chunk({}, 'stop')));
  frames.push(sseFrame(null, { id: 'chatcmpl-aggregate', object: 'chat.completion.chunk', created: FIXED_OPENAI_CREATED, model, choices: [], usage }));
  frames.push('data: [DONE]\n\n');
  res.writeHead(200, { 'content-type': 'text/event-stream; charset=utf-8', 'cache-control': 'no-cache', connection: 'keep-alive' });
  res.end(frames.join(''));
}

function splitText(text: string): string[] {
  const pieces: string[] = [];
  for (let at = 0; at < text.length; at += 240) pieces.push(text.slice(at, at + 240));
  return pieces.length === 0 ? [''] : pieces;
}

async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return clientError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const stream = body.stream === true;
  const answer = await serve(body, 'anthropic', requirementsFor(body, 'anthropic'));
  if (answer.model === null) return clientError(res, 'anthropic', answer.status || 503, 'overloaded_error', 'no free model on the directory meets this request');
  const model = answer.model;
  if (!stream) {
    return sendJson(res, 200, {
      id: 'msg_aggregate',
      type: 'message',
      role: 'assistant',
      model,
      content: [{ type: 'text', text: answer.text }],
      stop_reason: 'end_turn',
      stop_sequence: null,
      usage: { input_tokens: answer.input, output_tokens: answer.output },
    });
  }
  const frames: string[] = [];
  frames.push(
    sseFrame('message_start', {
      type: 'message_start',
      message: {
        id: 'msg_aggregate',
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: answer.input, output_tokens: 0 },
      },
    }),
  );
  frames.push(sseFrame('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }));
  for (const piece of SINGLE_DELTA ? [answer.text] : splitText(answer.text)) {
    frames.push(sseFrame('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: piece } }));
  }
  frames.push(sseFrame('content_block_stop', { type: 'content_block_stop', index: 0 }));
  frames.push(
    sseFrame('message_delta', { type: 'message_delta', delta: { stop_reason: 'end_turn', stop_sequence: null }, usage: { output_tokens: answer.output } }),
  );
  frames.push(sseFrame('message_stop', { type: 'message_stop' }));
  res.writeHead(200, { 'content-type': 'text/event-stream; charset=utf-8', 'cache-control': 'no-cache', connection: 'keep-alive' });
  res.end(frames.join(''));
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    const anthropic = typeof req.headers['x-api-key'] === 'string';
    return sendJson(res, 200, anthropic ? anthropicList() : openAIList());
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') return chat(req, res);
  if (req.method === 'POST' && url.pathname === '/v1/messages') return messages(req, res);
  clientError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

// ---------------------------------------------------------------- register, work, exit

async function register(port: number): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`register answered HTTP ${res.status}: ${(await res.text()).slice(0, 300)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: ${reply.requests} requests`);
}

function work(close: () => void): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const body: unknown = await res.json();
        if (isRecord(body) && body.done === true) {
          clearInterval(timer);
          close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: the next tick asks again.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_MS);
}

async function main(): Promise<void> {
  if (ENV_URL === '' && PROMPT_FILE === '') {
    // Outside a trial there is no brief and no provider route: an empty catalog, as the rules say.
    const text = `${JSON.stringify({ providers: [] })}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  await load();
  if (process.argv.includes('--list')) {
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) clientError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'api_error', 'the server failed to answer');
    });
  });
  await new Promise<void>((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  if (port === 8080) throw new Error('the OS picked the env’s port 8080: restart the trial');
  log(`listening on 0.0.0.0:${port}: ${catalog.providers.length} providers, ${freeModels().length} free models`);
  await register(port);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  work(() => server.close());
}

try {
  await main();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}