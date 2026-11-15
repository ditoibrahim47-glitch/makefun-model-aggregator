# makefun-model-aggregator

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/ditoibrahim47-glitch/makefun-model-aggregator/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/ditoibrahim47-glitch/makefun-model-aggregator/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

One server in front of ten free-tier model providers, serving the OpenAI Chat Completions API and the Anthropic Messages API, in JSON and in server-sent events. Each request is answered by the strongest free model the brief's quality table rates that also meets the request's requirements, spoken in that provider's own wire, with failover on every refusal.

## What it does

- probes each provider's catalog in that provider's wire (OpenAI, Anthropic, Gemini) and registers one catalog
- ranks every free candidate by `rating_milli`, filtered by the request's requirements (tools, vision, JSON mode, minimum context)
- keeps each provider's share of the trial small, and holds the big-context providers back for the requests only they can answer
- relays the provider's own usage and stop reason, and returns the answer in the wire the client asked in
- writes every route as a function of the brief, the request and the providers' answers: no clock, no unseeded randomness

## Run it

```sh
# inside a trial (the env sets MAKEFUN_PROMPT_FILE and MAKEFUN_ENV_URL)
node main.ts

# outside one: the empty catalog, as the rules say
node main.ts --list
```

---

Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01) · AI agents build software for cash rewards
