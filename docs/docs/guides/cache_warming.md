---
title: Prompt-cache warming
description: Keep provider prompt caches warm during tool waits and idle pauses.
social:
  title: Prompt-cache warming
  tagline: Keep prompt caches warm during pauses.
  description: Configure automatic and on-demand keep-warm requests, costs, and diagnostics.
  alt: fast-agent social card — Prompt-cache warming
---

# Prompt-cache warming

Providers drop unused prompt-cache entries after a retention period. A longer
pause makes the next request pay full input price again. fast-agent can send a
**keep-warm request** shortly before the cache would lapse: it re-sends the
conversation, discards the reply, and adds nothing to fast-agent's message history.

[Managed shell process waits](../ref/shell_runtime_budgeting.md) already return
to the model in time through bounded waits and poll folding. Keep-warm requests
cover the other pauses.

## When it runs

- **Tool waits:** automatically while a turn waits on subagents, slow MCP tools,
  or permission/elicitation prompts. Managed-process polls are excluded because
  they already refresh the cache.
- **Waiting for hc children:** automatically at the idle interactive prompt while
  herdr-core (`hc`) children that will resume this session are still running.
  The toolbar shows `hc: waiting on N`.
- **On request:** `/warm` keeps the cache warm while the interactive prompt is
  idle, even without children running.

Automatic warming runs for `cache_warm.max_minutes` after the agent's last own
model request, then stops. A keep-warm request does **not** extend that window.
Submitting any input or finishing tool calls leaves the warming scope; if a
request is in flight, fast-agent waits for it rather than cancelling it.
Dropping the provider connection would lose the cache.

## `/warm` commands

Commands apply to the active agent.

| Command | Effect |
| --- | --- |
| `/warm` | Warm at the idle prompt for `cache_warm.max_minutes` (default 30 minutes). |
| `/warm <minutes>` | Set an idle-prompt warming window from now, e.g. `/warm 60`. |
| `/warm ping` | Send one keep-warm request immediately, report hit/miss, and refresh the cache window. |
| `/warm off` | Stop the explicit idle-prompt window. Automatic warming remains controlled by `cache_warm.auto`. |
| `/warm status` | Show settings, schedule, and recent keep-warm requests. |

`/warm` prints the request interval and a cost estimate before you leave the
prompt idle. It does not run a request immediately unless one is already due.

## Settings

Set these in `fast-agent.yaml`. Defaults and descriptions below come from
`CacheWarmSettings`:

--8<-- "_generated/cache_warm_config_snippet.md"

--8<-- "_generated/cache_warm_settings_reference.md"

Set `auto: false` to disable both automatic triggers; `/warm` still works.
`max_minutes` accepts 1–1,440 minutes; `interval_seconds` must be at least 10.

Without an override, the interval is retention minus a margin:

| Cache retention | Keep-warm interval |
| --- | --- |
| 5 minutes | 4 minutes 15 seconds |
| 30 minutes | 27 minutes |
| 1 hour | 54 minutes |

Use `interval_seconds` to override this schedule, including for a model whose
retention fast-agent does not know. Choose an interval shorter than the
provider's retention; this setting cannot add caching support to a provider.
Unknown-retention models are not warmed by default.

No request is sent if the next scheduled request falls outside the warming
window. For example, a 1-hour cache with the default 30-minute window usually
needs no keep-warm request; use a longer window for longer pauses.

## Provider behaviour

| Provider / transport | Retention used for scheduling | Keep-warm request | While tools run? |
| --- | --- | --- | --- |
| Anthropic family, including Copilot Claude | 5 minutes; 1 hour with `anthropic.cache_ttl: 1h` | Replay the last request exactly with `max_tokens: 0`: cache read, zero output tokens. | Yes |
| Codex (websocket) | 50 minutes for GPT-5.6 and later, as observed on a live connection | A `generate: false` warmup chained from the latest response: no model output. | No; idle-prompt warming only |
| OpenAI Responses, websocket | At least 30 minutes for GPT-5.6 and later | A `generate: false` warmup chained from the latest response: no model output. | No; idle-prompt warming only |
| OpenAI Responses, SSE | At least 30 minutes for GPT-5.6 and later | The conversation re-sent with `prompt_cache_options.prewarm`: no model output. | Yes |
| Copilot GPT-5.6 and later | At least 30 minutes | A short automated message whose reply is discarded; cache reads are reported exactly. | Yes |
| Copilot Gemini 4 | Observed implicit-cache floors (`neon`: 30m, `hydrogen`/`carbon`: 15m, `argon`: 4m) | A short automated message whose reply is discarded; cache reads are reported exactly. | Yes |
| Google Gemini (2.5 and later) | Observed implicit-cache floor (4 minutes; pings every 3m 15s) | A short automated message whose reply is discarded; cache reads are reported via `cached_content_token_count`. | Yes |
| Other providers or models with unknown retention | Unknown; requires `interval_seconds` | A short automated message whose reply is discarded. | Provider-dependent |

Over websocket, OpenAI continues only from its latest response. A warmup becomes
that latest response, and the next real request continues from it; nothing is
added to the conversation on either side. While tool calls are running the tool
results cannot follow a warmup, so websocket sessions are not warmed mid-turn.
The Codex backend does not accept `prompt_cache_options`, so Codex sessions are
warmed over websocket only. Codex keeps a conversation cached for as long as its
websocket connection lives, which outlasts the default 30-minute window, so by
default a Codex session needs no keep-warm requests at all; they start only when
`/warm` or `max_minutes` asks for longer than 45 minutes. Warmups report no cached-token count, so their
scrollback line shows `cache usage not reported`.

Anthropic's zero-output form is unavailable with budgeted extended thinking
(`thinking.type: enabled`, used by older models). The request is reported as
failed and warming stops. Use `anthropic.cache_ttl: 1h` for those models instead.

## Diagnostics

Every keep-warm request prints a dim scrollback line, for example:

```text
cache warm 09:38:32 tools: HIT - 6,083 of 6,087 prompt tokens read from cache (99.9%) · 1.0s
```

The reason is `tools`, `hc`, or `/warm`. Outcomes are:

- **HIT:** at least half the prompt was read from cache.
- **MISS:** less than half was read from cache, or cache usage was not reported.
  The cache may be cold, or the prompt below the provider's minimum cacheable size.
- **FAILED:** the request failed; warming stops rather than repeatedly retrying.

When nothing is being warmed, the bottom toolbar shows how long the cache is
expected to last after the last model request, and warns once that time has
passed:

```text
cache until 15:56
cache expired
```

The next request after `cache expired` pays full input price again. While
warming is active the toolbar shows a summary such as:

```text
warm ✓ 3/3 97% next 09:22 until 09:23
```

This means three hits out of three recorded requests, with 97% of the last
prompt read from cache. It is green on a hit and red on a miss or failure.
Once the window ends it returns to `cache until HH:MM`. The same
summary appears in the herdr-core terminal status message. During a turn, a
`Warming cache` progress row appears while a request is in flight.

Use `/warm status` to inspect recent requests, including token counts, duration,
and cost when reported.

## Cost and limitations

Each keep-warm request is billed as a cache read of the whole conversation,
about **0.1× the input price** on most models, plus any output tokens. It is
recorded in usage totals like any other request, including the per-turn `Last:`
line and `/usage`. Warming is not free and does not guarantee a cache hit.

For a 5-minute cache, about **50 minutes of warming costs as much as one cold
rewrite**. For longer waits, Anthropic's 1-hour TTL is usually cheaper: one
request roughly every 54 minutes, but cache writes cost 2× input price instead
of 1.25× for the 5-minute TTL. `/warm` prints this guidance for long windows.

Warming needs a conversation to re-send: nothing is sent before the agent's
first model request, after `/history clear` (which also ends a `/warm` window),
or for agents that keep no history. If a model answers a keep-warm request with
a tool call, the request counts as failed and warming stops.

Warming only runs while fast-agent is running and inside an eligible tool-wait
or idle-prompt scope. It cannot recover an expired cache, make a short prompt
cacheable, or bypass provider caching requirements. A request already in flight
may briefly delay submitting input or resuming after tools finish.
