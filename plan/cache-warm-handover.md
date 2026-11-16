# Prompt-cache keep-warm: handover

Written 16 November 2026 for continuing on a machine with screen capture.

## Where the code is

- Branch `cache-warm` on `origin` (evalstate/fast-agent), commit `27fe44327`.
  `git fetch origin && git switch cache-warm && uv sync`.
- The branch sits on top of local `dev/0.10.55`, which was 11 commits ahead of
  and 1 behind `origin/dev/0.10.55`. That divergence is unresolved; nothing was
  pushed to `dev/0.10.55`.
- The commit also contains the herdr-core child-attachment work
  (`integrations/hc_children.py` and related prompt/toolbar changes) that was
  uncommitted in the checkout. Idle warming depends on it.
- Format, lint, typecheck and the unit suite (10,933 tests) passed at that
  commit.
- User guide: `docs/docs/guides/cache_warming.md`.

## What was built

A provider drops an unused prompt cache after a retention period. While an agent
makes no requests of its own, fast-agent now sends a keep-warm request shortly
before the cache would lapse. History is never modified.

Triggers:

1. Automatically while a turn waits on tools or subagents (not managed-process
   polls, which already return to the model in time).
2. Automatically at the idle prompt while herdr-core children that will resume
   the session are running.
3. On request: `/warm [minutes|off|status]`, default 30 minutes.

Settings (`cache_warm`): `auto` (true), `max_minutes` (30), `interval_seconds`
(unset; overrides the schedule and is how to test quickly).

Main files:

| Area | File |
| --- | --- |
| Scheduler, status, formatting | `src/fast_agent/core/cache_warmer.py` |
| Retention facts, keep-warm message | `src/fast_agent/llm/prompt_cache.py` |
| LLM entry point (`warm_cache`) | `src/fast_agent/llm/fastagent_llm.py` |
| Anthropic `max_tokens: 0` replay | `src/fast_agent/llm/provider/anthropic/llm_anthropic.py` |
| Responses warmup, planner | `src/fast_agent/llm/provider/openai/responses.py`, `responses_websocket.py` |
| Copilot GPT route | `src/fast_agent/llm/provider/copilot/responses.py` |
| Tool-wait scope | `src/fast_agent/agents/tool_runner.py` (`_cache_kept_warm`) |
| Idle-prompt scope, expiry redraw | `src/fast_agent/ui/prompt/cache_warm.py` |
| Toolbar segment | `src/fast_agent/ui/prompt/status_bar/renderer.py` (`_build_warm_segment`) |
| `/warm` | `src/fast_agent/commands/handlers/warm.py` |

## How each provider route behaves

| Route | Keep-warm request | While tools run | Verified |
| --- | --- | --- | --- |
| Anthropic family, incl. Copilot Claude | last request replayed with `max_tokens: 0`; zero output | yes | yes: 99.9% read from cache, idle and during a subagent run |
| Copilot GPT-5.6+ | short "reply OK" message; a few output tokens | yes | yes: exact cache reads reported |
| Codex (websocket) | `generate: false` warmup chained from the latest response; zero output | no | request works; cache benefit unproven |
| Platform Responses (API key) | `generate: false` over websocket, `prompt_cache_options.prewarm` over SSE | websocket no, SSE yes | no: no API key was available |

Findings behind that table:

- **Copilot GPT follows the documented GPT-5.6+ rules.** In a 40-minute test,
  two unwarmed sessions read 2,716 of about 23,650 tokens from cache and rewrote
  the rest; three sessions with one or two keep-warm requests read all but about
  20 tokens. Copilot ignores both `generate: false` and `prewarm` and generates
  anyway, hence the "reply OK" form.
- **Codex differs from the docs.** A new websocket connection was a total cache
  miss every time. Unwarmed sessions still hit the cache at 36 and 50 minutes on
  a live connection. Across two runs, 3 of 3 unwarmed sessions hit and 3 of 5
  warmed sessions hit (one miss after a reconnect, one unexplained). Codex
  rejects `stream_id` and `prompt_cache_options`, and reports zero cached tokens
  for a warmup, so warmups cannot be verified from their own usage.
- **Codex retention is therefore recorded as 50 minutes** (observed, not
  documented). With the default 30-minute window, Codex sends no keep-warm
  requests at all; they start only if `/warm` or `max_minutes` exceeds 45
  minutes.
- **Why Codex cannot warm mid-turn:** over websocket the provider continues only
  from its latest response. A warmup is accepted while tool calls are pending,
  but the tool results then cannot follow it, and a full re-send is a cache miss.
  The documented `stream_id` lanes would solve this on the platform API.

## What to look at on screen

These are the visible behaviours worth capturing. Use
`CACHE_WARM__INTERVAL_SECONDS=20` to make requests frequent.

Bottom toolbar, right-hand end:

| State | Text | Colour |
| --- | --- | --- |
| Nothing warming, cache believed live | `cache until 15:56` | green |
| Retention passed | `cache expired` | yellow |
| Warming, verified reads | `warm ✓ 2/2 98% next 09:30 until 09:59` | green |
| Warming on Codex (unverifiable) | `warm ✓ 1/1 next 10:32 until 10:32` | green |
| Last request read little from cache | `warm ✗ MISS 0/1 0% next … until …` | red |
| A request failed; warming stopped | `warm FAILED 09:31:10 · cache …` | red |
| Model with unknown retention | no segment | |

Scrollback, one dim line per request:

```text
cache warm 09:38:32 tools: HIT - 6,083 of 6,087 prompt tokens read from cache (99.9%) · 1.0s
cache warm 10:31:31 /warm: SENT - cache usage not reported · 0.5s
cache warm (hc): no further requests before the window ends at 09:59:10
```

Other places:

- A `Warming cache` row in the progress display while a request is in flight
  during a turn.
- The herdr terminal status message (`hc get <t>`, field `status.msg`) carries
  the same toolbar text.
- `/warm status` lists settings, expected cache lifetime and recent requests.

Suggested captures:

1. Copilot Haiku: send one message, leave idle; `cache until HH:MM` flips to
   `cache expired` after five minutes.
2. Copilot Haiku with the 20-second interval: `/warm 2`; HIT lines and the
   green `warm ✓` segment; then send a message while a request is in flight.
3. Copilot Haiku: ask for a subagent that runs `sleep 70`; `tools:` HIT lines
   appear mid-turn alongside the subagent table.
4. Codex: the same subagent request shows no warming; idle shows
   `cache until` about 50 minutes ahead.
5. Inside herdr-core: start `hc run --bg --name napper sleep 100` and end the
   turn; the toolbar shows `hc: waiting on 1: napper` next to the warm segment.

Known display issue: on Codex the green tick and `1/1` only mean the request was
sent and accepted. Proposed fix, not done: a neutral mark such as
`warm • sent 1 next 10:32`.

## Other changes in the same commit

- **Managed-process wait default is 300 seconds for GPT-5.6 and later**
  (`gpt-5.6*`, `gpt-6*`); GPT-5.5 and earlier stay at 240. A scan of 24,220
  bench-run trajectories showed models do use the wake-up to stop stuck or
  abandoned processes (759 processes in 642 trials, stopped a median of 7
  seconds after the wake). A 300-second wait pushed no trial past its deadline;
  900 seconds plausibly would have on seven TBLite trials with 10 to 15 minute
  deadlines. Pure "still running, wait again" polls were only 0.31% of model
  turns, so the saving from longer waits is small.
- **openai SDK 3.40.0 to 3.51.0.** No regressions in the unit suite. No large
  simplification follows from the upgrade; a possible follow-up is replacing the
  hand-written websocket message bridge with the SDK's typed events (roughly
  60 to 130 lines).

## Not done or untested

- Platform Responses route (needs an OpenAI API key): `prewarm`, `generate:
  false`, and `stream_id` forking for mid-turn warming.
- Warming at real intervals on large conversations; all live tests used 5k to
  24k token contexts.
- Keep-warm requests are counted in live usage but not persisted, so a resumed
  session under-reports them. They are also mixed into the per-turn `Last:`
  line rather than shown separately.
- A subagent launched with a Codex model override from one long-running parent
  session failed with "Provider 'codexresponses' does not support max_tokens".
  It worked from a fresh Haiku parent. Cause unknown; not investigated.
- Clock-change robustness in the scheduler (it uses wall-clock time).
- On `gpt-6-astra`, changing reasoning effort while idle and reverting it before
  the next message costs one full re-send.

## Open decisions

1. Whether to show the neutral, unverified mark for Codex warmups.
2. Whether keep-warm usage should get its own line and be persisted.
3. Whether the per-request scrollback line stays always-on or is limited to
   misses and failures after the pre-release period.
4. Whether coordinators need a longer automatic window than 30 minutes.
5. How to reconcile local `dev/0.10.55` with `origin/dev/0.10.55` before merging
   `cache-warm`.

## herdr-core integration: where it stands and what could improve it

Today fast-agent learns about hc children only by observation. The model types
`hc run --bg …` as shell text; fast-agent polls hc for the terminal's children
at prompt and turn boundaries and shows `hc: waiting on N`. Checked against
hc 0.1.18:

- Children belong to the hc terminal, not the fast-agent session. They end with
  the parent's program unless started with `--outlive-parent`. Resuming in a
  different terminal does not find them, and the API has no way to re-parent.
  Resuming in the same terminal with `--outlive-parent` should work but was not
  tested.
- hc has no usage or cost fields; only a free-text status message. Child cost
  cannot be added reliably today.
- hc does not ship an MCP server. It exposes an OpenRPC API (`hc schema`,
  `hc connect`) with about 30 methods, and an event stream (`hc events`).

Opportunities, smallest first:

1. Record child addresses in the session and label children with the session
   id, so a resumed session can find them. Prerequisite for most of the rest.
2. Use the hc event stream so the toolbar updates live and silently exited
   children stop showing as running.
3. On wake-up, show which child reported and how long it ran.
4. Show children in the progress display like the subagent table, using hc's
   per-terminal status.
5. Roll up cost for fast-agent children by passing a session id the parent can
   read back.
6. Surface `blocked` children in the parent's toolbar and status.
7. A thin first-class tool (spawn, send, wait, list) over hc's API, or an MCP
   adapter. Makes the above reliable but adds a surface to maintain against a
   fast-moving CLI.
8. A child-wait tool that returns to the model on a timer and is folded like
   process polls: the route to keeping a Codex coordinator's cache warm inside
   a turn.

Suggested starting point if this is picked up: items 1 and 2.

## Quick test recipes

```bash
# Idle warming with frequent requests
CACHE_WARM__INTERVAL_SECONDS=20 uv run fast-agent go --model copilot.claude-haiku-5.5
#   then: send a message, /warm 2, /warm status

# Codex: toolbar shows "cache until" about 50 minutes ahead; no requests by default
uv run fast-agent go --model codexresponses.gpt-6.1-terra

# Copilot GPT: verified keep-warm, including during tool calls
CACHE_WARM__INTERVAL_SECONDS=20 uv run fast-agent go --model copilot.gpt-6.1-terra

# See websocket request planning (create vs continuation) inline
FAST_AGENT_DEBUG_RESPONSES_WS=1 uv run fast-agent go --model codexresponses.gpt-6.1-terra

# Unit tests for the feature
uv run pytest tests/unit/fast_agent/core/test_cache_warmer.py \
  tests/unit/fast_agent/llm/test_warm_cache.py \
  tests/unit/fast_agent/llm/providers/test_responses_websocket.py -q
```

The experiment scripts and raw reports from this session lived under `/tmp` on
the original machine and are not in the repository; the numbers above are the
ones that matter.
