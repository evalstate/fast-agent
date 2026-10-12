---
title: Compaction
description: Keep long conversations within the context window by summarizing older turns.
social:
  title: Compaction
  tagline: Summarize older turns to stay within the context window.
  description: Keep long conversations within the context window by summarizing older turns.
  alt: fast-agent social card — Compaction
---

# Compaction

As a conversation grows, its history eventually approaches the model's context
window. **Compaction** keeps a session going by replacing older turns with a
single *checkpoint summary* produced by the agent's own model, while keeping the
most recent turns verbatim. The work done so far is preserved as a concise
handoff; the tokens it used are freed.

Compaction is built in. It runs automatically when context fills up, and you can
also trigger it on demand from the TUI or the [Harness API](../agents/defining/harness-api.md).

For OpenAI and Codex Responses, see [prompt caching and Astra effort changes](responses-caching.md)
for cache behavior across compaction. Fast-agent's summary compaction is distinct
from OpenAI's native compaction endpoints.

## How it works

A compaction does three things:

1. **Plans retention.** Leading template messages (system-prompt-like content)
   are always kept. The most recent `keep_turns` turns are kept verbatim. The
   turns in between are the *compact region*. The boundary always falls on a user
   turn, so an assistant tool call is never separated from its tool result.
2. **Summarizes.** The compact region is sent to the agent's model with a
   summarization prompt asking for a handoff summary for "another LLM that will
   resume the work." This is a side-channel call — it does not add the request to
   your history, and uses no tools.
3. **Replaces.** History becomes `templates + summary + recent turns`. The
   summary is a clearly-marked message (it shows as `compacted` in `/history`),
   not an ordinary user message. The original pre-compaction history is archived
   to a `compacted_*.json` file in the session directory when session history
   persistence is enabled.

If the summarization call fails or returns nothing, history is left untouched.
With session history enabled, an archive failure also leaves history unchanged.
Archives are published atomically under collision-resistant names; each new
summary records the archive filename, SHA-256 digest, and retention boundaries.
Prior summaries in that archive link to earlier archives.

## Automatic compaction

By default (`compaction.auto: true`), fast-agent checks context usage after each
completed turn and compacts when usage crosses `compaction.threshold` (see the
generated default below). The trigger uses **server-observed** token usage from
the last response, not an estimate, so it reflects what the provider actually
charged.

This applies everywhere agents run: the TUI, `fast.run()`, and the Harness API.
There is nothing to enable.

--8<-- "_generated/compaction_config_snippet.md"

To turn auto-compaction off and rely solely on manual `/compact`, set
`compaction.auto: false`.

`compaction.strategy` chooses how the summary is produced. With `auto`, a
provider that compacts natively can take over: if it compacts within normal
requests, fast-agent's own auto-compaction stands aside so the two never compete.
Set `strategy: client` to always use fast-agent's summary. Native provider
compaction is being introduced provider by provider:

- Claude models on [GitHub Copilot](../models/providers/copilot.md#compaction)
  compact within requests when the context window is known. A custom
  `compaction.prompt` selects fast-agent's summary instead.
- OpenAI GPT models on Codex Responses, Responses (API key), and GitHub Copilot use
  [OpenAI provider-native checkpoints](#openai-provider-native-checkpoints).

Elsewhere both settings behave the same.

### OpenAI provider-native checkpoints

On by default for OpenAI GPT models on Codex Responses (`codexresponses.*`),
Responses (`responses.*`, API key), and GitHub Copilot (`copilot.gpt-*`). fast-agent
still decides when to compact and what to keep, but the compacted region is
replaced by an opaque, encrypted checkpoint that OpenAI produces (a request ending
in a `compaction_trigger` item), instead of a written summary.

Choose per model with the model string, or per route in config:

```bash
fast-agent go --model 'sol?compaction=client'                # fast-agent's summary
fast-agent go --model 'copilot.gpt-6-sol?compaction=native'  # provider-native
```

```yaml
codexresponses:      # ChatGPT login (codexresponses.*)
  native_compaction: false
responses:           # OpenAI API key (responses.*)
  native_compaction: false
copilot:             # GitHub Copilot GPT models (copilot.gpt-*)
  native_compaction: false
```

Precedence: `compaction.strategy: client` always uses fast-agent's summary; otherwise
the model string's `compaction=native|client` wins over the route's
`native_compaction` setting, which defaults to `true`. `compaction=native` is rejected
for providers without native compaction. On Copilot Claude models, `compaction=client`
turns off server-side threshold compaction in favour of fast-agent's summary.

- Templates and the most recent `keep_turns` turns stay verbatim around the
  checkpoint, exactly as with a summary. Archiving, `/compact`, `/compact preview`,
  mid-turn compaction, and session save/resume work the same way.
- The checkpoint is not human-readable. `/history` shows its strategy, encrypted
  size, and archive file; the full prior history remains in that archive. The
  compaction result shows `—` for the size afterwards, because an opaque
  checkpoint's token cost is only known once the next request reports usage.
- How much detail survives depends on the model: in testing, `gpt-6-astra` and
  `gpt-6-sol` recalled a detail from a large tool result after compaction, while
  `gpt-6-luna` sometimes did not.
- A native checkpoint takes no prompt. A custom `compaction.prompt` or
  `/compact <instructions>` asks for a written summary, so either uses
  fast-agent's summary instead; `/compact preview` shows which applies.
- For `gpt-6-astra`, fast-agent re-sends the desired reasoning effort as a
  `configuration_update` before the next user message, as OpenAI requires after
  compaction.
- Only OpenAI can read the checkpoint. Switching this conversation to another
  provider leaves it with the placeholder text; restore the archive if needed.
- Checkpoints from `codexresponses` and `responses` can be read on all three routes.
  Copilot's checkpoints can only be read on Copilot: switching a Copilot-compacted
  conversation to `codexresponses` or `responses` stops with an explanation; switch back
  to a Copilot model or restore the archive with `/history load <archive>`.

## Tuning the trigger

The point at which compaction kicks in is `compaction.threshold`. Lower it to
compact earlier (more headroom, more frequent summarization); raise it to let
context fill closer to the limit before compacting.

--8<-- "_generated/compaction_settings_reference.md"

## The summarization prompt

The built-in prompt asks the model for a structured handoff summary covering
goals, decisions, progress, what remains, and any critical data needed to
continue. You can see the exact prompt in use at any time:

```
/compact prompt
```

To customize it, set `compaction.prompt` to either inline text or a path to a
text/markdown file. Relative file paths resolve from the directory of the loaded
config file (not the process working directory):

```yaml
compaction:
  prompt: ./prompts/compaction.md
```

## Manual compaction in the TUI

| Command | Effect |
|---|---|
| `/compact` | Compact now, showing the before → after context usage. |
| `/compact <instructions>` | Steer the summary, e.g. `/compact focus on the database migration`. |
| `/compact preview` | Show what would be kept and dropped — no model call. |
| `/compact prompt` | Print the active summarization prompt. |

`/compact preview` is free: it reports which turns would be summarized and an
estimated before → after token count without calling the model. Use it to decide
whether compaction is worthwhile before paying for it.

While `/compact` runs, the streaming token progress display is shown — the same
live indicator as a normal turn — so you can watch the summary being generated.

After compaction, the before/after context window is visualized, and the summary
remains in history. Because the summary is an ordinary part of the conversation,
you can correct or extend it with your next message if the model missed
something.

## How `/history` behaves after compaction

The checkpoint summary appears in `/history` as a `compacted` row (marked with
`≡`), distinct from user and assistant turns, with a short preview of the summary
content. Turns that were folded into the summary no longer appear individually —
they live in the `compacted_*.json` archive instead. The recent turns kept
verbatim continue to display normally.

To recover the full pre-compaction transcript, load the archive:

```
/history load compacted_20260613-120000_default.json
```

## ATIF export after compaction

Live ATIF output and persisted session ATIF export reconstruct archived history
through the same verifier. Original messages and tool results appear once, in
their original positions. Each summary becomes an ATIF `context_management`
boundary (`type: "compaction"`, `boundary: "replace"`) following the ATIF v1.7
convention:

- the boundary step's `observation` holds the summary the model saw;
- the system prompt, templates, and retained recent turns that stayed in the
  model's context follow the boundary as `is_copied_context` steps, which carry
  no usage and are excluded from SFT by spec-following consumers;
- `extra.context_management` lists `removed_step_ids`, `retained_step_ids`, and
  `copied_step_ids`, and each copy records `copied_from_step_id`;
- the summarization model call is embedded in `subagent_trajectories`, referenced
  from the boundary observation, and included in final metrics.

A consumer applying the spec's replace-boundary rule therefore reconstructs the
same context fast-agent sent to the model.

`context_management.strategy` is `summary` for fast-agent's own summary. A
provider-native checkpoint records its strategy (`openai_compaction`): its
observation describes the opaque checkpoint instead of quoting a summary, the
embedded call is the provider's compaction request, and
`final_metrics.extra.provider_checkpoint_compactions` counts these boundaries. Process-poll fold audits are still
expanded, including when reconciling a raw transient final turn.

Recovery follows linked archives and verifies exact template and retained-tail
sequences; it does not concatenate current/previous snapshots or globally
deduplicate messages. Legacy unlinked archives require a unique exact positive
tail overlap, timestamped originals, and an archive filename timestamp within
five seconds before the summary. Missing, malformed, conflicting, or ambiguous
evidence fails the full export rather than silently producing a partial trace.
This includes legacy summary-only histories and retained tails whose exact
overlap can no longer be verified.

Disabling session history still permits ordinary compaction, but a full ATIF
export cannot recover discarded history. Persisted recovery covers saved
evidence only, not unsaved messages lost on hard termination. It does not merge
existing ATIF files: preserve those separately when they contain extra transient
evidence. Checkpoints written before summary calls were recorded have no
summary usage: their boundaries report `summary_usage: "unavailable"` and
final metrics mark the accounting as excluding those calls.

## Manual compaction from the Harness API

Under the Harness API, auto-compaction is on by default just as in the TUI. To
compact a session explicitly:

```python
result = await session.compact()
print(f"{result.messages_before} → {result.messages_after} messages")
```

`compact()` accepts optional `instructions` (one-off focus for the summary) and
`agent_name` (target a specific agent in the session). It returns a
`CompactionResult` and raises `CompactionSkipped` when there is nothing worth
compacting. See the [Harness API](../agents/defining/harness-api.md#compacting-history)
guide for details.

For building custom triggers or tooling, the primitives in
`fast_agent.history.compaction` are importable directly — including
`plan_compaction` for a model-call-free retention preview and `should_auto_compact`
for the threshold check.

## What is kept, summarized, and archived

| | Kept verbatim | Summarized | Archived |
|---|:---:|:---:|:---:|
| Leading template / system messages | ✅ | | |
| Older turns (the compact region) | | ✅ | ✅ |
| Most recent `keep_turns` turns | ✅ | | |
| Full pre-compaction history | | | ✅ |

The summary itself is retained in the live history as the checkpoint. Tool calls
and their results are summarized together (never split). A single very large turn
is always eligible for compaction, even when `keep_turns` would otherwise retain
it, so a runaway tool loop can still be brought back under the limit.
