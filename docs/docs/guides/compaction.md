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
   to a `compacted_*.json` file in the session directory, so nothing is lost.

If the summarization call fails or returns nothing, history is left untouched.

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
