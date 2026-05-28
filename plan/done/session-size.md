---
title: "Session Size Indicator in /session list"
status: draft
---

# Session Size Indicator in `/session list`

## Goal

Add a compact visual size indicator to session list output (interactive + ACP markdown),
similar in spirit to history density shading.

Example target UX:

- small session: `░`
- medium session: `▒`
- large session: `▓`
- very large session: `█`

This should make large sessions obvious at a glance without opening each one.

---

## Current behavior and constraints

1. Session metadata is saved to `session.json` via `Session._save_metadata()`.
2. Session history autosave runs from the `after_turn_complete` hook (`save_session_history`).
3. `/session list` rendering paths:
   - interactive/TUI: `commands/handlers/sessions.py::_build_session_entries(...)`
   - ACP markdown: `commands/renderers/session_markdown.py`, fed by
     `session/formatting.py::format_session_entries(...)`
4. Current list rows already include timestamp, pin state, agent count, summary.

Design implication: compute/store a lightweight turn count during autosave and read it from
`session.json` during listing (no history file parsing during list).

---

## Proposed metadata additions

Add optional metadata keys in `session.json`:

- `turn_count_by_agent: dict[str, int]`
- `turn_count_total: int`

Notes:

- `turn_count_total` is the value used for list glyphs.
- Missing keys are valid for old sessions (fallback indicator).

---

## Turn counting rule

Use the same turn boundary semantics as conversation summaries:

- a turn starts on a `user` message with no `tool_results`

Implementation should use `split_into_turns(...)` from
`fast_agent.types.conversation_summary` for consistency.

---

## Indicator mapping (initial thresholds)

Define a small helper (in `session/formatting.py` or nearby) that maps `turn_count_total`
to a single character.

Initial mapping proposal:

- missing/unknown: `·`
- `0`: `·`
- `1-4`: `░`
- `5-14`: `▒`
- `15-39`: `▓`
- `40+`: `█`

Thresholds can be tuned later based on real usage.

---

## Implementation plan

## 1) Persist turn counts during save

Files:

- `src/fast_agent/session/session_manager.py`

Changes:

1. In `Session.save_history(...)` after saving history file and resolving `agent_name`:
   - compute `agent_turn_count` from `agent.message_history`
   - update `metadata["turn_count_by_agent"][agent_name]`
   - compute/update `metadata["turn_count_total"]` as sum of per-agent counts
2. Keep behavior safe when agent has no name or malformed metadata.
3. Ensure this remains part of existing `_save_metadata()` call path.

## 2) Preserve counts on fork

Files:

- `src/fast_agent/session/session_manager.py`

Changes:

1. In `fork_current_session(...)`, copy over `turn_count_by_agent` and `turn_count_total`
   (or recompute from copied map if safer).

## 3) Thread counts into list summaries

Files:

- `src/fast_agent/session/formatting.py`

Changes:

1. Extend `SessionEntrySummary` with optional `turn_count` and `size_indicator`.
2. In `build_session_entry_summaries(...)`, read `turn_count_total` from metadata.
3. Compute `size_indicator` from helper mapping.
4. For old sessions without metadata, default to unknown indicator.

## 4) Render indicator in interactive list

Files:

- `src/fast_agent/commands/handlers/sessions.py`

Changes:

1. In `_build_session_entries(...)`, add `entry.size_indicator` before session name,
   dim-styled so it is informative but not noisy.

## 5) Render indicator in compact/markdown list

Files:

- `src/fast_agent/session/formatting.py`

Changes:

1. In `format_session_entries(..., mode="compact")`, prepend indicator to each line
   so ACP markdown rendering automatically includes it.

No direct change should be required in `session_markdown.py` if compact lines already
contain the indicator.

---

## Backward compatibility

1. Existing sessions without new metadata continue to list correctly.
2. Unknown/missing count shows neutral marker (`·`).
3. No migration step required.

---

## Testing plan

## Unit tests

1. `tests/unit/fast_agent/session/test_session_formatting.py`
   - add/adjust assertions for indicator presence in compact/verbose outputs
   - include edge cases around threshold boundaries
2. Add/extend `session_manager` unit test(s)
   - verify `turn_count_by_agent` and `turn_count_total` are written on save
   - verify malformed metadata is handled robustly

## Integration tests

1. `tests/integration/sessions/test_sessions.py`
   - assert new metadata keys appear in `session.json` after autosave
2. ACP session list integration test
   - ensure response contains indicator in list lines

---

## Risks / trade-offs

1. **Accuracy vs cost:** storing turn counts is fast and cheap; parsing all histories on list
   would be more accurate for legacy edge cases but slower.
2. **Definition drift:** if turn definition changes elsewhere, this feature must stay aligned
   by relying on shared split helper.
3. **Visual noise:** indicators should be dim and stable; avoid overloading line with too many
   symbols.

---

## Rollout sequence

1. Land metadata persistence + tests.
2. Land formatting/rendering + tests.
3. Validate in both interactive prompt and ACP `/session list`.
4. Tune thresholds if needed after initial usage.

