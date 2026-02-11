# Plan: Capability-Removal Chip Semantics + Content Capture Controls

## Problem statement
When fast-agent removes unsupported content before an LLM call (for example: vision content for a text-only model), the toolbar currently often highlights the **T** chip red instead of **V**. This is misleading because the condition is a **handled capability downgrade**, not a generic text/error condition.

Example message:

> Removed unsupported content before sending to moonshotai/Kimi-K2-Instruct-0905: 1 vision block (image/webp). Missing capability: vision. Stored original content in 'fast-agent-error'.

Desired behavior:
- Red **V** chip for handled vision removal.
- Preserve red **T** fallback for true unclassified errors.

---

## Investigation summary

### Current flow
1. `LlmDecorator` sanitizes unsupported blocks and builds a `RemovedContentSummary`.
2. Detailed per-turn removal metadata is attached to:
   - `FAST_AGENT_REMOVED_METADATA_CHANNEL` (`fast-agent-removed-meta`)
3. Before persistence, `_strip_removed_metadata()` removes that channel.
4. Toolbar flag extraction (`_extract_alert_flags_from_meta`) reads persisted history.
5. No metadata remains, so toolbar falls back to:
   - if `FAST_AGENT_ERROR_CHANNEL` exists and no flags -> add `"T"`

### Root cause
The UI’s typed alert source (removed-meta) is ephemeral and stripped before history persistence, so the toolbar loses category context (`V`, `D`, `T`) and misclassifies handled removals as generic text/error.

---

## Design goals
1. Correct chip semantics for handled removals (**V/D/T by category**).
2. Keep explicit distinction between:
   - handled capability downgrade
   - generic execution/generation errors
3. Add binary content capture support under `.fast-agent/content`.
4. Expose runtime control with modes:
   - `none`
   - `auto`
   - `always`
5. Support on-the-fly control via slash commands (TUI + ACP parity).
6. Keep changes incremental and backward compatible.

---

## Proposed implementation plan

## Phase 1: Fix chip classification (small, focused)

### 1.1 Add persisted alert channel
Introduce a compact persisted channel for UI classification (separate from removed-meta):
- New constant (proposed): `FAST_AGENT_ALERT_CHANNEL = "fast-agent-alert"`

### 1.2 Write compact alert entries when removal occurs
In `LlmDecorator._sanitize_messages_for_llm` / summary assembly:
- add compact JSON block(s) to `fast-agent-alert`, e.g.:

```json
{
  "type": "unsupported_content_removed",
  "flags": ["V"],
  "categories": ["vision"],
  "handled": true
}
```

### 1.3 Preserve compact alert channel in history
- Keep stripping `fast-agent-removed-meta` (optional detailed channel),
- but do **not** strip `fast-agent-alert`.

### 1.4 Toolbar precedence update
In `enhanced_prompt.py` toolbar logic:
1. Read flags from `fast-agent-alert` first.
2. Fall back to old metadata if present (defensive/backward compatibility).
3. Only apply fallback `T` when:
   - error channel exists
   - and no structured alert flags were resolved.

### 1.5 Test coverage
Add unit tests to verify:
- vision removal => red **V**
- document removal => red **D**
- mixed removal => multiple red flags
- generic error without classification => fallback red **T**

---

## Phase 2: Content capture system (`none|auto|always`)

### 2.1 Add capture configuration
Add settings model (and optional agent-level override) for content capture:
- `mode: Literal["none", "auto", "always"]`
- `output_dir: str` default `.fast-agent/content`

Semantics:
- `none`: no capture to disk
- `auto`: capture binary only when content had to be removed for compatibility
- `always`: capture all eligible binary content encountered

### 2.2 Implement capture writer service
Create a focused utility/service:
- resolve environment content path under `.fast-agent/content`
- ensure dirs exist
- write binary files via atomic/safe naming
- derive extension from MIME when possible
- write sidecar metadata (`.json`) containing provenance:
  - timestamp
  - mime
  - source (`message`/`tool_result`)
  - tool_id
  - model name
  - mode
  - hash/size

### 2.3 Add channel references for traceability
When capture occurs:
- append lightweight reference block to a channel (e.g., alert/error channel), containing file id/path + metadata summary.
- avoid embedding large duplicated content.

### 2.4 Hook-friendly extension point
Provide optional extensibility without forcing custom hooks:
- built-in default capture pipeline
- optional hook function spec to customize naming/routing/retention
- keep default behavior deterministic if no hook configured

---

## Phase 3: Runtime slash command controls

## 3.1 TUI slash command
Add `/content` command family:
- `/content` -> show current mode and output dir
- `/content mode <none|auto|always>` -> set mode for current session/agent
- optional future: `/content dir <path>`

### 3.2 ACP slash command parity
Mirror behavior in ACP slash handler so remote clients can manage capture mode.

### 3.3 Command architecture
Follow existing `/model` pattern:
- payload type in `ui/command_payloads.py`
- parse in `ui/enhanced_prompt.py`
- execute in `commands/handlers/content.py`
- wire in `ui/interactive_prompt.py`
- wire in `acp/slash_commands.py`

---

## Phase 4: Documentation and migration

1. Document alert chip semantics:
   - handled removal flags vs generic errors
2. Document content capture configuration and modes.
3. Document `/content` slash commands (TUI + ACP).
4. Keep backward compatibility:
   - existing `fast-agent-error` behavior remains
   - `fast-agent-removed-meta` can remain internal/ephemeral

---

## Delivery strategy

### PR 1 (recommended first)
- Phase 1 only (chip correctness)
- Minimal risk, fast validation

### PR 2
- Phase 2 core capture service + config + tests

### PR 3
- Phase 3 slash commands + ACP parity + docs updates

This sequencing gives immediate UX improvement (correct V/D/T chips) while de-risking the broader capture feature rollout.
