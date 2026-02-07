# Compaction Strategy Plan (Internal)

## Goals

Provide first-class compaction strategies that can be configured in agent cards and
implemented via hook-based history manipulation. The initial set should be reliable,
well-tested, and easy to compose.

## Proposed strategies

### 1) Rolling window (turns)

Trim history to the last **N turns**. A turn starts at a user message that does not
contain `tool_results`.

- Source utility: `fast_agent.types.split_into_turns`
- Hook type: `after_turn_complete`
- Parameters: `turns: int`

### 2) Truncate over X

Trim history when context usage exceeds a threshold.

- Trigger: `usage.context_usage_percentage > threshold`
- Optional: `max_tokens` override when usage is unavailable
- Hook type: `after_turn_complete`
- Parameters: `threshold_percent: float`, `max_tokens: int | None`

### 3) Clear results

Strip tool-result payloads to reduce context size while preserving the flow.

Variants:

- **Soft**: remove `tool_results` content but keep tool call structure.
- **Hard**: remove intermediate tool results and tool calls (similar to the
  existing `trim_tool_loop_history`).

Hook type: `after_turn_complete`

### 4) Compaction prompt

Summarize history via a prompt and replace the message history with the summary.

Key steps:

1. Collect current history.
2. Run a compaction prompt (same agent or a dedicated compactor agent).
3. Replace history with summary + recent turn(s).

Hook type: `after_turn_complete`

## Hook requirements

- Use `ctx.usage` when available to make token-aware decisions.
- Use `ctx.load_message_history(...)` to replace history.
- Use `show_hook_message(...)` for user-visible compaction notices.

## Configuration approach

We plan to expose these strategies in agent cards via:

1) **Direct hook config** (short term):

```yaml
tool_hooks:
  after_turn_complete: my_hooks.py:rolling_window
```

2) **Dedicated compaction section** (longer term), mapped to hooks internally:

```yaml
compaction:
  strategy: rolling_window
  turns: 12
```

## Open design questions

- Token-based truncation: use usage stats when available, but add an internal
  fallback estimator if needed.
- Compaction prompt target: same agent vs a named agent; current hook context
  supports `ctx.get_agent(name)` for cross-agent access.
- Ordering/stacking strategies: determine whether multiple strategies can run
  in sequence or must be exclusive.
