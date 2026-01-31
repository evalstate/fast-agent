# Updated Hook Utilities (Internal)

## Overview

The hook system now includes:

- **A3-style hook messaging** helpers with bright prefix coloration and hook name display.
- **Hook failure banners** (bright red) with details in logs.
- **Cross-agent lookup** from hooks via a simple string → `AgentProtocol` lookup.
- **Hook smoke-test harness** for running hooks against recorded histories.

These utilities are intended for both production diagnostics and safe, repeatable testing.

## Hook UI helpers

### `show_hook_message`

```python
from fast_agent.hooks import show_hook_message

async def before_llm_call(ctx: HookContext) -> None:
    show_hook_message(
        ctx,
        "trimmed 6 messages",
        hook_name="before_llm_call",
        hook_kind="tool",
    )
```

Behavior:

- Uses **A3 style** by default (prefix `▎•`).
- The prefix inherits the **bright hook color** (yellow for normal hooks).
- Hook name is rendered **dim**.
- Multiline content is allowed; only the **first line** shows the prefix, later lines are indented.
- Accepts `str` or `rich.text.Text` for richer formatting.

### `show_hook_failure`

```python
from fast_agent.hooks import show_hook_failure

try:
    ...
except Exception as exc:
    show_hook_failure(ctx, hook_name="after_turn_complete", hook_kind="tool", error=exc)
    raise
```

Behavior:

- Shows a **bright red** hook failure banner using A3 styling.
- Includes a short message and the exception text.
- Full exception details are written to logs automatically by hook wrappers.

## Cross‑agent access from hooks

Hooks can now resolve other agents by name:

```python
companion = ctx.get_agent("hook-companion")
if companion:
    ...
```

This works because `AgentApp` now injects an agent registry into every agent instance.

## Token usage access from hooks

Hooks can access current context usage via:

```python
usage = ctx.usage
if usage and usage.context_usage_percentage is not None:
    pct = usage.context_usage_percentage
```

This uses real usage stats from the attached LLM when available.

## Failure handling

Tool and lifecycle hooks now wrap exceptions and:

1. Show a **bright red** hook failure banner.
2. Log the full exception details.
3. Re-raise (so failures propagate as before).

## Examples (in-repo)

- **Agent cards**
  - `.dev/agent-cards/hook-kimi.md`
  - `.dev/agent-cards/hook-companion.md`

- **Hook module**
  - `.dev/agent-cards/hook_demo_hooks.py` (multiline Rich output + hook failure trigger)

- **Function tools**
  - `.dev/agent-cards/hook_tools.py` (echo, uppercase, fail tool)

- **Smoke test**
  - `scripts/hook_smoke_test.py` (run hook against a saved history file)

- **End-to-end runner**
  - `scripts/hook_demo_run.py`

## Running the demo

```bash
uv run scripts/hook_demo_run.py
```

Suggested prompts:

- Normal: `Hello hook test`
- Tool failure + hook failure: `tool-fail`
- Hook failure only: `hook-fail`

## Hook smoke test usage

```bash
uv run scripts/hook_smoke_test.py \
  --hook path/to/hooks.py:after_turn_complete \
  --history ./history.json \
  --hook-type after_turn_complete \
  --output ./history-trimmed.json
```
