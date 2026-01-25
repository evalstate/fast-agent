# Agent Lifecycle Hooks Plan

## Goal
Add **agent lifecycle hooks** that can be defined in **agent cards**, with a strongly typed hook context that exposes the agent, its config, and its execution context. Hooks should run for normal agents **and** detached clones created via `spawn_detached_instance()`.

## Non-goals
- No global application startup/shutdown hooks (per-agent only).
- No changes to existing tool hooks behavior.

---

## Proposed API

### Agent card schema
Add a new optional field to agent cards:

```yaml
lifecycle_hooks:
  on_start: hooks.py:start_agent
  on_shutdown: hooks.py:stop_agent
```

Hook keys:
- `on_start`
- `on_shutdown`

### AgentConfig
Add to `AgentConfig` in `src/fast_agent/agents/agent_types.py`:

```py
lifecycle_hooks: dict[str, str] | None = None
```

---

## Hook Context (Type-safe)
Create a dedicated lifecycle hook context rather than reusing `HookContext`.

**New file:** `src/fast_agent/hooks/lifecycle_hook_context.py`

```py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.context import Context
from fast_agent.interfaces import AgentProtocol


@dataclass
class AgentLifecycleContext:
    agent: AgentProtocol
    context: Context | None
    config: AgentConfig
    hook_type: Literal["on_start", "on_shutdown"]

    @property
    def agent_name(self) -> str:
        return self.agent.name

    @property
    def has_context(self) -> bool:
        return self.context is not None
```

**Why:** gives hooks safe access to `ctx.context` and `ctx.config` without `getattr`/`cast`.

---

## Hook Loader

**New file:** `src/fast_agent/hooks/lifecycle_hook_loader.py`

- Valid types: `{"on_start", "on_shutdown"}`
- Define a dedicated protocol for lifecycle hooks (don’t reuse `HookFunction`):

```py
class LifecycleHookFunction(Protocol):
    __name__: str

    def __call__(self, ctx: AgentLifecycleContext) -> Awaitable[None]: ...
```
- Uses existing `load_hook_function()` from `tools/hook_loader.py` to resolve `module.py:func`.
- Returns a small dataclass:

```py
@dataclass(frozen=True)
class AgentLifecycleHooks:
    on_start: Callable[[AgentLifecycleContext], Awaitable[None]] | None = None
    on_shutdown: Callable[[AgentLifecycleContext], Awaitable[None]] | None = None
```

---

## Lifecycle Hook Execution

### Base helper (no mixin needed)
Because `McpAgent` is in the `LlmDecorator` inheritance chain, and we will update
`McpAgent.initialize()`/`shutdown()` to call `super()`, the lifecycle hooks can live
directly on `LlmDecorator`. This keeps the implementation simple and avoids a shared
mixin/utility module.

Add a private helper on `LlmDecorator`:

```py
async def _run_lifecycle_hook(self, hook_type: Literal["on_start", "on_shutdown"]) -> None:
    hooks = self._lifecycle_hooks  # lazy-loaded
    hook = hooks.on_start if hook_type == "on_start" else hooks.on_shutdown
    if hook is None:
        return
    ctx = AgentLifecycleContext(
        agent=self,
        context=self.context,
        config=self.config,
        hook_type=hook_type,
    )
    try:
        await hook(ctx)
    except Exception as exc:  # noqa: BLE001
        if hook_type == "on_start":
            raise AgentConfigError(
                f"Lifecycle hook '{hook_type}' failed", str(exc)
            ) from exc
        logger.exception("Lifecycle hook failed during shutdown", hook_type=hook_type)
```

### Lazy hook loading
In the same base class, lazily load hooks from `self.config.lifecycle_hooks` using the new loader.

**Why:** ensures detached clones automatically pick up hooks without factory wiring.

### Integration points
- `LlmDecorator.initialize()` → call `_run_lifecycle_hook("on_start")`
- `LlmDecorator.shutdown()` → call `_run_lifecycle_hook("on_shutdown")`
- Update `McpAgent.initialize()` to call `super().initialize()` after MCP setup. The base implementation runs `_run_lifecycle_hook("on_start")`.
- Update `McpAgent.shutdown()` to call `_run_lifecycle_hook("on_shutdown")` **before** closing the MCP aggregator, then call `super().shutdown()` to clear `_initialized`.

Rationale: today `McpAgent` never flips `_initialized` because it bypasses the base lifecycle. Calling `super()` keeps `initialized` accurate and ensures lifecycle hooks run consistently without extra mixins/utilities.

---

## Agent Card Parsing / Dumping

Update `src/fast_agent/core/agent_card_loader.py`:
- Parse `lifecycle_hooks` similarly to `tool_hooks`.
- Validate it is a dict of strings.
- Validate keys against `VALID_LIFECYCLE_HOOK_TYPES`.
- Include it in the dumped card (if present).

---

## Tests

### Unit tests
Add to `tests/unit/fast_agent/hooks/`:
- `test_lifecycle_hooks_invalid_type_raises()`
- `test_lifecycle_hooks_loads_and_calls()` (hook receives typed context with `.context` and `.config`)

### Integration test (passthrough + detached clones + thread-safety)

**New test file:** `tests/integration/agent_hooks/test_agent_lifecycle_hooks.py`

**Scenario:**
- Use a child agent card with `lifecycle_hooks`.
- Use a parent agent that calls the child as a tool (which triggers `spawn_detached_instance()` per call).
- Inject a `PassthroughLLM` subclass into the parent that always calls the child tool.
- Run two parent requests concurrently (`asyncio.gather`) to create **two detached clones** at the same time.

**Hook behavior:**
- Hook functions append JSON lines to a marker file with:
  - `agent_name`
  - `agent_id = id(ctx.agent)`
  - `hook_type`

**Assertions:**
- For events where `agent_name` ends with `[tool]`, there are **two distinct `agent_id`s**.
- Each `agent_id` has exactly **one `on_start` and one `on_shutdown`**.
- No cross-talk: every `on_shutdown` corresponds to a prior `on_start` of the same `agent_id`.

This validates:
- Hooks load from cards.
- Hooks run for detached clones.
- Hooks are safe under concurrent clone creation.

---

## Implementation Checklist

1. Add `lifecycle_hooks` to `AgentConfig` and agent card loader/dumper.
2. Add `AgentLifecycleContext` (typed) and `AgentLifecycleHooks` + loader with `LifecycleHookFunction` protocol.
3. Add lazy loader + `_run_lifecycle_hook()` to `LlmDecorator` (no mixin needed).
4. Call lifecycle hooks from `initialize()`/`shutdown()` in `LlmDecorator` and update `McpAgent` to call `super()`.
5. Add error-handling policy (fail on `on_start`, log on `on_shutdown`).
6. Validate lifecycle hook keys against `VALID_LIFECYCLE_HOOK_TYPES` in card parsing.
7. Add unit tests for loader and context.
8. Add integration test with concurrent detached clones (passthrough model).
9. Run lint + typecheck:
   - `uv run scripts/lint.py --fix`
   - `uv run scripts/typecheck.py`

---

## Notes on Type Safety
- Avoid `getattr` and `cast` inside hooks.
- Use `AgentProtocol` + explicit `Context | None` on the lifecycle context.
- Use `Literal["on_start", "on_shutdown"]` for hook typing.
