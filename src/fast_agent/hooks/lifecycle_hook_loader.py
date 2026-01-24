"""Lifecycle hook loader for agent lifecycle hooks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, cast

from fast_agent.core.exceptions import AgentConfigError
from fast_agent.tools.hook_loader import load_hook_function

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from pathlib import Path

    from fast_agent.hooks.lifecycle_hook_context import AgentLifecycleContext

VALID_LIFECYCLE_HOOK_TYPES = frozenset({"on_start", "on_shutdown"})


class LifecycleHookFunction(Protocol):
    __name__: str

    def __call__(self, ctx: AgentLifecycleContext) -> Awaitable[None]: ...


@dataclass(frozen=True)
class AgentLifecycleHooks:
    on_start: Callable[[AgentLifecycleContext], Awaitable[None]] | None = None
    on_shutdown: Callable[[AgentLifecycleContext], Awaitable[None]] | None = None


def load_lifecycle_hooks(
    hooks_config: dict[str, str] | None,
    base_path: Path | None = None,
) -> AgentLifecycleHooks:
    if not hooks_config:
        return AgentLifecycleHooks()

    invalid_types = set(hooks_config.keys()) - VALID_LIFECYCLE_HOOK_TYPES
    if invalid_types:
        raise AgentConfigError(
            f"Invalid lifecycle hook types: {invalid_types}",
            f"Valid types are: {sorted(VALID_LIFECYCLE_HOOK_TYPES)}",
        )

    hooks: dict[str, Callable[[AgentLifecycleContext], Awaitable[None]]] = {}
    for hook_type, spec in hooks_config.items():
        hook_func = load_hook_function(spec, base_path)
        hooks[hook_type] = cast("Callable[[AgentLifecycleContext], Awaitable[None]]", hook_func)

    return AgentLifecycleHooks(
        on_start=hooks.get("on_start"),
        on_shutdown=hooks.get("on_shutdown"),
    )
