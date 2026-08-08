"""Hook utilities for fast-agent."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

__all__ = [
    "AgentLifecycleContext",
    "HookContext",
    "auto_compact_history",
    "save_session_history",
    "show_hook_failure",
    "show_hook_message",
    "trim_tool_loop_history",
]

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "AgentLifecycleContext": (
        "fast_agent.hooks.lifecycle_hook_context",
        "AgentLifecycleContext",
    ),
    "HookContext": ("fast_agent.hooks.hook_context", "HookContext"),
    "auto_compact_history": ("fast_agent.hooks.compaction", "auto_compact_history"),
    "save_session_history": (
        "fast_agent.hooks.session_history",
        "save_session_history",
    ),
    "show_hook_failure": ("fast_agent.hooks.hook_messages", "show_hook_failure"),
    "show_hook_message": ("fast_agent.hooks.hook_messages", "show_hook_message"),
    "trim_tool_loop_history": (
        "fast_agent.hooks.history_trimmer",
        "trim_tool_loop_history",
    ),
}

if TYPE_CHECKING:
    from fast_agent.hooks.compaction import auto_compact_history as auto_compact_history
    from fast_agent.hooks.history_trimmer import (
        trim_tool_loop_history as trim_tool_loop_history,
    )
    from fast_agent.hooks.hook_context import HookContext as HookContext
    from fast_agent.hooks.hook_messages import show_hook_failure as show_hook_failure
    from fast_agent.hooks.hook_messages import show_hook_message as show_hook_message
    from fast_agent.hooks.lifecycle_hook_context import (
        AgentLifecycleContext as AgentLifecycleContext,
    )
    from fast_agent.hooks.session_history import save_session_history as save_session_history


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'") from exc
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
