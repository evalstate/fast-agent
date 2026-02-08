"""Hook utilities for fast-agent."""

from fast_agent.hooks.history_trimmer import trim_tool_loop_history
from fast_agent.hooks.hook_context import HookContext
from fast_agent.hooks.hook_messages import show_hook_failure, show_hook_message
from fast_agent.hooks.lifecycle_hook_context import AgentLifecycleContext
from fast_agent.hooks.session_history import save_session_history

__all__ = [
    "AgentLifecycleContext",
    "HookContext",
    "save_session_history",
    "show_hook_failure",
    "show_hook_message",
    "trim_tool_loop_history",
]
