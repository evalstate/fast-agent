"""Hook utilities for fast-agent."""

from fast_agent.hooks.history_trimmer import trim_tool_loop_history
from fast_agent.hooks.hook_context import HookContext
from fast_agent.hooks.session_history import save_session_history

__all__ = ["HookContext", "save_session_history", "trim_tool_loop_history"]
