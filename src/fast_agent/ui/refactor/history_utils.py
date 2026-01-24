"""History utilities for interactive prompt (wrapper for refactor utilities)."""

from fast_agent.commands.handlers.history import (
    _collect_user_turns,
    _group_turns_for_history_actions,
    _trim_history_for_rewind,
)
from fast_agent.ui.history_actions import display_history_turn as _display_history_turn

__all__ = [
    "_group_turns_for_history_actions",
    "_collect_user_turns",
    "_trim_history_for_rewind",
    "_display_history_turn",
]
