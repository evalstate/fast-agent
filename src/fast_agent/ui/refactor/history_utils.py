"""
History utilities for interactive prompt (initial split - wrappers to existing implementations).
This file currently re-exports functions from the original interactive_prompt module to keep
behavior unchanged while enabling incremental refactor.
"""

from fast_agent.ui.interactive_prompt import (
    _collect_user_turns,
    _display_history_turn,
    _group_turns_for_history_actions,
    _trim_history_for_rewind,
)

__all__ = [
    "_group_turns_for_history_actions",
    "_collect_user_turns",
    "_trim_history_for_rewind",
    "_display_history_turn",
]
