"""Compatibility wrapper for prompt input runtime helpers.

Prefer ``fast_agent.ui.prompt.input_runtime`` for prompt input lifecycle code.
"""

from fast_agent.ui.prompt import input_runtime as _input_runtime
from fast_agent.ui.prompt.input_runtime import (
    _ERASE_PREVIOUS_LINE_SEQ,
    _clear_prompt_echo_line,
    build_prompt_style,
    cleanup_prompt_session,
    create_prompt_session,
    run_prompt_once,
    start_toolbar_switch_task,
)

__all__ = [
    "_ERASE_PREVIOUS_LINE_SEQ",
    "_clear_prompt_echo_line",
    "build_prompt_style",
    "cleanup_prompt_session",
    "create_prompt_session",
    "run_prompt_once",
    "start_toolbar_switch_task",
]


def __getattr__(name: str):
    return getattr(_input_runtime, name)
