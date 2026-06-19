"""Prompt parsing/completion package for interactive UI."""

from __future__ import annotations

from typing import TYPE_CHECKING

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "AgentCompleter": ("fast_agent.ui.prompt.completer", "AgentCompleter"),
    "get_argument_input": ("fast_agent.ui.prompt.input", "get_argument_input"),
    "get_enhanced_input": ("fast_agent.ui.prompt.input", "get_enhanced_input"),
    "get_selection_input": ("fast_agent.ui.prompt.input", "get_selection_input"),
    "handle_special_commands": (
        "fast_agent.ui.prompt.special_commands",
        "handle_special_commands",
    ),
    "parse_special_input": ("fast_agent.ui.prompt.parser", "parse_special_input"),
}


def __getattr__(name: str):
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'") from exc
    from importlib import import_module

    return getattr(import_module(module_name), attr_name)


if TYPE_CHECKING:
    from .completer import AgentCompleter as AgentCompleter
    from .input import get_argument_input as get_argument_input
    from .input import get_enhanced_input as get_enhanced_input
    from .input import get_selection_input as get_selection_input
    from .parser import parse_special_input as parse_special_input
    from .special_commands import handle_special_commands as handle_special_commands

__all__ = [
    "AgentCompleter",
    "get_argument_input",
    "get_enhanced_input",
    "get_selection_input",
    "handle_special_commands",
    "parse_special_input",
]
