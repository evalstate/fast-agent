"""Compatibility surface for enhanced prompt APIs.

Core implementation now lives under ``fast_agent.ui.prompt`` modules.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from prompt_toolkit.history import InMemoryHistory

    from fast_agent.core.agent_app import AgentApp
    from fast_agent.ui.command_payloads import CommandPayload
    from fast_agent.ui.prompt.completer import AgentCompleter as AgentCompleter
    from fast_agent.ui.prompt.keybindings import AgentKeyBindings as AgentKeyBindings
    from fast_agent.ui.prompt.keybindings import ShellPrefixLexer as ShellPrefixLexer

# Legacy mutable globals preserved for compatibility with tests and external imports.
agent_histories: dict[str, InMemoryHistory] = {}
available_agents: set[str] = set()
in_multiline_mode: bool = False
_last_copyable_output: str | None = None
_copy_notice: str | None = None
_copy_notice_until: float = 0.0
help_message_shown: bool = False
_agent_info_shown: set[str] = set()
_startup_notices: list[object] = []

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "AgentCompleter": ("fast_agent.ui.prompt.completer", "AgentCompleter"),
    "AgentKeyBindings": ("fast_agent.ui.prompt.keybindings", "AgentKeyBindings"),
    "ShellPrefixLexer": ("fast_agent.ui.prompt.keybindings", "ShellPrefixLexer"),
}


def __getattr__(name: str):
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'") from exc
    from importlib import import_module

    return getattr(import_module(module_name), attr_name)


def _lazy_call(module_name: str, attr_name: str, *args, **kwargs):
    from importlib import import_module

    return getattr(import_module(module_name), attr_name)(*args, **kwargs)


def _extract_alert_flags_from_alert(*args, **kwargs):
    return _lazy_call(
        "fast_agent.ui.prompt.alert_flags",
        "_extract_alert_flags_from_alert",
        *args,
        **kwargs,
    )


def _extract_alert_flags_from_meta(*args, **kwargs):
    return _lazy_call(
        "fast_agent.ui.prompt.alert_flags",
        "_extract_alert_flags_from_meta",
        *args,
        **kwargs,
    )


def _resolve_alert_flags_from_history(*args, **kwargs):
    return _lazy_call(
        "fast_agent.ui.prompt.alert_flags",
        "_resolve_alert_flags_from_history",
        *args,
        **kwargs,
    )


def get_text_from_editor(*args, **kwargs):
    return _lazy_call(
        "fast_agent.ui.prompt.editor",
        "get_text_from_editor",
        *args,
        **kwargs,
    )


def create_keybindings(*args, **kwargs):
    return _lazy_call(
        "fast_agent.ui.prompt.keybindings",
        "create_keybindings",
        *args,
        **kwargs,
    )


def _can_fit_shell_path_and_version(*args, **kwargs):
    return _lazy_call(
        "fast_agent.ui.prompt.toolbar",
        "_can_fit_shell_path_and_version",
        *args,
        **kwargs,
    )


def _fit_shell_identity_for_toolbar(*args, **kwargs):
    return _lazy_call(
        "fast_agent.ui.prompt.toolbar",
        "_fit_shell_identity_for_toolbar",
        *args,
        **kwargs,
    )


def _fit_shell_path_for_toolbar(*args, **kwargs):
    return _lazy_call(
        "fast_agent.ui.prompt.toolbar",
        "_fit_shell_path_for_toolbar",
        *args,
        **kwargs,
    )


def _format_context_usage_percent_for_toolbar(*args, **kwargs):
    return _lazy_call(
        "fast_agent.ui.prompt.toolbar",
        "_format_context_usage_percent_for_toolbar",
        *args,
        **kwargs,
    )


def _format_parent_current_path(*args, **kwargs):
    return _lazy_call(
        "fast_agent.ui.prompt.toolbar",
        "_format_parent_current_path",
        *args,
        **kwargs,
    )


def _format_toolbar_agent_identity(*args, **kwargs):
    return _lazy_call(
        "fast_agent.ui.prompt.toolbar",
        "_format_toolbar_agent_identity",
        *args,
        **kwargs,
    )


def _left_truncate_with_ellipsis(*args, **kwargs):
    return _lazy_call(
        "fast_agent.ui.prompt.toolbar",
        "_left_truncate_with_ellipsis",
        *args,
        **kwargs,
    )


def _render_model_gauges(*args, **kwargs):
    return _lazy_call(
        "fast_agent.ui.prompt.toolbar",
        "_render_model_gauges",
        *args,
        **kwargs,
    )


def _sync_to_input_module() -> None:
    from fast_agent.ui.prompt import input as _input

    _input.agent_histories = agent_histories
    _input.available_agents = available_agents
    _input.in_multiline_mode = in_multiline_mode
    _input._last_copyable_output = _last_copyable_output
    _input._copy_notice = _copy_notice
    _input._copy_notice_until = _copy_notice_until
    _input.help_message_shown = help_message_shown
    _input._agent_info_shown = _agent_info_shown
    _input._startup_notices = _startup_notices


def _sync_from_input_module() -> None:
    global agent_histories, available_agents, in_multiline_mode
    global _last_copyable_output, _copy_notice, _copy_notice_until
    global help_message_shown, _agent_info_shown, _startup_notices

    from fast_agent.ui.prompt import input as _input

    agent_histories = _input.agent_histories
    available_agents = _input.available_agents
    in_multiline_mode = _input.in_multiline_mode
    _last_copyable_output = _input._last_copyable_output
    _copy_notice = _input._copy_notice
    _copy_notice_until = _input._copy_notice_until
    help_message_shown = _input.help_message_shown
    _agent_info_shown = _input._agent_info_shown
    _startup_notices = _input._startup_notices


def set_last_copyable_output(output: str) -> None:
    global _last_copyable_output
    _last_copyable_output = output
    _sync_to_input_module()


async def _display_agent_info_helper(*args, **kwargs) -> None:
    from fast_agent.ui.prompt.input import (
        _display_agent_info_helper as _display_agent_info_helper_impl,
    )

    await _display_agent_info_helper_impl(*args, **kwargs)


async def show_mcp_status(*args, **kwargs) -> None:
    from fast_agent.ui.prompt.input import show_mcp_status as _show_mcp_status

    await _show_mcp_status(*args, **kwargs)


async def get_argument_input(*args, **kwargs):
    from fast_agent.ui.prompt.input import get_argument_input as _get_argument_input

    return await _get_argument_input(*args, **kwargs)


async def get_selection_input(*args, **kwargs):
    from fast_agent.ui.prompt.input import get_selection_input as _get_selection_input

    return await _get_selection_input(*args, **kwargs)


def parse_special_input(text: str) -> str | CommandPayload:
    from fast_agent.ui.prompt.parser import parse_special_input as _parse_special_input

    return _parse_special_input(text)


def queue_startup_notice(notice: object) -> None:
    from fast_agent.ui.prompt.input import queue_startup_notice as _queue_startup_notice

    _sync_to_input_module()
    _queue_startup_notice(notice)
    _sync_from_input_module()


def queue_startup_markdown_notice(
    text: str,
    *,
    title: str | None = None,
    style: str | None = None,
    right_info: str | None = None,
    agent_name: str | None = None,
) -> None:
    from fast_agent.ui.prompt.input import (
        queue_startup_markdown_notice as _queue_startup_markdown_notice,
    )

    _sync_to_input_module()
    _queue_startup_markdown_notice(
        text,
        title=title,
        style=style,
        right_info=right_info,
        agent_name=agent_name,
    )
    _sync_from_input_module()


async def get_enhanced_input(*args, **kwargs) -> str | CommandPayload:
    from fast_agent.ui.prompt.input import get_enhanced_input as _get_enhanced_input

    _sync_to_input_module()
    result = await _get_enhanced_input(*args, **kwargs)
    _sync_from_input_module()
    return result


async def handle_special_commands(
    command: str | CommandPayload | None, agent_app: "AgentApp | bool | None" = None
) -> bool | CommandPayload:
    from fast_agent.ui.prompt.input import handle_special_commands as _handle_special_commands

    _sync_to_input_module()
    result = await _handle_special_commands(command, agent_app)
    _sync_from_input_module()
    return result


__all__ = [
    "AgentCompleter",
    "AgentKeyBindings",
    "ShellPrefixLexer",
    "_can_fit_shell_path_and_version",
    "_display_agent_info_helper",
    "_extract_alert_flags_from_alert",
    "_extract_alert_flags_from_meta",
    "_fit_shell_identity_for_toolbar",
    "_fit_shell_path_for_toolbar",
    "_format_context_usage_percent_for_toolbar",
    "_format_parent_current_path",
    "_format_toolbar_agent_identity",
    "_left_truncate_with_ellipsis",
    "_render_model_gauges",
    "_resolve_alert_flags_from_history",
    "agent_histories",
    "available_agents",
    "create_keybindings",
    "get_argument_input",
    "get_enhanced_input",
    "get_selection_input",
    "get_text_from_editor",
    "handle_special_commands",
    "in_multiline_mode",
    "parse_special_input",
    "queue_startup_markdown_notice",
    "queue_startup_notice",
    "set_last_copyable_output",
    "show_mcp_status",
]
