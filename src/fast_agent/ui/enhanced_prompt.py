"""Compatibility surface for enhanced prompt APIs.

Core implementation now lives under ``fast_agent.ui.prompt`` modules.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fast_agent.ui.prompt.alert_flags import (
    _extract_alert_flags_from_alert,
    _extract_alert_flags_from_meta,
    _resolve_alert_flags_from_history,
)
from fast_agent.ui.prompt.completer import AgentCompleter
from fast_agent.ui.prompt.editor import get_text_from_editor
from fast_agent.ui.prompt.keybindings import AgentKeyBindings, create_keybindings
from fast_agent.ui.prompt.parser import parse_special_input
from fast_agent.ui.prompt.session import (
    ShellPrefixLexer,
    _display_agent_info_helper,
    get_argument_input,
    get_selection_input,
    show_mcp_status,
)
from fast_agent.ui.prompt.session import (
    get_enhanced_input as _get_enhanced_input,
)
from fast_agent.ui.prompt.session import (
    handle_special_commands as _handle_special_commands,
)
from fast_agent.ui.prompt.toolbar import (
    _can_fit_shell_path_and_version,
    _fit_shell_identity_for_toolbar,
    _fit_shell_path_for_toolbar,
    _format_parent_current_path,
    _format_toolbar_agent_identity,
    _left_truncate_with_ellipsis,
)

if TYPE_CHECKING:
    from prompt_toolkit.history import InMemoryHistory

    from fast_agent.core.agent_app import AgentApp
    from fast_agent.ui.command_payloads import CommandPayload

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


def _sync_to_session() -> None:
    from fast_agent.ui.prompt import session as _session

    _session.agent_histories = agent_histories
    _session.available_agents = available_agents
    _session.in_multiline_mode = in_multiline_mode
    _session._last_copyable_output = _last_copyable_output
    _session._copy_notice = _copy_notice
    _session._copy_notice_until = _copy_notice_until
    _session.help_message_shown = help_message_shown
    _session._agent_info_shown = _agent_info_shown
    _session._startup_notices = _startup_notices


def _sync_from_session() -> None:
    global agent_histories, available_agents, in_multiline_mode
    global _last_copyable_output, _copy_notice, _copy_notice_until
    global help_message_shown, _agent_info_shown, _startup_notices

    from fast_agent.ui.prompt import session as _session

    agent_histories = _session.agent_histories
    available_agents = _session.available_agents
    in_multiline_mode = _session.in_multiline_mode
    _last_copyable_output = _session._last_copyable_output
    _copy_notice = _session._copy_notice
    _copy_notice_until = _session._copy_notice_until
    help_message_shown = _session.help_message_shown
    _agent_info_shown = _session._agent_info_shown
    _startup_notices = _session._startup_notices


def set_last_copyable_output(output: str) -> None:
    global _last_copyable_output
    _last_copyable_output = output
    _sync_to_session()


def queue_startup_notice(notice: object) -> None:
    from fast_agent.ui.prompt.session import queue_startup_notice as _queue_startup_notice

    _sync_to_session()
    _queue_startup_notice(notice)
    _sync_from_session()


def queue_startup_markdown_notice(
    text: str,
    *,
    title: str | None = None,
    style: str | None = None,
    right_info: str | None = None,
    agent_name: str | None = None,
) -> None:
    from fast_agent.ui.prompt.session import (
        queue_startup_markdown_notice as _queue_startup_markdown_notice,
    )

    _sync_to_session()
    _queue_startup_markdown_notice(
        text,
        title=title,
        style=style,
        right_info=right_info,
        agent_name=agent_name,
    )
    _sync_from_session()


async def get_enhanced_input(*args, **kwargs) -> str | CommandPayload:
    _sync_to_session()
    result = await _get_enhanced_input(*args, **kwargs)
    _sync_from_session()
    return result


async def handle_special_commands(
    command: str | CommandPayload | None, agent_app: "AgentApp | bool | None" = None
) -> bool | CommandPayload:
    _sync_to_session()
    result = await _handle_special_commands(command, agent_app)
    _sync_from_session()
    return result


__all__ = [
    "AgentCompleter",
    "AgentKeyBindings",
    "ShellPrefixLexer",
    "create_keybindings",
    "get_argument_input",
    "get_enhanced_input",
    "get_selection_input",
    "get_text_from_editor",
    "handle_special_commands",
    "parse_special_input",
    "queue_startup_notice",
    "queue_startup_markdown_notice",
    "set_last_copyable_output",
    "show_mcp_status",
    "_can_fit_shell_path_and_version",
    "_extract_alert_flags_from_alert",
    "_extract_alert_flags_from_meta",
    "_fit_shell_identity_for_toolbar",
    "_fit_shell_path_for_toolbar",
    "_format_parent_current_path",
    "_format_toolbar_agent_identity",
    "_left_truncate_with_ellipsis",
    "_resolve_alert_flags_from_history",
    "_display_agent_info_helper",
    "agent_histories",
    "available_agents",
    "in_multiline_mode",
]
