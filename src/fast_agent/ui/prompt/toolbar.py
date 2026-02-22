"""Toolbar formatting utilities."""

from __future__ import annotations

import shutil
from typing import TYPE_CHECKING

from prompt_toolkit.application.current import get_app_or_none
from prompt_toolkit.formatted_text import HTML, to_formatted_text
from prompt_toolkit.formatted_text.utils import fragment_list_width

from fast_agent.agents.agent_types import AgentType

if TYPE_CHECKING:
    from pathlib import Path

_ELLIPSIS = "…"


def _left_truncate_with_ellipsis(text: str, max_length: int) -> str:
    """Truncate text from the left using a single-character ellipsis."""
    if max_length <= 0:
        return ""
    if len(text) <= max_length:
        return text
    if max_length == 1:
        return _ELLIPSIS
    return f"{_ELLIPSIS}{text[-(max_length - 1) :]}"


def _format_parent_current_path(path: Path) -> str:
    """Render a path as parent/current when both names are available."""
    current = path.name or str(path)
    parent = path.parent.name
    if parent:
        return f"{parent}/{current}"
    return current


def _fit_shell_path_for_toolbar(path: Path, max_length: int) -> str:
    """Fit a shell path for toolbar display without spilling."""
    if max_length <= 0:
        return ""

    parent_current = _format_parent_current_path(path)
    if len(parent_current) <= max_length:
        return parent_current

    current = path.name or str(path)
    if len(current) <= max_length:
        return current

    return _left_truncate_with_ellipsis(current, max_length)


def _fit_shell_identity_for_toolbar(path: Path, version_segment: str, max_length: int) -> str:
    """Fit shell path and optional version segment without spilling toolbar width."""
    if max_length <= 0:
        return ""

    parent_current = _format_parent_current_path(path)
    current = path.name or str(path)

    parent_current_with_version = f"{parent_current} | {version_segment}"
    if len(parent_current_with_version) <= max_length:
        return parent_current_with_version

    current_with_version = f"{current} | {version_segment}"
    if len(current_with_version) <= max_length:
        return current_with_version

    return _fit_shell_path_for_toolbar(path, max_length)


def _can_fit_shell_path_and_version(path: Path, version_segment: str, max_length: int) -> bool:
    """Return whether path + version can fit in the available toolbar width."""
    if max_length <= 0:
        return False

    parent_current = _format_parent_current_path(path)
    parent_current_with_version = f"{parent_current} | {version_segment}"
    if len(parent_current_with_version) <= max_length:
        return True

    current = path.name or str(path)
    current_with_version = f"{current} | {version_segment}"
    return len(current_with_version) <= max_length


def _toolbar_markup_width(markup: str) -> int:
    """Compute visible width for a prompt_toolkit HTML markup fragment."""
    if not markup:
        return 0
    try:
        return fragment_list_width(to_formatted_text(HTML(markup)))
    except Exception:
        return len(markup)


def _resolve_toolbar_width() -> int:
    """Resolve current toolbar width from prompt-toolkit app or terminal fallback."""
    app = get_app_or_none()
    if app is not None:
        try:
            return max(1, app.output.get_size().columns)
        except Exception:
            pass
    return max(1, shutil.get_terminal_size((80, 20)).columns)


def _is_smart_agent(agent: object | None) -> bool:
    """Return True when the provided agent instance is a smart agent."""
    if agent is None:
        return False
    agent_type = getattr(agent, "agent_type", None)
    normalized = getattr(agent_type, "value", agent_type)
    if isinstance(normalized, str):
        return normalized.lower() == AgentType.SMART.value
    return normalized == AgentType.SMART


def _format_toolbar_agent_identity(
    agent_name: str, toolbar_color: str, agent: object | None
) -> str:
    """Render toolbar agent identity, suffixing [S] for smart agents."""
    label = f"{agent_name}[S]" if _is_smart_agent(agent) else agent_name
    return f"<style fg='{toolbar_color}' bg='ansiblack'> {label} </style>"
