"""Shared Rich text formatting helpers for command handlers."""

from __future__ import annotations

import textwrap
from contextlib import suppress
from dataclasses import dataclass
from shutil import get_terminal_size
from typing import TYPE_CHECKING

from rich.text import Text

from fast_agent.marketplace.update_status import (
    COMMON_UPDATE_STATUS_STYLES,
    MarketplaceUpdateStatus,
    format_update_status_text,
)
from fast_agent.utils.path_display import format_relative_path
from fast_agent.utils.text import strip_to_none

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


@dataclass(frozen=True, slots=True)
class StatusText:
    text: str
    style: str | None = None


_UPDATE_STATUS_STYLES = COMMON_UPDATE_STATUS_STYLES


def append_heading(content: Text, heading: str) -> None:
    """Append a bold heading, separated from prior content when needed."""
    if content.plain:
        content.append("\n")
    content.append_text(Text(heading, style="bold"))
    content.append("\n\n")


def _wrap_width_for_indent(indent: str, *, width: int = 72) -> int:
    return max(20, width - len(indent))


def append_wrapped_text(content: Text, value: str, *, indent: str = "") -> None:
    """Append wrapped text lines with optional indentation."""
    text = strip_to_none(value)
    if text is None:
        return
    wrapped_lines = textwrap.wrap(text, width=_wrap_width_for_indent(indent))
    for line in wrapped_lines:
        content.append(indent)
        content.append_text(Text(line))
        content.append("\n")


def append_warning_line(content: Text, message: str) -> None:
    content.append_text(Text(message, style="yellow"))


def indexed_row(index: int) -> Text:
    row = Text()
    row.append(f"[{index:2}] ", style="dim cyan")
    return row


def append_indexed_name_line(content: Text, index: int, name: str) -> None:
    row = indexed_row(index)
    row.append(name, style="bright_blue bold")
    content.append_text(row)
    content.append("\n")


def append_indexed_current_line(
    content: Text,
    index: int,
    value: str,
    *,
    is_current: bool,
) -> None:
    row = indexed_row(index)
    row.append(value, style="bright_blue bold")
    if is_current:
        row.append(" • ", style="dim")
        row.append("current", style="dim green")
    content.append_text(row)
    content.append("\n")


def append_detail_line(
    content: Text,
    label: str,
    value: str,
    *,
    prefix: str = "  - ",
    style: str | None = "dim",
    value_style: str | None = None,
) -> None:
    content.append(prefix, style=style)
    content.append(f"{label}: ", style=style)
    content.append(value, style=value_style)
    content.append("\n")


def append_revision_line(
    content: Text,
    current_revision: str | None,
    available_revision: str | None,
    *,
    format_revision: Callable[[str | None], str],
) -> None:
    if strip_to_none(current_revision) is None and strip_to_none(available_revision) is None:
        return
    current = format_revision(current_revision)
    available = format_revision(available_revision)
    append_detail_line(
        content,
        "revision",
        f"{current} -> {available}",
        value_style="dim",
    )


def append_status_line(
    content: Text,
    status: StatusText,
    *,
    prefix: str = "  - ",
) -> None:
    content.append(prefix, style="dim")
    content.append("status: ", style="dim")
    content.append(status.text, style=status.style)
    content.append("\n")


def format_display_path(path: Path, *, cwd: Path | None = None) -> str:
    return format_relative_path(path, cwd=cwd)


def resolve_terminal_width() -> int:
    console_width = _console_width()
    if console_width > 0:
        return console_width
    return get_terminal_size(fallback=(100, 20)).columns


def _console_width() -> int:
    with suppress(Exception):
        from fast_agent.ui.console import console

        width = console.size.width
        return width if isinstance(width, int) else 0
    return 0


def update_status_text(
    status: MarketplaceUpdateStatus,
    *,
    detail: str | None = None,
    labels: dict[MarketplaceUpdateStatus, str] | None = None,
    detail_statuses: set[MarketplaceUpdateStatus] | None = None,
) -> StatusText:
    return StatusText(
        text=format_update_status_text(
            status,
            detail=detail,
            labels=labels,
            detail_statuses=detail_statuses,
        ),
        style=_UPDATE_STATUS_STYLES.get(status),
    )
