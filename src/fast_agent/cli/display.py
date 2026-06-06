"""Shared Rich display helpers for top-level CLI commands."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from rich.table import Table
from rich.text import Text

from fast_agent.marketplace.update_status import MarketplaceUpdateStatus, format_update_status_text
from fast_agent.ui.a3_headers import build_a3_section_header
from fast_agent.utils.path_display import format_relative_path

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from rich.console import Console


def print_section_header(console: Console, title: str, *, color: str = "blue") -> None:
    """Render a compact A3-style section header without horizontal rules."""
    combined = build_a3_section_header(title, color=color, include_dot=False)
    console.print()
    console.print(combined)
    console.print()


def print_hint(console: Console, message: str) -> None:
    """Render a low-emphasis CLI hint line."""
    console.print(Text(f"  {message}", style="dim"))


def print_warning(console: Console, message: str) -> None:
    """Render a warning line with the CLI's standard warning style."""
    console.print(Text(message, style="yellow"))


def indexed_table(*columns: tuple[str, str]) -> Table:
    """Create a standard CLI table with the leading one-based index column."""
    table = Table(show_header=True, box=None)
    table.add_column("#", justify="right", style="dim", header_style="bold bright_white")
    for name, style in columns:
        table.add_column(name, style=style, header_style="bold bright_white")
    return table


def print_detail_line(
    console: Console,
    label: str,
    value: object,
    *,
    label_style: str = "dim",
    value_style: str | None = "cyan",
) -> None:
    """Render a simple label/value detail line without bullets."""
    line = Text()
    line.append(f"{label}: ", style=label_style)
    line.append(str(value), style=value_style)
    console.print(line)


@dataclass(frozen=True)
class DetailDisplayRow:
    label: str
    value: object
    label_style: str = "dim"
    value_style: str | None = "cyan"


def print_detail_section(
    console: Console,
    title: str,
    rows: Sequence[DetailDisplayRow],
    *,
    color: str = "blue",
) -> None:
    """Render a section header followed by simple detail rows."""
    print_section_header(console, title, color=color)
    for row in rows:
        print_detail_line(
            console,
            row.label,
            row.value,
            label_style=row.label_style,
            value_style=row.value_style,
        )


def format_display_path(path: Path, *, cwd: Path | None = None) -> str:
    """Format a path relative to the current working directory when possible."""
    return format_relative_path(path, cwd=cwd)


@dataclass(frozen=True)
class UpdateDisplayRow:
    index: int
    name: str
    source_path: Path
    current_revision: str | None
    available_revision: str | None
    status: MarketplaceUpdateStatus
    detail: str | None = None


def print_update_table(
    console: Console,
    rows: Sequence[UpdateDisplayRow],
    *,
    format_revision_short: Callable[[str | None], str],
) -> None:
    """Render the shared update status table used by marketplace-style commands."""
    table = Table(show_header=True, box=None)
    table.add_column("#", justify="right", style="dim", header_style="bold bright_white")
    table.add_column("Name", style="cyan", header_style="bold bright_white")
    table.add_column("Source", style="dim", header_style="bold bright_white")
    table.add_column("Revision", style="white", header_style="bold bright_white")
    table.add_column("Status", style="green", header_style="bold bright_white")

    for row in rows:
        revision_display = ""
        if row.current_revision or row.available_revision:
            current = format_revision_short(row.current_revision)
            available = format_revision_short(row.available_revision)
            revision_display = f"{current} -> {available}"

        status = format_update_status_text(row.status, detail=row.detail)

        table.add_row(
            str(row.index),
            row.name,
            format_display_path(row.source_path),
            revision_display,
            status,
        )

    console.print(table)
