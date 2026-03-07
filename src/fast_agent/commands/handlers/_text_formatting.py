"""Shared Rich text formatting helpers for command handlers."""

from __future__ import annotations

import textwrap

from rich.text import Text


def append_heading(content: Text, heading: str) -> None:
    """Append a bold heading, separated from prior content when needed."""
    if content.plain:
        content.append("\n")
    content.append_text(Text.from_markup(f"[bold]{heading}[/bold]\n\n"))


def append_wrapped_text(content: Text, value: str, *, indent: str = "") -> None:
    """Append wrapped text lines with optional indentation."""
    wrapped_lines = textwrap.wrap(value.strip(), width=72)
    for line in wrapped_lines:
        content.append(indent)
        content.append_text(Text(line))
        content.append("\n")
