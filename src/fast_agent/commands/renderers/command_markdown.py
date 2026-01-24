"""Markdown rendering helpers for command outcomes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

from rich.text import Text

if TYPE_CHECKING:
    from fast_agent.commands.results import CommandMessage, CommandOutcome


def render_command_outcome_markdown(
    outcome: "CommandOutcome",
    *,
    heading: str,
    extra_messages: Iterable["CommandMessage"] | None = None,
) -> str:
    normalized_heading = heading.lstrip("# ").strip()
    lines: list[str] = []
    if normalized_heading:
        lines.extend([f"# {normalized_heading}", ""])

    messages = list(outcome.messages)
    if extra_messages:
        messages.extend(extra_messages)

    for message in messages:
        content = message.text
        if isinstance(content, Text):
            content = content.plain
        text = str(content)

        if message.title:
            lines.append(f"## {message.title}")
            lines.append("")

        if message.channel == "error":
            text = f"**Error:** {text}"
        elif message.channel == "warning":
            text = f"**Warning:** {text}"

        lines.append(text)
        lines.append("")

    return "\n".join(lines).rstrip()
