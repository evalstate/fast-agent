"""Markdown rendering helpers for command outcomes."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fast_agent.commands.renderers.markdown_blocks import markdown_heading
from fast_agent.commands.results import command_channel_label
from fast_agent.utils.markdown import escape_markdown_text
from fast_agent.utils.text import strip_to_none

if TYPE_CHECKING:
    from collections.abc import Iterable

    from fast_agent.commands.results import CommandMessage, CommandOutcome


def _formatted_message_text(message: "CommandMessage") -> str | None:
    text = strip_to_none(message.plain_text())
    if text is None:
        return None
    if not message.render_markdown:
        text = escape_markdown_text(text)
    label = command_channel_label(message.channel)
    if label is not None and message.channel != "info":
        return f"**{label}:** {text}"
    return text


def _message_markdown_lines(message: "CommandMessage") -> list[str]:
    lines: list[str] = []
    title = markdown_heading(message.title or "", level=2)
    if title:
        lines.extend([title, ""])

    message_text = _formatted_message_text(message)
    if message_text is not None:
        lines.extend([message_text, ""])
    return lines


def render_command_outcome_markdown(
    outcome: "CommandOutcome",
    *,
    heading: str,
    extra_messages: Iterable["CommandMessage"] | None = None,
) -> str:
    heading_line = markdown_heading(heading)
    lines: list[str] = []
    if heading_line:
        lines.extend([heading_line, ""])

    messages = list(outcome.messages)
    if extra_messages:
        messages.extend(extra_messages)

    for message in messages:
        lines.extend(_message_markdown_lines(message))

    return "\n".join(lines).rstrip()
