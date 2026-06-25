"""Shared command outputs and message containers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from rich.text import Text

if TYPE_CHECKING:
    from collections.abc import Mapping

    from rich.console import RenderableType

CommandChannel = Literal["system", "info", "warning", "error"]
COMMAND_CHANNEL_LABELS: dict[CommandChannel, str] = {
    "info": "Info",
    "warning": "Warning",
    "error": "Error",
}


def command_channel_label(channel: CommandChannel) -> str | None:
    return COMMAND_CHANNEL_LABELS.get(channel)


@dataclass(slots=True)
class CommandMessage:
    """A displayable message returned by a command handler."""

    text: str | Text
    channel: CommandChannel = "system"
    title: str | None = None
    right_info: str | None = None
    agent_name: str | None = None
    render_markdown: bool = False
    metadata: dict[str, object] = field(default_factory=dict)
    post_content: RenderableType | None = None

    def plain_text(self) -> str:
        content = self.text
        if isinstance(content, Text):
            return content.plain
        return str(content)


@dataclass(slots=True)
class CommandOutcome:
    """Result object returned from a command handler."""

    handled: bool = True
    messages: list[CommandMessage] = field(default_factory=list)
    buffer_prefill: str | None = None
    switch_agent: str | None = None
    requires_refresh: bool = False

    def add_message(
        self,
        text: str | Text,
        *,
        channel: CommandChannel = "system",
        title: str | None = None,
        right_info: str | None = None,
        agent_name: str | None = None,
        render_markdown: bool = False,
        metadata: Mapping[str, object] | None = None,
        post_content: RenderableType | None = None,
    ) -> None:
        self.messages.append(
            CommandMessage(
                text=text,
                channel=channel,
                title=title,
                right_info=right_info,
                agent_name=agent_name,
                render_markdown=render_markdown,
                metadata=dict(metadata) if metadata is not None else {},
                post_content=post_content,
            )
        )
