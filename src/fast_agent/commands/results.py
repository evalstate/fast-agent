"""Shared command outputs and message containers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from rich.text import Text


CommandChannel = Literal["system", "info", "warning", "error"]


@dataclass(slots=True)
class CommandMessage:
    """A displayable message returned by a command handler."""

    text: str | "Text"
    channel: CommandChannel = "system"
    title: str | None = None
    right_info: str | None = None
    agent_name: str | None = None
    render_markdown: bool = False


@dataclass(slots=True)
class CommandOutcome:
    """Result object returned from a command handler."""

    handled: bool = True
    messages: list[CommandMessage] = field(default_factory=list)
    buffer_prefill: str | None = None
    switch_agent: str | None = None
    requires_refresh: bool = False
    halt_loop: bool = False

    def add_message(
        self,
        text: str | "Text",
        *,
        channel: CommandChannel = "system",
        title: str | None = None,
        right_info: str | None = None,
        agent_name: str | None = None,
        render_markdown: bool = False,
    ) -> None:
        self.messages.append(
            CommandMessage(
                text=text,
                channel=channel,
                title=title,
                right_info=right_info,
                agent_name=agent_name,
                render_markdown=render_markdown,
            )
        )
