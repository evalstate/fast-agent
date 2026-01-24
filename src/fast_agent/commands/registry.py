"""Command handler registry helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Awaitable, Callable, Mapping

from fast_agent.commands.context import CommandContext
from fast_agent.commands.results import CommandOutcome

CommandHandler = Callable[[CommandContext, object], Awaitable[CommandOutcome]]


@dataclass(slots=True)
class CommandRegistry:
    """Maps command keys to handler callables."""

    handlers: dict[str, CommandHandler] = field(default_factory=dict)

    def register(self, name: str, handler: CommandHandler) -> None:
        self.handlers[name] = handler

    def as_mapping(self) -> Mapping[str, CommandHandler]:
        return dict(self.handlers)
