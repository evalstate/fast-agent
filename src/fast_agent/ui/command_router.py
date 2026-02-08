"""TUI command router skeleton for shared command handlers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Awaitable, Callable

from fast_agent.commands.results import CommandOutcome
from fast_agent.ui.command_payloads import CommandPayload

if TYPE_CHECKING:
    from fast_agent.commands.context import CommandContext


CommandDispatcher = Callable[[CommandPayload], Awaitable[CommandOutcome]]


@dataclass(slots=True)
class CommandRouter:
    """Dispatches parsed command payloads to shared command handlers."""

    context: CommandContext
    dispatcher: CommandDispatcher

    async def dispatch(self, payload: CommandPayload) -> CommandOutcome:
        return await self.dispatcher(payload)
