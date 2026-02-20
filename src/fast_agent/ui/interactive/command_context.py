"""Context helpers for interactive prompt command handling."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from fast_agent.commands.context import CommandContext
from fast_agent.config import get_settings
from fast_agent.ui.adapters import TuiCommandIO

if TYPE_CHECKING:
    from fast_agent.commands.context import AgentProvider
    from fast_agent.commands.results import CommandOutcome
    from fast_agent.core.agent_app import AgentApp


def build_command_context(prompt_provider: "AgentApp", agent_name: str) -> CommandContext:
    settings = get_settings()
    noenv_mode = bool(getattr(prompt_provider, "_noenv_mode", False))
    io = TuiCommandIO(
        prompt_provider=cast("AgentProvider", prompt_provider),
        agent_name=agent_name,
        settings=settings,
    )
    return CommandContext(
        agent_provider=cast("AgentProvider", prompt_provider),
        current_agent_name=agent_name,
        io=io,
        settings=settings,
        noenv=noenv_mode,
    )


async def emit_command_outcome(context: CommandContext, outcome: "CommandOutcome") -> None:
    for message in outcome.messages:
        await context.io.emit(message)
