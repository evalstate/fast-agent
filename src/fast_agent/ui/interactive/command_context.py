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
    from fast_agent.session.session_manager import SessionManager


def build_command_context(
    prompt_provider: "AgentApp",
    agent_name: str,
    *,
    session_manager: "SessionManager | None" = None,
) -> CommandContext:
    settings = get_settings()
    try:
        noenv_mode = prompt_provider.noenv_mode
    except AttributeError:
        noenv_mode = prompt_provider._noenv_mode
    io = TuiCommandIO(
        prompt_provider=cast("AgentProvider", prompt_provider),
        agent_name=agent_name,
        settings=settings,
    )
    effective_session_manager = session_manager
    if effective_session_manager is None and not noenv_mode:
        from fast_agent.session import get_session_manager

        effective_session_manager = get_session_manager()
    return CommandContext(
        agent_provider=cast("AgentProvider", prompt_provider),
        current_agent_name=agent_name,
        io=io,
        settings=settings,
        noenv=noenv_mode,
        session_manager=effective_session_manager,
    )


async def emit_command_outcome(context: CommandContext, outcome: "CommandOutcome") -> None:
    for message in outcome.messages:
        await context.io.emit(message)
