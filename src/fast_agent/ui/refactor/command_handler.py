"""
CommandHandler skeleton that delegates a few read-only commands to
shared command handlers. This is an incremental migration step; more
commands will be moved here in later commits.
"""

from typing import Any, cast

from fast_agent.commands.context import AgentProvider, CommandContext
from fast_agent.commands.handlers import prompts as prompt_handlers
from fast_agent.commands.handlers import skills as skills_handlers
from fast_agent.commands.handlers import tools as tools_handlers
from fast_agent.config import get_settings
from fast_agent.ui.adapters import TuiCommandIO
from fast_agent.ui.command_payloads import (
    CommandPayload,
    ListPromptsCommand,
    ListSkillsCommand,
    ListToolsCommand,
)


class CommandHandler:
    """Minimal CommandHandler that delegates a few read-only commands."""

    def __init__(self, agent_types: dict[str, Any] | None = None) -> None:
        self._agent_types = agent_types

    async def handle(
        self, payload: CommandPayload, agent: str, prompt_provider, display
    ) -> None:  # type: ignore[type-arg]
        """Handle a parsed CommandPayload for read-only commands.

        Currently supports: /prompts, /tools, /skills. Others will raise
        NotImplementedError until migrated.
        """
        match payload:
            case ListPromptsCommand():
                settings = get_settings()
                io = TuiCommandIO(
                    prompt_provider=cast("AgentProvider", prompt_provider),
                    agent_name=agent,
                    settings=settings,
                )
                context = CommandContext(
                    agent_provider=cast("AgentProvider", prompt_provider),
                    current_agent_name=agent,
                    io=io,
                    settings=settings,
                )
                outcome = await prompt_handlers.handle_list_prompts(
                    context,
                    agent_name=agent,
                )
                for message in outcome.messages:
                    await io.emit(message)
                return
            case ListToolsCommand():
                settings = get_settings()
                io = TuiCommandIO(
                    prompt_provider=cast("AgentProvider", prompt_provider),
                    agent_name=agent,
                    settings=settings,
                )
                context = CommandContext(
                    agent_provider=cast("AgentProvider", prompt_provider),
                    current_agent_name=agent,
                    io=io,
                    settings=settings,
                )
                outcome = await tools_handlers.handle_list_tools(
                    context,
                    agent_name=agent,
                )
                for message in outcome.messages:
                    await io.emit(message)
                return
            case ListSkillsCommand():
                settings = get_settings()
                io = TuiCommandIO(
                    prompt_provider=cast("AgentProvider", prompt_provider),
                    agent_name=agent,
                    settings=settings,
                )
                context = CommandContext(
                    agent_provider=cast("AgentProvider", prompt_provider),
                    current_agent_name=agent,
                    io=io,
                    settings=settings,
                )
                outcome = await skills_handlers.handle_list_skills(
                    context,
                    agent_name=agent,
                )
                for message in outcome.messages:
                    await io.emit(message)
                return
            case _:
                raise NotImplementedError("CommandHandler.handle: command not migrated yet")


__all__ = ["CommandHandler"]
