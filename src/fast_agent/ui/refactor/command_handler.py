"""
CommandHandler skeleton that delegates a few read-only commands to
the existing InteractivePrompt implementation. This is an incremental
migration step; more commands will be moved here in later commits.
"""

from typing import Any

from fast_agent.ui.interactive_prompt import InteractivePrompt
from fast_agent.ui.command_payloads import (
    ListPromptsCommand,
    ListToolsCommand,
    ListSkillsCommand,
    CommandPayload,
)


class CommandHandler:
    """Minimal CommandHandler that delegates a few read-only commands.
    """

    def __init__(self, agent_types: dict[str, Any] | None = None) -> None:
        self._impl = InteractivePrompt(agent_types=agent_types)

    async def handle(self, payload: CommandPayload, agent: str, prompt_provider, display) -> None:  # type: ignore[type-arg]
        """Handle a parsed CommandPayload for read-only commands.

        Currently supports: /prompts, /tools, /skills. Others will raise
        NotImplementedError until migrated.
        """
        match payload:
            case ListPromptsCommand():
                await self._impl._list_prompts(prompt_provider, agent)
                return
            case ListToolsCommand():
                await self._impl._list_tools(prompt_provider, agent)
                return
            case ListSkillsCommand():
                await self._impl._list_skills(prompt_provider, agent)
                return
            case _:
                raise NotImplementedError("CommandHandler.handle: command not migrated yet")


__all__ = ["CommandHandler"]
