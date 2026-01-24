"""TUI adapter implementation for shared command handlers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from rich.text import Text

from fast_agent.commands.context import CommandIO
from fast_agent.config import Settings, get_settings
from fast_agent.ui.enhanced_prompt import get_argument_input, get_selection_input
from fast_agent.ui.history_actions import display_history_turn
from fast_agent.ui.message_primitives import MessageType

if TYPE_CHECKING:
    from collections.abc import Sequence

    from fast_agent.commands.context import AgentProvider
    from fast_agent.commands.results import CommandMessage
    from fast_agent.llm.usage_tracking import UsageAccumulator
    from fast_agent.types import PromptMessageExtended


@dataclass(slots=True)
class TuiCommandIO(CommandIO):
    """Command IO implementation backed by the interactive TUI."""

    prompt_provider: "AgentProvider"
    agent_name: str
    settings: Settings | None = None

    def _resolve_display(self, agent_name: str | None):
        from fast_agent.ui.console_display import ConsoleDisplay

        target_agent = None
        if agent_name and hasattr(self.prompt_provider, "_agent"):
            try:
                target_agent = self.prompt_provider._agent(agent_name)
            except Exception:
                target_agent = None

        display = getattr(target_agent, "display", None) if target_agent is not None else None
        if display is not None:
            return display

        config = None
        if target_agent is not None:
            agent_context = getattr(target_agent, "context", None)
            config = getattr(agent_context, "config", None) if agent_context else None

        if config is None:
            config = self.settings or get_settings()

        return ConsoleDisplay(config=config)

    async def emit(self, message: CommandMessage) -> None:
        display = self._resolve_display(message.agent_name or self.agent_name)
        content = message.text

        if not isinstance(content, Text):
            content = Text(str(content))

        if message.title:
            header = Text(message.title, style="bold")
            if content.plain:
                header.append("\n")
                header.append_text(content)
            content = header

        if message.channel == "error":
            content.stylize("red")
        elif message.channel == "warning":
            content.stylize("yellow")
        elif message.channel == "info":
            content.stylize("cyan")

        display.display_message(
            content=content,
            message_type=MessageType.SYSTEM,
            name=message.agent_name or self.agent_name,
            right_info=message.right_info or "",
            truncate_content=False,
        )

    async def prompt_text(
        self,
        prompt: str,
        *,
        default: str | None = None,
        allow_empty: bool = True,
    ) -> str | None:
        arg_name = prompt.rstrip(":")
        value = await get_argument_input(
            arg_name=arg_name,
            description=None,
            required=not allow_empty,
        )
        if value is None or value == "":
            return default if default is not None else value
        return value

    async def prompt_selection(
        self,
        prompt: str,
        *,
        options: Sequence[str],
        allow_cancel: bool = False,
        default: str | None = None,
    ) -> str | None:
        return await get_selection_input(
            prompt,
            options=list(options),
            allow_cancel=allow_cancel,
            default=default,
        )

    async def prompt_argument(
        self,
        arg_name: str,
        *,
        description: str | None = None,
        required: bool = True,
    ) -> str | None:
        return await get_argument_input(
            arg_name=arg_name,
            description=description,
            required=required,
        )

    async def display_history_turn(
        self,
        agent_name: str,
        turn: list[PromptMessageExtended],
        *,
        turn_index: int | None = None,
        total_turns: int | None = None,
    ) -> None:
        await display_history_turn(
            agent_name,
            turn,
            config=self.settings or get_settings(),
            turn_index=turn_index,
            total_turns=total_turns,
        )

    async def display_history_overview(
        self,
        agent_name: str,
        history: list[PromptMessageExtended],
        usage: "UsageAccumulator" | None = None,
    ) -> None:
        from fast_agent.ui.history_display import display_history_overview

        display_history_overview(agent_name, history, usage)

    async def display_usage_report(self, agents: dict[str, object]) -> None:
        from fast_agent.ui.usage_display import display_usage_report

        display_usage_report(agents, show_if_progress_disabled=True)

    async def display_system_prompt(
        self,
        agent_name: str,
        system_prompt: str,
        *,
        server_count: int = 0,
    ) -> None:
        display = self._resolve_display(agent_name)
        show_system = getattr(display, "show_system_message", None)
        if callable(show_system):
            show_system(system_prompt, agent_name=agent_name, server_count=server_count)
            return

        from fast_agent.ui.console_display import ConsoleDisplay

        ConsoleDisplay(config=self.settings or get_settings()).show_system_message(
            system_prompt,
            agent_name=agent_name,
            server_count=server_count,
        )
