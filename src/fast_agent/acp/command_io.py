"""ACP command IO adapter for shared command handlers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Sequence

from fast_agent.commands.context import CommandIO
from fast_agent.commands.history_summaries import HistoryOverview, build_history_overview
from fast_agent.commands.status_summaries import SystemPromptSummary

if TYPE_CHECKING:
    from fast_agent.commands.results import CommandMessage
    from fast_agent.llm.usage_tracking import UsageAccumulator
    from fast_agent.types import PromptMessageExtended


@dataclass(slots=True)
class ACPCommandIO(CommandIO):
    """Minimal ACP IO adapter that captures emitted messages."""

    messages: list["CommandMessage"] = field(default_factory=list)
    history_overview: HistoryOverview | None = None
    system_prompt: SystemPromptSummary | None = None

    async def emit(self, message: "CommandMessage") -> None:
        self.messages.append(message)

    async def prompt_text(
        self,
        prompt: str,
        *,
        default: str | None = None,
        allow_empty: bool = True,
    ) -> str | None:
        return default if allow_empty else None

    async def prompt_selection(
        self,
        prompt: str,
        *,
        options: Sequence[str],
        allow_cancel: bool = False,
        default: str | None = None,
    ) -> str | None:
        return None

    async def prompt_argument(
        self,
        arg_name: str,
        *,
        description: str | None = None,
        required: bool = True,
    ) -> str | None:
        return None

    async def display_history_turn(
        self,
        agent_name: str,
        turn: list[PromptMessageExtended],
        *,
        turn_index: int | None = None,
        total_turns: int | None = None,
    ) -> None:
        return None

    async def display_history_overview(
        self,
        agent_name: str,
        history: list[PromptMessageExtended],
        usage: "UsageAccumulator" | None = None,
    ) -> None:
        self.history_overview = build_history_overview(history)

    async def display_usage_report(self, agents: dict[str, object]) -> None:
        return None

    async def display_system_prompt(
        self,
        agent_name: str,
        system_prompt: str,
        *,
        server_count: int = 0,
    ) -> None:
        self.system_prompt = SystemPromptSummary(
            agent_name=agent_name,
            system_prompt=system_prompt,
            server_count=server_count,
        )
