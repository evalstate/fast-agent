"""ACP command IO adapter for shared command handlers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from fast_agent.commands.context import (
    NonInteractiveCommandIOBase,
)
from fast_agent.commands.history_summaries import HistoryOverview, build_history_overview
from fast_agent.commands.status_summaries import SystemPromptSummary

if TYPE_CHECKING:
    from fast_agent.commands.results import CommandMessage
    from fast_agent.llm.usage_tracking import UsageAccumulator
    from fast_agent.types import PromptMessageExtended


@dataclass(slots=True)
class ACPCommandIO(NonInteractiveCommandIOBase):
    """Minimal ACP IO adapter that captures emitted messages."""

    messages: list["CommandMessage"] = field(default_factory=list)
    history_overview: HistoryOverview | None = None
    system_prompt: SystemPromptSummary | None = None

    async def emit(self, message: "CommandMessage") -> None:
        self.messages.append(message)

    async def display_history_overview(
        self,
        agent_name: str,
        history: list[PromptMessageExtended],
        usage: "UsageAccumulator" | None = None,
    ) -> None:
        self.history_overview = build_history_overview(history)

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
