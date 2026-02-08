"""Context and IO abstraction for shared command handlers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Awaitable, Iterable, Protocol, Sequence

from fast_agent.config import Settings, get_settings

if TYPE_CHECKING:
    from fast_agent.commands.results import CommandMessage
    from fast_agent.llm.usage_tracking import UsageAccumulator
    from fast_agent.types import PromptMessageExtended


class CommandIO(Protocol):
    """UI/transport specific IO operations used by shared command handlers."""

    async def emit(self, message: CommandMessage) -> None:
        """Display a message in the current UI."""

    async def prompt_text(
        self,
        prompt: str,
        *,
        default: str | None = None,
        allow_empty: bool = True,
    ) -> str | None:
        """Prompt for free-form text input."""

    async def prompt_selection(
        self,
        prompt: str,
        *,
        options: Sequence[str],
        allow_cancel: bool = False,
        default: str | None = None,
    ) -> str | None:
        """Prompt for a selection from a list of options."""

    async def prompt_argument(
        self,
        arg_name: str,
        *,
        description: str | None = None,
        required: bool = True,
    ) -> str | None:
        """Prompt for a prompt argument value."""

    async def display_history_turn(
        self,
        agent_name: str,
        turn: list[PromptMessageExtended],
        *,
        turn_index: int | None = None,
        total_turns: int | None = None,
    ) -> None:
        """Display a history turn with rich formatting."""

    async def display_history_overview(
        self,
        agent_name: str,
        history: list[PromptMessageExtended],
        usage: "UsageAccumulator" | None = None,
    ) -> None:
        """Display a conversation history overview."""

    async def display_usage_report(self, agents: dict[str, object]) -> None:
        """Display a usage report for the provided agents."""

    async def display_system_prompt(
        self,
        agent_name: str,
        system_prompt: str,
        *,
        server_count: int = 0,
    ) -> None:
        """Display the system prompt for the active agent."""


class AgentProvider(Protocol):
    """Minimum provider surface for shared command handlers (expand as needed)."""

    def _agent(self, name: str): ...

    def agent_names(self) -> Iterable[str]: ...

    def list_prompts(
        self,
        namespace: str | None,
        agent_name: str | None = None,
    ) -> Awaitable[object]: ...


@dataclass(slots=True)
class CommandContext:
    """Context passed to shared command handlers."""

    agent_provider: AgentProvider
    current_agent_name: str
    io: CommandIO
    settings: Settings | None = None
    noenv: bool = False

    def resolve_settings(self) -> Settings:
        return self.settings or get_settings()
