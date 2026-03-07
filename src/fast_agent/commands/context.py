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

    async def prompt_model_selection(
        self,
        *,
        initial_provider: str | None = None,
        default_model: str | None = None,
    ) -> str | None:
        """Prompt for a model selection and return a resolved model spec."""

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


async def noninteractive_prompt_selection(
    prompt: str,
    *,
    options: Sequence[str],
    allow_cancel: bool = False,
    default: str | None = None,
) -> str | None:
    """Default no-op selection prompt for non-interactive command IO."""
    del prompt, options, allow_cancel, default
    return None


async def noninteractive_prompt_model_selection(
    *,
    initial_provider: str | None = None,
    default_model: str | None = None,
) -> str | None:
    """Default no-op model picker for non-interactive command IO."""
    del initial_provider, default_model
    return None


async def noninteractive_prompt_argument(
    arg_name: str,
    *,
    description: str | None = None,
    required: bool = True,
) -> str | None:
    """Default no-op prompt-argument handler for non-interactive command IO."""
    del arg_name, description, required
    return None


class NonInteractiveCommandIOBase(CommandIO):
    """Shared no-op prompt/display behavior for non-interactive command IO."""

    async def prompt_text(
        self,
        prompt: str,
        *,
        default: str | None = None,
        allow_empty: bool = True,
    ) -> str | None:
        del prompt
        return default if allow_empty else None

    async def prompt_selection(
        self,
        prompt: str,
        *,
        options: Sequence[str],
        allow_cancel: bool = False,
        default: str | None = None,
    ) -> str | None:
        return await noninteractive_prompt_selection(
            prompt,
            options=options,
            allow_cancel=allow_cancel,
            default=default,
        )

    async def prompt_model_selection(
        self,
        *,
        initial_provider: str | None = None,
        default_model: str | None = None,
    ) -> str | None:
        return await noninteractive_prompt_model_selection(
            initial_provider=initial_provider,
            default_model=default_model,
        )

    async def prompt_argument(
        self,
        arg_name: str,
        *,
        description: str | None = None,
        required: bool = True,
    ) -> str | None:
        return await noninteractive_prompt_argument(
            arg_name,
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
        del agent_name, turn, turn_index, total_turns

    async def display_history_overview(
        self,
        agent_name: str,
        history: list[PromptMessageExtended],
        usage: "UsageAccumulator" | None = None,
    ) -> None:
        del agent_name, history, usage

    async def display_usage_report(self, agents: dict[str, object]) -> None:
        del agents

    async def display_system_prompt(
        self,
        agent_name: str,
        system_prompt: str,
        *,
        server_count: int = 0,
    ) -> None:
        del agent_name, system_prompt, server_count


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
