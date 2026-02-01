"""HookContext provides a rich context object passed to card hooks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping, Protocol

from fast_agent.types.llm_stop_reason import LlmStopReason

if TYPE_CHECKING:
    from fast_agent.context import Context
    from fast_agent.interfaces import AgentProtocol, MessageHistoryAgentProtocol
    from fast_agent.llm.usage_tracking import UsageAccumulator
    from fast_agent.types import PromptMessageExtended


class HookRunner(Protocol):
    """Protocol for tool runners used by hooks."""

    @property
    def iteration(self) -> int: ...


@dataclass
class HookContext:
    """
    Rich context passed to card hooks.

    Provides access to the tool runner, agent, current message, and helper
    methods for common operations like history manipulation.

    Example usage in a hook function:
        async def my_hook(ctx: HookContext) -> None:
            print(f"Iteration {ctx.iteration}: {len(ctx.message_history)} messages")
            if ctx.is_turn_complete:
                # Manipulate history after turn completes
                history = ctx.message_history
                # ... modify history ...
                ctx.load_message_history(trimmed_history)
    """

    runner: HookRunner
    agent: MessageHistoryAgentProtocol
    message: PromptMessageExtended
    hook_type: str  # "before_llm_call", "after_llm_call", "after_turn_complete", etc.

    @property
    def agent_name(self) -> str:
        """Name of the agent running this hook."""
        return self.agent.name

    @property
    def iteration(self) -> int:
        """Current iteration number in the tool loop."""
        return self.runner.iteration

    @property
    def is_turn_complete(self) -> bool:
        """Whether the turn is complete (stop_reason is not TOOL_USE)."""
        return self.message.stop_reason not in (LlmStopReason.TOOL_USE, None)

    @property
    def message_history(self) -> list[PromptMessageExtended]:
        """Get the agent's current message history."""
        return self.agent.message_history

    @property
    def usage(self) -> "UsageAccumulator | None":
        """Return the usage accumulator when available (token stats)."""
        return getattr(self.agent, "usage_accumulator", None)

    @property
    def agent_registry(self) -> "Mapping[str, AgentProtocol] | None":
        """Return the active agent registry when configured."""
        return getattr(self.agent, "agent_registry", None)

    @property
    def context(self) -> "Context | None":
        """Return the agent's context if available."""
        return getattr(self.agent, "context", None)

    def get_agent(self, name: str) -> "AgentProtocol | None":
        """Lookup another agent by name when a registry is available."""
        getter = getattr(self.agent, "get_agent", None)
        if callable(getter):
            return getter(name)
        return None

    def load_message_history(self, messages: list[PromptMessageExtended]) -> None:
        """Replace the agent's message history with the given messages."""
        self.agent.load_message_history(messages)
