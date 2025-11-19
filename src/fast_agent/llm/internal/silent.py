"""Silent LLM implementation that suppresses display output while maintaining functionality."""

from typing import Any

from fast_agent.llm.internal.passthrough import PassthroughLLM
from fast_agent.llm.provider_types import Provider
from fast_agent.llm.usage_tracking import TurnUsage, UsageAccumulator


class ZeroUsageAccumulator(UsageAccumulator):
    """Usage accumulator that always reports zero usage."""

    def add_turn(self, turn: TurnUsage) -> None:
        """Override to do nothing - no usage accumulation."""
        pass


# TODO -- this won't work anymore
class SilentLLM(PassthroughLLM):
    """
    A specialized LLM that processes messages like PassthroughLLM but suppresses all display output.

    This is particularly useful for parallel agent workflows where the fan-in agent
    should aggregate results without polluting the console with intermediate output.
    Token counting is disabled - the model always reports zero usage.
    """

    def __init__(
        self, provider=Provider.FAST_AGENT, name: str = "Silent", **kwargs: dict[str, Any]
    ) -> None:
        super().__init__(name=name, provider=provider, **kwargs)
        # Override with zero usage accumulator - silent model reports no usage
        self.usage_accumulator = ZeroUsageAccumulator()

    def show_tool_calls(self, tool_calls: Any, **kwargs) -> None:
        """Override to suppress tool call display."""
        pass

    def show_tool_results(self, tool_results: Any, **kwargs) -> None:
        """Override to suppress tool result display."""
        pass
