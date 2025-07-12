"""Silent LLM implementation that suppresses display output while maintaining functionality."""

from typing import Any

from mcp_agent.llm.augmented_llm_passthrough import PassthroughLLM
from mcp_agent.llm.provider_types import Provider
from mcp_agent.llm.usage_tracking import TurnUsage, UsageAccumulator


class ZeroUsageAccumulator(UsageAccumulator):
    """Usage accumulator that always reports zero usage."""
    
    def add_turn(self, turn: TurnUsage) -> None:
        """Override to do nothing - no usage accumulation."""
        pass


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
    
    def show_user_message(self, message: Any, **kwargs) -> None:
        """Override to suppress user message display."""
        pass
    
    async def show_assistant_message(self, message: Any, **kwargs) -> None:
        """Override to suppress assistant message display."""
        pass
    
    def show_tool_calls(self, tool_calls: Any, **kwargs) -> None:
        """Override to suppress tool call display."""
        pass
    
    def show_tool_results(self, tool_results: Any, **kwargs) -> None:
        """Override to suppress tool result display."""
        pass