"""
Conversation statistics and analysis utilities.

This module provides ConversationSummary for analyzing message history
and extracting useful statistics like tool call counts, error rates, etc.
"""

from collections import Counter
from typing import Dict, List

from pydantic import BaseModel, computed_field

from fast_agent.mcp.prompt_message_extended import PromptMessageExtended


class ConversationSummary(BaseModel):
    """
    Analyzes a conversation's message history and provides computed statistics.

    This class takes a list of PromptMessageExtended messages and provides
    convenient computed properties for common statistics like tool call counts,
    error rates, and per-tool breakdowns.

    Example:
        ```python
        from fast_agent import ConversationSummary

        # After running an agent
        summary = ConversationSummary(agent.message_history)

        # Access computed statistics
        print(f"Tool calls: {summary.tool_calls}")
        print(f"Tool errors: {summary.tool_errors}")
        print(f"Error rate: {summary.tool_error_rate:.1%}")
        print(f"Tool breakdown: {summary.tool_call_map}")

        # Export to dict for CSV/JSON
        data = summary.model_dump()
        ```

    All computed properties are included in .model_dump() for easy serialization.
    """

    messages: List[PromptMessageExtended]

    @computed_field
    @property
    def message_count(self) -> int:
        """Total number of messages in the conversation."""
        return len(self.messages)

    @computed_field
    @property
    def user_message_count(self) -> int:
        """Number of messages from the user."""
        return sum(1 for msg in self.messages if msg.role == "user")

    @computed_field
    @property
    def assistant_message_count(self) -> int:
        """Number of messages from the assistant."""
        return sum(1 for msg in self.messages if msg.role == "assistant")

    @computed_field
    @property
    def tool_calls(self) -> int:
        """Total number of tool calls made across all messages."""
        return sum(
            len(msg.tool_calls) for msg in self.messages if msg.tool_calls
        )

    @computed_field
    @property
    def tool_errors(self) -> int:
        """Total number of tool calls that resulted in errors."""
        return sum(
            sum(1 for result in msg.tool_results.values() if result.isError)
            for msg in self.messages if msg.tool_results
        )

    @computed_field
    @property
    def tool_successes(self) -> int:
        """Total number of tool calls that completed successfully."""
        return sum(
            sum(1 for result in msg.tool_results.values() if not result.isError)
            for msg in self.messages if msg.tool_results
        )

    @computed_field
    @property
    def tool_error_rate(self) -> float:
        """
        Proportion of tool calls that resulted in errors (0.0 to 1.0).
        Returns 0.0 if there were no tool calls.
        """
        total_results = self.tool_errors + self.tool_successes
        if total_results == 0:
            return 0.0
        return self.tool_errors / total_results

    @computed_field
    @property
    def tool_call_map(self) -> Dict[str, int]:
        """
        Mapping of tool names to the number of times they were called.

        Example: {"fetch_weather": 3, "calculate": 1}
        """
        tool_names: List[str] = []
        for msg in self.messages:
            if msg.tool_calls:
                tool_names.extend(
                    call.params.name for call in msg.tool_calls.values()
                )
        return dict(Counter(tool_names))

    @computed_field
    @property
    def tool_error_map(self) -> Dict[str, int]:
        """
        Mapping of tool names to the number of errors they produced.

        Example: {"fetch_weather": 1, "invalid_tool": 2}

        Note: This maps tool call IDs back to their original tool names by
        finding corresponding CallToolRequest entries in assistant messages.
        """
        # First, build a map from tool_id -> tool_name by scanning tool_calls
        tool_id_to_name: Dict[str, str] = {}
        for msg in self.messages:
            if msg.tool_calls:
                for tool_id, call in msg.tool_calls.items():
                    tool_id_to_name[tool_id] = call.params.name

        # Then, count errors by tool name
        error_names: List[str] = []
        for msg in self.messages:
            if msg.tool_results:
                for tool_id, result in msg.tool_results.items():
                    if result.isError:
                        # Look up the tool name from the tool_id
                        tool_name = tool_id_to_name.get(tool_id, "unknown")
                        error_names.append(tool_name)

        return dict(Counter(error_names))

    @computed_field
    @property
    def has_tool_calls(self) -> bool:
        """Whether any tool calls were made in this conversation."""
        return self.tool_calls > 0

    @computed_field
    @property
    def has_tool_errors(self) -> bool:
        """Whether any tool errors occurred in this conversation."""
        return self.tool_errors > 0
