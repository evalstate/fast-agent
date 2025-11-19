"""
Conversation statistics and analysis utilities.

This module provides ConversationSummary for analyzing message history
and extracting useful statistics like tool call counts, error rates, etc.
"""

import json
from collections import Counter
from typing import Dict, List

from pydantic import BaseModel, computed_field

from fast_agent.constants import FAST_AGENT_TIMING
from fast_agent.mcp.helpers.content_helpers import get_text
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended


class ConversationSummary(BaseModel):
    """
    Analyzes a conversation's message history and provides computed statistics.

    This class takes a list of PromptMessageExtended messages and provides
    convenient computed properties for common statistics like tool call counts,
    error rates, per-tool breakdowns, and timing information.

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

        # Timing statistics
        print(f"Total time: {summary.total_elapsed_time_ms}ms")
        print(f"Avg response time: {summary.average_assistant_response_time_ms}ms")

        # Export to dict for CSV/JSON
        data = summary.model_dump()
        ```

    All computed properties are included in .model_dump() for easy serialization.
    """

    messages: List[PromptMessageExtended]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def message_count(self) -> int:
        """Total number of messages in the conversation."""
        return len(self.messages)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def user_message_count(self) -> int:
        """Number of messages from the user."""
        return sum(1 for msg in self.messages if msg.role == "user")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def assistant_message_count(self) -> int:
        """Number of messages from the assistant."""
        return sum(1 for msg in self.messages if msg.role == "assistant")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def tool_calls(self) -> int:
        """Total number of tool calls made across all messages."""
        return sum(
            len(msg.tool_calls) for msg in self.messages if msg.tool_calls
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def tool_errors(self) -> int:
        """Total number of tool calls that resulted in errors."""
        return sum(
            sum(1 for result in msg.tool_results.values() if result.isError)
            for msg in self.messages if msg.tool_results
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def tool_successes(self) -> int:
        """Total number of tool calls that completed successfully."""
        return sum(
            sum(1 for result in msg.tool_results.values() if not result.isError)
            for msg in self.messages if msg.tool_results
        )

    @computed_field  # type: ignore[prop-decorator]
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

    @computed_field  # type: ignore[prop-decorator]
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

    @computed_field  # type: ignore[prop-decorator]
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

    @computed_field  # type: ignore[prop-decorator]
    @property
    def has_tool_calls(self) -> bool:
        """Whether any tool calls were made in this conversation."""
        return self.tool_calls > 0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def has_tool_errors(self) -> bool:
        """Whether any tool errors occurred in this conversation."""
        return self.tool_errors > 0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_elapsed_time_ms(self) -> float:
        """
        Total elapsed time in milliseconds across all assistant message generations.

        This sums the duration_ms from timing data stored in message channels.
        Only messages with FAST_AGENT_TIMING channel data are included.

        Returns:
            Total time in milliseconds, or 0.0 if no timing data is available.
        """
        total = 0.0
        for msg in self.messages:
            if msg.role == "assistant" and msg.channels:
                timing_blocks = msg.channels.get(FAST_AGENT_TIMING, [])
                if timing_blocks:
                    try:
                        # Parse timing data from first block
                        timing_text = get_text(timing_blocks[0])
                        if timing_text:
                            timing_data = json.loads(timing_text)
                            total += timing_data.get("duration_ms", 0)
                    except (json.JSONDecodeError, KeyError, IndexError):
                        # Skip messages with invalid timing data
                        continue
        return total

    @computed_field  # type: ignore[prop-decorator]
    @property
    def assistant_message_timings(self) -> List[Dict[str, float]]:
        """
        List of timing data for each assistant message.

        Returns a list of dicts containing start_time, end_time, and duration_ms
        for each assistant message that has timing data.

        Example:
            [
                {"start_time": 1234567890.123, "end_time": 1234567892.456, "duration_ms": 2333.0},
                {"start_time": 1234567893.789, "end_time": 1234567895.012, "duration_ms": 1223.0},
            ]
        """
        timings = []
        for msg in self.messages:
            if msg.role == "assistant" and msg.channels:
                timing_blocks = msg.channels.get(FAST_AGENT_TIMING, [])
                if timing_blocks:
                    try:
                        timing_text = get_text(timing_blocks[0])
                        if timing_text:
                            timing_data = json.loads(timing_text)
                            timings.append(timing_data)
                    except (json.JSONDecodeError, KeyError, IndexError):
                        # Skip messages with invalid timing data
                        continue
        return timings

    @computed_field  # type: ignore[prop-decorator]
    @property
    def average_assistant_response_time_ms(self) -> float:
        """
        Average response time in milliseconds for assistant messages.

        Returns:
            Average time in milliseconds, or 0.0 if no timing data is available.
        """
        timings = self.assistant_message_timings
        if not timings:
            return 0.0
        total = sum(t.get("duration_ms", 0) for t in timings)
        return total / len(timings)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def first_llm_start_time(self) -> float | None:
        """
        Timestamp when the first LLM call started.

        Returns:
            Unix timestamp (from perf_counter) or None if no timing data.
        """
        timings = self.assistant_message_timings
        if not timings:
            return None
        return timings[0].get("start_time")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def last_llm_end_time(self) -> float | None:
        """
        Timestamp when the last LLM call ended.

        Returns:
            Unix timestamp (from perf_counter) or None if no timing data.
        """
        timings = self.assistant_message_timings
        if not timings:
            return None
        return timings[-1].get("end_time")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def conversation_span_ms(self) -> float:
        """
        Wall-clock time from first LLM call start to last LLM call end.

        This represents the active conversation time, including:
        - All LLM inference time
        - All tool execution time between LLM calls
        - Agent orchestration overhead between turns

        This is different from total_elapsed_time_ms which only sums LLM call durations.

        Example:
            If you have 3 LLM calls (2s, 1.5s, 1s) with tool execution in between:
            - total_elapsed_time_ms = 4500ms (sum of LLM times only)
            - conversation_span_ms = 9000ms (first start to last end, includes everything)

        Returns:
            Time in milliseconds, or 0.0 if no timing data is available.
        """
        first_start = self.first_llm_start_time
        last_end = self.last_llm_end_time

        if first_start is None or last_end is None:
            return 0.0

        return round((last_end - first_start) * 1000, 2)
