"""LLM-related type definitions for fast-agent."""

from enum import Enum
from typing import Union


class LlmStopReason(str, Enum):
    """
    Enumeration of stop reasons for LLM message generation.

    Extends the MCP SDK's standard stop reasons with additional custom values.
    Inherits from str to ensure compatibility with string-based APIs.
    Used primarily in PromptMessageExtended and LLM response handling.
    """

    # MCP SDK standard values (from mcp.types.StopReason)
    END_TURN = "endTurn"
    STOP_SEQUENCE = "stopSequence"
    MAX_TOKENS = "maxTokens"
    TOOL_USE = "toolUse"  # Used when LLM stops to call tools
    PAUSE = "pause"

    # Custom extensions for fast-agent
    ERROR = "error"  # Used when there's an error in generation

    TIMEOUT = "timeout"  # Used when generation times out
    SAFETY = "safety"  # a safety or content warning was triggered

    def __eq__(self, other: object) -> bool:
        """
        Allow comparison with both enum members and raw strings.

        This enables code like:
        - result.stopReason == LlmStopReason.END_TURN
        - result.stopReason == "endTurn"
        """
        if isinstance(other, str):
            return self.value == other
        return super().__eq__(other)

    @classmethod
    def from_string(cls, value: Union[str, "LlmStopReason"]) -> "LlmStopReason":
        """
        Convert a string to a LlmStopReason enum member.

        Args:
            value: A string or LlmStopReason enum member

        Returns:
            The corresponding LlmStopReason enum member

        Raises:
            ValueError: If the string doesn't match any enum value
        """
        if isinstance(value, cls):
            return value

        for member in cls:
            if member.value == value:
                return member

        raise ValueError(
            f"Invalid stop reason: {value}. Valid values are: {[m.value for m in cls]}"
        )

    @classmethod
    def is_valid(cls, value: str) -> bool:
        """
        Check if a string is a valid stop reason.

        Args:
            value: A string to check

        Returns:
            True if the string matches a valid stop reason, False otherwise
        """
        return value in [member.value for member in cls]
