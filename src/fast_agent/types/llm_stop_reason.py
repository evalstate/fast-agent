from __future__ import annotations

from enum import Enum


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
    CANCELLED = "cancelled"  # Used when generation is cancelled by user

    TIMEOUT = "timeout"  # Used when generation times out
    SAFETY = "safety"  # a safety or content warning was triggered

    @classmethod
    def from_string(cls, value: str | "LlmStopReason") -> "LlmStopReason":
        """
        Convert a string to a LlmStopReason enum member.

        Args:
            value: A string or LlmStopReason enum member

        Returns:
            The corresponding LlmStopReason enum member

        Raises:
            ValueError: If the string doesn't match any enum value
        """
        try:
            return cls(value)
        except ValueError as exc:
            valid_values = [member.value for member in cls]
            raise ValueError(
                f"Invalid stop reason: {value}. Valid values are: {valid_values}"
            ) from exc

    @classmethod
    def is_valid(cls, value: str) -> bool:
        """
        Check if a string is a valid stop reason.

        Args:
            value: A string to check

        Returns:
            True if the string matches a valid stop reason, False otherwise
        """
        try:
            cls(value)
        except ValueError:
            return False
        return True
