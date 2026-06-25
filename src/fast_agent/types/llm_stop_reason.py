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
