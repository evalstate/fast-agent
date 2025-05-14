"""
Sampling format converter for Amazon Bedrock.
Handles conversion between Fast-Agent message formats and Bedrock API formats.
"""

from typing import Any, Dict

from mcp import StopReason
from mcp.types import PromptMessage

from mcp_agent.llm.providers.multipart_converter_bedrock import BedrockConverter
from mcp_agent.llm.sampling_format_converter import ProviderFormatConverter
from mcp_agent.logging.logger import get_logger

_logger = get_logger(__name__)

# Type aliases for Bedrock message formats
BedrockMessageParam = Dict[str, Any]
BedrockMessage = Dict[str, Any]


class BedrockSamplingConverter(ProviderFormatConverter[BedrockMessageParam, BedrockMessage]):
    """
    Convert between Bedrock and MCP types for sampling.
    """

    @classmethod
    def from_prompt_message(cls, message: PromptMessage) -> BedrockMessageParam:
        """Convert an MCP PromptMessage to a Bedrock API message."""
        return BedrockConverter.convert_prompt_message_to_bedrock(message)


def mcp_stop_reason_to_bedrock_stop_reason(stop_reason: StopReason) -> str:
    """
    Convert MCP stop reason to Bedrock stop reason.
    
    Args:
        stop_reason: MCP stop reason
        
    Returns:
        Equivalent Bedrock stop reason
    """
    if not stop_reason:
        return "end_turn"
    elif stop_reason == "endTurn":
        return "end_turn"
    elif stop_reason == "maxTokens":
        return "max_tokens"
    elif stop_reason == "stopSequence":
        return "stop_sequence"
    elif stop_reason == "toolUse":
        return "tool_use"
    else:
        return stop_reason


def bedrock_stop_reason_to_mcp_stop_reason(stop_reason: str) -> StopReason:
    """
    Convert Bedrock stop reason to MCP stop reason.
    
    Args:
        stop_reason: Bedrock stop reason
        
    Returns:
        Equivalent MCP stop reason
    """
    if not stop_reason:
        return "endTurn"
    elif stop_reason == "end_turn":
        return "endTurn"
    elif stop_reason == "max_tokens":
        return "maxTokens"
    elif stop_reason == "stop_sequence":
        return "stopSequence"
    elif stop_reason == "tool_use":
        return "toolUse"
    else:
        return stop_reason