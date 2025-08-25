from anthropic.types import (
    Message,
    MessageParam,
)
from mcp import StopReason
from mcp.types import (
    PromptMessage,
)

from mcp_agent.llm.providers.multipart_converter_anthropic import (
    AnthropicConverter,
)
from mcp_agent.llm.sampling_format_converter import ProviderFormatConverter
from mcp_agent.logging.logger import get_logger

_logger = get_logger(__name__)


class AnthropicSamplingConverter(ProviderFormatConverter[MessageParam, Message]):
    """
    Convert between Anthropic and MCP types.
    """

    @classmethod
    def from_prompt_message(cls, message: PromptMessage) -> MessageParam:
        """Convert an MCP PromptMessage to an Anthropic MessageParam."""
        return AnthropicConverter.convert_prompt_message_to_anthropic(message)


def mcp_stop_reason_to_anthropic_stop_reason(stop_reason: StopReason):
    if not stop_reason:
        return None
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
