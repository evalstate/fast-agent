
from mcp.types import (
    PromptMessage,
)
from openai.types.chat import ChatCompletionMessage, ChatCompletionMessageParam

from mcp_agent.llm.sampling_format_converter import (
    ProviderFormatConverter,
)
from mcp_agent.logging.logger import get_logger

_logger = get_logger(__name__)


class OpenAISamplingConverter(
    ProviderFormatConverter[ChatCompletionMessageParam, ChatCompletionMessage]
):
    @classmethod
    def from_prompt_message(cls, message: PromptMessage) -> ChatCompletionMessageParam:
        """Convert an MCP PromptMessage to an OpenAI message dict."""
        from mcp_agent.llm.providers.multipart_converter_openai import (
            OpenAIConverter,
        )

        # Use the full-featured OpenAI converter for consistent handling
        return OpenAIConverter.convert_prompt_message_to_openai(message)
