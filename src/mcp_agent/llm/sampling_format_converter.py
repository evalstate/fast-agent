from typing import Generic, List, Protocol, TypeVar

from mcp.types import PromptMessage

from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart

# Define covariant type variables
MessageParamT_co = TypeVar("MessageParamT_co", covariant=True)
MessageT_co = TypeVar("MessageT_co", covariant=True)


class ProviderFormatConverter(Protocol, Generic[MessageParamT_co, MessageT_co]):
    """Conversions between LLM provider and MCP types"""

    @classmethod
    def from_prompt_message(cls, message: PromptMessage) -> MessageParamT_co:
        """Convert an MCP PromptMessage to a provider-specific message parameter."""
        ...

    @classmethod
    def from_mutlipart_prompts(
        cls, messages: List[PromptMessageMultipart]
    ) -> List[MessageParamT_co]:
        """Convert a list of PromptMessageMultiparts to a list of provider-specific implementations"""
        ...


class BasicFormatConverter(ProviderFormatConverter[PromptMessage, PromptMessage]):
    @classmethod
    def from_prompt_message(cls, message: PromptMessage) -> PromptMessage:
        return message

    @classmethod
    def from_multipart_prompts(
        cls, messages: List[PromptMessageMultipart]
    ) -> List[PromptMessageMultipart]:
        return messages
