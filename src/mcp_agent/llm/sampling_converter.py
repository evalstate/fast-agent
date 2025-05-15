"""
Simplified converter between MCP sampling types and PromptMessageMultipart.
This replaces the more complex provider-specific converters with direct conversions.
"""

from typing import List, Optional

from mcp.types import (
    CreateMessageRequestParams,
    CreateMessageResult,
    SamplingMessage,
    TextContent,
)

from mcp_agent.mcp.interfaces import RequestParams
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart


class SamplingConverter:
    """
    Simplified converter between MCP sampling types and internal LLM types.

    This handles converting between:
    - SamplingMessage and PromptMessageMultipart
    - CreateMessageRequestParams and RequestParams
    - LLM responses and CreateMessageResult
    """

    @staticmethod
    def sampling_message_to_prompt_message(
        message: SamplingMessage,
    ) -> PromptMessageMultipart:
        """
        Convert a SamplingMessage to a PromptMessageMultipart.

        Args:
            message: MCP SamplingMessage to convert

        Returns:
            PromptMessageMultipart suitable for use with LLMs
        """
        return PromptMessageMultipart(role=message.role, content=[message.content])

    @staticmethod
    def extract_request_params(params: CreateMessageRequestParams) -> RequestParams:
        """
        Extract parameters from CreateMessageRequestParams into RequestParams.

        Args:
            params: MCP request parameters

        Returns:
            RequestParams suitable for use with LLM.generate_prompt
        """
        return RequestParams(
            maxTokens=params.maxTokens,
            systemPrompt=params.systemPrompt,
            temperature=params.temperature,
            stopSequences=params.stopSequences,
            modelPreferences=params.modelPreferences,
            # Add any other parameters needed
        )

    @staticmethod
    def error_result(error_message: str, model: Optional[str] = None) -> CreateMessageResult:
        """
        Create an error result.

        Args:
            error_message: Error message text
            model: Optional model identifier

        Returns:
            CreateMessageResult with error information
        """
        return CreateMessageResult(
            role="assistant",
            content=TextContent(type="text", text=error_message),
            model=model or "unknown",
            stopReason="error",
        )

    @staticmethod
    def convert_messages(
        messages: List[SamplingMessage],
    ) -> List[PromptMessageMultipart]:
        """
        Convert multiple SamplingMessages to PromptMessageMultipart objects.

        Args:
            messages: List of SamplingMessages to convert

        Returns:
            List of PromptMessageMultipart objects, each with a single content item
        """
        return [SamplingConverter.sampling_message_to_prompt_message(msg) for msg in messages]
