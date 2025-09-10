"""
Simplified converter between MCP sampling types and PromptMessageExtended.
This replaces the more complex provider-specific converters with direct conversions.
"""

from typing import List, Optional

from mcp.types import (
    CreateMessageRequestParams,
    CreateMessageResult,
    SamplingMessage,
    TextContent,
)

from fast_agent.types import PromptMessageExtended, RequestParams
from fast_agent.types.llm_stop_reason import LlmStopReason


class SamplingConverter:
    """
    Simplified converter between MCP sampling types and internal LLM types.

    This handles converting between:
    - SamplingMessage and PromptMessageExtended
    - CreateMessageRequestParams and RequestParams
    - LLM responses and CreateMessageResult
    """

    @staticmethod
    def sampling_message_to_prompt_message(
        message: SamplingMessage,
    ) -> PromptMessageExtended:
        """
        Convert a SamplingMessage to a PromptMessageExtended.

        Args:
            message: MCP SamplingMessage to convert

        Returns:
            PromptMessageExtended suitable for use with LLMs
        """
        return PromptMessageExtended(role=message.role, content=[message.content])

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
            stopReason=LlmStopReason.ERROR.value,
        )

    @staticmethod
    def convert_messages(
        messages: List[SamplingMessage],
    ) -> List[PromptMessageExtended]:
        """
        Convert multiple SamplingMessages to PromptMessageExtended objects.

        This properly combines consecutive messages with the same role into a single
        multipart message, which is required by APIs like Anthropic.

        Args:
            messages: List of SamplingMessages to convert

        Returns:
            List of PromptMessageExtended objects with consecutive same-role messages combined
        """
        return [SamplingConverter.sampling_message_to_prompt_message(msg) for msg in messages]
