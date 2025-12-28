"""
Simplified converter between MCP sampling types and PromptMessageExtended.
This replaces the more complex provider-specific converters with direct conversions.
"""

from mcp.types import (
    AudioContent,
    CallToolRequest,
    CallToolRequestParams,
    CallToolResult,
    ContentBlock,
    CreateMessageRequestParams,
    CreateMessageResult,
    EmbeddedResource,
    ImageContent,
    ResourceLink,
    SamplingMessage,
    SamplingMessageContentBlock,
    TextContent,
    ToolResultContent,
    ToolUseContent,
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

        Handles all content types including ToolUseContent and ToolResultContent
        for multi-turn tool conversations.

        Args:
            message: MCP SamplingMessage to convert

        Returns:
            PromptMessageExtended suitable for use with LLMs
        """
        # Normalize content to a list
        content = message.content
        if not isinstance(content, list):
            content_list: list[SamplingMessageContentBlock] = [content]
        else:
            content_list = content

        supported_content: list[ContentBlock] = []
        tool_calls: dict[str, CallToolRequest] = {}
        tool_results: dict[str, CallToolResult] = {}

        for item in content_list:
            if isinstance(
                item, (TextContent, ImageContent, AudioContent, ResourceLink, EmbeddedResource)
            ):
                supported_content.append(item)
            elif isinstance(item, ToolUseContent):
                # Convert ToolUseContent to internal CallToolRequest format
                tool_calls[item.id] = CallToolRequest(
                    method="tools/call",
                    params=CallToolRequestParams(
                        name=item.name,
                        arguments=item.input,
                    ),
                )
            elif isinstance(item, ToolResultContent):
                # Convert ToolResultContent to internal CallToolResult format
                # Extract text from content list if present
                result_content = item.content if item.content else []
                # isError can be None, so we need to handle that
                is_error = getattr(item, "isError", None)
                tool_results[item.toolUseId] = CallToolResult(
                    content=result_content,
                    isError=is_error if is_error is not None else False,
                )

        return PromptMessageExtended(
            role=message.role,
            content=supported_content,
            tool_calls=tool_calls if tool_calls else None,
            tool_results=tool_results if tool_results else None,
        )

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
    def error_result(error_message: str, model: str | None = None) -> CreateMessageResult:
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
        messages: list[SamplingMessage],
    ) -> list[PromptMessageExtended]:
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

    @staticmethod
    def llm_response_to_sampling_content(
        response: PromptMessageExtended,
    ) -> list[SamplingMessageContentBlock]:
        """
        Convert an LLM response to sampling message content blocks.

        Handles both text content and tool calls, converting them to the
        appropriate MCP types for sampling responses.

        Args:
            response: The LLM response message

        Returns:
            List of content blocks suitable for CreateMessageResult
        """
        content_blocks: list[SamplingMessageContentBlock] = []

        # Add text and other standard content
        for item in response.content:
            if isinstance(item, (TextContent, ImageContent, AudioContent)):
                content_blocks.append(item)

        # Convert tool_calls to ToolUseContent
        if response.tool_calls:
            for tool_id, call_request in response.tool_calls.items():
                content_blocks.append(
                    ToolUseContent(
                        type="tool_use",
                        id=tool_id,
                        name=call_request.params.name,
                        input=call_request.params.arguments or {},
                    )
                )

        return content_blocks
