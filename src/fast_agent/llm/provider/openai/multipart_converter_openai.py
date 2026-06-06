import json
from collections.abc import Iterable, Mapping
from typing import Any, Union

from mcp.types import (
    AudioContent,
    BlobResourceContents,
    CallToolRequest,
    CallToolResult,
    EmbeddedResource,
    ImageContent,
    PromptMessage,
    ResourceLink,
    TextContent,
    TextResourceContents,
)
from mcp.types import (
    ContentBlock as MCPContentBlock,
)
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCallParam,
    ChatCompletionToolMessageParam,
    ChatCompletionUserMessageParam,
)
from openai.types.chat.chat_completion_assistant_message_param import ContentArrayOfContentPart
from openai.types.chat.chat_completion_tool_message_param import ChatCompletionContentPartTextParam
from openai.types.chat.chat_completion_user_message_param import ChatCompletionContentPartParam

from fast_agent.core.logging.logger import get_logger
from fast_agent.mcp.helpers.content_helpers import (
    canonicalize_tool_result_content_for_llm,
    get_image_data,
    get_resource_uri,
    get_text,
    is_image_content,
    is_resource_content,
    is_resource_link,
    is_text_content,
)
from fast_agent.mcp.mime_utils import (
    guess_mime_type,
    is_document_mime_type,
    is_image_mime_type,
    is_text_mime_type,
)
from fast_agent.types import PromptMessageExtended

_logger = get_logger("multipart_converter_openai")

# Define type aliases for content blocks
ContentBlock = dict[str, Any]
OpenAIMessage = dict[str, Any]
McpResourceContents = BlobResourceContents | TextResourceContents
type OpenAITextExtractableBlock = (
    ChatCompletionContentPartParam
    | ContentArrayOfContentPart
    | ChatCompletionContentPartTextParam
    | Mapping[str, Any]
)
type OpenAITextExtractableContent = str | Iterable[OpenAITextExtractableBlock] | None


class OpenAIConverter:
    """Converts MCP message types to OpenAI API format."""

    @staticmethod
    def _make_message(role: str, content: Any) -> ChatCompletionMessageParam:
        """Create a properly typed message based on role."""
        if role == "assistant":
            return ChatCompletionAssistantMessageParam(role="assistant", content=content)
        if role == "user":
            return ChatCompletionUserMessageParam(role="user", content=content)
        if role == "tool":
            # Tool messages need tool_call_id, but this helper is for simple content messages
            # Tool messages are handled separately in convert_tool_result_to_openai
            return ChatCompletionUserMessageParam(role="user", content=content)
        # Default to user for unknown roles (system messages handled elsewhere)
        return ChatCompletionUserMessageParam(role="user", content=content)

    @staticmethod
    def _is_supported_image_type(mime_type: str) -> bool:
        """
        Check if the given MIME type is supported by OpenAI's image API.

        Args:
            mime_type: The MIME type to check

        Returns:
            True if the MIME type is generally supported, False otherwise
        """
        return (
            mime_type is not None and is_image_mime_type(mime_type) and mime_type != "image/svg+xml"
        )

    @staticmethod
    def convert_to_openai(
        multipart_msg: PromptMessageExtended, concatenate_text_blocks: bool = False
    ) -> list[ChatCompletionMessageParam]:
        """
        Convert a PromptMessageExtended message to OpenAI API format.

        Args:
            multipart_msg: The PromptMessageExtended message to convert
            concatenate_text_blocks: If True, adjacent text blocks will be combined

        Returns:
            A list of OpenAI API message objects
        """
        # If this is an assistant message that contains tool_calls, convert to an
        # assistant message with tool_calls per OpenAI format to establish the
        # required call IDs before tool responses appear.
        if multipart_msg.role == "assistant" and multipart_msg.tool_calls:
            return [
                ChatCompletionAssistantMessageParam(
                    role="assistant",
                    tool_calls=OpenAIConverter._convert_tool_calls_to_openai(
                        multipart_msg.tool_calls
                    ),
                    content=OpenAIConverter._assistant_tool_call_content(multipart_msg.content),
                )
            ]

        # Handle tool_results first if present
        if multipart_msg.tool_results:
            messages = OpenAIConverter.convert_function_results_to_openai(
                multipart_msg.tool_results, concatenate_text_blocks
            )

            # If there's also content, convert and append it
            if multipart_msg.content:
                role = multipart_msg.role
                content_msg = OpenAIConverter._convert_content_to_message(
                    multipart_msg.content, role, concatenate_text_blocks
                )
                if content_msg:  # Only append if non-empty
                    messages.append(content_msg)

            return messages

        # Regular content conversion (no tool_results)
        role = multipart_msg.role
        content_msg = OpenAIConverter._convert_content_to_message(
            multipart_msg.content, role, concatenate_text_blocks
        )
        return [content_msg] if content_msg else []

    @staticmethod
    def _convert_tool_calls_to_openai(
        tool_calls: Mapping[str, CallToolRequest],
    ) -> list[ChatCompletionMessageToolCallParam]:
        return [
            ChatCompletionMessageToolCallParam(
                id=tool_id,
                type="function",
                function={
                    "name": request.params.name,
                    "arguments": json.dumps(request.params.arguments or {}),
                },
            )
            for tool_id, request in tool_calls.items()
        ]

    @staticmethod
    def _assistant_tool_call_content(content: list) -> str:
        if not content:
            return ""
        content_message = OpenAIConverter._convert_content_to_message(content, "assistant")
        if content_message is None:
            return ""
        return OpenAIConverter._extract_text_from_content_blocks(content_message.get("content"))

    @staticmethod
    def _convert_content_to_message(
        content: list, role: str, concatenate_text_blocks: bool = False
    ) -> ChatCompletionMessageParam | None:
        """
        Convert content blocks to a single OpenAI message.

        Args:
            content: List of content blocks
            role: The message role
            concatenate_text_blocks: If True, adjacent text blocks will be combined

        Returns:
            An OpenAI message dict or None if content is empty
        """
        # Handle empty content
        if not content:
            return OpenAIConverter._make_message(role, "")

        # single text block
        if len(content) == 1 and is_text_content(content[0]):
            return OpenAIConverter._make_message(role, get_text(content[0]))

        # For user messages, convert each content block
        content_blocks: list[ContentBlock] = []

        _logger.debug(f"Converting {len(content)} content items for role '{role}'")

        for item in content:
            content_blocks.extend(OpenAIConverter._convert_content_item_to_blocks(item))

        if not content_blocks:
            return OpenAIConverter._make_message(role, "")

        # If concatenate_text_blocks is True, combine adjacent text blocks
        if concatenate_text_blocks:
            content_blocks = OpenAIConverter._concatenate_text_blocks(content_blocks)

        if role == "assistant":
            return OpenAIConverter._make_message(
                role,
                OpenAIConverter._extract_text_from_content_blocks(content_blocks),
            )

        # Return message with content blocks
        _logger.debug(f"Final message for role '{role}': {len(content_blocks)} content blocks")
        return OpenAIConverter._make_message(role, content_blocks)

    @staticmethod
    def _convert_content_item_to_blocks(item: Any) -> list[ContentBlock]:
        try:
            block = OpenAIConverter._convert_content_item(item)
            if block is None:
                return []
            return [block]
        except Exception as e:
            _logger.warning(f"Error converting content item: {e}")
            fallback_text = f"[Content conversion error: {e!s}]"
            return [{"type": "text", "text": fallback_text}]

    @staticmethod
    def _convert_content_item(item: Any) -> ContentBlock | None:
        if is_text_content(item):
            return {"type": "text", "text": get_text(item)}

        if is_image_content(item):
            image_block = OpenAIConverter._convert_image_content(item)
            _logger.debug(f"Added image content block: {image_block.get('type', 'unknown')}")
            return image_block

        if isinstance(item, AudioContent):
            return OpenAIConverter._convert_audio_content(item)

        if is_resource_content(item):
            return OpenAIConverter._convert_embedded_resource(item)

        if is_resource_link(item):
            return OpenAIConverter._convert_resource_link_content(item)

        _logger.warning(f"Unsupported content type: {type(item)}")
        return {"type": "text", "text": f"[Unsupported content type: {type(item).__name__}]"}

    @staticmethod
    def _convert_audio_content(item: AudioContent) -> ContentBlock:
        mime_type = item.mimeType or "audio"
        return {"type": "text", "text": f"[Unsupported audio content: {mime_type}]"}

    @staticmethod
    def _convert_resource_link_content(item: ResourceLink) -> ContentBlock | None:
        uri = item.uri
        mime_type = item.mimeType
        if uri and mime_type and OpenAIConverter._is_supported_image_type(mime_type):
            return {"type": "image_url", "image_url": {"url": str(uri)}}
        if uri and mime_type and is_document_mime_type(mime_type):
            return OpenAIConverter._convert_resource_link_document(item, str(uri))

        text = get_text(item)
        if text:
            return {"type": "text", "text": text}
        return None

    @staticmethod
    def _concatenate_text_blocks(blocks: list[ContentBlock]) -> list[ContentBlock]:
        """
        Combine adjacent text blocks into single blocks.

        Args:
            blocks: List of content blocks

        Returns:
            List with adjacent text blocks combined
        """
        if not blocks:
            return []

        combined_blocks: list[ContentBlock] = []
        current_text = ""

        for block in blocks:
            if block["type"] == "text":
                # Add to current text accumulator
                if current_text:
                    current_text += " " + block["text"]
                else:
                    current_text = block["text"]
            else:
                # Non-text block found, flush accumulated text if any
                if current_text:
                    combined_blocks.append({"type": "text", "text": current_text})
                    current_text = ""
                # Add the non-text block
                combined_blocks.append(block)

        # Don't forget any remaining text
        if current_text:
            combined_blocks.append({"type": "text", "text": current_text})

        return combined_blocks

    @staticmethod
    def convert_prompt_message_to_openai(
        message: PromptMessage, concatenate_text_blocks: bool = False
    ) -> ChatCompletionMessageParam:
        """
        Convert a standard PromptMessage to OpenAI API format.

        Args:
            message: The PromptMessage to convert
            concatenate_text_blocks: If True, adjacent text blocks will be combined

        Returns:
            An OpenAI API message object
        """
        # Convert the PromptMessage to a PromptMessageExtended containing a single content item
        multipart = PromptMessageExtended(role=message.role, content=[message.content])

        # Use the existing conversion method with the specified concatenation option
        # Since convert_to_openai now returns a list, we return the first element
        messages = OpenAIConverter.convert_to_openai(multipart, concatenate_text_blocks)
        return messages[0] if messages else OpenAIConverter._make_message(message.role, "")

    @staticmethod
    def _convert_image_content(content: ImageContent) -> ContentBlock:
        """Convert ImageContent to OpenAI image_url content block."""
        # Get image data using helper
        image_data = get_image_data(content)

        # OpenAI requires image URLs or data URIs for images
        image_url = {"url": f"data:{content.mimeType};base64,{image_data}"}

        return {"type": "image_url", "image_url": image_url}

    @staticmethod
    def _determine_mime_type(resource_content: McpResourceContents) -> str:
        """
        Determine the MIME type of a resource.

        Args:
            resource_content: The resource content to check

        Returns:
            The determined MIME type as a string
        """
        if resource_content.mimeType:
            return resource_content.mimeType

        return guess_mime_type(str(resource_content.uri))

    @staticmethod
    def _is_binary_resource_content(resource_content: McpResourceContents) -> bool:
        return isinstance(resource_content, BlobResourceContents)

    @staticmethod
    def _convert_embedded_resource(
        resource: EmbeddedResource,
    ) -> ContentBlock | None:
        """
        Convert EmbeddedResource to appropriate OpenAI content block.

        Args:
            resource: The embedded resource to convert

        Returns:
            An appropriate OpenAI content block or None if conversion failed
        """
        resource_content = resource.resource
        uri_str = get_resource_uri(resource)
        uri = resource_content.uri
        is_url = uri and str(uri).startswith(("http://", "https://"))
        from fast_agent.mcp.resource_utils import extract_title_from_uri

        title = extract_title_from_uri(uri) if uri else "resource"
        mime_type = OpenAIConverter._determine_mime_type(resource_content)

        if OpenAIConverter._is_supported_image_type(mime_type):
            return OpenAIConverter._convert_embedded_image_resource(
                resource,
                title=title,
                mime_type=mime_type,
                uri_str=uri_str,
                is_url=bool(is_url),
            )

        if mime_type == "application/pdf":
            return OpenAIConverter._convert_embedded_pdf_resource(
                resource_content,
                title=title,
                uri_str=uri_str,
                is_url=bool(is_url),
            )

        if mime_type == "image/svg+xml" or is_text_mime_type(mime_type):
            return OpenAIConverter._convert_embedded_text_resource(
                resource, title=title, mime_type=mime_type
            )

        text = get_text(resource)
        if text:
            return {"type": "text", "text": text}

        # Default fallback for binary resources
        if OpenAIConverter._is_binary_resource_content(resource_content):
            return {
                "type": "text",
                "text": f"[Binary resource: {title} ({mime_type})]",
            }

        # Last resort fallback
        return {
            "type": "text",
            "text": f"[Unsupported resource: {title} ({mime_type})]",
        }

    @staticmethod
    def _convert_embedded_image_resource(
        resource: EmbeddedResource,
        *,
        title: str,
        mime_type: str,
        uri_str: str | None,
        is_url: bool,
    ) -> ContentBlock:
        if is_url and uri_str:
            return {"type": "image_url", "image_url": {"url": uri_str}}

        image_data = get_image_data(resource)
        if image_data:
            return {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{image_data}"},
            }
        return {"type": "text", "text": f"[Image missing data: {title}]"}

    @staticmethod
    def _convert_embedded_pdf_resource(
        resource_content: McpResourceContents,
        *,
        title: str,
        uri_str: str | None,
        is_url: bool,
    ) -> ContentBlock | None:
        filename = title or "document.pdf"
        if is_url and uri_str:
            return OpenAIConverter._build_file_part(filename, file_url=uri_str)
        if isinstance(resource_content, BlobResourceContents):
            return {
                "type": "file",
                "file": {
                    "filename": filename,
                    "file_data": f"data:application/pdf;base64,{resource_content.blob}",
                },
            }
        return None

    @staticmethod
    def _convert_embedded_text_resource(
        resource: EmbeddedResource, *, title: str, mime_type: str
    ) -> ContentBlock | None:
        text = get_text(resource)
        if not text:
            return None

        file_text = (
            f'<fastagent:file title="{title}" mimetype="{mime_type}">\n{text}\n</fastagent:file>'
        )
        return {"type": "text", "text": file_text}

    @staticmethod
    def _build_file_part(
        filename: str,
        *,
        file_data: str | None = None,
        file_url: str | None = None,
    ) -> ContentBlock:
        file_block: dict[str, str] = {"filename": filename}
        if file_data:
            file_block["file_data"] = file_data
        if file_url:
            file_block["file_url"] = file_url
        return {"type": "file", "file": file_block}

    @staticmethod
    def _convert_resource_link_document(
        resource,
        uri_str: str,
    ) -> ContentBlock:
        from fast_agent.mcp.resource_utils import extract_title_from_uri

        filename = resource.name or extract_title_from_uri(resource.uri) or "document"
        return OpenAIConverter._build_file_part(
            filename,
            file_url=uri_str,
        )

    @staticmethod
    def _extract_text_from_content_blocks(
        content: OpenAITextExtractableContent,
    ) -> str:
        """
        Extract and combine text from content blocks.

        Args:
            content: Content blocks or string

        Returns:
            Combined text as a string
        """
        if content is None:
            return ""
        if isinstance(content, str):
            return content

        # Extract only text blocks
        text_parts: list[str] = []
        for block in content:
            if isinstance(block, Mapping) and block.get("type") == "text":
                text = block.get("text")
                if isinstance(text, str):
                    text_parts.append(text)

        return " ".join(text_parts) if text_parts else "[Complex content converted to text]"

    @staticmethod
    def convert_tool_result_to_openai(
        tool_result: CallToolResult,
        tool_call_id: str,
        concatenate_text_blocks: bool = False,
    ) -> Union[
        ChatCompletionMessageParam,
        tuple[ChatCompletionMessageParam, list[ChatCompletionMessageParam]],
    ]:
        """
        Convert a CallToolResult to an OpenAI tool message.

        If the result contains non-text elements, those are converted to separate user messages
        since OpenAI tool messages can only contain text.

        Args:
            tool_result: The tool result from a tool call
            tool_call_id: The ID of the associated tool use
            concatenate_text_blocks: If True, adjacent text blocks will be combined

        Returns:
            Either a single OpenAI message for the tool response (if text only),
            or a tuple containing the tool message and a list of additional messages for non-text content
        """
        canonical_content = canonicalize_tool_result_content_for_llm(
            tool_result,
            logger=_logger,
            source="openai.chat",
        )

        # Handle empty content case
        if not canonical_content:
            return ChatCompletionToolMessageParam(
                role="tool",
                tool_call_id=tool_call_id,
                content="[Tool completed successfully]",
            )

        # Separate text and non-text content
        text_content: list[MCPContentBlock] = []
        non_text_content: list[MCPContentBlock] = []

        for item in canonical_content:
            if isinstance(item, TextContent):
                text_content.append(item)
            else:
                non_text_content.append(item)

        # Create tool message with text content
        tool_message_content = ""
        if text_content:
            # Convert text content to OpenAI format
            temp_multipart = PromptMessageExtended(role="user", content=text_content)
            converted_messages = OpenAIConverter.convert_to_openai(
                temp_multipart, concatenate_text_blocks=concatenate_text_blocks
            )

            # Extract text from content blocks (convert_to_openai now returns a list)
            if converted_messages:
                tool_message_content = OpenAIConverter._extract_text_from_content_blocks(
                    converted_messages[0].get("content", "")
                )

        # Ensure we always have non-empty content for compatibility
        if not tool_message_content or tool_message_content.strip() == "":
            tool_message_content = "[Tool completed successfully]"

        # Create the tool message with just the text
        tool_message = ChatCompletionToolMessageParam(
            role="tool",
            tool_call_id=tool_call_id,
            content=tool_message_content,
        )

        # If there's no non-text content, return just the tool message
        if not non_text_content:
            return tool_message

        # Process non-text content as a separate user message
        non_text_multipart = PromptMessageExtended(role="user", content=non_text_content)

        # Convert to OpenAI format (returns a list now)
        user_messages = OpenAIConverter.convert_to_openai(non_text_multipart)

        # Debug logging to understand what's happening with image conversion
        _logger.debug(
            f"Tool result conversion: non_text_content={len(non_text_content)} items, "
            f"user_messages={len(user_messages)} messages"
        )
        if not user_messages:
            _logger.warning(
                f"No user messages generated for non-text content: {[type(item).__name__ for item in non_text_content]}"
            )

        return (tool_message, user_messages)

    @staticmethod
    def convert_function_results_to_openai(
        results: dict[str, CallToolResult],
        concatenate_text_blocks: bool = False,
    ) -> list[ChatCompletionMessageParam]:
        """
        Convert function call results to OpenAI messages.

        Args:
            results: Dictionary mapping tool_call_id to CallToolResult
            concatenate_text_blocks: If True, adjacent text blocks will be combined

        Returns:
            List of OpenAI API messages for tool responses
        """
        tool_messages = []
        user_messages = []
        has_mixed_content = False

        for tool_call_id, result in results.items():
            try:
                converted = OpenAIConverter.convert_tool_result_to_openai(
                    tool_result=result,
                    tool_call_id=tool_call_id,
                    concatenate_text_blocks=concatenate_text_blocks,
                )

                # Handle the case where we have mixed content and get back a tuple
                if isinstance(converted, tuple):
                    tool_message, additional_messages = converted
                    tool_messages.append(tool_message)
                    user_messages.extend(additional_messages)
                    has_mixed_content = True
                else:
                    # Single message case (text-only)
                    tool_messages.append(converted)
            except Exception as e:
                _logger.error(f"Failed to convert tool_call_id={tool_call_id}: {e}")
                # Create a basic tool response to prevent missing tool_call_id error
                fallback_message = ChatCompletionToolMessageParam(
                    role="tool",
                    tool_call_id=tool_call_id,
                    content=f"[Conversion error: {e!s}]",
                )
                tool_messages.append(fallback_message)

        # CONDITIONAL REORDERING: Only reorder if there are user messages (mixed content)
        if has_mixed_content and user_messages:
            # Reorder: All tool messages first (OpenAI sequence), then user messages (vision context)
            messages = tool_messages + user_messages
        else:
            # Pure tool responses - keep original order to preserve context (snapshots, etc.)
            messages = tool_messages
        return messages
