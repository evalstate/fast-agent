import json
from typing import Any, Dict, List, Optional, Tuple, Union

from mcp.types import (
    CallToolResult,
    EmbeddedResource,
    ImageContent,
    PromptMessage,
    TextContent,
)
from openai.types.chat import ChatCompletionMessageParam

from fast_agent.core.logging.logger import get_logger
from fast_agent.mcp.helpers.content_helpers import (
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
    is_image_mime_type,
    is_text_mime_type,
)
from fast_agent.types import PromptMessageExtended

_logger = get_logger("multipart_converter_openai")

# Define type aliases for content blocks
ContentBlock = Dict[str, Any]
OpenAIMessage = Dict[str, Any]


class OpenAIConverter:
    """Converts MCP message types to OpenAI API format."""

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
    ) -> List[Dict[str, Any]]:
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
            tool_calls_list: List[Dict[str, Any]] = []
            for tool_id, req in multipart_msg.tool_calls.items():
                name = None
                arguments = {}
                try:
                    params = getattr(req, "params", None)
                    if params is not None:
                        name = getattr(params, "name", None)
                        arguments = getattr(params, "arguments", {}) or {}
                except Exception:
                    pass

                tool_calls_list.append(
                    {
                        "id": tool_id,
                        "type": "function",
                        "function": {
                            "name": name or "unknown_tool",
                            "arguments": json.dumps(arguments),
                        },
                    }
                )

            return [{"role": "assistant", "tool_calls": tool_calls_list, "content": ""}]

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
    def _convert_content_to_message(
        content: list, role: str, concatenate_text_blocks: bool = False
    ) -> Dict[str, Any] | None:
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
            return {"role": role, "content": ""}

        # single text block
        if 1 == len(content) and is_text_content(content[0]):
            return {"role": role, "content": get_text(content[0])}

        # For user messages, convert each content block
        content_blocks: List[ContentBlock] = []

        _logger.debug(f"Converting {len(content)} content items for role '{role}'")

        for item in content:
            try:
                if is_text_content(item):
                    text = get_text(item)
                    content_blocks.append({"type": "text", "text": text})

                elif is_image_content(item):
                    image_block = OpenAIConverter._convert_image_content(item)
                    content_blocks.append(image_block)
                    _logger.debug(
                        f"Added image content block: {image_block.get('type', 'unknown')}"
                    )

                elif is_resource_content(item):
                    block = OpenAIConverter._convert_embedded_resource(item)
                    if block:
                        content_blocks.append(block)

                elif is_resource_link(item):
                    text = get_text(item)
                    if text:
                        content_blocks.append({"type": "text", "text": text})

                else:
                    _logger.warning(f"Unsupported content type: {type(item)}")
                    # Create a text block with information about the skipped content
                    fallback_text = f"[Unsupported content type: {type(item).__name__}]"
                    content_blocks.append({"type": "text", "text": fallback_text})

            except Exception as e:
                _logger.warning(f"Error converting content item: {e}")
                # Create a text block with information about the conversion error
                fallback_text = f"[Content conversion error: {str(e)}]"
                content_blocks.append({"type": "text", "text": fallback_text})

        if not content_blocks:
            return {"role": role, "content": ""}

        # If concatenate_text_blocks is True, combine adjacent text blocks
        if concatenate_text_blocks:
            content_blocks = OpenAIConverter._concatenate_text_blocks(content_blocks)

        # Return user message with content blocks
        result = {"role": role, "content": content_blocks}
        _logger.debug(f"Final message for role '{role}': {len(content_blocks)} content blocks")
        return result

    @staticmethod
    def _concatenate_text_blocks(blocks: List[ContentBlock]) -> List[ContentBlock]:
        """
        Combine adjacent text blocks into single blocks.

        Args:
            blocks: List of content blocks

        Returns:
            List with adjacent text blocks combined
        """
        if not blocks:
            return []

        combined_blocks: List[ContentBlock] = []
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
        return messages[0] if messages else {"role": message.role, "content": ""}

    @staticmethod
    def _convert_image_content(content: ImageContent) -> ContentBlock:
        """Convert ImageContent to OpenAI image_url content block."""
        # Get image data using helper
        image_data = get_image_data(content)

        # OpenAI requires image URLs or data URIs for images
        image_url = {"url": f"data:{content.mimeType};base64,{image_data}"}

        # Check if the image has annotations for detail level
        if hasattr(content, "annotations") and content.annotations:
            if hasattr(content.annotations, "detail"):
                detail = content.annotations.detail
                if detail in ("auto", "low", "high"):
                    image_url["detail"] = detail

        return {"type": "image_url", "image_url": image_url}

    @staticmethod
    def _determine_mime_type(resource_content) -> str:
        """
        Determine the MIME type of a resource.

        Args:
            resource_content: The resource content to check

        Returns:
            The determined MIME type as a string
        """
        if hasattr(resource_content, "mimeType") and resource_content.mimeType:
            return resource_content.mimeType

        if hasattr(resource_content, "uri") and resource_content.uri:
            mime_type = guess_mime_type(str(resource_content.uri))
            return mime_type

        if hasattr(resource_content, "blob"):
            return "application/octet-stream"

        return "text/plain"

    @staticmethod
    def _convert_embedded_resource(
        resource: EmbeddedResource,
    ) -> Optional[ContentBlock]:
        """
        Convert EmbeddedResource to appropriate OpenAI content block.

        Args:
            resource: The embedded resource to convert

        Returns:
            An appropriate OpenAI content block or None if conversion failed
        """
        resource_content = resource.resource
        uri_str = get_resource_uri(resource)
        uri = getattr(resource_content, "uri", None)
        is_url = uri and str(uri).startswith(("http://", "https://"))
        from fast_agent.mcp.resource_utils import extract_title_from_uri

        title = extract_title_from_uri(uri) if uri else "resource"
        mime_type = OpenAIConverter._determine_mime_type(resource_content)

        # Handle different resource types based on MIME type

        # Handle images
        if OpenAIConverter._is_supported_image_type(mime_type):
            if is_url and uri_str:
                return {"type": "image_url", "image_url": {"url": uri_str}}

            # Try to get image data
            image_data = get_image_data(resource)
            if image_data:
                return {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{image_data}"},
                }
            else:
                return {"type": "text", "text": f"[Image missing data: {title}]"}

        # Handle PDFs
        elif mime_type == "application/pdf":
            if is_url and uri_str:
                # OpenAI doesn't directly support PDF URLs, explain this limitation
                return {
                    "type": "text",
                    "text": f"[PDF URL: {uri_str}]\nOpenAI requires PDF files to be uploaded or provided as base64 data.",
                }
            elif hasattr(resource_content, "blob"):
                return {
                    "type": "file",
                    "file": {
                        "filename": title or "document.pdf",
                        "file_data": f"data:application/pdf;base64,{resource_content.blob}",
                    },
                }

        # Handle SVG (convert to text)
        elif mime_type == "image/svg+xml":
            text = get_text(resource)
            if text:
                file_text = (
                    f'<fastagent:file title="{title}" mimetype="{mime_type}">\n'
                    f"{text}\n"
                    f"</fastagent:file>"
                )
                return {"type": "text", "text": file_text}

        # Handle text files
        elif is_text_mime_type(mime_type):
            text = get_text(resource)
            if text:
                file_text = (
                    f'<fastagent:file title="{title}" mimetype="{mime_type}">\n'
                    f"{text}\n"
                    f"</fastagent:file>"
                )
                return {"type": "text", "text": file_text}

        # Default fallback for text resources
        text = get_text(resource)
        if text:
            return {"type": "text", "text": text}

        # Default fallback for binary resources
        elif hasattr(resource_content, "blob"):
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
    def _extract_text_from_content_blocks(
        content: Union[str, List[ContentBlock]],
    ) -> str:
        """
        Extract and combine text from content blocks.

        Args:
            content: Content blocks or string

        Returns:
            Combined text as a string
        """
        if isinstance(content, str):
            return content

        if not content:
            return ""

        # Extract only text blocks
        text_parts = []
        for block in content:
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))

        return " ".join(text_parts) if text_parts else "[Complex content converted to text]"

    @staticmethod
    def convert_tool_result_to_openai(
        tool_result: CallToolResult,
        tool_call_id: str,
        concatenate_text_blocks: bool = False,
    ) -> Union[Dict[str, Any], Tuple[Dict[str, Any], List[Dict[str, Any]]]]:
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
        # Handle empty content case
        if not tool_result.content:
            return {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": "[Tool completed successfully]",
            }

        # Separate text and non-text content
        text_content = []
        non_text_content = []

        for item in tool_result.content:
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
        tool_message = {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": tool_message_content,
        }

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
        results: Dict[str, CallToolResult],
        concatenate_text_blocks: bool = False,
    ) -> List[Dict[str, Any]]:
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
                fallback_message = {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": f"[Conversion error: {str(e)}]",
                }
                tool_messages.append(fallback_message)

        # CONDITIONAL REORDERING: Only reorder if there are user messages (mixed content)
        if has_mixed_content and user_messages:
            # Reorder: All tool messages first (OpenAI sequence), then user messages (vision context)
            messages = tool_messages + user_messages
        else:
            # Pure tool responses - keep original order to preserve context (snapshots, etc.)
            messages = tool_messages
        return messages
