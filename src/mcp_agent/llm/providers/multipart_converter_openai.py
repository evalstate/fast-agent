import base64
import json
from typing import Any, Dict, List, Optional, Tuple, Union

from mcp.types import (
    CallToolRequestParams,
    CallToolResult,
    EmbeddedResource,

    PromptMessage,
    ResourceLink,
    TextContent,
    ImageContent,

)
from openai.types.chat import (
    ChatCompletionMessage,
    ChatCompletionUserMessageParam,

    ChatCompletionMessageParam,
    ChatCompletionMessageToolCall,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolParam,
)
from openai.types.chat.chat_completion_message_tool_call import (
    Function, 
)

"""
openai/types/chat/chat_completion_message_param.py

ChatCompletionMessageParam: TypeAlias = Union[
    ChatCompletionDeveloperMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionFunctionMessageParam,
]
"""

from mcp_agent.logging.logger import get_logger
from mcp_agent.mcp.helpers.content_helpers import (
    get_image_data,
    get_resource_uri,
    get_text,
    is_image_content,
    is_resource_content,
    is_resource_link,
    is_text_content,
)
from mcp_agent.mcp.mime_utils import (
    guess_mime_type,
    is_image_mime_type,
    is_text_mime_type,
)
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart
from mcp_agent.mcp.resource_utils import extract_title_from_uri

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
        multipart_msg: PromptMessageMultipart, concatenate_text_blocks: bool = False
    ) -> Dict[str, str | ContentBlock | List[ContentBlock]]:
        """
        Convert a PromptMessageMultipart message to OpenAI API format.

        Args:
            multipart_msg: The PromptMessageMultipart message to convert
            concatenate_text_blocks: If True, adjacent text blocks will be combined

        Returns:
            An OpenAI API message object
        """
        role = multipart_msg.role

        # Handle empty content
        if not multipart_msg.content:
            return {"role": role, "content": ""}

        # single text block
        if 1 == len(multipart_msg.content) and is_text_content(multipart_msg.content[0]):
            return {"role": role, "content": get_text(multipart_msg.content[0])}

        # For user messages, convert each content block
        content_blocks: List[ContentBlock] = []

        for item in multipart_msg.content:
            try:
                if is_text_content(item):
                    text = get_text(item)
                    content_blocks.append({"type": "text", "text": text})

                elif is_image_content(item):
                    content_blocks.append(OpenAIConverter._convert_image_content(item))

                elif is_resource_content(item):
                    block = OpenAIConverter._convert_embedded_resource(item)
                    if block:
                        content_blocks.append(block)

                elif is_resource_link(item):
                    block = OpenAIConverter._convert_resource_link(item)
                    if block:
                        content_blocks.append(block)

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
        return {"role": role, "content": content_blocks}

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
    ) -> ChatCompletionMessage:
        """
        Convert a standard PromptMessage to OpenAI API format.

        Args:
            message: The PromptMessage to convert
            concatenate_text_blocks: If True, adjacent text blocks will be combined

        Returns:
            An OpenAI API message object
        """
        # Convert the PromptMessage to a PromptMessageMultipart containing a single content item
        multipart = PromptMessageMultipart(role=message.role, content=[message.content])

        # Use the existing conversion method with the specified concatenation option
        return OpenAIConverter.convert_to_openai(multipart, concatenate_text_blocks)

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
    def _convert_resource_link(
        resource: ResourceLink,
    ) -> Optional[ContentBlock]:
        """
        Convert ResourceLink to OpenAI content block.

        Args:
            resource: The resource link to convert

        Returns:
            An OpenAI content block or None if conversion failed
        """
        name = resource.name or "unknown"
        uri_str = str(resource.uri)
        mime_type = resource.mimeType or "unknown"
        description = resource.description or "No description"

        # Create a text block with the resource link information
        return {
            "type": "text",
            "text": f"Linked Resource ${name} MIME type {mime_type}>\n"
            f"Resource Link: {uri_str}\n"
            f"${description}\n",
        }

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
    ) -> Tuple[str, Optional[List]]:
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
            temp_multipart = PromptMessageMultipart(role="user", content=text_content)
            converted = OpenAIConverter.convert_to_openai(
                temp_multipart, concatenate_text_blocks=concatenate_text_blocks
            )

            # Extract text from content blocks
            tool_message_content = OpenAIConverter._extract_text_from_content_blocks(
                converted.get("content", "")
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
            return (tool_message, None)

        # Process non-text content as a separate user message
        non_text_multipart = PromptMessageMultipart(role="user", content=non_text_content)

        # Convert to OpenAI format
        user_message = OpenAIConverter.convert_to_openai(non_text_multipart)

        # We need to add tool_call_id manually
        # user_message["tool_call_id"] = tool_call_id

        return (tool_message, [user_message])

    @staticmethod
    def convert_function_results_to_openai(
        results: List[CallToolResult],
        concatenate_text_blocks: bool,
        arguments: str,
        function_name: str,
    ) -> List[ChatCompletionMessage]:
        """
        Convert a list of function call results to OpenAI messages.

        Args:
            results: List of (tool_call_id, result) tuples
            concatenate_text_blocks: If True, adjacent text blocks will be combined

        Returns:
            List of OpenAI API messages for tool responses
        """
        messages: List[ChatCompletionMessage] = []

        for tool_call_id, result in results:
            tool_message, additional_messages = OpenAIConverter.convert_tool_result_to_openai(
                tool_result=result,
                tool_call_id=tool_call_id,
                concatenate_text_blocks=concatenate_text_blocks,
            )

            openai_function = Function(
                    arguments=arguments,
                    name=function_name
                )

            message_tool_call = ChatCompletionMessageToolCall(
                id=tool_call_id,
                function=openai_function
            )

            messages.append(ChatCompletionMessage(
                content=tool_message["content"],
                role=tool_message["role"],
                tool_calls=[message_tool_call]

            ))
            if additional_messages:
                for additional_message in additional_messages:
                    messages.append(ChatCompletionMessage(content=additional_message))

        return messages


    @staticmethod
    def convert_from_openai_list_to_multipart(
        messages: List[ChatCompletionMessage],
    ) -> List[PromptMessageMultipart]:
        """Converts a list of OpenAI messages to a list of PromptMessageMultipart."""
        return [OpenAIConverter.convert_from_openai_to_multipart(msg) for msg in messages]

    @staticmethod
    def _convert_openai_content_to_mcp_parts(
        content: Union[str, List[Dict[str, Any]]]
    ) -> List[Union[TextContent, ImageContent]]:
        """
        Converts the content part of an OpenAI message into a list of MCP ContentBlocks.

        Args:
            content: The content from an OpenAI message, which can be a string or a list of parts.

        Returns:
            A list of MCP ContentBlock objects (e.g., TextContent, ImageContent).
        """
        if not content:
            return []

        # Handle simple string content
        if isinstance(content, str):
            return [TextContent(type="text", text=content)]

        mcp_parts = []
        # Handle a list of content blocks
        for part in content:
            part_type = part.get("type")

            if part_type == "text":
                mcp_parts.append(TextContent(type="text", text=part.get("text", "")))

            elif part_type == "image_url":
                image_url_data = part.get("image_url", {})
                url = image_url_data.get("url", "")

                # Check for and parse base64 data URIs
                if url.startswith("data:"):
                    try:
                        # e.g., "data:image/jpeg;base64,iVBORw0KGgo..."
                        header, encoded_data = url.split(",", 1)
                        mime_type = header.split(":")[1].split(";")[0]
                        mcp_parts.append(
                            ImageContent(type="image", mimeType=mime_type, data=encoded_data)
                        )
                    except (ValueError, IndexError) as e:
                        _logger.warning(f"Could not parse image data URI: {url}, error: {e}")
                else:
                    # Handle regular URLs by converting them to a descriptive text block
                    mcp_parts.append(TextContent(type="text", text=f"[Image at URL: {url}]"))
            else:
                _logger.warning(f"Skipping unsupported OpenAI content part type: {part_type}")

        return mcp_parts

    @staticmethod
    def convert_from_openai_to_multipart(message: ChatCompletionMessage) -> PromptMessageMultipart:
        """Converts a single OpenAI message to a PromptMessageMultipart."""
        role = message["role"]
        mcp_content: List[Union[TextContent, ImageContent, CallToolResult]] = []

        # Handle tool results, which have a unique structure
        if role == "tool":
            text_content = str(message.get("content", ""))
            # The content should be a TextContent block, not a CallToolResult
            mcp_content.append(TextContent(text=text_content))
            # The role can remain "tool" if your system handles it, or be mapped to "user"
            return PromptMessageMultipart(role=role, content=mcp_content)

        # Handle standard content (text, images) for user/assistant messages
        mcp_content.extend(
            OpenAIConverter._convert_openai_content_to_mcp_parts(message.get("content"))
        )

        # Handle tool call requests from assistant messages
        if tool_calls := message.get("tool_calls"):
            for tool_call in tool_calls:
                if tool_call.get("type") == "function":
                    function = tool_call.get("function", {})
                    name = function.get("name")
                    arguments_str = function.get("arguments", "{}")
                    try:
                        arguments_dict = json.loads(arguments_str)
                    except json.JSONDecodeError:
                        _logger.error(f"Failed to decode tool arguments: {arguments_str}")
                        arguments_dict = {
                            "error": "failed to decode arguments",
                            "raw_arguments": arguments_str,
                        }

                    # Use CallToolRequestParams instead of CallTool
                    mcp_content.append(
                        CallToolRequestParams(
                            id=tool_call.get("id"),  # Pass id as an extra field
                            name=name,
                            arguments=arguments_dict,  # Use 'arguments' field
                        )
                    )

        return PromptMessageMultipart(role=role, content=mcp_content)