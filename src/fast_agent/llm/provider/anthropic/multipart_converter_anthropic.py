from typing import List, Sequence, Union

from anthropic.types import (
    Base64ImageSourceParam,
    Base64PDFSourceParam,
    ContentBlockParam,
    DocumentBlockParam,
    ImageBlockParam,
    MessageParam,
    PlainTextSourceParam,
    TextBlockParam,
    ToolResultBlockParam,
    ToolUseBlockParam,
    URLImageSourceParam,
    URLPDFSourceParam,
)
from mcp.types import (
    BlobResourceContents,
    CallToolResult,
    ContentBlock,
    EmbeddedResource,
    ImageContent,
    PromptMessage,
    TextContent,
    TextResourceContents,
)

from fast_agent.core.logging.logger import get_logger
from fast_agent.mcp.helpers.content_helpers import (
    get_image_data,
    get_resource_uri,
    get_text,
    is_image_content,
    is_resource_content,
    is_text_content,
)
from fast_agent.mcp.mime_utils import (
    guess_mime_type,
    is_image_mime_type,
    is_text_mime_type,
)
from fast_agent.types import PromptMessageExtended

_logger = get_logger("multipart_converter_anthropic")

# List of image MIME types supported by Anthropic API
SUPPORTED_IMAGE_MIME_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}


class AnthropicConverter:
    """Converts MCP message types to Anthropic API format."""

    @staticmethod
    def _is_supported_image_type(mime_type: str) -> bool:
        """Check if the given MIME type is supported by Anthropic's image API.

        Args:
            mime_type: The MIME type to check

        Returns:
            True if the MIME type is supported, False otherwise
        """
        return mime_type in SUPPORTED_IMAGE_MIME_TYPES

    @staticmethod
    def convert_to_anthropic(multipart_msg: PromptMessageExtended) -> MessageParam:
        """
        Convert a PromptMessageExtended message to Anthropic API format.

        Args:
            multipart_msg: The PromptMessageExtended message to convert

        Returns:
            An Anthropic API MessageParam object
        """
        role = multipart_msg.role
        all_content_blocks = []

        # If this is an assistant message that contains tool_calls, convert
        # those into Anthropic tool_use blocks so the next user message can
        # legally include corresponding tool_result blocks.
        if role == "assistant" and multipart_msg.tool_calls:
            for tool_use_id, req in multipart_msg.tool_calls.items():
                name = None
                args = None
                try:
                    params = getattr(req, "params", None)
                    if params is not None:
                        name = getattr(params, "name", None)
                        args = getattr(params, "arguments", None)
                except Exception:
                    pass

                all_content_blocks.append(
                    ToolUseBlockParam(
                        type="tool_use",
                        id=tool_use_id,
                        name=name or "unknown_tool",
                        input=args or {},
                    )
                )

            return MessageParam(role=role, content=all_content_blocks)

        # Handle tool_results if present (for user messages with tool results)
        # Tool results must come FIRST in the content array per Anthropic API requirements
        if multipart_msg.tool_results:
            # Convert dict to list of tuples for create_tool_results_message
            tool_results_list = list(multipart_msg.tool_results.items())
            tool_msg = AnthropicConverter.create_tool_results_message(tool_results_list)
            # Extract the content blocks from the tool results message
            all_content_blocks.extend(tool_msg["content"])

        # Then handle regular content blocks if present
        if multipart_msg.content:
            # Convert content blocks
            anthropic_blocks = AnthropicConverter._convert_content_items(
                multipart_msg.content, document_mode=True
            )

            # Filter blocks based on role (assistant can only have text blocks)
            if role == "assistant":
                text_blocks = []
                for block in anthropic_blocks:
                    if block.get("type") == "text":
                        text_blocks.append(block)
                    else:
                        _logger.warning(
                            f"Removing non-text block from assistant message: {block.get('type')}"
                        )
                anthropic_blocks = text_blocks

            all_content_blocks.extend(anthropic_blocks)

        # Handle empty content case
        if not all_content_blocks:
            return MessageParam(role=role, content=[])

        # Create the Anthropic message
        return MessageParam(role=role, content=all_content_blocks)

    @staticmethod
    def convert_prompt_message_to_anthropic(message: PromptMessage) -> MessageParam:
        """
        Convert a standard PromptMessage to Anthropic API format.

        Args:
            message: The PromptMessage to convert

        Returns:
            An Anthropic API MessageParam object
        """
        # Convert the PromptMessage to a PromptMessageExtended containing a single content item
        multipart = PromptMessageExtended(role=message.role, content=[message.content])

        # Use the existing conversion method
        return AnthropicConverter.convert_to_anthropic(multipart)

    @staticmethod
    def _convert_content_items(
        content_items: Sequence[ContentBlock],
        document_mode: bool = True,
    ) -> List[ContentBlockParam]:
        """
        Convert a list of content items to Anthropic content blocks.

        Args:
            content_items: Sequence of MCP content items
            document_mode: Whether to convert text resources to document blocks (True) or text blocks (False)

        Returns:
            List of Anthropic content blocks
        """
        anthropic_blocks: List[ContentBlockParam] = []

        for content_item in content_items:
            if is_text_content(content_item):
                # Handle text content
                text = get_text(content_item)
                anthropic_blocks.append(TextBlockParam(type="text", text=text))

            elif is_image_content(content_item):
                # Handle image content
                image_content = content_item  # type: ImageContent
                # Check if image MIME type is supported
                if not AnthropicConverter._is_supported_image_type(image_content.mimeType):
                    data_size = len(image_content.data) if image_content.data else 0
                    anthropic_blocks.append(
                        TextBlockParam(
                            type="text",
                            text=f"Image with unsupported format '{image_content.mimeType}' ({data_size} bytes)",
                        )
                    )
                else:
                    image_data = get_image_data(image_content)
                    anthropic_blocks.append(
                        ImageBlockParam(
                            type="image",
                            source=Base64ImageSourceParam(
                                type="base64",
                                media_type=image_content.mimeType,
                                data=image_data,
                            ),
                        )
                    )

            elif is_resource_content(content_item):
                # Handle embedded resource
                block = AnthropicConverter._convert_embedded_resource(content_item, document_mode)
                anthropic_blocks.append(block)

        return anthropic_blocks

    @staticmethod
    def _convert_embedded_resource(
        resource: EmbeddedResource,
        document_mode: bool = True,
    ) -> ContentBlockParam:
        """
        Convert EmbeddedResource to appropriate Anthropic block type.

        Args:
            resource: The embedded resource to convert
            document_mode: Whether to convert text resources to Document blocks (True) or Text blocks (False)

        Returns:
            An appropriate ContentBlockParam for the resource
        """
        resource_content = resource.resource
        uri_str = get_resource_uri(resource)
        uri = getattr(resource_content, "uri", None)
        is_url: bool = uri and uri.scheme in ("http", "https")

        # Determine MIME type
        mime_type = AnthropicConverter._determine_mime_type(resource_content)

        # Extract title from URI
        from fast_agent.mcp.resource_utils import extract_title_from_uri

        title = extract_title_from_uri(uri) if uri else "resource"

        # Convert based on MIME type
        if mime_type == "image/svg+xml":
            return AnthropicConverter._convert_svg_resource(resource_content)

        elif is_image_mime_type(mime_type):
            if not AnthropicConverter._is_supported_image_type(mime_type):
                return AnthropicConverter._create_fallback_text(
                    f"Image with unsupported format '{mime_type}'", resource
                )

            if is_url and uri_str:
                return ImageBlockParam(
                    type="image", source=URLImageSourceParam(type="url", url=uri_str)
                )

            # Try to get image data
            image_data = get_image_data(resource)
            if image_data:
                return ImageBlockParam(
                    type="image",
                    source=Base64ImageSourceParam(
                        type="base64", media_type=mime_type, data=image_data
                    ),
                )

            return AnthropicConverter._create_fallback_text("Image missing data", resource)

        elif mime_type == "application/pdf":
            if is_url and uri_str:
                return DocumentBlockParam(
                    type="document",
                    title=title,
                    source=URLPDFSourceParam(type="url", url=uri_str),
                )
            elif isinstance(resource_content, BlobResourceContents):
                return DocumentBlockParam(
                    type="document",
                    title=title,
                    source=Base64PDFSourceParam(
                        type="base64",
                        media_type="application/pdf",
                        data=resource_content.blob,
                    ),
                )
            return TextBlockParam(type="text", text=f"[PDF resource missing data: {title}]")

        elif is_text_mime_type(mime_type):
            text = get_text(resource)
            if not text:
                return TextBlockParam(
                    type="text",
                    text=f"[Text content could not be extracted from {title}]",
                )

            # Create document block when in document mode
            if document_mode:
                return DocumentBlockParam(
                    type="document",
                    title=title,
                    source=PlainTextSourceParam(
                        type="text",
                        media_type="text/plain",
                        data=text,
                    ),
                )

            # Return as simple text block when not in document mode
            return TextBlockParam(type="text", text=text)

        # Default fallback - convert to text if possible
        text = get_text(resource)
        if text:
            return TextBlockParam(type="text", text=text)

        # This is for binary resources - match the format expected by the test
        if isinstance(resource.resource, BlobResourceContents) and hasattr(
            resource.resource, "blob"
        ):
            blob_length = len(resource.resource.blob)
            return TextBlockParam(
                type="text",
                text=f"Embedded Resource {uri._url} with unsupported format {mime_type} ({blob_length} characters)",
            )

        return AnthropicConverter._create_fallback_text(
            f"Unsupported resource ({mime_type})", resource
        )

    @staticmethod
    def _determine_mime_type(
        resource: Union[TextResourceContents, BlobResourceContents],
    ) -> str:
        """
        Determine the MIME type of a resource.

        Args:
            resource: The resource to check

        Returns:
            The MIME type as a string
        """
        if getattr(resource, "mimeType", None):
            return resource.mimeType

        if getattr(resource, "uri", None):
            return guess_mime_type(resource.uri.serialize_url)

        if hasattr(resource, "blob"):
            return "application/octet-stream"

        return "text/plain"

    @staticmethod
    def _convert_svg_resource(resource_content) -> TextBlockParam:
        """
        Convert SVG resource to text block with XML code formatting.

        Args:
            resource_content: The resource content containing SVG data

        Returns:
            A TextBlockParam with formatted SVG content
        """
        # Use get_text helper to extract text from various content types
        svg_content = get_text(resource_content)
        if svg_content:
            return TextBlockParam(type="text", text=f"```xml\n{svg_content}\n```")
        return TextBlockParam(type="text", text="[SVG content could not be extracted]")

    @staticmethod
    def _create_fallback_text(
        message: str, resource: Union[TextContent, ImageContent, EmbeddedResource]
    ) -> TextBlockParam:
        """
        Create a fallback text block for unsupported resource types.

        Args:
            message: The fallback message
            resource: The resource that couldn't be converted

        Returns:
            A TextBlockParam with the fallback message
        """
        if isinstance(resource, EmbeddedResource) and hasattr(resource.resource, "uri"):
            uri = resource.resource.uri
            return TextBlockParam(type="text", text=f"[{message}: {uri._url}]")

        return TextBlockParam(type="text", text=f"[{message}]")

    @staticmethod
    def create_tool_results_message(
        tool_results: List[tuple[str, CallToolResult]],
    ) -> MessageParam:
        """
        Create a user message containing tool results.

        Args:
            tool_results: List of (tool_use_id, tool_result) tuples

        Returns:
            A MessageParam with role='user' containing all tool results
        """
        content_blocks = []

        for tool_use_id, result in tool_results:
            # Process each tool result
            tool_result_blocks = []

            # Process each content item in the result
            for item in result.content:
                if isinstance(item, (TextContent, ImageContent)):
                    blocks = AnthropicConverter._convert_content_items([item], document_mode=False)
                    tool_result_blocks.extend(blocks)
                elif isinstance(item, EmbeddedResource):
                    resource_content = item.resource
                    document_mode: bool = not isinstance(resource_content, TextResourceContents)
                    # With  Anthropic SDK 0.66, documents can be inside tool results
                    # Text resources remain inline within the tool_result
                    block = AnthropicConverter._convert_embedded_resource(
                        item, document_mode=document_mode
                    )
                    tool_result_blocks.append(block)

            # Create the tool result block if we have content
            if tool_result_blocks:
                content_blocks.append(
                    ToolResultBlockParam(
                        type="tool_result",
                        tool_use_id=tool_use_id,
                        content=tool_result_blocks,
                        is_error=result.isError,
                    )
                )
            else:
                # If there's no content, still create a placeholder
                content_blocks.append(
                    ToolResultBlockParam(
                        type="tool_result",
                        tool_use_id=tool_use_id,
                        content=[TextBlockParam(type="text", text="[No content in tool result]")],
                        is_error=result.isError,
                    )
                )

            # All content is now included within the tool_result block.

        return MessageParam(role="user", content=content_blocks)
