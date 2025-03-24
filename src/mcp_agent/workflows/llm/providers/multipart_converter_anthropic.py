from typing import List, Union, Sequence, Optional

from mcp.types import (
    TextContent,
    ImageContent,
    EmbeddedResource,
    CallToolResult,
    TextResourceContents,
    BlobResourceContents,
    PromptMessage,
)
from pydantic import AnyUrl
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart
from mcp_agent.mcp.mime_utils import (
    guess_mime_type,
    is_text_mime_type,
    is_image_mime_type,
)

from anthropic.types import (
    MessageParam,
    TextBlockParam,
    ImageBlockParam,
    DocumentBlockParam,
    Base64ImageSourceParam,
    URLImageSourceParam,
    Base64PDFSourceParam,
    URLPDFSourceParam,
    PlainTextSourceParam,
    ToolResultBlockParam,
    ContentBlockParam,
)
from mcp_agent.logging.logger import get_logger
from mcp_agent.mcp.resource_utils import extract_title_from_uri

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
    def convert_to_anthropic(multipart_msg: PromptMessageMultipart) -> MessageParam:
        """
        Convert a PromptMessageMultipart message to Anthropic API format.

        Args:
            multipart_msg: The PromptMessageMultipart message to convert

        Returns:
            An Anthropic API MessageParam object
        """
        role = multipart_msg.role

        # Handle empty content case - create an empty list instead of a text block
        if not multipart_msg.content:
            return MessageParam(role=role, content=[])

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

        # Create the Anthropic message
        return MessageParam(role=role, content=anthropic_blocks)

    @staticmethod
    def convert_prompt_message_to_anthropic(message: PromptMessage) -> MessageParam:
        """
        Convert a standard PromptMessage to Anthropic API format.

        Args:
            message: The PromptMessage to convert

        Returns:
            An Anthropic API MessageParam object
        """
        # Convert the PromptMessage to a PromptMessageMultipart containing a single content item
        multipart = PromptMessageMultipart(role=message.role, content=[message.content])

        # Use the existing conversion method
        return AnthropicConverter.convert_to_anthropic(multipart)

    @staticmethod
    def _convert_content_items(
        content_items: Sequence[Union[TextContent, ImageContent, EmbeddedResource]],
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
            if isinstance(content_item, TextContent):
                anthropic_blocks.append(
                    TextBlockParam(type="text", text=content_item.text)
                )

            elif isinstance(content_item, ImageContent):
                # Check if image MIME type is supported
                if not AnthropicConverter._is_supported_image_type(
                    content_item.mimeType
                ):
                    anthropic_blocks.append(
                        TextBlockParam(
                            type="text",
                            text=f"Image with unsupported format '{content_item.mimeType}' ({len(content_item.data)} bytes)",
                        )
                    )
                else:
                    anthropic_blocks.append(
                        ImageBlockParam(
                            type="image",
                            source=Base64ImageSourceParam(
                                type="base64",
                                media_type=content_item.mimeType,
                                data=content_item.data,
                            ),
                        )
                    )

            elif isinstance(content_item, EmbeddedResource):
                block = AnthropicConverter._convert_embedded_resource(
                    content_item, document_mode
                )
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
        uri: Optional[AnyUrl] = getattr(resource_content, "uri", None)
        is_url: bool = uri and uri.scheme in ("http", "https")

        # Determine MIME type
        mime_type = AnthropicConverter._determine_mime_type(resource_content)

        # Extract title from URI
        title = extract_title_from_uri(uri) if uri else "resource"

        # Convert based on MIME type
        if mime_type == "image/svg+xml":
            return AnthropicConverter._convert_svg_resource(resource_content)

        elif is_image_mime_type(mime_type):
            if not AnthropicConverter._is_supported_image_type(mime_type):
                return AnthropicConverter._create_fallback_text(
                    f"Image with unsupported format '{mime_type}'", resource
                )

            if is_url:
                return ImageBlockParam(
                    type="image", source=URLImageSourceParam(type="url", url=str(uri))
                )
            elif hasattr(resource_content, "blob"):
                return ImageBlockParam(
                    type="image",
                    source=Base64ImageSourceParam(
                        type="base64", media_type=mime_type, data=resource_content.blob
                    ),
                )
            return AnthropicConverter._create_fallback_text(
                "Image missing data", resource
            )

        elif mime_type == "application/pdf":
            if is_url:
                return DocumentBlockParam(
                    type="document",
                    title=title,
                    source=URLPDFSourceParam(type="url", url=str(uri)),
                )
            elif hasattr(resource_content, "blob"):
                return DocumentBlockParam(
                    type="document",
                    title=title,
                    source=Base64PDFSourceParam(
                        type="base64",
                        media_type="application/pdf",
                        data=resource_content.blob,
                    ),
                )
            return TextBlockParam(
                type="text", text=f"[PDF resource missing data: {title}]"
            )

        elif is_text_mime_type(mime_type):
            if not hasattr(resource_content, "text"):
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
                        data=resource_content.text,
                    ),
                )

            # Return as simple text block when not in document mode
            return TextBlockParam(type="text", text=resource_content.text)

        # Default fallback - convert to text if possible
        if hasattr(resource_content, "text"):
            return TextBlockParam(type="text", text=resource_content.text)

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
        if hasattr(resource_content, "text"):
            svg_content = resource_content.text
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
    def convert_tool_result_to_anthropic(
        tool_result: CallToolResult, tool_use_id: str
    ) -> ToolResultBlockParam:
        """
        Convert an MCP CallToolResult to an Anthropic ToolResultBlockParam.

        Args:
            tool_result: The tool result from a tool call
            tool_use_id: The ID of the associated tool use

        Returns:
            An Anthropic ToolResultBlockParam ready to be included in a user message
        """
        # For tool results, always use document_mode=False to get text blocks instead of document blocks
        anthropic_content = []

        for item in tool_result.content:
            if isinstance(item, EmbeddedResource):
                # For embedded resources, always use text mode in tool results
                resource_block = AnthropicConverter._convert_embedded_resource(
                    item, document_mode=False
                )
                anthropic_content.append(resource_block)
            elif isinstance(item, (TextContent, ImageContent)):
                # For text and image, use standard conversion
                blocks = AnthropicConverter._convert_content_items(
                    [item], document_mode=False
                )
                anthropic_content.extend(blocks)

        # If we ended up with no valid content blocks, create a placeholder
        if not anthropic_content:
            anthropic_content = [
                TextBlockParam(type="text", text="[No content in tool result]")
            ]

        # Create the tool result block
        return ToolResultBlockParam(
            type="tool_result",
            tool_use_id=tool_use_id,
            content=anthropic_content,
            is_error=tool_result.isError,
        )

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
            separate_blocks = []

            # Process each content item in the result
            for item in result.content:
                if isinstance(item, (TextContent, ImageContent)):
                    blocks = AnthropicConverter._convert_content_items(
                        [item], document_mode=False
                    )
                    tool_result_blocks.extend(blocks)
                elif isinstance(item, EmbeddedResource):
                    resource_content = item.resource

                    # Text resources go in tool results, others go as separate blocks
                    if isinstance(resource_content, TextResourceContents):
                        block = AnthropicConverter._convert_embedded_resource(
                            item, document_mode=False
                        )
                        tool_result_blocks.append(block)
                    else:
                        # For binary resources like PDFs, add as separate block
                        block = AnthropicConverter._convert_embedded_resource(
                            item, document_mode=True
                        )
                        separate_blocks.append(block)

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
                        content=[
                            TextBlockParam(
                                type="text", text="[No content in tool result]"
                            )
                        ],
                        is_error=result.isError,
                    )
                )

            # Add separate blocks directly to the message
            content_blocks.extend(separate_blocks)

        return MessageParam(role="user", content=content_blocks)
