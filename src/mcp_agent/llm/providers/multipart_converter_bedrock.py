"""
Converter for Amazon Bedrock message formats.
Adapts Fast-Agent multipart messages to the format required by Bedrock's API,
supporting multiple model families including Claude, Amazon Titan, and Meta Llama.
"""

from typing import Any, Dict, List, Optional, Sequence, Union

from mcp.types import (
    BlobResourceContents,
    CallToolResult,
    EmbeddedResource,
    ImageContent,
    PromptMessage,
    TextContent,
    TextResourceContents,
)

from mcp_agent.logging.logger import get_logger
from mcp_agent.mcp.helpers.content_helpers import (
    get_image_data,
    get_resource_uri,
    get_text,
    is_image_content,
    is_resource_content,
    is_text_content,
)
from mcp_agent.mcp.mime_utils import (
    guess_mime_type,
    is_image_mime_type,
    is_text_mime_type,
)
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart
from mcp_agent.mcp.resource_utils import extract_title_from_uri

_logger = get_logger("multipart_converter_bedrock")

# List of image MIME types supported by Bedrock's Claude models
SUPPORTED_IMAGE_MIME_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}

# Type alias for Bedrock message
BedrockMessage = Dict[str, Any]

# Constants for model family detection
CLAUDE_MODEL_PREFIX = "anthropic."
NOVA_MODEL_PREFIX = "amazon.nova-"
META_MODEL_PREFIX = "meta.llama"

# Model families supported
class ModelFamily:
    """Enum-like class for model families"""
    CLAUDE = "claude"
    NOVA = "nova"
    META = "meta"
    UNKNOWN = "unknown"


class BedrockConverter:
    """Converts MCP message types to Bedrock API format for various model families."""
    
    @staticmethod
    def detect_model_family(model_id: str) -> str:
        """
        Detect the model family based on the model ID.
        
        Args:
            model_id: The Bedrock model ID
            
        Returns:
            The model family as a string
        """
        # Strip region prefix if present (us. or eu.)
        if model_id.startswith("us.") or model_id.startswith("eu."):
            model_id_no_prefix = model_id[3:]
        else:
            model_id_no_prefix = model_id
            
        if model_id_no_prefix.startswith(CLAUDE_MODEL_PREFIX) or "claude" in model_id_no_prefix.lower():
            return ModelFamily.CLAUDE
        elif model_id_no_prefix.startswith(NOVA_MODEL_PREFIX) or "nova" in model_id_no_prefix.lower():
            return ModelFamily.NOVA
        elif model_id_no_prefix.startswith(META_MODEL_PREFIX) or "llama" in model_id_no_prefix.lower():
            return ModelFamily.META
        else:
            return ModelFamily.UNKNOWN

    @staticmethod
    def _is_supported_image_type(mime_type: str) -> bool:
        """Check if the given MIME type is supported by Bedrock's Claude models.

        Args:
            mime_type: The MIME type to check

        Returns:
            True if the MIME type is supported, False otherwise
        """
        return mime_type in SUPPORTED_IMAGE_MIME_TYPES

    @staticmethod
    def convert_to_bedrock(multipart_msg: PromptMessageMultipart, model_family: str = ModelFamily.CLAUDE) -> BedrockMessage:
        """
        Convert a PromptMessageMultipart message to Bedrock API format based on model family.

        Args:
            multipart_msg: The PromptMessageMultipart message to convert
            model_family: The model family (claude, titan, meta, etc.)

        Returns:
            A Bedrock API message object in the appropriate format for the model family
        """
        role = multipart_msg.role
        
        # Choose the conversion method based on model family
        if model_family == ModelFamily.CLAUDE:
            return BedrockConverter._convert_to_claude_format(multipart_msg)
        elif model_family == ModelFamily.NOVA:
            return BedrockConverter._convert_to_nova_format(multipart_msg)
        elif model_family == ModelFamily.META:
            return BedrockConverter._convert_to_meta_format(multipart_msg)
        else:
            # Default to Claude format
            return BedrockConverter._convert_to_claude_format(multipart_msg)
    
    @staticmethod
    def _convert_to_claude_format(multipart_msg: PromptMessageMultipart) -> BedrockMessage:
        """
        Convert message to Claude format for Bedrock.
        
        Args:
            multipart_msg: The message to convert
            
        Returns:
            Message in Claude format
        """
        role = multipart_msg.role
        
        # Bedrock Claude expects "user" and "assistant" roles
        # Map "system" to a special anthropic_message format
        if role == "system":
            return {
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": f"<admin>\n{BedrockConverter._extract_system_text(multipart_msg)}\n</admin>"
                }]
            }

        # Handle empty content case - create an empty list 
        if not multipart_msg.content:
            return {"role": role, "content": []}

        # Convert content blocks
        bedrock_content = BedrockConverter._convert_content_items(
            multipart_msg.content, document_mode=True
        )

        # Create the Claude message
        return {"role": role, "content": bedrock_content}
        
    @staticmethod
    def _convert_to_nova_format(multipart_msg: PromptMessageMultipart) -> BedrockMessage:
        """
        Convert message to Amazon Nova format for Bedrock.
        
        Args:
            multipart_msg: The message to convert
            
        Returns:
            Message in Nova format
        """
        role = multipart_msg.role
        text_content = []
        
        # For Nova, we combine all text content into a single string
        # based on the role type
        if multipart_msg.content:
            for content_item in multipart_msg.content:
                if is_text_content(content_item):
                    text = get_text(content_item)
                    if text:
                        text_content.append(text)
                elif is_resource_content(content_item):
                    text = get_text(content_item)
                    if text:
                        text_content.append(text)
        
        text = "\n".join(text_content)
        
        # Amazon Nova uses a different format with "inputText" for user messages
        # and no role distinction in their basic format
        if role == "user":
            return {"text": text}
        elif role == "assistant":
            return {"text": text}
        elif role == "system":
            # For Nova, system prompts are prepended to the user message
            # with a special format
            return {"text": f"<<SYS>>\n{text}\n<</SYS>>"}
            
        # Default case
        return {"text": text}
    
    @staticmethod
    def _convert_to_meta_format(multipart_msg: PromptMessageMultipart) -> BedrockMessage:
        """
        Convert message to Meta Llama format for Bedrock.
        
        Args:
            multipart_msg: The message to convert
            
        Returns:
            Message in Meta Llama format
        """
        role = multipart_msg.role
        text_content = []
        
        # For Meta Llama models, extract all text content
        if multipart_msg.content:
            for content_item in multipart_msg.content:
                if is_text_content(content_item):
                    text = get_text(content_item)
                    if text:
                        text_content.append(text)
                elif is_resource_content(content_item):
                    text = get_text(content_item)
                    if text:
                        text_content.append(text)
        
        text = "\n".join(text_content)
        
        # Meta Llama on Bedrock generally uses a format similar to this
        if role == "user":
            return {"role": "user", "content": text}
        elif role == "assistant":
            return {"role": "assistant", "content": text}
        elif role == "system":
            return {"role": "system", "content": text}
        
        # Default case
        return {"role": role, "content": text}

    @staticmethod
    def _extract_system_text(multipart_msg: PromptMessageMultipart) -> str:
        """
        Extract system prompt text from a multipart message.
        
        Args:
            multipart_msg: The system message to extract text from
            
        Returns:
            The combined text from all text content items
        """
        text_parts = []
        
        for content_item in multipart_msg.content:
            if is_text_content(content_item):
                text = get_text(content_item)
                if text:
                    text_parts.append(text)
            elif is_resource_content(content_item):
                text = get_text(content_item)
                if text:
                    text_parts.append(text)
        
        return "\n".join(text_parts)

    @staticmethod
    def convert_prompt_message_to_bedrock(message: PromptMessage) -> BedrockMessage:
        """
        Convert a standard PromptMessage to Bedrock API format.

        Args:
            message: The PromptMessage to convert

        Returns:
            A Bedrock API message object
        """
        # Convert the PromptMessage to a PromptMessageMultipart containing a single content item
        multipart = PromptMessageMultipart(role=message.role, content=[message.content])

        # Use the existing conversion method
        return BedrockConverter.convert_to_bedrock(multipart)

    @staticmethod
    def _convert_content_items(
        content_items: Sequence[Union[TextContent, ImageContent, EmbeddedResource]],
        document_mode: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Convert a list of content items to Bedrock content blocks.

        Args:
            content_items: Sequence of MCP content items
            document_mode: Whether to convert text resources to document blocks (True) or text blocks (False)

        Returns:
            List of Bedrock content blocks
        """
        bedrock_blocks: List[Dict[str, Any]] = []

        for content_item in content_items:
            if is_text_content(content_item):
                # Handle text content
                text = get_text(content_item)
                bedrock_blocks.append({"type": "text", "text": text})

            elif is_image_content(content_item):
                # Handle image content
                image_content = content_item  # type: ImageContent
                # Check if image MIME type is supported
                if not BedrockConverter._is_supported_image_type(image_content.mimeType):
                    data_size = len(image_content.data) if image_content.data else 0
                    bedrock_blocks.append({
                        "type": "text",
                        "text": f"Image with unsupported format '{image_content.mimeType}' ({data_size} bytes)"
                    })
                else:
                    image_data = get_image_data(image_content)
                    bedrock_blocks.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": image_content.mimeType,
                            "data": image_data,
                        }
                    })

            elif is_resource_content(content_item):
                # Handle embedded resource
                block = BedrockConverter._convert_embedded_resource(content_item, document_mode)
                bedrock_blocks.append(block)

        return bedrock_blocks

    @staticmethod
    def _convert_embedded_resource(
        resource: EmbeddedResource,
        document_mode: bool = True,
    ) -> Dict[str, Any]:
        """
        Convert EmbeddedResource to appropriate Bedrock block type.

        Args:
            resource: The embedded resource to convert
            document_mode: Whether to convert text resources to Document blocks (True) or Text blocks (False)

        Returns:
            An appropriate content block for the resource
        """
        resource_content = resource.resource
        uri_str = get_resource_uri(resource)
        uri = getattr(resource_content, "uri", None)
        is_url: bool = uri and uri.scheme in ("http", "https")

        # Determine MIME type
        mime_type = BedrockConverter._determine_mime_type(resource_content)

        # Extract title from URI
        title = extract_title_from_uri(uri) if uri else "resource"

        # Convert based on MIME type
        if mime_type == "image/svg+xml":
            return BedrockConverter._convert_svg_resource(resource_content)

        elif is_image_mime_type(mime_type):
            if not BedrockConverter._is_supported_image_type(mime_type):
                return BedrockConverter._create_fallback_text(
                    f"Image with unsupported format '{mime_type}'", resource
                )

            if is_url and uri_str:
                return {
                    "type": "image", 
                    "source": {"type": "url", "url": uri_str}
                }
            
            # Try to get image data
            image_data = get_image_data(resource)
            if image_data:
                return {
                    "type": "image",
                    "source": {
                        "type": "base64", 
                        "media_type": mime_type, 
                        "data": image_data
                    },
                }
            
            return BedrockConverter._create_fallback_text("Image missing data", resource)

        elif mime_type == "application/pdf":
            # Bedrock does not support document blocks like direct Anthropic API, 
            # so we convert to text description
            return {"type": "text", "text": f"[PDF document: {title}]"}

        elif is_text_mime_type(mime_type):
            text = get_text(resource)
            if not text:
                return {
                    "type": "text",
                    "text": f"[Text content could not be extracted from {title}]",
                }

            # Return as simple text block - Bedrock doesn't support document blocks
            return {"type": "text", "text": text}

        # Default fallback - convert to text if possible
        text = get_text(resource)
        if text:
            return {"type": "text", "text": text}

        # This is for binary resources
        if isinstance(resource.resource, BlobResourceContents) and hasattr(
            resource.resource, "blob"
        ):
            blob_length = len(resource.resource.blob)
            return {
                "type": "text",
                "text": f"Embedded Resource {uri._url} with unsupported format {mime_type} ({blob_length} characters)",
            }

        return BedrockConverter._create_fallback_text(
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
    def _convert_svg_resource(resource_content) -> Dict[str, Any]:
        """
        Convert SVG resource to text block with XML code formatting.

        Args:
            resource_content: The resource content containing SVG data

        Returns:
            A text block with formatted SVG content
        """
        if hasattr(resource_content, "text"):
            svg_content = resource_content.text
            return {"type": "text", "text": f"```xml\n{svg_content}\n```"}
        return {"type": "text", "text": "[SVG content could not be extracted]"}

    @staticmethod
    def _create_fallback_text(
        message: str, resource: Union[TextContent, ImageContent, EmbeddedResource]
    ) -> Dict[str, Any]:
        """
        Create a fallback text block for unsupported resource types.

        Args:
            message: The fallback message
            resource: The resource that couldn't be converted

        Returns:
            A text block with the fallback message
        """
        if isinstance(resource, EmbeddedResource) and hasattr(resource.resource, "uri"):
            uri = resource.resource.uri
            return {"type": "text", "text": f"[{message}: {uri._url}]"}

        return {"type": "text", "text": f"[{message}]"}

    @staticmethod
    def convert_tool_result_to_bedrock(
        tool_result: CallToolResult, tool_use_id: str, model_family: str = ModelFamily.CLAUDE
    ) -> Dict[str, Any]:
        """
        Convert an MCP CallToolResult to a Bedrock tool result block.

        Args:
            tool_result: The tool result from a tool call
            tool_use_id: The ID of the associated tool use
            model_family: The model family (claude, titan, meta)

        Returns:
            A Bedrock tool result block ready to be included in a user message
        """
        # Convert tool results to content items
        bedrock_content = []

        for item in tool_result.content:
            if isinstance(item, EmbeddedResource):
                # For embedded resources, always use text mode in tool results
                resource_block = BedrockConverter._convert_embedded_resource(
                    item, document_mode=False
                )
                bedrock_content.append(resource_block)
            elif isinstance(item, (TextContent, ImageContent)):
                # For text and image, use standard conversion
                blocks = BedrockConverter._convert_content_items([item], document_mode=False)
                bedrock_content.extend(blocks)

        # If we ended up with no valid content blocks, create a placeholder
        if not bedrock_content:
            bedrock_content = [{"type": "text", "text": "[No content in tool result]"}]

        # Create the tool result block - Bedrock format
        return {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": bedrock_content,
            "is_error": tool_result.isError,
        }

    @staticmethod
    def create_tool_results_message(
        tool_results: List[tuple[str, CallToolResult]],
        model_family: str = ModelFamily.CLAUDE,
    ) -> BedrockMessage:
        """
        Create a user message containing tool results.

        Args:
            tool_results: List of (tool_use_id, tool_result) tuples

        Returns:
            A Bedrock message with role='user' containing all tool results
        """
        content_blocks = []

        for tool_use_id, result in tool_results:
            # Process each tool result
            tool_result_blocks = []
            separate_blocks = []

            # Process each content item in the result
            for item in result.content:
                if isinstance(item, (TextContent, ImageContent)):
                    blocks = BedrockConverter._convert_content_items([item], document_mode=False)
                    tool_result_blocks.extend(blocks)
                elif isinstance(item, EmbeddedResource):
                    resource_content = item.resource

                    # Text resources go in tool results, others go as separate blocks
                    if isinstance(resource_content, TextResourceContents):
                        block = BedrockConverter._convert_embedded_resource(
                            item, document_mode=False
                        )
                        tool_result_blocks.append(block)
                    else:
                        # For binary resources like PDFs, add as separate block
                        block = BedrockConverter._convert_embedded_resource(
                            item, document_mode=True
                        )
                        separate_blocks.append(block)

            # Create the tool result block if we have content
            if tool_result_blocks:
                content_blocks.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": tool_result_blocks,
                    "is_error": result.isError,
                })
            else:
                # If there's no content, still create a placeholder
                content_blocks.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": [{"type": "text", "text": "[No content in tool result]"}],
                    "is_error": result.isError,
                })

            # Add separate blocks directly to the message
            content_blocks.extend(separate_blocks)

        return {"role": "user", "content": content_blocks}