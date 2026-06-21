import json
import os
import re
from collections.abc import Mapping, Sequence
from typing import Any, Literal, Protocol, cast, runtime_checkable
from urllib.parse import urlparse

from anthropic.types.beta import (
    BetaBase64ImageSourceParam,
    BetaBase64PDFSourceParam,
    BetaContentBlockParam,
    BetaFileDocumentSourceParam,
    BetaImageBlockParam,
    BetaMessageParam,
    BetaPlainTextSourceParam,
    BetaRedactedThinkingBlock,
    BetaRedactedThinkingBlockParam,
    BetaRequestDocumentBlockParam,
    BetaTextBlockParam,
    BetaThinkingBlock,
    BetaThinkingBlockParam,
    BetaToolResultBlockParam,
    BetaToolUseBlockParam,
    BetaURLImageSourceParam,
    BetaURLPDFSourceParam,
)
from mcp.types import (
    BlobResourceContents,
    CallToolResult,
    ContentBlock,
    EmbeddedResource,
    ImageContent,
    PromptMessage,
    ResourceLink,
    TextContent,
    TextResourceContents,
)
from pydantic import TypeAdapter

from fast_agent.constants import (
    ANTHROPIC_ASSISTANT_RAW_CONTENT,
    ANTHROPIC_SERVER_TOOLS_CHANNEL,
    ANTHROPIC_THINKING_BLOCKS,
)
from fast_agent.core.logging.logger import get_logger
from fast_agent.llm.provider.anthropic.web_tools import is_server_tool_trace_payload
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
    DOCUMENT_MIME_TYPES,
    guess_mime_type,
    is_image_mime_type,
    is_text_mime_type,
)
from fast_agent.types import PromptMessageExtended

_logger = get_logger("multipart_converter_anthropic")
ANTHROPIC_FILE_ID_META_KEY = "fast_agent_anthropic_file_id"

# Validate and normalize replay blocks against *input* content block params.
# Using output block schemas preserves output-only fields (for example
# `parsed_output`) that Anthropic rejects when sent back in message history.
_ANTHROPIC_CONTENT_BLOCK_LIST_ADAPTER = TypeAdapter(list[BetaContentBlockParam])

# List of image MIME types supported by Anthropic API
SUPPORTED_IMAGE_MIME_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}


@runtime_checkable
class _SupportsModelDump(Protocol):
    def model_dump(self, *args: Any, **kwargs: Any) -> Any: ...


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
    def convert_to_anthropic(multipart_msg: PromptMessageExtended) -> BetaMessageParam:
        """
        Convert a PromptMessageExtended message to Anthropic API format.

        Args:
            multipart_msg: The PromptMessageExtended message to convert

        Returns:
            An Anthropic API BetaMessageParam object
        """
        role = multipart_msg.role
        all_content_blocks: list = []

        if role == "assistant":
            raw_assistant_content = AnthropicConverter._assistant_raw_content(
                multipart_msg.channels
            )
            if raw_assistant_content is not None:
                return BetaMessageParam(role=role, content=raw_assistant_content)

        if role == "assistant" and multipart_msg.tool_calls:
            return AnthropicConverter._assistant_tool_call_message(multipart_msg)

        if multipart_msg.tool_results:
            all_content_blocks.extend(AnthropicConverter._tool_result_content_blocks(multipart_msg))

        if role == "assistant" and multipart_msg.channels:
            AnthropicConverter._append_assistant_channel_blocks(
                multipart_msg.channels,
                all_content_blocks,
            )

        # Then handle regular content blocks if present
        if multipart_msg.content:
            # Convert content blocks
            anthropic_blocks = AnthropicConverter._convert_content_items(
                multipart_msg.content, document_mode=True
            )

            if role == "assistant":
                anthropic_blocks = AnthropicConverter._assistant_text_blocks(anthropic_blocks)

            all_content_blocks.extend(anthropic_blocks)

        # Handle empty content case
        if not all_content_blocks:
            return BetaMessageParam(role=role, content=[])

        # Create the Anthropic message
        return BetaMessageParam(role=role, content=all_content_blocks)

    @staticmethod
    def _assistant_raw_content(
        channels: Mapping[str, Sequence[ContentBlock]] | None,
    ) -> list[BetaContentBlockParam] | None:
        if not channels:
            return None
        raw_assistant_content = AnthropicConverter._deserialize_assistant_raw_blocks(channels)
        return raw_assistant_content or None

    @staticmethod
    def _assistant_tool_call_message(multipart_msg: PromptMessageExtended) -> BetaMessageParam:
        all_content_blocks: list[BetaContentBlockParam] = []
        if multipart_msg.channels:
            AnthropicConverter._append_assistant_channel_blocks(
                multipart_msg.channels,
                all_content_blocks,
            )

        if multipart_msg.content:
            anthropic_blocks = AnthropicConverter._convert_content_items(
                multipart_msg.content,
                document_mode=True,
            )
            all_content_blocks.extend(AnthropicConverter._assistant_text_blocks(anthropic_blocks))

        all_content_blocks.extend(AnthropicConverter._tool_use_blocks(multipart_msg.tool_calls))
        return BetaMessageParam(role="assistant", content=all_content_blocks)

    @staticmethod
    def _assistant_text_blocks(
        blocks: Sequence[BetaContentBlockParam],
    ) -> list[BetaContentBlockParam]:
        text_blocks: list[BetaContentBlockParam] = []
        for block in blocks:
            block_type = block.get("type") if isinstance(block, dict) else None
            if block_type == "text":
                text_blocks.append(block)
                continue
            _logger.warning(f"Removing non-text block from assistant message: {block_type}")
        return text_blocks

    @staticmethod
    def _tool_use_blocks(tool_calls: Mapping[str, Any] | None) -> list[BetaContentBlockParam]:
        if not tool_calls:
            return []

        blocks: list[BetaContentBlockParam] = []
        for tool_use_id, req in tool_calls.items():
            sanitized_id = AnthropicConverter._sanitize_tool_id(tool_use_id)
            params = req.params
            name = params.name if params else None
            args = params.arguments if params else None
            blocks.append(
                BetaToolUseBlockParam(
                    type="tool_use",
                    id=sanitized_id,
                    name=name or "unknown_tool",
                    input=args or {},
                )
            )
        return blocks

    @staticmethod
    def _tool_result_content_blocks(
        multipart_msg: PromptMessageExtended,
    ) -> list[BetaContentBlockParam]:
        if not multipart_msg.tool_results:
            return []
        tool_results_list = list(multipart_msg.tool_results.items())
        tool_msg = AnthropicConverter.create_tool_results_message(tool_results_list)
        content = tool_msg["content"]
        if isinstance(content, str):
            return [BetaTextBlockParam(type="text", text=content)]
        return list(content)

    @staticmethod
    def _append_assistant_channel_blocks(
        channels: Mapping[str, Sequence[ContentBlock]],
        destination: list[BetaContentBlockParam],
    ) -> None:
        thinking_blocks = AnthropicConverter._deserialize_thinking_channel_blocks(channels)
        server_tool_blocks: list[BetaContentBlockParam] = []
        AnthropicConverter._append_server_tool_channel_blocks(channels, server_tool_blocks)

        # Legacy history fallback:
        # Before exact raw-content replay was added, thinking and server-tool payloads
        # were persisted in separate channels and lost relative ordering. For turns that
        # include multiple thinking blocks and server-tool activity, Anthropic generally
        # expects first-thought -> server-tool blocks -> follow-up thoughts.
        if thinking_blocks and server_tool_blocks and len(thinking_blocks) > 1:
            destination.append(thinking_blocks[0])
            destination.extend(server_tool_blocks)
            destination.extend(thinking_blocks[1:])
            return

        destination.extend(thinking_blocks)
        destination.extend(server_tool_blocks)

    @staticmethod
    def _deserialize_thinking_channel_blocks(
        channels: Mapping[str, Sequence[ContentBlock]],
    ) -> list[BetaContentBlockParam]:
        raw_thinking = channels.get(ANTHROPIC_THINKING_BLOCKS)
        if not raw_thinking:
            return []

        thinking_blocks: list[BetaContentBlockParam] = []
        for thinking_block in raw_thinking:
            # Pass through raw BetaThinkingBlock/BetaRedactedThinkingBlock.
            # These contain signatures or encrypted data needed for API verification.
            if isinstance(thinking_block, BetaThinkingBlock):
                thinking_blocks.append(
                    BetaThinkingBlockParam(
                        type="thinking",
                        thinking=thinking_block.thinking,
                        signature=thinking_block.signature,
                    )
                )
                continue

            if isinstance(thinking_block, BetaRedactedThinkingBlock):
                thinking_blocks.append(thinking_block)
                continue

            if not isinstance(thinking_block, TextContent):
                continue

            try:
                payload = json.loads(thinking_block.text)
            except (TypeError, json.JSONDecodeError):
                payload = None

            if not isinstance(payload, dict):
                continue

            block_type = payload.get("type")
            if block_type == "thinking":
                thinking_blocks.append(
                    BetaThinkingBlockParam(
                        type="thinking",
                        thinking=payload.get("thinking", ""),
                        signature=payload.get("signature", ""),
                    )
                )
            elif block_type == "redacted_thinking":
                thinking_blocks.append(
                    BetaRedactedThinkingBlockParam(
                        type="redacted_thinking",
                        data=payload.get("data", ""),
                    )
                )

        return thinking_blocks

    @staticmethod
    def _deserialize_assistant_raw_blocks(
        channels: Mapping[str, Sequence[ContentBlock]],
    ) -> list[BetaContentBlockParam]:
        raw_blocks = channels.get(ANTHROPIC_ASSISTANT_RAW_CONTENT)
        if not raw_blocks:
            return []

        deserialized: list[BetaContentBlockParam] = []
        for block in raw_blocks:
            payload = AnthropicConverter._text_content_json_payload(block)
            if payload is None:
                continue

            normalized = AnthropicConverter._validated_anthropic_content_block(
                payload,
                warning_message="Skipping invalid assistant replay payload",
            )
            if normalized is None:
                continue

            if normalized.get("type") == "text":
                normalized.pop("parsed_output", None)

            deserialized.append(cast("BetaContentBlockParam", normalized))

        return deserialized

    @staticmethod
    def _text_content_json_payload(block: ContentBlock) -> dict[str, Any] | None:
        if not isinstance(block, TextContent) or not block.text:
            return None
        try:
            payload = json.loads(block.text)
        except Exception:
            return None
        return payload if isinstance(payload, dict) else None

    @staticmethod
    def _validated_anthropic_content_block(
        payload: dict[str, Any],
        *,
        warning_message: str,
    ) -> dict[str, Any] | None:
        candidate = cast("BetaContentBlockParam", payload)
        try:
            validated_blocks = _ANTHROPIC_CONTENT_BLOCK_LIST_ADAPTER.validate_python([candidate])
        except Exception as error:
            payload_type = payload.get("type")
            _logger.warning(
                warning_message,
                data={
                    "payload_type": payload_type,
                    "error": str(error),
                },
            )
            return None

        normalized = validated_blocks[0]
        if isinstance(normalized, _SupportsModelDump):
            try:
                normalized = normalized.model_dump(mode="json", exclude_none=False)
            except TypeError:
                normalized = normalized.model_dump()

        if not isinstance(normalized, dict):
            return None
        return cast("dict[str, Any]", normalized)

    @staticmethod
    def _append_server_tool_channel_blocks(
        channels: Mapping[str, Sequence[ContentBlock]] | None,
        destination: list[BetaContentBlockParam],
    ) -> None:
        if not channels:
            return
        raw_blocks = channels.get(ANTHROPIC_SERVER_TOOLS_CHANNEL)
        if not raw_blocks:
            return

        for block in raw_blocks:
            payload = AnthropicConverter._text_content_json_payload(block)
            if payload is None:
                continue
            if not is_server_tool_trace_payload(payload):
                continue

            normalized = AnthropicConverter._validated_anthropic_content_block(
                payload,
                warning_message="Skipping invalid server-tool payload from assistant channel",
            )
            if normalized is None:
                if os.environ.get("FAST_AGENT_WEBDEBUG"):
                    print(
                        "[webdebug] skipped invalid server-tool channel payload "
                        f"type={payload.get('type')} validation failed"
                    )
                continue

            if isinstance(normalized, dict):
                destination.append(cast("BetaContentBlockParam", normalized))

    @staticmethod
    def convert_prompt_message_to_anthropic(message: PromptMessage) -> BetaMessageParam:
        """
        Convert a standard PromptMessage to Anthropic API format.

        Args:
            message: The PromptMessage to convert

        Returns:
            An Anthropic API BetaMessageParam object
        """
        # Convert the PromptMessage to a PromptMessageExtended containing a single content item
        multipart = PromptMessageExtended(role=message.role, content=[message.content])

        # Use the existing conversion method
        return AnthropicConverter.convert_to_anthropic(multipart)

    @staticmethod
    def _convert_content_items(
        content_items: Sequence[ContentBlock],
        document_mode: bool = True,
    ) -> list[BetaContentBlockParam]:
        """
        Convert a list of content items to Anthropic content blocks.

        Args:
            content_items: Sequence of MCP content items
            document_mode: Whether to convert text resources to document blocks (True) or text blocks (False)

        Returns:
            List of Anthropic content blocks
        """
        anthropic_blocks: list[BetaContentBlockParam] = []

        for content_item in content_items:
            if is_text_content(content_item):
                # Handle text content
                text = get_text(content_item)
                if text:
                    anthropic_blocks.append(BetaTextBlockParam(type="text", text=text))

            elif is_image_content(content_item):
                # Handle image content
                image_content = content_item
                mime_type = image_content.mimeType or ""
                # Check if image MIME type is supported
                if not AnthropicConverter._is_supported_image_type(mime_type):
                    data_size = len(image_content.data) if image_content.data else 0
                    anthropic_blocks.append(
                        BetaTextBlockParam(
                            type="text",
                            text=f"Image with unsupported format '{mime_type}' ({data_size} bytes)",
                        )
                    )
                else:
                    image_data = get_image_data(image_content)
                    if image_data and mime_type in SUPPORTED_IMAGE_MIME_TYPES:
                        anthropic_blocks.append(
                            BetaImageBlockParam(
                                type="image",
                                source=BetaBase64ImageSourceParam(
                                    type="base64",
                                    media_type=cast(
                                        "Literal['image/jpeg', 'image/png', 'image/gif', 'image/webp']",
                                        mime_type,
                                    ),
                                    data=image_data,
                                ),
                            )
                        )

            elif is_resource_content(content_item):
                # Handle embedded resource
                block = AnthropicConverter._convert_embedded_resource(content_item, document_mode)
                anthropic_blocks.append(block)
            elif is_resource_link(content_item):
                anthropic_blocks.append(
                    AnthropicConverter._convert_resource_link(content_item, document_mode)
                )

        return anthropic_blocks

    @staticmethod
    def _convert_embedded_resource(
        resource: EmbeddedResource,
        document_mode: bool = True,
    ) -> BetaContentBlockParam:
        """
        Convert EmbeddedResource to appropriate Anthropic block type.

        Args:
            resource: The embedded resource to convert
            document_mode: Whether to convert text resources to Document blocks (True) or Text blocks (False)

        Returns:
            An appropriate BetaContentBlockParam for the resource
        """
        resource_content = resource.resource
        uri_str = get_resource_uri(resource)
        uri = resource_content.uri
        parsed_uri = urlparse(uri_str) if uri_str else None
        is_url: bool = bool(parsed_uri and parsed_uri.scheme in ("http", "https"))

        # Determine MIME type
        mime_type = AnthropicConverter._determine_mime_type(resource_content)

        # Extract title from URI
        from fast_agent.mcp.resource_utils import extract_title_from_uri

        title = extract_title_from_uri(uri) if uri else "resource"
        meta = getattr(resource_content, "meta", None)
        file_id = meta.get(ANTHROPIC_FILE_ID_META_KEY) if isinstance(meta, dict) else None

        if AnthropicConverter._should_use_anthropic_file_id(file_id, mime_type):
            return BetaRequestDocumentBlockParam(
                type="document",
                title=title,
                source=BetaFileDocumentSourceParam(type="file", file_id=cast("str", file_id)),
            )

        return AnthropicConverter._convert_resource_by_mime(
            resource,
            resource_content=resource_content,
            mime_type=mime_type,
            title=title,
            uri=uri,
            uri_str=uri_str,
            is_url=is_url,
            document_mode=document_mode,
        )

    @staticmethod
    def _convert_resource_by_mime(
        resource: EmbeddedResource,
        *,
        resource_content: Any,
        mime_type: str,
        title: str,
        uri: Any,
        uri_str: str | None,
        is_url: bool,
        document_mode: bool,
    ) -> BetaContentBlockParam:
        if mime_type == "image/svg+xml":
            block = AnthropicConverter._convert_svg_resource(resource_content)
        elif is_image_mime_type(mime_type):
            block = AnthropicConverter._convert_image_resource(
                resource,
                mime_type=mime_type,
                uri_str=uri_str,
                is_url=is_url,
            )
        elif mime_type == "application/pdf":
            block = AnthropicConverter._convert_pdf_resource(
                resource_content,
                title=title,
                uri_str=uri_str,
                is_url=is_url,
            )
        elif is_text_mime_type(mime_type):
            block = AnthropicConverter._convert_text_resource(
                resource,
                title=title,
                document_mode=document_mode,
            )
        elif text := get_text(resource):
            block = BetaTextBlockParam(type="text", text=text)
        elif isinstance(resource.resource, BlobResourceContents):
            block = AnthropicConverter._unsupported_blob_resource_text(
                resource.resource,
                uri=uri,
                uri_str=uri_str,
                mime_type=mime_type,
            )
        else:
            block = AnthropicConverter._create_fallback_text(
                f"Unsupported resource ({mime_type})", resource
            )
        return block

    @staticmethod
    def _should_use_anthropic_file_id(file_id: object, mime_type: str) -> bool:
        return (
            isinstance(file_id, str)
            and bool(file_id)
            and mime_type in DOCUMENT_MIME_TYPES
            and mime_type != "application/pdf"
        )

    @staticmethod
    def _convert_image_resource(
        resource: EmbeddedResource,
        *,
        mime_type: str,
        uri_str: str | None,
        is_url: bool,
    ) -> BetaContentBlockParam:
        if not AnthropicConverter._is_supported_image_type(mime_type):
            return AnthropicConverter._create_fallback_text(
                f"Image with unsupported format '{mime_type}'", resource
            )

        if is_url and uri_str:
            return BetaImageBlockParam(
                type="image", source=BetaURLImageSourceParam(type="url", url=uri_str)
            )

        image_data = get_image_data(resource)
        if image_data:
            return BetaImageBlockParam(
                type="image",
                source=BetaBase64ImageSourceParam(
                    type="base64",
                    media_type=cast(
                        "Literal['image/jpeg', 'image/png', 'image/gif', 'image/webp']",
                        mime_type,
                    ),
                    data=image_data,
                ),
            )

        return AnthropicConverter._create_fallback_text("Image missing data", resource)

    @staticmethod
    def _convert_pdf_resource(
        resource_content: Any,
        *,
        title: str,
        uri_str: str | None,
        is_url: bool,
    ) -> BetaContentBlockParam:
        if is_url and uri_str:
            return BetaRequestDocumentBlockParam(
                type="document",
                title=title,
                source=BetaURLPDFSourceParam(type="url", url=uri_str),
            )
        if isinstance(resource_content, BlobResourceContents):
            return BetaRequestDocumentBlockParam(
                type="document",
                title=title,
                source=BetaBase64PDFSourceParam(
                    type="base64",
                    media_type="application/pdf",
                    data=resource_content.blob,
                ),
            )
        return BetaTextBlockParam(type="text", text=f"[PDF resource missing data: {title}]")

    @staticmethod
    def _convert_text_resource(
        resource: EmbeddedResource, *, title: str, document_mode: bool
    ) -> BetaContentBlockParam:
        text = get_text(resource)
        if not text:
            return BetaTextBlockParam(
                type="text",
                text=f"[Text content could not be extracted from {title}]",
            )

        if document_mode:
            return BetaRequestDocumentBlockParam(
                type="document",
                title=title,
                source=BetaPlainTextSourceParam(
                    type="text",
                    media_type="text/plain",
                    data=text,
                ),
            )

        return BetaTextBlockParam(type="text", text=text)

    @staticmethod
    def _unsupported_blob_resource_text(
        resource_content: BlobResourceContents,
        *,
        uri: Any,
        uri_str: str | None,
        mime_type: str,
    ) -> BetaContentBlockParam:
        blob_length = len(resource_content.blob)
        uri_display = uri._url if uri else (uri_str or "<unknown>")
        return BetaTextBlockParam(
            type="text",
            text=(
                f"Embedded Resource {uri_display} with unsupported format "
                f"{mime_type} ({blob_length} characters)"
            ),
        )

    @staticmethod
    def _convert_resource_link(
        resource: ResourceLink,
        document_mode: bool = True,
    ) -> BetaContentBlockParam:
        """Convert ResourceLink to an Anthropic block when URL sources are supported."""
        del document_mode
        uri_str = str(resource.uri) if resource.uri else None
        parsed_uri = urlparse(uri_str) if uri_str else None
        is_url: bool = bool(parsed_uri and parsed_uri.scheme in ("http", "https"))
        mime_type = resource.mimeType or (guess_mime_type(uri_str) if uri_str else None) or ""

        from fast_agent.mcp.resource_utils import extract_title_from_uri

        title = (
            extract_title_from_uri(resource.uri) if resource.uri else (resource.name or "resource")
        )

        if is_url and is_image_mime_type(mime_type):
            assert uri_str is not None
            if not AnthropicConverter._is_supported_image_type(mime_type):
                return BetaTextBlockParam(
                    type="text",
                    text=f"Image with unsupported format '{mime_type}'",
                )
            return BetaImageBlockParam(
                type="image",
                source=BetaURLImageSourceParam(type="url", url=uri_str),
            )

        if is_url and mime_type == "application/pdf":
            assert uri_str is not None
            return BetaRequestDocumentBlockParam(
                type="document",
                title=title,
                source=BetaURLPDFSourceParam(type="url", url=uri_str),
            )

        text = get_text(resource)
        if text:
            return BetaTextBlockParam(type="text", text=text)

        return BetaTextBlockParam(type="text", text=f"[Resource link: {title}]")

    @staticmethod
    def _determine_mime_type(
        resource: TextResourceContents | BlobResourceContents,
    ) -> str:
        """
        Determine the MIME type of a resource.

        Args:
            resource: The resource to check

        Returns:
            The MIME type as a string
        """
        if resource.mimeType:
            return resource.mimeType

        if resource.uri:
            return guess_mime_type(str(resource.uri))

        if isinstance(resource, BlobResourceContents):
            return "application/octet-stream"

        return "text/plain"

    @staticmethod
    def _convert_svg_resource(resource_content) -> BetaTextBlockParam:
        """
        Convert SVG resource to text block with XML code formatting.

        Args:
            resource_content: The resource content containing SVG data

        Returns:
            A BetaTextBlockParam with formatted SVG content
        """
        # Use get_text helper to extract text from various content types
        svg_content = get_text(resource_content)
        if svg_content:
            return BetaTextBlockParam(type="text", text=f"```xml\n{svg_content}\n```")
        return BetaTextBlockParam(type="text", text="[SVG content could not be extracted]")

    @staticmethod
    def _create_fallback_text(message: str, resource: ContentBlock) -> BetaTextBlockParam:
        """
        Create a fallback text block for unsupported resource types.

        Args:
            message: The fallback message
            resource: The resource that couldn't be converted

        Returns:
            A BetaTextBlockParam with the fallback message
        """
        if isinstance(resource, EmbeddedResource):
            uri = resource.resource.uri
            if uri:
                return BetaTextBlockParam(type="text", text=f"[{message}: {uri._url}]")
            if uri_str := get_resource_uri(resource):
                return BetaTextBlockParam(type="text", text=f"[{message}: {uri_str}]")

        return BetaTextBlockParam(type="text", text=f"[{message}]")

    @staticmethod
    def create_tool_results_message(
        tool_results: list[tuple[str, CallToolResult]],
    ) -> BetaMessageParam:
        """
        Create a user message containing tool results.

        Args:
            tool_results: List of (tool_use_id, tool_result) tuples

        Returns:
            A BetaMessageParam with role='user' containing all tool results
        """
        content_blocks = []

        for tool_use_id, result in tool_results:
            sanitized_id = AnthropicConverter._sanitize_tool_id(tool_use_id)
            # Process each tool result
            tool_result_blocks: list[Any] = []

            # Process each content item in the result
            for item in canonicalize_tool_result_content_for_llm(
                result,
                logger=_logger,
                source="anthropic",
            ):
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
                elif isinstance(item, ResourceLink):
                    block = AnthropicConverter._convert_resource_link(item, document_mode=True)
                    tool_result_blocks.append(block)

            # Create the tool result block if we have content
            if tool_result_blocks:
                content_blocks.append(
                    BetaToolResultBlockParam(
                        type="tool_result",
                        tool_use_id=sanitized_id,
                        content=tool_result_blocks,
                        is_error=result.isError,
                    )
                )
            else:
                # If there's no content, still create a placeholder
                content_blocks.append(
                    BetaToolResultBlockParam(
                        type="tool_result",
                        tool_use_id=sanitized_id,
                        content=[
                            BetaTextBlockParam(type="text", text="[No content in tool result]")
                        ],
                        is_error=result.isError,
                    )
                )

            # All content is now included within the tool_result block.

        return BetaMessageParam(role="user", content=content_blocks)

    @staticmethod
    def _sanitize_tool_id(tool_id: str | None) -> str:
        """
        Anthropic tool_use ids must match ^[a-zA-Z0-9_-]+$.
        Clean any other characters to underscores and provide a stable fallback.
        """
        if not tool_id:
            return "tool"
        cleaned = re.sub(r"[^a-zA-Z0-9_-]", "_", tool_id)
        return cleaned or "tool"
