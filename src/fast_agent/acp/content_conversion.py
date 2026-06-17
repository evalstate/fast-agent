"""
Content block conversion from ACP to MCP format.

This module handles conversion of content blocks from the Agent Client Protocol (ACP)
to Model Context Protocol (MCP) format for processing by fast-agent.
"""

import re
from dataclasses import dataclass
from typing import cast
from urllib.parse import urlparse

import acp.schema as acp_schema
import mcp.types as mcp_types
from acp.helpers import (
    ContentBlock as ACPContentBlock,
)
from acp.helpers import (
    audio_block,
    embedded_blob_resource,
    embedded_text_resource,
    image_block,
    resource_block,
    resource_link_block,
    text_block,
)
from mcp.types import ContentBlock as MCPContentBlock
from pydantic import AnyUrl

from fast_agent.io.path_uri import file_uri_to_path
from fast_agent.utils.commandline import quote_commandline_token


def convert_acp_content_to_mcp(acp_content: ACPContentBlock) -> MCPContentBlock | None:
    """
    Convert an ACP content block to MCP format.

    Args:
        acp_content: Content block from ACP (Agent Client Protocol)

    Returns:
        Corresponding MCP content block, or None if conversion is not supported

    Supported conversions:
        - TextContentBlock -> TextContent
        - ImageContentBlock -> ImageContent
        - AudioContentBlock -> AudioContent
        - ResourceLink -> ResourceLink
        - EmbeddedResourceContentBlock -> EmbeddedResource
    """
    match acp_content:
        case acp_schema.TextContentBlock():
            return _convert_text_content(acp_content)
        case acp_schema.ImageContentBlock():
            return _convert_image_content(acp_content)
        case acp_schema.AudioContentBlock():
            return _convert_audio_content(acp_content)
        case acp_schema.ResourceContentBlock():
            return _convert_resource_link(acp_content)
        case acp_schema.EmbeddedResourceContentBlock():
            return _convert_embedded_resource(acp_content)
        case _:
            return None


def _convert_text_content(
    acp_text: acp_schema.TextContentBlock,
) -> mcp_types.TextContent:
    """Convert ACP TextContentBlock to MCP TextContent."""
    return mcp_types.TextContent(
        type="text",
        text=acp_text.text,
        annotations=_convert_annotations(acp_text.annotations),
    )


def _convert_image_content(
    acp_image: acp_schema.ImageContentBlock,
) -> mcp_types.ImageContent:
    """Convert ACP ImageContentBlock to MCP ImageContent."""
    return mcp_types.ImageContent(
        type="image",
        data=acp_image.data,
        mimeType=acp_image.mime_type,
        annotations=_convert_annotations(acp_image.annotations),
    )


def _convert_audio_content(
    acp_audio: acp_schema.AudioContentBlock,
) -> mcp_types.AudioContent:
    """Convert ACP AudioContentBlock to MCP AudioContent."""
    return mcp_types.AudioContent(
        type="audio",
        data=acp_audio.data,
        mimeType=acp_audio.mime_type,
        annotations=_convert_annotations(acp_audio.annotations),
    )


def _convert_resource_link(
    acp_link: acp_schema.ResourceContentBlock,
) -> mcp_types.ResourceLink:
    """Convert ACP ResourceLink to MCP ResourceLink."""
    return mcp_types.ResourceLink(
        type="resource_link",
        name=acp_link.name,
        uri=AnyUrl(acp_link.uri),
        mimeType=acp_link.mime_type,
        size=acp_link.size,
        description=acp_link.description,
        title=acp_link.title,
        annotations=_convert_annotations(acp_link.annotations),
    )


def _convert_embedded_resource(
    acp_resource: acp_schema.EmbeddedResourceContentBlock,
) -> mcp_types.EmbeddedResource:
    """Convert ACP EmbeddedResourceContentBlock to MCP EmbeddedResource."""
    return mcp_types.EmbeddedResource(
        type="resource",
        resource=_convert_resource_contents(acp_resource.resource),
        annotations=_convert_annotations(acp_resource.annotations),
    )


def _convert_resource_contents(
    acp_resource: acp_schema.TextResourceContents | acp_schema.BlobResourceContents,
) -> mcp_types.TextResourceContents | mcp_types.BlobResourceContents:
    """Convert ACP resource contents to MCP resource contents."""
    match acp_resource:
        case acp_schema.TextResourceContents():
            return mcp_types.TextResourceContents(
                uri=AnyUrl(acp_resource.uri),
                mimeType=acp_resource.mime_type or None,
                text=acp_resource.text,
            )
        case acp_schema.BlobResourceContents():
            return mcp_types.BlobResourceContents(
                uri=AnyUrl(acp_resource.uri),
                mimeType=acp_resource.mime_type or None,
                blob=acp_resource.blob,
            )
        case _:
            raise ValueError(f"Unsupported resource type: {type(acp_resource)}")


def _convert_annotations(
    acp_annotations: acp_schema.Annotations | None,
) -> mcp_types.Annotations | None:
    """Convert ACP annotations to MCP annotations."""
    if not acp_annotations:
        return None

    audience = (
        cast("list[mcp_types.Role]", list(acp_annotations.audience))
        if acp_annotations.audience
        else None
    )
    return mcp_types.Annotations(
        audience=audience,
        priority=acp_annotations.priority,
    )


def _file_uri_to_path(uri: str) -> str:
    """
    Convert a file:// URI to a local filesystem path.

    Examples:
        file:///home/user/foo.txt -> /home/user/foo.txt
        file:///C:/Users/test/foo.txt -> C:/Users/test/foo.txt
        /already/a/path.txt -> /already/a/path.txt (unchanged)
    """
    parsed = urlparse(uri)
    if parsed.scheme == "file":
        if parsed.netloc and not parsed.path:
            return parsed.netloc
        return _normalize_acp_file_path(str(file_uri_to_path(parsed)))
    return uri


def _normalize_acp_file_path(path: str) -> str:
    if re.match(r"^/[A-Za-z]:", path):
        return path[1:]
    return path


def _quote_slash_command_path(path: str) -> str:
    return quote_commandline_token(path, syntax="posix")


@dataclass(frozen=True, slots=True)
class _SlashCommandToken:
    start: int
    end: int
    value: str


def _slash_command_tokens(text: str) -> list[_SlashCommandToken]:
    tokens: list[_SlashCommandToken] = []
    index = 0
    length = len(text)

    while index < length:
        while index < length and text[index].isspace():
            index += 1
        if index >= length:
            break

        start = index
        value: list[str] = []
        quote: str | None = None
        while index < length:
            char = text[index]
            if quote is None:
                if char.isspace():
                    break
                if char in {'"', "'"}:
                    quote = char
                    index += 1
                    continue
                if char == "\\" and index + 1 < length:
                    value.append(text[index + 1])
                    index += 2
                    continue
                value.append(char)
                index += 1
                continue

            if char == quote:
                quote = None
                index += 1
                continue
            if quote == '"' and char == "\\" and index + 1 < length:
                value.append(text[index + 1])
                index += 2
                continue
            value.append(char)
            index += 1

        tokens.append(_SlashCommandToken(start=start, end=index, value="".join(value)))

    return tokens


def _inline_referenced_resource_paths(
    text: str,
    path_by_filename: dict[str, str],
) -> tuple[str, int]:
    inlined_parts: list[str] = []
    replacement_count = 0
    last_index = 0

    for token in _slash_command_tokens(text):
        if not token.value.startswith("@"):
            continue
        path = path_by_filename.get(token.value[1:])
        if path is None:
            continue
        inlined_parts.append(text[last_index : token.start])
        inlined_parts.append(path)
        last_index = token.end
        replacement_count += 1

    if replacement_count == 0:
        return text, 0

    inlined_parts.append(text[last_index:])
    return "".join(inlined_parts), replacement_count


def inline_resources_for_slash_command(
    acp_prompt: list[ACPContentBlock],
) -> list[ACPContentBlock]:
    """
    If the prompt starts with a slash command, inline resource paths into the text.

    When ACP clients attach files via "@" syntax, they come as separate resource
    blocks. This function detects slash commands and converts file:// URIs to
    local paths so slash command handlers receive usable file paths.

    Handles two client behaviors:
    1. Text has "@filename" references that match resource URIs by filename:
       Input:  [TextBlock("/card @foo.txt"), ResourceBlock(uri="file:///path/foo.txt")]
       Output: [TextBlock("/card /path/foo.txt")]

    2. Text ends with trailing space and resources follow without "@" references:
       Input:  [TextBlock("/card "), ResourceBlock(uri="file:///foo.txt")]
       Output: [TextBlock("/card /foo.txt")]

    Args:
        acp_prompt: List of ACP content blocks

    Returns:
        Modified prompt with resource paths inlined if it's a slash command,
        otherwise returns the original prompt unchanged.
    """
    if not acp_prompt:
        return acp_prompt

    first_block = acp_prompt[0]

    # Only process if first block is text starting with "/"
    if not isinstance(first_block, acp_schema.TextContentBlock):
        return acp_prompt

    text = first_block.text
    if not text.lstrip().startswith("/"):
        return acp_prompt

    path_by_filename, paths = _resource_paths_by_filename(acp_prompt[1:])
    if not paths:
        return acp_prompt

    inlined_text, replacement_count = _inline_referenced_resource_paths(
        text,
        path_by_filename,
    )

    if replacement_count == 0:
        # No matching @references found, append paths to the end
        inlined_text = text.rstrip() + " " + " ".join(paths)

    return [acp_schema.TextContentBlock(type="text", text=inlined_text)]


def _resource_paths_by_filename(
    blocks: list[ACPContentBlock],
) -> tuple[dict[str, str], list[str]]:
    path_by_filename: dict[str, str] = {}
    paths: list[str] = []

    for block in blocks:
        if not isinstance(block, acp_schema.EmbeddedResourceContentBlock):
            continue
        if not block.resource.uri:
            continue

        path = _file_uri_to_path(str(block.resource.uri))
        quoted_path = _quote_slash_command_path(path)
        paths.append(quoted_path)

        filename = path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
        if filename:
            path_by_filename[filename] = quoted_path

    return path_by_filename, paths


def convert_acp_prompt_to_mcp_content_blocks(
    acp_prompt: list[ACPContentBlock],
) -> list[MCPContentBlock]:
    """
    Convert a list of ACP content blocks to MCP content blocks.

    Args:
        acp_prompt: List of content blocks from ACP prompt

    Returns:
        List of MCP content blocks (only supported types are converted)
    """
    mcp_blocks = []

    for acp_block in acp_prompt:
        mcp_block = convert_acp_content_to_mcp(acp_block)
        if mcp_block is not None:
            mcp_blocks.append(mcp_block)

    return mcp_blocks


def convert_mcp_content_to_acp(mcp_content: MCPContentBlock) -> ACPContentBlock | None:
    """
    Convert an MCP content block to ACP format.

    Args:
        mcp_content: Content block from MCP (TextContent, ImageContent, etc.)

    Returns:
        Corresponding ACP ContentBlock, or None if conversion is not supported.
    """
    match mcp_content:
        case mcp_types.TextContent():
            return text_block(mcp_content.text)
        case mcp_types.ImageContent():
            return image_block(mcp_content.data, mcp_content.mimeType)
        case mcp_types.AudioContent():
            return audio_block(mcp_content.data, mcp_content.mimeType)
        case mcp_types.ResourceLink():
            return resource_link_block(
                name=mcp_content.name,
                uri=str(mcp_content.uri),
                mime_type=mcp_content.mimeType,
                size=mcp_content.size,
                description=mcp_content.description,
                title=mcp_content.title,
            )
        case mcp_types.EmbeddedResource():
            return _convert_mcp_embedded_resource_to_acp(mcp_content)
        case _:
            return None


def _convert_mcp_embedded_resource_to_acp(
    mcp_content: mcp_types.EmbeddedResource,
) -> ACPContentBlock | None:
    match mcp_content.resource:
        case mcp_types.TextResourceContents():
            embedded = embedded_text_resource(
                uri=str(mcp_content.resource.uri),
                text=mcp_content.resource.text,
                mime_type=mcp_content.resource.mimeType,
            )
        case mcp_types.BlobResourceContents():
            embedded = embedded_blob_resource(
                uri=str(mcp_content.resource.uri),
                blob=mcp_content.resource.blob,
                mime_type=mcp_content.resource.mimeType,
            )
        case _:
            return None
    return resource_block(embedded)
