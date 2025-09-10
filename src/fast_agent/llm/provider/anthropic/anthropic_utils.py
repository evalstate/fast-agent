"""
Utility functions for Anthropic integration with MCP.

Provides conversion between Anthropic message formats and PromptMessageExtended,
leveraging existing code for resource handling and delimited formats.
"""

from anthropic.types import (
    MessageParam,
)
from mcp.types import (
    EmbeddedResource,
    ImageContent,
    TextContent,
    TextResourceContents,
)

from fast_agent.types import PromptMessageExtended


# TODO -- only used for saving, but this will be driven directly from PromptMessages
def anthropic_message_param_to_prompt_message_multipart(
    message_param: MessageParam,
) -> PromptMessageExtended:
    """
    Convert an Anthropic MessageParam to a PromptMessageExtended.

    Args:
        message_param: The Anthropic MessageParam to convert

    Returns:
        A PromptMessageExtended representation
    """
    role = message_param["role"]
    content = message_param["content"]

    # Handle string content (user messages can be simple strings)
    if isinstance(content, str):
        return PromptMessageExtended(role=role, content=[TextContent(type="text", text=content)])

    # Convert content blocks to MCP content types
    mcp_contents = []

    for block in content:
        if isinstance(block, dict):
            if block.get("type") == "text":
                text = block.get("text", "")

                # Check if this is a resource marker
                if (
                    text
                    and (text.startswith("[Resource:") or text.startswith("[Binary Resource:"))
                    and "\n" in text
                ):
                    header, content_text = text.split("\n", 1)
                    if "MIME:" in header:
                        mime_match = header.split("MIME:", 1)[1].split("]")[0].strip()
                        if mime_match != "text/plain":  # Only process non-plain text resources
                            if "Resource:" in header and "Binary Resource:" not in header:
                                uri = header.split("Resource:", 1)[1].split(",")[0].strip()
                                mcp_contents.append(
                                    EmbeddedResource(
                                        type="resource",
                                        resource=TextResourceContents(
                                            uri=uri,
                                            mimeType=mime_match,
                                            text=content_text,
                                        ),
                                    )
                                )
                                continue

                # Regular text content
                mcp_contents.append(TextContent(type="text", text=text))

            elif block.get("type") == "image":
                # Image content
                source = block.get("source", {})
                if isinstance(source, dict) and source.get("type") == "base64":
                    media_type = source.get("media_type", "image/png")
                    data = source.get("data", "")
                    mcp_contents.append(ImageContent(type="image", data=data, mimeType=media_type))

    return PromptMessageExtended(role=role, content=mcp_contents)
