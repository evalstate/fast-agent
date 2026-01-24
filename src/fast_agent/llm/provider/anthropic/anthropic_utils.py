"""
Utility functions for Anthropic integration with MCP.

Provides conversion between Anthropic message formats and PromptMessageExtended,
leveraging existing code for resource handling and delimited formats.
"""

from anthropic.types import (
    MessageParam,
)
from mcp.types import (
    ImageContent,
    TextContent,
)

from fast_agent.mcp.resource_utils import parse_resource_marker
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

                resource_marker = parse_resource_marker(text)
                if resource_marker:
                    mcp_contents.append(resource_marker)
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
