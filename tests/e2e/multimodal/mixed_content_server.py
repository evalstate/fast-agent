#!/usr/bin/env python3
"""
MCP server that reproduces the OpenAI tool call validation issue scenario.

This server provides two tools:
1. get_page_data: Returns pure text (simulates browser_snapshot)
2. take_screenshot: Returns text + image (simulates browser_take_screenshot)

When both tools are called in parallel, the mixed content from take_screenshot
used to cause OpenAI API validation errors before the fix.
"""

import base64
import logging

from mcp.server.fastmcp import FastMCP
from mcp.types import ImageContent, TextContent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create the FastMCP server
app = FastMCP(name="MixedContentServer", debug=True)


@app.tool(
    name="get_page_data",
    description="Gets current page data and navigation state (returns pure text)",
)
def get_page_data() -> str:
    """
    Simulates browser_snapshot - returns pure text data.
    This represents a tool that returns only text content.
    """
    return "Page snapshot: Navigation complete, DOM ready, elements loaded successfully"


@app.tool(
    name="take_screenshot",
    description="Takes a screenshot of the current page (returns text description + image)",
)
def take_screenshot() -> list[TextContent | ImageContent]:
    """
    Simulates browser_take_screenshot - returns mixed content (text + image).
    This represents a tool that returns both text and image content,
    which used to cause the OpenAI validation issue.
    """
    try:
        # Create a valid minimal 1x1 pixel transparent PNG
        fake_image_data = base64.b64encode(
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
            b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01"
            b"\x00\x00\x05\x00\x01\r\n-\xdb\x00\x00\x00\x00IEND\xaeB`\x82"
        ).decode("utf-8")

        return [
            TextContent(type="text", text="Screenshot captured successfully"),
            ImageContent(type="image", data=fake_image_data, mimeType="image/png"),
        ]
    except Exception as e:
        logger.exception(f"Error creating screenshot: {e}")
        return [TextContent(type="text", text=f"Error taking screenshot: {str(e)}")]


@app.tool(
    name="get_both_data",
    description="Gets both page data and screenshot in one call (for testing single tool with mixed content)",
)
def get_both_data() -> list[TextContent | ImageContent]:
    """
    Returns both text and image content in a single tool call.
    This can be used to test mixed content handling in non-parallel scenarios.
    """
    try:
        # Same valid minimal PNG as take_screenshot
        fake_image_data = base64.b64encode(
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
            b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01"
            b"\x00\x00\x05\x00\x01\r\n-\xdb\x00\x00\x00\x00IEND\xaeB`\x82"
        ).decode("utf-8")

        return [
            TextContent(type="text", text="Combined data: Page loaded, screenshot taken"),
            TextContent(type="text", text="Navigation state: Ready"),
            ImageContent(type="image", data=fake_image_data, mimeType="image/png"),
        ]
    except Exception as e:
        logger.exception(f"Error getting combined data: {e}")
        return [TextContent(type="text", text=f"Error getting data: {str(e)}")]


if __name__ == "__main__":
    # Run the server using stdio transport
    app.run(transport="stdio")
