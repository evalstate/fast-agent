#!/usr/bin/env python3
"""
Simple MCP server that responds to tool calls with text and image content.
"""

import logging
from pathlib import Path

from mcp.server.fastmcp import Context, FastMCP, Image
from mcp.types import ImageContent, TextContent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create the FastMCP server
app = FastMCP(name="ImageToolServer", debug=True)


@app.tool(name="get_image", description="Returns an image with a descriptive text")
async def get_image(
    image_name: str = "default", ctx: Context = None
) -> list[TextContent | ImageContent]:
    """
    Returns an image file along with a descriptive text.

    Args:
        image_name: Name of the image to return (default just returns image.jpg)

    Returns:
        A list containing a text message and the requested image
    """
    try:
        # Read the image file and convert to base64
        # Create the response with text and image
        return [
            TextContent(type="text", text="Here's your image:"),
            Image(path="image.jpg").to_image_content(),
        ]
    except Exception as e:
        logger.exception(f"Error processing image: {e}")
        return [TextContent(type="text", text=f"Error processing image: {str(e)}")]


if __name__ == "__main__":
    # Check if the default image exists
    if not Path("image.jpg").exists():
        logger.warning("Default image file 'image.jpg' not found in the current directory")
        logger.warning("Please add an image file named 'image.jpg' to the current directory")

    # Run the server using stdio transport
    app.run(transport="stdio")
