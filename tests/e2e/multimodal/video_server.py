#!/usr/bin/env python3
"""
Simple MCP server that returns ResourceLinks to video content for testing.
"""

import logging
import sys

from mcp.server.fastmcp import FastMCP
from mcp.types import ResourceLink, TextContent

from fast_agent.mcp.helpers.content_helpers import text_content, video_link

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create the FastMCP server
app = FastMCP(name="VideoLinkServer", debug=True)

# Global variable to store the video URL
video_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Mystery Video


@app.tool(
    name="get_video_link",
    description="Returns a ResourceLink to a sample video for analysis",
    structured_output=False,
)
async def get_video_link() -> list[TextContent | ResourceLink]:
    """Return a ResourceLink to a video."""
    return [
        text_content("Here's a video link for analysis:"),
        video_link(video_url, name="Mystery Video"),
    ]


if __name__ == "__main__":
    # Get video URL from command line argument or use default
    if len(sys.argv) > 1:
        video_url = sys.argv[1]
        logger.info(f"Using video URL: {video_url}")
    else:
        logger.info(f"No video URL provided, using default: {video_url}")

    # Run the server using stdio transport
    app.run(transport="stdio")
