#!/usr/bin/env python3
"""
Simple MCP server that responds to tool calls with text and image content.
"""

import logging

from mcp.server.fastmcp import FastMCP
from mcp.types import ResourceLink
from pydantic import AnyUrl

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create the FastMCP server
app = FastMCP(name="Linked Resources", instructions="For integration tests with linked resources.")


@app.tool(name="getlink", description="Returns the colour of a shirt.")
def getlink() -> ResourceLink:
    return ResourceLink(
        name="linked resource",
        type="resource_link",
        uri=AnyUrl("resource://fast-agent/linked-resource"),
        description="A description, perhaps for the LLM",
        mimeType="text/plain",
    )


if __name__ == "__main__":
    # Run the server using stdio transport
    app.run(transport="stdio")
