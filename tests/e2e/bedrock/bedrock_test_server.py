#!/usr/bin/env python3
"""
Bedrock-specific MCP server that matches the main smoke test server functionality.
"""

import logging

from mcp.server.fastmcp import FastMCP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create the FastMCP server
app = FastMCP(name="Bedrock Test Server")


@app.tool(
    name="check_weather",
    description="Returns the weather for a specified location.",
)
def check_weather(location: str) -> str:
    # Return sunny weather condition
    return "It's sunny in " + location


@app.tool(name="shirt_colour", description="returns the colour of the shirt being worn")
def shirt_colour() -> str:
    return "blue polka dots"


if __name__ == "__main__":
    # Run the server using stdio transport
    app.run(transport="stdio")
