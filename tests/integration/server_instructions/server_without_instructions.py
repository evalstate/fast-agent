#!/usr/bin/env python3
"""
MCP server WITHOUT instructions for testing server instructions feature.
"""

import logging

from mcp.server.fastmcp import FastMCP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create the FastMCP server WITHOUT instructions
app = FastMCP(
    name="Server Without Instructions"
    # No instructions parameter provided
)


# Simple utility tools
@app.tool(
    name="echo",
    description="Echo back the input",
)
def echo(message: str) -> str:
    return f"Echo: {message}"


@app.tool(
    name="ping",
    description="Simple ping tool",
)
def ping() -> str:
    return "pong"


@app.tool(
    name="get_status",
    description="Get server status",
)
def get_status() -> str:
    return "Server is running"


@app.tool(
    name="random_number",
    description="Generate a random number between 1 and 100",
)
def random_number() -> str:
    import random
    return f"Random number: {random.randint(1, 100)}"


if __name__ == "__main__":
    import sys
    app.run(transport="stdio")
    sys.exit(0)