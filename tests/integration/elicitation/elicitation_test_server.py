"""
Enhanced test server for sampling functionality
"""

import logging
import sys

from fastmcp import FastMCP
from fastmcp.server.dependencies import get_context
from fastmcp.server.elicitation import (
    AcceptedElicitation,
    CancelledElicitation,
    DeclinedElicitation,
)
from pydantic import BaseModel, Field

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("elicitation_server")

# Create MCP server
mcp = FastMCP("MCP Elicitation Server")


@mcp.resource(uri="elicitation://generate")
async def get() -> str:
    """Tool that echoes back the input parameter"""

    class ServerRating(BaseModel):
        rating: bool = Field(description="Server Rating")

    get_context()
    result = await get_context().elicit("Rate this server 5 stars?", ServerRating)
    ret = "nothing"
    match result:
        case AcceptedElicitation(data=data):
            assert isinstance(data, ServerRating)
            if data.rating:
                ret = str(data.rating)
        case DeclinedElicitation():
            ret = "declined"
        case CancelledElicitation():
            ret = "cancelled"

    return f"Result: {ret}"


if __name__ == "__main__":
    logger.info("Starting elicitation test server...")
    mcp.run()
