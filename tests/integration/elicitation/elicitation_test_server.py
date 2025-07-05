"""
Enhanced test server for sampling functionality
"""

import logging
import sys

from mcp import (
    ReadResourceResult,
)
from mcp.server.elicitation import (
    AcceptedElicitation,
    CancelledElicitation,
    DeclinedElicitation,
)
from mcp.server.fastmcp import FastMCP
from mcp.types import TextResourceContents
from pydantic import AnyUrl, BaseModel, Field

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("elicitation_server")

# Create MCP server
mcp = FastMCP("MCP Elicitation Server", log_level="DEBUG")


@mcp.resource(uri="elicitation://generate")
async def get() -> ReadResourceResult:
    """Tool that echoes back the input parameter"""

    class ServerRating(BaseModel):
        rating: bool = Field(description="Server Rating")

    mcp.get_context()
    result = await mcp.get_context().elicit("Rate this server 5 stars?", schema=ServerRating)
    ret = "nothing"
    match result:
        case AcceptedElicitation(data=data):
            if data.rating:
                ret = str(data.rating)
        case DeclinedElicitation():
            ret = "declined"
        case CancelledElicitation():
            ret = "cancelled"

    # Return the result directly, without nesting
    return ReadResourceResult(
        contents=[
            TextResourceContents(
                mimeType="text/plain", uri=AnyUrl("elicitation://generate"), text=f"Result: {ret}"
            )
        ]
    )


if __name__ == "__main__":
    logger.info("Starting elicitation test server...")
    mcp.run()
