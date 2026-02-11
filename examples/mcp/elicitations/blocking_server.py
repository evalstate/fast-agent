"""
MCP Server for Testing Elicitation Blocking Scenarios

This server provides tools to test the concurrent POST fix for elicitation.
It demonstrates both:
1. Associated elicitation (via POST response SSE stream)
2. Dissociated elicitation (via GET stream)

The blocking issue occurs when:
- Client sends tools/call POST
- Server sends elicitation request to client
- Client tries to POST elicitation response
- Client's response is blocked by HTTP client connection pool

Without the concurrent POST fix, the elicitation response cannot be sent
until the original tools/call POST completes, causing timeouts.

Run with: python examples/mcp/elicitations/blocking_elicitation_server.py
Connect to: http://127.0.0.1:8000/mcp
"""

import logging
import sys
from typing import TYPE_CHECKING, Any

from mcp.server.elicitation import (
    AcceptedElicitation,
    CancelledElicitation,
    DeclinedElicitation,
)
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from mcp.types import ElicitResult

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("blocking_elicitation_server")

# Create MCP server (host/port are configured here, not in run())
mcp = FastMCP(
    "Blocking Elicitation Test Server",
    log_level="DEBUG",
    host="127.0.0.1",
    port=8000,
)


class DeploymentConfig(BaseModel):
    """Schema for deployment confirmation elicitation."""

    environment: str = Field(
        description="Target environment for deployment",
        json_schema_extra={"enum": ["development", "staging", "production"]},
    )
    confirm: bool = Field(description="Confirm the deployment?")


@mcp.tool()
async def deploy_associated() -> str:
    """
    Test elicitation via POST response SSE stream (associated with request).

    This uses FastMCP's ctx.elicit() which automatically sets related_request_id,
    routing the elicitation through the POST response stream.

    Expected behavior:
    - Without fix: Client blocks, elicitation may timeout
    - With fix: Elicitation completes immediately
    """
    ctx = mcp.get_context()
    logger.info("deploy_associated: Sending elicitation via POST response SSE")

    result = await ctx.elicit(
        "Confirm deployment configuration (associated - via POST SSE)",
        schema=DeploymentConfig,
    )

    match result:
        case AcceptedElicitation(data=data):
            logger.info(f"Elicitation accepted: {data}")
            return f"Deployed to {data.environment} (confirm={data.confirm})"
        case DeclinedElicitation():
            logger.info("Elicitation declined")
            return "Deployment declined by user"
        case CancelledElicitation():
            logger.info("Elicitation cancelled")
            return "Deployment cancelled"
        case _:
            return f"Unexpected result: {result}"


@mcp.tool()
async def deploy_dissociated() -> str:
    """
    Test elicitation via GET stream (dissociated from request).

    This bypasses FastMCP and calls session.elicit_form() directly WITHOUT
    setting related_request_id, routing the elicitation through the GET stream.

    This matches the scenario from the user's logs where elicitation
    goes via GET and the response is blocked.

    Expected behavior:
    - Without fix: Client blocks, elicitation times out after ~20s
    - With fix: Elicitation completes immediately
    """
    ctx = mcp.get_context()
    session = ctx.request_context.session

    logger.info("deploy_dissociated: Sending elicitation via GET stream (no related_request_id)")

    # Call session.elicit_form WITHOUT related_request_id
    # This routes the elicitation to the GET stream instead of POST response
    requested_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "environment": {
                "type": "string",
                "title": "Environment",
                "enum": ["development", "staging", "production"],
            },
            "confirm": {
                "type": "boolean",
                "title": "Confirm deployment?",
            },
        },
        "required": ["environment", "confirm"],
    }

    result: ElicitResult = await session.elicit_form(
        message="Confirm deployment configuration (dissociated - via GET stream)",
        requestedSchema=requested_schema,
        related_request_id=None,  # <-- KEY: No related_request_id = routes to GET stream
    )

    logger.info(f"Elicitation result: {result}")

    match result.action:
        case "accept":
            content = result.content or {}
            env = content.get("environment", "unknown")
            confirm = content.get("confirm", False)
            return f"Deployed to {env} (confirm={confirm})"
        case "decline":
            return "Deployment declined by user"
        case "cancel":
            return "Deployment cancelled"
        case _:
            return f"Unexpected action: {result.action}"


@mcp.tool()
async def ping() -> str:
    """Simple ping tool for testing basic connectivity."""
    return "pong"


if __name__ == "__main__":
    logger.info("Starting blocking elicitation test server...")
    logger.info("Connect to: http://127.0.0.1:8000/mcp")
    logger.info("")
    logger.info("Available tools:")
    logger.info("  - deploy_associated: Elicitation via POST response SSE")
    logger.info("  - deploy_dissociated: Elicitation via GET stream")
    logger.info("  - ping: Basic connectivity test")
    mcp.run(transport="streamable-http")

