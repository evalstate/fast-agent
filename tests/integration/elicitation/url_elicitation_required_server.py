"""Integration test server for URL elicitation required error handling."""

import json

from fastmcp import FastMCP
from mcp_types import ElicitRequestURLParams

mcp = FastMCP("URL Elicitation Required Test Server")
_PREFIX = "fast-agent-url-elicitation-required:"


@mcp.tool()
async def url_required_valid_tool() -> str:
    """Return URL elicitation required error with well-formed data."""
    raise RuntimeError(
        _PREFIX
        + json.dumps(
            {
                "elicitations": [
                    ElicitRequestURLParams(
                        mode="url",
                        message="Complete authorization to continue.",
                        url="https://example.com/auth/first",
                        elicitation_id="valid-1",
                    ).model_dump(by_alias=True),
                    ElicitRequestURLParams(
                        mode="url",
                        message="Confirm payment in your browser.",
                        url="https://example.com/pay/second",
                        elicitation_id="valid-2",
                    ).model_dump(by_alias=True),
                ]
            }
        )
    )


@mcp.tool()
async def url_required_malformed_tool() -> str:
    """Return URL elicitation required error with malformed data payload."""
    raise RuntimeError(_PREFIX + json.dumps({"elicitations": []}))


if __name__ == "__main__":
    mcp.run()
