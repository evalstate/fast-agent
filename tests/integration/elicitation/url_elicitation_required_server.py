"""Integration test server for URL elicitation required error handling."""

from mcp.server.fastmcp import FastMCP
from mcp.shared.exceptions import UrlElicitationRequiredError
from mcp.types import ElicitRequestURLParams

mcp = FastMCP("URL Elicitation Required Test Server", log_level="INFO")


@mcp.tool()
async def url_required_valid_tool() -> str:
    """Return URL elicitation required error with well-formed data."""
    raise UrlElicitationRequiredError(
        [
            ElicitRequestURLParams(
                mode="url",
                message="Complete authorization to continue.",
                url="https://example.com/auth/first",
                elicitationId="valid-1",
            ),
            ElicitRequestURLParams(
                mode="url",
                message="Confirm payment in your browser.",
                url="https://example.com/pay/second",
                elicitationId="valid-2",
            ),
        ],
        message="This request requires URL elicitations.",
    )


@mcp.tool()
async def url_required_malformed_tool() -> str:
    """Return URL elicitation required error with malformed data payload."""
    raise UrlElicitationRequiredError(
        [],
        message="Malformed URL elicitation payload",
    )


if __name__ == "__main__":
    mcp.run()
