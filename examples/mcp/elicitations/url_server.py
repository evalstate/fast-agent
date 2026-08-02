"""Modern, request-scoped URL elicitation example."""

from urllib.parse import quote

from mcp.server.mcpserver import Context, MCPServer
from mcp_types import (
    ElicitRequest,
    ElicitRequestURLParams,
    ElicitResult,
    InputRequiredResult,
)

RESPONSE_KEY = "authorize_console"

server = MCPServer("Sandbox Console URL Elicitation")


@server.tool()
def request_console_access(
    sandbox_id: str,
    ctx: Context,
) -> str | InputRequiredResult:
    """Request navigation consent without claiming external authorization completed."""
    responses = ctx.input_responses
    request_state = f"console-access:{sandbox_id}"

    if responses is None:
        authorization_url = f"https://example.com/authorize?sandbox={quote(sandbox_id, safe='')}"
        return InputRequiredResult(
            input_requests={
                RESPONSE_KEY: ElicitRequest(
                    params=ElicitRequestURLParams(
                        message=f"Authorize browser access to sandbox {sandbox_id}.",
                        url=authorization_url,
                    )
                )
            },
            request_state=request_state,
        )

    if ctx.request_state != request_state:
        raise ValueError("URL elicitation request state did not match")

    response = responses.get(RESPONSE_KEY)
    if not isinstance(response, ElicitResult):
        raise ValueError("URL elicitation response was missing")

    match response.action:
        case "accept":
            return (
                f"SIMULATED: navigation accepted for {sandbox_id}; "
                "external completion was not verified and no console access was granted"
            )
        case "decline":
            return f"SIMULATED: navigation declined for {sandbox_id}; no URL was opened"
        case "cancel":
            return f"SIMULATED: navigation cancelled for {sandbox_id}; no URL was opened"


if __name__ == "__main__":
    server.run(transport="stdio")
