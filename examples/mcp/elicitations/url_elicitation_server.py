"""
MCP Server for URL Elicitation Demo

This server demonstrates URL mode elicitation, which allows servers to direct
users to external URLs for sensitive interactions that should not pass through
the MCP client (OAuth flows, payment processing, API key entry, etc.).

URL mode elicitation is "out-of-band" - the user interaction happens in their
browser, not through the MCP protocol.
"""

import logging
import sys
import uuid

from mcp import ReadResourceResult
from mcp.server.fastmcp import FastMCP
from mcp.types import TextResourceContents
from pydantic import AnyUrl

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("url_elicitation_server")

# Create MCP server
mcp = FastMCP("URL Elicitation Demo Server", log_level="INFO")

# Simulated state - in a real server, this would be persistent storage
_authorized_sessions: set[str] = set()


@mcp.tool()
async def authorize_api_access(service_name: str) -> str:
    """
    Request API access authorization from the user.

    This tool demonstrates URL elicitation for OAuth-like flows.
    The user will be directed to an external URL to complete authorization.

    Args:
        service_name: The name of the service to authorize access for
    """
    ctx = mcp.get_context()

    # Generate a unique elicitation ID for tracking
    elicitation_id = str(uuid.uuid4())

    # In a real implementation, this URL would point to your OAuth authorization endpoint
    # with proper state parameter, client_id, redirect_uri, etc.
    auth_url = f"https://example.com/oauth/authorize?service={service_name}&state={elicitation_id}"

    # Request URL elicitation - user will be shown this URL to navigate to
    result = await ctx.elicit_url(
        message=f"Please authorize access to {service_name} to continue. "
        "You will be redirected to a secure authorization page.",
        url=auth_url,
        elicitation_id=elicitation_id,
    )

    if result.action == "accept":
        # User was shown the URL and acknowledged it
        # In a real implementation, you would wait for the OAuth callback
        # and verify the authorization was completed
        return (
            f"Authorization URL displayed for {service_name}. "
            f"Please complete the authorization in your browser. "
            f"Elicitation ID: {elicitation_id}"
        )
    elif result.action == "decline":
        return f"User declined to authorize access to {service_name}."
    else:
        return f"Authorization request was cancelled for {service_name}."


@mcp.tool()
async def enter_api_key(api_name: str) -> str:
    """
    Request the user to enter an API key via secure URL.

    Sensitive credentials should never be collected through form elicitation
    as they would pass through the MCP client/LLM context. URL elicitation
    ensures credentials stay outside the MCP message flow.

    Args:
        api_name: The name of the API requiring a key
    """
    ctx = mcp.get_context()

    elicitation_id = str(uuid.uuid4())

    # In a real implementation, this would be your secure credential entry page
    credential_url = f"https://example.com/credentials/enter?api={api_name}&id={elicitation_id}"

    result = await ctx.elicit_url(
        message=f"Please enter your {api_name} API key securely. "
        "Your credentials will NOT be shared with the AI assistant.",
        url=credential_url,
        elicitation_id=elicitation_id,
    )

    if result.action == "accept":
        return (
            f"Credential entry page displayed for {api_name}. "
            "Please enter your API key in the secure form."
        )
    elif result.action == "decline":
        return f"User declined to enter credentials for {api_name}."
    else:
        return f"Credential entry was cancelled for {api_name}."


@mcp.tool()
async def initiate_payment(amount: float, currency: str, description: str) -> str:
    """
    Initiate a payment flow via URL elicitation.

    Payment processing should always use URL elicitation to ensure
    payment details stay outside the MCP message context.

    Args:
        amount: The payment amount
        currency: The currency code (e.g., USD, EUR)
        description: Description of what the payment is for
    """
    ctx = mcp.get_context()

    elicitation_id = str(uuid.uuid4())

    # In a real implementation, this would be your payment processor's checkout URL
    payment_url = (
        f"https://pay.example.com/checkout?"
        f"amount={amount}&currency={currency}&id={elicitation_id}"
    )

    result = await ctx.elicit_url(
        message=f"Complete payment of {amount} {currency} for: {description}. "
        "You will be redirected to a secure payment page.",
        url=payment_url,
        elicitation_id=elicitation_id,
    )

    if result.action == "accept":
        return (
            f"Payment page displayed for {amount} {currency}. "
            "Please complete the payment in your browser."
        )
    elif result.action == "decline":
        return "User declined to proceed with payment."
    else:
        return "Payment request was cancelled."


@mcp.resource(uri="elicitation://url-demo")
async def url_demo_resource() -> ReadResourceResult:
    """
    Demonstrate URL elicitation via a resource.

    This resource shows how URL elicitation can be triggered from
    a resource read operation.
    """
    ctx = mcp.get_context()

    elicitation_id = str(uuid.uuid4())

    result = await ctx.elicit_url(
        message="This resource requires external verification. "
        "Please verify your identity to access the data.",
        url=f"https://verify.example.com/identity?id={elicitation_id}",
        elicitation_id=elicitation_id,
    )

    if result.action == "accept":
        response = (
            "Verification URL displayed.\n\n"
            "In a real implementation, the server would wait for verification "
            "completion before returning the protected resource data."
        )
    elif result.action == "decline":
        response = "Access declined - verification required to view this resource."
    else:
        response = "Verification cancelled."

    return ReadResourceResult(
        contents=[
            TextResourceContents(
                mimeType="text/plain",
                uri=AnyUrl("elicitation://url-demo"),
                text=response,
            )
        ]
    )


if __name__ == "__main__":
    logger.info("Starting URL elicitation demo server...")
    mcp.run()
