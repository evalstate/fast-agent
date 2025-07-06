"""
MCP Server for Account Creation Demo

This server provides an account signup form that can be triggered
by tools, demonstrating LLM-initiated elicitations.

Note: Following MCP spec, we don't collect sensitive information like passwords.
"""

import logging
import sys

from mcp.server.elicitation import (
    AcceptedElicitation,
    CancelledElicitation,
    DeclinedElicitation,
)
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("elicitation_account_server")

# Create MCP server
mcp = FastMCP("Account Creation Server", log_level="INFO")


@mcp.tool()
async def create_user_account(service_name: str = "MyApp") -> str:
    """
    Create a new user account for the specified service.

    Args:
        service_name: The name of the service to create an account for

    Returns:
        Status message about the account creation
    """
    # This tool triggers the elicitation form
    logger.info(f"Creating account for service: {service_name}")

    class AccountSignup(BaseModel):
        username: str = Field(description="Choose a username", min_length=3, max_length=20)
        email: str = Field(description="Your email address", json_schema_extra={"format": "email"})
        full_name: str = Field(description="Your full name", max_length=30)

        language: str = Field(
            default="en",
            description="Preferred language",
            json_schema_extra={
                "enum": [
                    "en",
                    "zh",
                    "es",
                    "fr",
                    "de",
                    "ja",
                ],
                "enumNames": ["English", "中文", "Español", "Français", "Deutsch", "日本語"],
            },
        )
        agree_terms: bool = Field(description="I agree to the terms of service")
        marketing_emails: bool = Field(False, description="Send me product updates")

    result = await mcp.get_context().elicit(
        f"Create Your {service_name} Account", schema=AccountSignup
    )

    match result:
        case AcceptedElicitation(data=data):
            if not data.agree_terms:
                return "❌ Account creation failed: You must agree to the terms of service"
            else:
                return f"✅ Account created successfully for {service_name}!\nUsername: {data.username}\nEmail: {data.email}"
        case DeclinedElicitation():
            return f"❌ Account creation for {service_name} was declined by user"
        case CancelledElicitation():
            return f"❌ Account creation for {service_name} was cancelled by user"


if __name__ == "__main__":
    logger.info("Starting account creation server...")
    mcp.run()
