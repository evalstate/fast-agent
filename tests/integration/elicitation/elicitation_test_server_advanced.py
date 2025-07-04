"""
Advanced test server for comprehensive elicitation functionality
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
from mcp.server.fastmcp import Context, FastMCP
from mcp.types import TextResourceContents
from pydantic import AnyUrl, BaseModel, Field
from typing import Optional, List

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("elicitation_server_advanced")

# Create MCP server
mcp = FastMCP("MCP Advanced Elicitation Server", log_level="DEBUG")


@mcp.resource(uri="elicitation://simple-rating")
async def simple_rating() -> ReadResourceResult:
    """Simple boolean rating elicitation"""
    
    class ServerRating(BaseModel):
        rating: bool = Field(description="Do you like this server?")

    result = await mcp.get_context().elicit("Please rate this server", schema=ServerRating)
    
    match result:
        case AcceptedElicitation(data=data):
            response = f"You {'liked' if data.rating else 'did not like'} the server"
        case DeclinedElicitation():
            response = "Rating declined"
        case CancelledElicitation():
            response = "Rating cancelled"

    return ReadResourceResult(
        contents=[
            TextResourceContents(
                mimeType="text/plain", 
                uri=AnyUrl("elicitation://simple-rating"), 
                text=response
            )
        ]
    )


@mcp.resource(uri="elicitation://user-profile")
async def user_profile() -> ReadResourceResult:
    """Complex form with multiple field types"""
    
    class UserProfile(BaseModel):
        name: str = Field(description="Your full name", min_length=2, max_length=50)
        age: int = Field(description="Your age", ge=0, le=150)
        role: str = Field(
            description="Your job role",
            json_schema_extra={
                "enum": ["developer", "designer", "manager", "qa", "other"],
                "enumNames": ["Software Developer", "UI/UX Designer", "Project Manager", "Quality Assurance", "Other"]
            }
        )
        email: Optional[str] = Field(None, description="Your email address (optional)")
        subscribe_newsletter: bool = Field(False, description="Subscribe to our newsletter?")
        
    result = await mcp.get_context().elicit(
        "Please provide your user profile information", 
        schema=UserProfile
    )
    
    match result:
        case AcceptedElicitation(data=data):
            lines = [
                f"Name: {data.name}",
                f"Age: {data.age}",
                f"Role: {data.role.title()}",
                f"Email: {data.email or 'Not provided'}",
                f"Newsletter: {'Yes' if data.subscribe_newsletter else 'No'}"
            ]
            response = "Profile received:\n" + "\n".join(lines)
        case DeclinedElicitation():
            response = "Profile declined"
        case CancelledElicitation():
            response = "Profile cancelled"

    return ReadResourceResult(
        contents=[
            TextResourceContents(
                mimeType="text/plain", 
                uri=AnyUrl("elicitation://user-profile"), 
                text=response
            )
        ]
    )


@mcp.resource(uri="elicitation://preferences")
async def preferences() -> ReadResourceResult:
    """Enum-based preference selection"""
    
    from enum import Enum
    
    class Theme(str, Enum):
        LIGHT = "light"
        DARK = "dark"
        AUTO = "auto"
        
    class Language(str, Enum):
        EN = "en"
        ES = "es"
        FR = "fr"
        DE = "de"
        
    class Preferences(BaseModel):
        theme: Theme = Field(description="Choose your preferred theme")
        language: Language = Field(description="Select your language")
        notifications: bool = Field(True, description="Enable notifications?")
        
    result = await mcp.get_context().elicit(
        "Configure your preferences", 
        schema=Preferences
    )
    
    match result:
        case AcceptedElicitation(data=data):
            response = f"Preferences set: Theme={data.theme.value}, Language={data.language.value}, Notifications={data.notifications}"
        case DeclinedElicitation():
            response = "Preferences declined"
        case CancelledElicitation():
            response = "Preferences cancelled"

    return ReadResourceResult(
        contents=[
            TextResourceContents(
                mimeType="text/plain", 
                uri=AnyUrl("elicitation://preferences"), 
                text=response
            )
        ]
    )


@mcp.resource(uri="elicitation://feedback")
async def feedback() -> ReadResourceResult:
    """Feedback form with number ratings"""
    
    class Feedback(BaseModel):
        overall_rating: int = Field(description="Overall rating (1-5)", ge=1, le=5)
        ease_of_use: float = Field(description="Ease of use (0.0-10.0)", ge=0.0, le=10.0)
        would_recommend: bool = Field(description="Would you recommend to others?")
        comments: Optional[str] = Field(None, description="Additional comments", max_length=500)
        
    result = await mcp.get_context().elicit(
        "We'd love your feedback!", 
        schema=Feedback
    )
    
    match result:
        case AcceptedElicitation(data=data):
            lines = [
                f"Overall: {data.overall_rating}/5",
                f"Ease of use: {data.ease_of_use}/10.0",
                f"Would recommend: {'Yes' if data.would_recommend else 'No'}",
            ]
            if data.comments:
                lines.append(f"Comments: {data.comments}")
            response = "Feedback received:\n" + "\n".join(lines)
        case DeclinedElicitation():
            response = "Feedback declined"
        case CancelledElicitation():
            response = "Feedback cancelled"

    return ReadResourceResult(
        contents=[
            TextResourceContents(
                mimeType="text/plain", 
                uri=AnyUrl("elicitation://feedback"), 
                text=response
            )
        ]
    )


if __name__ == "__main__":
    logger.info("Starting advanced elicitation test server...")
    mcp.run()