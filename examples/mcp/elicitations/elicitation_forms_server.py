"""
MCP Server for Elicitation Examples

This server provides various elicitation resources that demonstrate
different form types and data collection patterns.
"""

import logging
import sys
from typing import Optional

from mcp import ReadResourceResult
from mcp.server.elicitation import (
    AcceptedElicitation,
    CancelledElicitation,
    DeclinedElicitation,
)
from mcp.server.fastmcp import FastMCP
from mcp.types import TextResourceContents
from pydantic import AnyUrl, BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("elicitation_server")

# Create MCP server
mcp = FastMCP("Elicitation Examples Server", log_level="INFO")


@mcp.resource(uri="elicitation://user-profile")
async def user_profile() -> ReadResourceResult:
    """Collect comprehensive user profile information."""

    class UserProfile(BaseModel):
        name: str = Field(description="Your full name", min_length=2, max_length=50)
        age: int = Field(description="Your age", ge=13, le=120)
        role: str = Field(
            description="Your job role",
            json_schema_extra={
                "enum": ["developer", "designer", "manager", "qa", "student", "other"],
                "enumNames": [
                    "Software Developer",
                    "UI/UX Designer",
                    "Project Manager",
                    "Quality Assurance",
                    "Student",
                    "Other",
                ],
            },
        )
        email: Optional[str] = Field(None, description="Your email address (optional)")
        subscribe_newsletter: bool = Field(False, description="Subscribe to our newsletter?")

    result = await mcp.get_context().elicit(
        "Please provide your user profile information", schema=UserProfile
    )

    match result:
        case AcceptedElicitation(data=data):
            lines = [
                f"Name: {data.name}",
                f"Age: {data.age}",
                f"Role: {data.role.replace('_', ' ').title()}",
                f"Email: {data.email or 'Not provided'}",
                f"Newsletter: {'Yes' if data.subscribe_newsletter else 'No'}",
            ]
            response = "Profile received:\n" + "\n".join(lines)
        case DeclinedElicitation():
            response = "Profile declined - no data collected"
        case CancelledElicitation():
            response = "Profile cancelled - operation aborted"

    return ReadResourceResult(
        contents=[
            TextResourceContents(
                mimeType="text/plain", uri=AnyUrl("elicitation://user-profile"), text=response
            )
        ]
    )


@mcp.resource(uri="elicitation://preferences")
async def preferences() -> ReadResourceResult:
    """Collect user preferences for application settings."""

    class Preferences(BaseModel):
        theme: str = Field(
            description="Choose your preferred theme",
            json_schema_extra={
                "enum": ["light", "dark", "auto"],
                "enumNames": ["Light Theme", "Dark Theme", "Auto (System)"],
            },
        )
        language: str = Field(
            description="Select your language",
            json_schema_extra={
                "enum": ["en", "es", "fr", "de", "ja", "zh"],
                "enumNames": ["English", "Español", "Français", "Deutsch", "日本語", "中文"],
            },
        )
        notifications: bool = Field(True, description="Enable notifications?")
        font_size: str = Field(
            default="medium",
            description="Preferred font size",
            json_schema_extra={
                "enum": ["small", "medium", "large", "xlarge"],
                "enumNames": ["Small", "Medium", "Large", "Extra Large"],
            },
        )

    result = await mcp.get_context().elicit(
        "Configure your application preferences", schema=Preferences
    )

    match result:
        case AcceptedElicitation(data=data):
            response = f"Preferences set: Theme={data.theme}, Language={data.language}, Notifications={data.notifications}, Font={data.font_size}"
        case DeclinedElicitation():
            response = "Preferences declined - using defaults"
        case CancelledElicitation():
            response = "Preferences cancelled - no changes made"

    return ReadResourceResult(
        contents=[
            TextResourceContents(
                mimeType="text/plain", uri=AnyUrl("elicitation://preferences"), text=response
            )
        ]
    )


@mcp.resource(uri="elicitation://simple-rating")
async def simple_rating() -> ReadResourceResult:
    """Simple yes/no rating question."""

    class ServerRating(BaseModel):
        rating: bool = Field(description="Do you like this elicitation demo?")

    result = await mcp.get_context().elicit("Quick question!", schema=ServerRating)

    match result:
        case AcceptedElicitation(data=data):
            response = f"You {'liked' if data.rating else 'did not like'} the demo"
        case DeclinedElicitation():
            response = "Rating declined"
        case CancelledElicitation():
            response = "Rating cancelled"

    return ReadResourceResult(
        contents=[
            TextResourceContents(
                mimeType="text/plain", uri=AnyUrl("elicitation://simple-rating"), text=response
            )
        ]
    )


@mcp.resource(uri="elicitation://feedback")
async def feedback() -> ReadResourceResult:
    """Detailed feedback form with ratings and comments."""

    class Feedback(BaseModel):
        overall_rating: int = Field(description="Overall rating (1-5 stars)", ge=1, le=5)
        ease_of_use: float = Field(description="Ease of use (0.0-10.0)", ge=0.0, le=10.0)
        would_recommend: bool = Field(description="Would you recommend this to others?")
        favorite_feature: str = Field(
            description="What was your favorite feature?",
            json_schema_extra={
                "enum": ["forms", "validation", "ui", "documentation", "examples"],
                "enumNames": [
                    "Form Design",
                    "Input Validation",
                    "User Interface",
                    "Documentation",
                    "Example Code",
                ],
            },
        )
        comments: Optional[str] = Field(
            None, description="Additional comments (optional)", max_length=500
        )

    result = await mcp.get_context().elicit(
        "We'd love your feedback on the elicitation system!", schema=Feedback
    )

    match result:
        case AcceptedElicitation(data=data):
            lines = [
                f"Overall: {'⭐' * data.overall_rating} ({data.overall_rating}/5)",
                f"Ease of use: {data.ease_of_use}/10.0",
                f"Would recommend: {'Yes' if data.would_recommend else 'No'}",
                f"Favorite feature: {data.favorite_feature.replace('_', ' ').title()}",
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
                mimeType="text/plain", uri=AnyUrl("elicitation://feedback"), text=response
            )
        ]
    )


@mcp.resource(uri="elicitation://account-signup")
async def account_signup() -> ReadResourceResult:
    """Account creation form for the real LLM example."""

    class AccountSignup(BaseModel):
        username: str = Field(
            description="Choose a username", min_length=3, max_length=20, pattern="^[a-zA-Z0-9_]+$"
        )
        password: str = Field(description="Create a password (min 8 characters)", min_length=8)
        confirm_password: str = Field(description="Confirm your password", min_length=8)
        email: str = Field(
            description="Your email address",
            pattern=r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
        )
        full_name: str = Field(description="Your full name")
        agree_terms: bool = Field(description="I agree to the terms of service")
        marketing_emails: bool = Field(False, description="Send me marketing emails")

    result = await mcp.get_context().elicit("Create Your Account", schema=AccountSignup)

    match result:
        case AcceptedElicitation(data=data):
            if data.password != data.confirm_password:
                response = "❌ Account creation failed: Passwords do not match"
            elif not data.agree_terms:
                response = "❌ Account creation failed: You must agree to the terms of service"
            else:
                lines = [
                    "✅ Account created successfully!",
                    f"Username: {data.username}",
                    f"Email: {data.email}",
                    f"Name: {data.full_name}",
                    f"Marketing emails: {'Enabled' if data.marketing_emails else 'Disabled'}",
                ]
                response = "\n".join(lines)
        case DeclinedElicitation():
            response = "❌ Account creation was declined by user"
        case CancelledElicitation():
            response = "❌ Account creation was cancelled by user"

    return ReadResourceResult(
        contents=[
            TextResourceContents(
                mimeType="text/plain", uri=AnyUrl("elicitation://account-signup"), text=response
            )
        ]
    )


@mcp.resource(uri="elicitation://game-character")
async def game_character() -> ReadResourceResult:
    """Fun game character creation form for the whimsical example."""

    class GameCharacter(BaseModel):
        character_name: str = Field(description="Name your character", min_length=2, max_length=30)
        character_class: str = Field(
            description="Choose your class",
            json_schema_extra={
                "enum": ["warrior", "mage", "rogue", "ranger", "paladin", "bard"],
                "enumNames": [
                    "⚔️ Warrior",
                    "🔮 Mage",
                    "🗡️ Rogue",
                    "🏹 Ranger",
                    "🛡️ Paladin",
                    "🎵 Bard",
                ],
            },
        )
        strength: int = Field(description="Strength (3-18)", ge=3, le=18, default=10)
        intelligence: int = Field(description="Intelligence (3-18)", ge=3, le=18, default=10)
        dexterity: int = Field(description="Dexterity (3-18)", ge=3, le=18, default=10)
        charisma: int = Field(description="Charisma (3-18)", ge=3, le=18, default=10)
        lucky_dice: bool = Field(False, description="Roll for a lucky bonus?")

    result = await mcp.get_context().elicit("🎮 Create Your Game Character!", schema=GameCharacter)

    match result:
        case AcceptedElicitation(data=data):
            import random

            lines = [
                f"🎭 Character Created: {data.character_name}",
                f"Class: {data.character_class.title()}",
                f"Stats: STR:{data.strength} INT:{data.intelligence} DEX:{data.dexterity} CHA:{data.charisma}",
            ]

            if data.lucky_dice:
                dice_roll = random.randint(1, 20)
                if dice_roll >= 15:
                    bonus = random.choice(
                        [
                            "🎁 Lucky! +2 to all stats!",
                            "🌟 Critical! Found a magic item!",
                            "💰 Jackpot! +100 gold!",
                        ]
                    )
                    lines.append(f"🎲 Dice Roll: {dice_roll} - {bonus}")
                else:
                    lines.append(f"🎲 Dice Roll: {dice_roll} - No bonus this time!")

            total_stats = data.strength + data.intelligence + data.dexterity + data.charisma
            if total_stats > 50:
                lines.append("💪 Powerful character build!")
            elif total_stats < 30:
                lines.append("🎯 Challenging build - good luck!")

            response = "\n".join(lines)
        case DeclinedElicitation():
            response = "Character creation declined - returning to menu"
        case CancelledElicitation():
            response = "Character creation cancelled"

    return ReadResourceResult(
        contents=[
            TextResourceContents(
                mimeType="text/plain", uri=AnyUrl("elicitation://game-character"), text=response
            )
        ]
    )


if __name__ == "__main__":
    logger.info("Starting elicitation examples server...")
    mcp.run()
