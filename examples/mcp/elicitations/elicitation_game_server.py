"""
MCP Server for Game Character Creation

This server provides a fun game character creation form
that can be used with custom handlers.
"""

import logging
import random
import sys

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
logger = logging.getLogger("elicitation_game_server")

# Create MCP server
mcp = FastMCP("Game Character Creation Server", log_level="INFO")


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


@mcp.tool()
async def roll_new_character(campaign_name: str = "Adventure") -> str:
    """
    Roll a new character for your campaign.

    Args:
        campaign_name: The name of the campaign

    Returns:
        Character details or status message
    """

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

    result = await mcp.get_context().elicit(
        f"🎮 Create Character for {campaign_name}!", schema=GameCharacter
    )

    match result:
        case AcceptedElicitation(data=data):
            response = f"🎭 {data.character_name} the {data.character_class.title()} joins {campaign_name}!\n"
            response += f"Stats: STR:{data.strength} INT:{data.intelligence} DEX:{data.dexterity} CHA:{data.charisma}"

            if data.lucky_dice:
                dice_roll = random.randint(1, 20)
                if dice_roll >= 15:
                    response += f"\n🎲 Lucky roll ({dice_roll})! Starting with bonus equipment!"
                else:
                    response += f"\n🎲 Rolled {dice_roll} - Standard starting gear."

            return response
        case DeclinedElicitation():
            return f"Character creation for {campaign_name} was declined"
        case CancelledElicitation():
            return f"Character creation for {campaign_name} was cancelled"


if __name__ == "__main__":
    logger.info("Starting game character creation server...")
    mcp.run()
