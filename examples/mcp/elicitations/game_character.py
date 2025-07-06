#!/usr/bin/env python3
"""
Demonstration of Custom Elicitation Handler

This example demonstrates a custom elicitation handler that creates
an interactive game character creation experience with dice rolls,
visual gauges, and fun interactions.
"""

import asyncio

# Import our custom handler from the separate module
from game_character_handler import game_character_elicitation_handler
from rich.console import Console
from rich.panel import Panel

from mcp_agent.core.fastagent import FastAgent
from mcp_agent.mcp.helpers.content_helpers import get_resource_text

fast = FastAgent("Game Character Creator", quiet=True)
console = Console()


@fast.agent(
    "character-creator",
    servers=["elicitation_game_server"],
    # Register our handler from game_character_handler.py
    elicitation_handler=game_character_elicitation_handler,
)
async def main():
    """Run the game character creator with custom elicitation handler."""
    async with fast.run() as agent:
        console.print(
            Panel(
                "[bold cyan]Welcome to the Character Creation Studio![/bold cyan]\n\n"
                "Create your hero with our magical character generator.\n"
                "Watch as the cosmic dice determine your fate!",
                title="🎮 Game Time 🎮",
                border_style="magenta",
            )
        )

        # Trigger the character creation
        result = await agent.get_resource("elicitation://game-character")

        if result_text := get_resource_text(result):
            character_panel = Panel(
                result_text, title="📜 Your Character 📜", border_style="green", expand=False
            )
            console.print(character_panel)

            console.print("\n[italic]Your character is ready for adventure![/italic]")
            console.print("[dim]The tavern door opens, and your journey begins...[/dim]\n")

            # Fun ending based on character
            if "Powerful character" in result_text:
                console.print("⚔️  [bold]The realm trembles at your might![/bold]")
            elif "Challenging build" in result_text:
                console.print("🎯 [bold]True heroes are forged through adversity![/bold]")
            else:
                console.print("🗡️  [bold]Your legend begins now![/bold]")


if __name__ == "__main__":
    asyncio.run(main())
