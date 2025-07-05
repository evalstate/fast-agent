"""
Custom Elicitation Handler for Game Character Creation

This module provides a whimsical custom elicitation handler that creates
an interactive game character creation experience with dice rolls,
visual gauges, and animated effects.
"""

import asyncio
import random
from typing import TYPE_CHECKING, Any, Dict

from mcp.shared.context import RequestContext
from mcp.types import ElicitRequestParams, ElicitResult
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn
from rich.table import Table

from mcp_agent.logging.logger import get_logger

if TYPE_CHECKING:
    from mcp import ClientSession

logger = get_logger(__name__)
console = Console()


async def game_character_elicitation_handler(
    context: RequestContext["ClientSession", Any],
    params: ElicitRequestParams,
) -> ElicitResult:
    """Custom handler that creates an interactive character creation experience."""
    logger.info(f"Game character elicitation handler called: {params.message}")

    if params.requestedSchema:
        properties = params.requestedSchema.get("properties", {})
        content: Dict[str, Any] = {}

        console.print("\n[bold magenta]🎮 Character Creation Studio 🎮[/bold magenta]\n")

        # Character name with typewriter effect
        if "character_name" in properties:
            console.print("[cyan]✨ Generating your character's name...[/cyan] ", end="")
            name_prefixes = ["Hero", "Legend", "Epic", "Mighty", "Brave", "Noble"]
            name_suffixes = ["blade", "heart", "storm", "fire", "shadow", "star"]

            name = f"{random.choice(name_prefixes)}{random.choice(name_suffixes)}{random.randint(1, 999)}"

            for char in name:
                console.print(char, end="", style="bold green")
                await asyncio.sleep(0.03)
            console.print("\n")
            content["character_name"] = name

        # Class selection with visual menu and fate dice
        if "character_class" in properties:
            class_enum = properties["character_class"].get("enum", [])
            class_names = properties["character_class"].get("enumNames", class_enum)

            table = Table(title="🎯 Choose Your Destiny", show_header=False, box=None)
            table.add_column("Option", style="cyan", width=8)
            table.add_column("Class", style="yellow", width=20)
            table.add_column("Description", style="dim", width=30)

            descriptions = [
                "Master of sword and shield",
                "Wielder of arcane mysteries",
                "Silent shadow striker",
                "Nature's deadly archer",
                "Holy warrior of light",
                "Inspiring magical performer",
            ]

            for i, (cls, name, desc) in enumerate(zip(class_enum, class_names, descriptions)):
                table.add_row(f"[{i + 1}]", name, desc)

            console.print(table)

            # Dramatic fate dice roll
            console.print("\n[bold yellow]🎲 The Fates decide your path...[/bold yellow]")
            for _ in range(8):
                dice_face = random.choice(["⚀", "⚁", "⚂", "⚃", "⚄", "⚅"])
                console.print(f"\r  Rolling... {dice_face}", end="")
                await asyncio.sleep(0.2)

            fate_roll = random.randint(1, 6)
            selected_idx = (fate_roll - 1) % len(class_enum)
            console.print(f"\n  🎲 Fate dice: [bold red]{fate_roll}[/bold red]!")
            console.print(
                f"✨ Destiny has chosen: [bold yellow]{class_names[selected_idx]}[/bold yellow]!\n"
            )
            content["character_class"] = class_enum[selected_idx]

        # Stats rolling with animated progress bars and cosmic effects
        stat_names = ["strength", "intelligence", "dexterity", "charisma"]
        stats_info = {
            "strength": {"emoji": "💪", "desc": "Physical power"},
            "intelligence": {"emoji": "🧠", "desc": "Mental acuity"},
            "dexterity": {"emoji": "🏃", "desc": "Agility & speed"},
            "charisma": {"emoji": "✨", "desc": "Personal magnetism"},
        }

        console.print("[bold]🌟 Rolling cosmic dice for your abilities...[/bold]\n")

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=25, style="cyan", complete_style="green"),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
        ) as progress:
            for stat in stat_names:
                if stat in properties:
                    # Roll 3d6 for classic D&D feel with bonus potential
                    rolls = [random.randint(1, 6) for _ in range(3)]
                    total = sum(rolls)

                    # Add cosmic bonus chance
                    if random.random() < 0.15:  # 15% chance for cosmic boost
                        cosmic_bonus = random.randint(1, 3)
                        total = min(18, total + cosmic_bonus)
                        cosmic_text = f" ✨+{cosmic_bonus} COSMIC✨"
                    else:
                        cosmic_text = ""

                    stat_info = stats_info.get(stat, {"emoji": "📊", "desc": stat.title()})
                    task = progress.add_task(
                        f"{stat_info['emoji']} {stat.capitalize()}: {stat_info['desc']}", total=18
                    )

                    # Animate the progress bar with suspense
                    for i in range(total + 1):
                        progress.update(task, completed=i)
                        await asyncio.sleep(0.04)

                    content[stat] = total
                    console.print(
                        f"   🎲 Rolled: {rolls} = [bold green]{total}[/bold green]{cosmic_text}"
                    )

        # Lucky dice legendary challenge
        if "lucky_dice" in properties:
            console.print("\n" + "=" * 60)
            console.print("[bold yellow]🎰 LEGENDARY CHALLENGE: Lucky Dice! 🎰[/bold yellow]")
            console.print("The ancient dice of fortune whisper your name...")
            console.print("Do you dare tempt fate for legendary power?")
            console.print("=" * 60)

            # Epic dice rolling sequence
            console.print("\n[cyan]🌟 Rolling the Dice of Destiny...[/cyan]")

            for i in range(15):
                dice_faces = ["⚀", "⚁", "⚂", "⚃", "⚄", "⚅"]
                d20_faces = ["🎲"] * 19 + ["💎"]  # Special diamond for 20

                if i < 10:
                    face = random.choice(dice_faces)
                else:
                    face = random.choice(d20_faces)

                console.print(f"\r  [bold]{face}[/bold] Rolling...", end="")
                await asyncio.sleep(0.15)

            final_roll = random.randint(1, 20)

            if final_roll == 20:
                console.print("\r  [bold red]💎 NATURAL 20! 💎[/bold red]")
                console.print("  [bold green]🌟 LEGENDARY SUCCESS! 🌟[/bold green]")
                console.print("  [gold1]You have been blessed by the gods themselves![/gold1]")
                bonus_text = "🏆 Divine Champion status unlocked!"
            elif final_roll >= 18:
                console.print(f"\r  [bold yellow]⭐ {final_roll} - EPIC ROLL! ⭐[/bold yellow]")
                bonus_text = "🎁 Epic treasure discovered!"
            elif final_roll >= 15:
                console.print(f"\r  [green]🎲 {final_roll} - Great success![/green]")
                bonus_text = "🌟 Rare magical item found!"
            elif final_roll >= 10:
                console.print(f"\r  [yellow]🎲 {final_roll} - Good fortune.[/yellow]")
                bonus_text = "🗡️ Modest blessing received."
            elif final_roll == 1:
                console.print("\r  [bold red]💀 CRITICAL FUMBLE! 💀[/bold red]")
                bonus_text = "😅 Learning experience gained... try again!"
            else:
                console.print(f"\r  [dim]🎲 {final_roll} - The dice are silent.[/dim]")
                bonus_text = "🎯 Your destiny remains unwritten."

            console.print(f"  [italic]{bonus_text}[/italic]")
            content["lucky_dice"] = final_roll >= 10

        # Epic character summary with theatrical flair
        console.print("\n" + "=" * 70)
        console.print("[bold cyan]📜 Your Legend Begins! 📜[/bold cyan]")
        console.print("=" * 70)

        # Add some flourish based on the character's total power
        total_stats = sum(content.get(stat, 10) for stat in stat_names if stat in content)
        if total_stats > 60:
            console.print("✨ [bold gold1]The realm trembles before your might![/bold gold1] ✨")
        elif total_stats > 50:
            console.print("⚔️ [bold green]A formidable hero rises![/bold green] ⚔️")
        elif total_stats < 35:
            console.print("🎯 [bold blue]The underdog's tale begins![/bold blue] 🎯")
        else:
            console.print("🗡️ [bold white]Adventure awaits the worthy![/bold white] 🗡️")

        console.print()

        return ElicitResult(action="accept", content=content)
    else:
        # No schema, return a fun message
        content = {"response": "⚔️ Ready for adventure! ⚔️"}
        return ElicitResult(action="accept", content=content)
