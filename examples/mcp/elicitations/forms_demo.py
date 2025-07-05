"""
Quick Start: Elicitation Forms Demo

This example demonstrates the elicitation forms feature with rich console output.
It uses the passthrough model to display forms to the user and shows the results
in an attractive format using the rich library.
"""

import asyncio

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from mcp_agent.core.fastagent import FastAgent
from mcp_agent.mcp.helpers.content_helpers import get_resource_text

# Create the application with quiet mode enabled for cleaner demo output
fast = FastAgent("Elicitation Forms Demo", quiet=True)
console = Console()


@fast.agent(
    "forms-demo",
    servers=[
        "elicitation_forms_mode",
    ],
)
async def main():
    """Run the forms demo with rich output."""
    async with fast.run() as agent:
        console.print("\n[bold cyan]Welcome to the Elicitation Forms Demo![/bold cyan]\n")
        console.print("This demo shows how to collect structured data using MCP elicitations.")
        console.print("We'll present several forms and display the results collected for each.\n")

        # Example 1: User Profile
        console.print("[bold yellow]Example 1: User Profile Form[/bold yellow]")
        result = await agent.get_resource("elicitation://user-profile")

        if result_text := get_resource_text(result):
            panel = Panel(
                result_text, title="Profile Data Received", border_style="green", expand=False
            )
            console.print(panel)
        else:
            console.print("[red]No profile data received[/red]")

        console.print("\n" + "─" * 50 + "\n")

        # Example 2: Preferences
        console.print("[bold yellow]Example 2: User Preferences[/bold yellow]")
        result = await agent.get_resource("elicitation://preferences")

        if result_text := get_resource_text(result):
            # Parse the result to show in a table
            table = Table(title="User Preferences", show_header=True, header_style="bold magenta")
            table.add_column("Setting", style="cyan")
            table.add_column("Value", style="green")

            # Simple parsing of the result text
            for line in result_text.split("\n"):
                if ":" in line and "Preferences set" in line:
                    prefs = line.split(": ", 1)[1]
                    for pref in prefs.split(", "):
                        if "=" in pref:
                            key, value = pref.split("=", 1)
                            table.add_row(key, value)

            if table.row_count > 0:
                console.print(table)
            else:
                console.print(Panel(result_text, border_style="green"))

        console.print("\n" + "─" * 50 + "\n")

        # Example 3: Simple Rating
        console.print("[bold yellow]Example 3: Quick Rating[/bold yellow]")
        result = await agent.get_resource("elicitation://simple-rating")

        if result_text := get_resource_text(result):
            # Create a visual representation of the rating
            if "liked" in result_text:
                rating_display = "⭐⭐⭐⭐⭐ Thank you for the positive feedback!"
                style = "green"
            elif "did not like" in result_text:
                rating_display = "⭐ We'll work on improving!"
                style = "yellow"
            else:
                rating_display = result_text
                style = "blue"

            console.print(
                Panel(rating_display, title="Rating Result", border_style=style, expand=False)
            )

        console.print("\n" + "─" * 50 + "\n")

        # Example 4: Detailed Feedback
        console.print("[bold yellow]Example 4: Detailed Feedback Form[/bold yellow]")
        result = await agent.get_resource("elicitation://feedback")

        if result_text := get_resource_text(result):
            feedback_panel = Panel(
                result_text, title="📝 Feedback Summary", border_style="cyan", expand=False
            )
            console.print(feedback_panel)

        console.print("\n[bold green]✅ Demo Complete![/bold green]")
        console.print("\nThis demo showed how elicitation forms can collect structured data")
        console.print("and present results in an attractive, user-friendly format.")


if __name__ == "__main__":
    # Quiet mode is enabled via the FastAgent constructor for cleaner demo output
    asyncio.run(main())
