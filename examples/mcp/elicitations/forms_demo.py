"""
Quick Start: Elicitation Forms Demo

This example demonstrates the elicitation forms feature of fast-agent.

When Read Resource requests are sent to the MCP Server, it generates an Elicitation
which creates a form for the user to fill out.
The results are returned to the demo program which prints out the results in a rich format.
"""

import asyncio

from rich.console import Console
from rich.panel import Panel

from mcp_agent.core.fastagent import FastAgent
from mcp_agent.mcp.helpers.content_helpers import get_resource_text

fast = FastAgent("Elicitation Forms Demo", quiet=True)
console = Console()


@fast.agent(
    "forms-demo",
    servers=[
        "elicitation_forms_server",
    ],
)
async def main():
    """Run the improved forms demo showcasing all elicitation features."""
    async with fast.run() as agent:
        console.print("\n[bold cyan]Welcome to the Elicitation Forms Demo![/bold cyan]\n")
        console.print("This demo shows how to collect structured data using MCP Elicitations.")
        console.print("We'll present several forms and display the results collected for each.\n")

        # Example 1: Event Registration
        console.print("[bold yellow]Example 1: Event Registration Form[/bold yellow]")
        console.print(
            "[dim]Demonstrates: string validation, email format, URL format, date format[/dim]"
        )
        result = await agent.get_resource("elicitation://event-registration")

        if result_text := get_resource_text(result):
            panel = Panel(
                result_text,
                title="🎫 Registration Confirmation",
                border_style="green",
                expand=False,
            )
            console.print(panel)
        else:
            console.print("[red]No registration data received[/red]")

        # Example 2: Product Review
        console.print("[bold yellow]Example 2: Product Review Form[/bold yellow]")
        console.print(
            "[dim]Demonstrates: number validation (range), radio selection, multiline text[/dim]"
        )
        result = await agent.get_resource("elicitation://product-review")

        if result_text := get_resource_text(result):
            review_panel = Panel(
                result_text, title="🛍️ Product Review", border_style="cyan", expand=False
            )
            console.print(review_panel)

        # Example 3: Account Settings
        console.print("[bold yellow]Example 3: Account Settings Form[/bold yellow]")
        console.print(
            "[dim]Demonstrates: boolean selections, radio selection, number validation[/dim]"
        )
        result = await agent.get_resource("elicitation://account-settings")

        if result_text := get_resource_text(result):
            settings_panel = Panel(
                result_text, title="⚙️ Account Settings", border_style="blue", expand=False
            )
            console.print(settings_panel)

        # Example 4: Service Appointment
        console.print("[bold yellow]Example 4: Service Appointment Booking[/bold yellow]")
        console.print(
            "[dim]Demonstrates: string validation, radio selection, boolean, datetime format[/dim]"
        )
        result = await agent.get_resource("elicitation://service-appointment")

        if result_text := get_resource_text(result):
            appointment_panel = Panel(
                result_text, title="🔧 Appointment Confirmed", border_style="magenta", expand=False
            )
            console.print(appointment_panel)

        console.print("\n[bold green]✅ Demo Complete![/bold green]")
        console.print("\n[bold cyan]Features Demonstrated:[/bold cyan]")
        console.print("• [green]String validation[/green] (min/max length)")
        console.print("• [green]Number validation[/green] (range constraints)")
        console.print("• [green]Radio selections[/green] (enum dropdowns)")
        console.print("• [green]Boolean selections[/green] (checkboxes)")
        console.print("• [green]Format validation[/green] (email, URL, date, datetime)")
        console.print("• [green]Multiline text[/green] (expandable text areas)")
        console.print("\nThese forms demonstrate natural, user-friendly data collection patterns!")


if __name__ == "__main__":
    asyncio.run(main())
