"""
Account Creation with LLM Integration

This example demonstrates using elicitations with a real LLM to handle
an account creation workflow. The LLM will guide the user through the
process and handle the form data automatically.
"""

import asyncio

from rich.console import Console
from rich.panel import Panel

from mcp_agent.core.fastagent import FastAgent

# Create the application
fast = FastAgent("Account Creation Assistant")
console = Console()


@fast.agent(
    "account-assistant",
    # Uses the forms mode so user fills out the form when LLM requests it
    servers=["elicitation_forms_mode"],
    # This example requires a real LLM model
    # You can override with: model="gpt-4o" or another model
)
async def main():
    """LLM-powered account creation assistant."""
    async with fast.run() as agent:
        console.print("\n[bold cyan]🤖 AI Account Creation Assistant[/bold cyan]\n")

        # Initial greeting and instruction to LLM
        await agent.send(
            "Hello! I'd like to create a new user account. "
            "Please help me start the account creation process using the available tools."
        )

        console.print("\n[dim]The AI assistant will now attempt to create an account...[/dim]\n")


if __name__ == "__main__":
    console.print(
        Panel(
            "[yellow]Note:[/yellow] This example requires a real LLM model to be configured.\n"
            "Edit fastagent.config.yaml and set 'default_model' to a valid model like 'gpt-4o'.\n"
            "Or run with: [cyan]uv run account_creation.py --model gpt-4o[/cyan]",
            title="Configuration Required",
            border_style="yellow",
        )
    )
    asyncio.run(main())
