"""
Quick Start: URL Elicitation Demo

This example demonstrates URL mode elicitation, where MCP servers direct
users to external URLs for sensitive interactions that should stay outside
the MCP client/LLM context.

Use cases for URL elicitation:
- OAuth authorization flows
- Payment processing
- Credential entry (API keys, passwords)
- Identity verification
- Any interaction where sensitive data should not pass through the LLM

When URL elicitation requests are made, the client displays the URL inline
with clear context about which MCP server is making the request and why.
"""

import asyncio

from rich.console import Console
from rich.panel import Panel

from fast_agent import FastAgent
from fast_agent.mcp.helpers.content_helpers import get_resource_text

fast = FastAgent("URL Elicitation Demo", quiet=True)
console = Console()


@fast.agent(
    "url-demo",
    servers=[
        "url_elicitation_server",
    ],
)
async def main():
    """Run the URL elicitation demo showcasing out-of-band interactions."""
    async with fast.run() as agent:
        console.print("\n[bold cyan]Welcome to the URL Elicitation Demo![/bold cyan]\n")
        console.print(
            "This demo shows how MCP servers can direct users to external URLs "
            "for sensitive interactions."
        )
        console.print(
            "Unlike form elicitation, URL elicitation keeps sensitive data "
            "[bold]outside[/bold] the LLM context.\n"
        )

        # Example 1: OAuth-like authorization via tool call
        console.print("[bold yellow]Example 1: OAuth Authorization Flow[/bold yellow]")
        console.print("[dim]The server will request authorization for an external service.[/dim]\n")
        result = await agent.send('***CALL_TOOL authorize_api_access {"service_name": "GitHub"}')
        panel = Panel(
            result,
            title="Authorization Result",
            border_style="green",
            expand=False,
        )
        console.print(panel)

        console.print()

        # Example 2: Secure credential entry via tool call
        console.print("[bold yellow]Example 2: Secure Credential Entry[/bold yellow]")
        console.print(
            "[dim]The server will request API key entry via a secure external page.[/dim]\n"
        )
        result = await agent.send('***CALL_TOOL enter_api_key {"api_name": "OpenAI"}')
        panel = Panel(
            result,
            title="Credential Entry Result",
            border_style="cyan",
            expand=False,
        )
        console.print(panel)

        console.print()

        # Example 3: Payment flow via tool call
        console.print("[bold yellow]Example 3: Payment Processing[/bold yellow]")
        console.print("[dim]The server will initiate a payment flow via external checkout.[/dim]\n")
        result = await agent.send(
            '***CALL_TOOL initiate_payment {"amount": 29.99, "currency": "USD", '
            '"description": "Pro Plan Subscription"}'
        )
        panel = Panel(
            result,
            title="Payment Result",
            border_style="magenta",
            expand=False,
        )
        console.print(panel)

        console.print()

        # Example 4: Resource with URL elicitation
        console.print("[bold yellow]Example 4: Protected Resource Access[/bold yellow]")
        console.print("[dim]The resource requires identity verification via external URL.[/dim]\n")
        resource_result = await agent.get_resource("elicitation://url-demo")
        if result_text := get_resource_text(resource_result):
            panel = Panel(
                result_text,
                title="Protected Resource",
                border_style="blue",
                expand=False,
            )
            console.print(panel)

        console.print("\n[bold green]Demo Complete![/bold green]")
        console.print("\n[bold cyan]Key Takeaways:[/bold cyan]")
        console.print(
            "- [green]URL elicitation[/green] directs users to external URLs for sensitive interactions"
        )
        console.print(
            "- [green]Security[/green]: Sensitive data (credentials, payments) stays outside LLM context"
        )
        console.print(
            "- [green]Clear messaging[/green]: Users see which MCP server is requesting navigation"
        )
        console.print(
            "- [green]Domain visibility[/green]: The target domain is highlighted for security awareness"
        )
        console.print(
            "\nUse URL elicitation for OAuth, payments, credential entry, and identity verification!"
        )


if __name__ == "__main__":
    asyncio.run(main())
