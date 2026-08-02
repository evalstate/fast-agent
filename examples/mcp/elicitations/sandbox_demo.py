"""Start a simulated t4-small sandbox using request-scoped MCP elicitation."""

import asyncio
from typing import TYPE_CHECKING, cast

from rich.console import Console
from rich.panel import Panel

from fast_agent import FastAgent
from fast_agent.mcp.helpers.content_helpers import get_text

if TYPE_CHECKING:
    from fast_agent.agents.mcp_agent import McpAgent

fast = FastAgent("MCP Elicitation Quickstart", quiet=True)
console = Console()


@fast.agent(
    "sandbox",
    servers=["sandbox_server"],
)
async def main() -> None:
    async with fast.run() as agent:
        console.print("\n[bold cyan]MCP Elicitation Quickstart[/bold cyan]\n")
        console.print("[bold]Request:[/bold] Start t4-small sandbox at $0.40 per hour?")
        console.print(
            "[dim]The active tool request pauses for the region, duration, and maximum budget. "
            "Accept, decline, or cancel the elicitation to continue.[/dim]\n"
        )

        sandbox = cast("McpAgent", agent.sandbox)
        result = await sandbox.call_tool("start_t4_small_sandbox", {})
        result_text = get_text(result.content[0])
        console.print(
            Panel(
                result_text or "No result returned",
                title="Provisioning result",
                border_style="green",
                expand=False,
            )
        )


if __name__ == "__main__":
    asyncio.run(main())
