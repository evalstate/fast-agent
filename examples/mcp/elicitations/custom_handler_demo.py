"""Override form elicitation with a deterministic development handler."""

import asyncio
from typing import TYPE_CHECKING, cast

from mcp.client.session import ClientRequestContext
from mcp_types import ElicitRequestFormParams, ElicitRequestParams, ElicitResult
from rich.console import Console
from rich.panel import Panel

from fast_agent import FastAgent
from fast_agent.mcp.helpers.content_helpers import get_text

if TYPE_CHECKING:
    from fast_agent.agents.mcp_agent import McpAgent

fast = FastAgent("Custom Elicitation Handler Demo", quiet=True)
console = Console()


async def development_sandbox_handler(
    context: ClientRequestContext,
    params: ElicitRequestParams,
) -> ElicitResult:
    """Supply deterministic form content for a simulated development flow."""
    del context
    assert isinstance(params, ElicitRequestFormParams)
    assert params.message == "Start t4-small sandbox at $0.40 per hour?"
    console.print("[dim]Custom handler selected eu-west-1, two hours, and a $1.00 budget.[/dim]")
    return ElicitResult(
        action="accept",
        content={
            "region": "eu-west-1",
            "duration_hours": 2,
            "max_budget_usd": 1.00,
        },
    )


@fast.agent(
    "sandbox_custom",
    servers=["sandbox_server"],
    elicitation_handler=development_sandbox_handler,
)
async def main() -> None:
    async with fast.run() as agent:
        sandbox = cast("McpAgent", agent.sandbox_custom)
        result = await sandbox.call_tool("start_t4_small_sandbox", {})
        result_text = get_text(result.content[0])
        console.print(Panel(result_text or "No result returned", title="Tool result", expand=False))


if __name__ == "__main__":
    asyncio.run(main())
