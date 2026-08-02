"""Handle modern URL elicitation with an explicit per-agent consent handler."""

import asyncio
from typing import TYPE_CHECKING, cast

from mcp.client.session import ClientRequestContext
from mcp_types import ElicitRequestParams, ElicitRequestURLParams, ElicitResult
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from fast_agent import FastAgent
from fast_agent.mcp.helpers.content_helpers import get_text

if TYPE_CHECKING:
    from fast_agent.agents.mcp_agent import McpAgent

fast = FastAgent("Modern URL Elicitation Demo", quiet=True)
console = Console()


def read_url_action() -> str:
    try:
        return input("Consent to navigate? [y]es / [n]o / [c]ancel: ")
    except EOFError:
        return "cancel"


async def url_consent_handler(
    context: ClientRequestContext,
    params: ElicitRequestParams,
) -> ElicitResult:
    """Display a URL and wait for an explicit protocol action."""
    del context
    assert isinstance(params, ElicitRequestURLParams)
    console.print(
        Panel(
            Text.assemble(
                (params.message, "bold"),
                "\n\n",
                "Open this URL in your browser:\n",
                (str(params.url), "cyan underline"),
                "\n\nAccepting confirms navigation only, not external completion.",
            ),
            title="URL elicitation",
            border_style="cyan",
        )
    )

    choice = (await asyncio.to_thread(read_url_action)).strip().casefold()
    match choice:
        case "y" | "yes":
            return ElicitResult(action="accept")
        case "n" | "no":
            return ElicitResult(action="decline")
        case _:
            return ElicitResult(action="cancel")


@fast.agent(
    "url_demo",
    servers=["url_server"],
    elicitation_handler=url_consent_handler,
)
async def main() -> None:
    async with fast.run() as agent:
        url_agent = cast("McpAgent", agent.url_demo)
        result = await url_agent.call_tool(
            "request_console_access",
            {"sandbox_id": "sim-t4-small-001"},
        )
        result_text = get_text(result.content[0])
        console.print(Panel(result_text or "No result returned", title="Tool result", expand=False))


if __name__ == "__main__":
    asyncio.run(main())
