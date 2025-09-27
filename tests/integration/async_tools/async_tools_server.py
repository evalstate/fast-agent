import asyncio

from mcp.server.fastmcp import Context, FastMCP

mcp = FastMCP("Async Tools Test Server")


@mcp.tool()
def sync_ping() -> str:
    """Simple synchronous tool to confirm server availability."""
    return "pong"


@mcp.tool(invocation_modes=["async"], keep_alive=5)
async def async_uppercase(message: str, ctx: Context) -> str:  # type: ignore[type-arg]
    """Async tool that uppercases text after a short delay."""
    await ctx.info(f"Starting async processing for: {message}")
    await asyncio.sleep(0.2)
    await ctx.report_progress(0.5, 1.0, "Halfway there")
    await asyncio.sleep(0.2)
    await ctx.report_progress(1.0, 1.0, "Completed async processing")
    return message.upper()


if __name__ == "__main__":
    mcp.run()
