from typing import TYPE_CHECKING

from mcp.server.fastmcp import Context, FastMCP

if TYPE_CHECKING:
    from mcp.types import ListRootsResult
mcp = FastMCP("MCP Root Tester", log_level="DEBUG")


@mcp.tool()
async def show_roots(ctx: Context) -> str:
    result: ListRootsResult = await ctx.session.list_roots()
    return result.model_dump_json()


if __name__ == "__main__":
    mcp.run()
