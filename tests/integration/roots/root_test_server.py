from typing import TYPE_CHECKING

from fastmcp import Context, FastMCP

if TYPE_CHECKING:
    from mcp_types import ListRootsResult

mcp = FastMCP("MCP Root Tester")


@mcp.tool()
async def show_roots(ctx: Context) -> str:
    result: ListRootsResult = await ctx.session.list_roots()
    return result.model_dump_json()


if __name__ == "__main__":
    mcp.run()
