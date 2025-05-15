from mcp.server.fastmcp import Context, FastMCP

mcp = FastMCP("MCP Root Tester", log_level="DEBUG")


@mcp.tool()
async def show_roots(ctx: Context) -> str:
    return await ctx.session.list_roots()


if __name__ == "__main__":
    mcp.run()
