from mcp.server.fastmcp import FastMCP, Context

mcp = FastMCP("MCP Root Tester", log_level="DEBUG")


@mcp.resource(uri="resource://fast-agent/size/{thing}")
async def get_size(thing: str, ctx: Context) -> str:
    # Now you can use ctx here
    return thing


if __name__ == "__main__":
    mcp.run()
