import os

from mcp.server.mcpserver import Context, MCPServer

server = MCPServer("modern-subscriptions")


@server.tool()
async def add_dynamic_tool(ctx: Context) -> str:
    if "dynamic_echo" not in {tool.name for tool in await ctx.mcp_server.list_tools()}:

        @ctx.mcp_server.tool(name="dynamic_echo")
        def dynamic_echo(message: str) -> str:
            return message

    await ctx.notify_tools_changed()
    return "added"


if __name__ == "__main__":
    server.run(
        transport="streamable-http",
        host="127.0.0.1",
        port=int(os.environ["FAST_AGENT_TEST_HTTP_PORT"]),
    )
