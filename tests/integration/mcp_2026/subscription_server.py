import os

from mcp.server.mcpserver import Context, MCPServer


class CachingMCPServer(MCPServer):
    list_tools_calls = 0

    async def _handle_list_tools(self, ctx, params):  # noqa: ANN001, ANN201
        self.list_tools_calls += 1
        result = await super()._handle_list_tools(ctx, params)
        result.ttl_ms = 60_000
        result.cache_scope = "private"
        return result


server = CachingMCPServer("modern-subscriptions")


@server.tool()
async def add_dynamic_tool(ctx: Context) -> str:
    if "dynamic_echo" not in {tool.name for tool in await ctx.mcp_server.list_tools()}:

        @ctx.mcp_server.tool(name="dynamic_echo")
        def dynamic_echo(message: str) -> str:
            return message

    await ctx.notify_tools_changed()
    return "added"


@server.tool()
def list_tools_call_count() -> int:
    return server.list_tools_calls


if __name__ == "__main__":
    server.run(
        transport="streamable-http",
        host="127.0.0.1",
        port=int(os.environ["FAST_AGENT_TEST_HTTP_PORT"]),
    )
