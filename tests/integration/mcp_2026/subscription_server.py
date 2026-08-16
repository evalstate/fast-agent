import os

from mcp.server.mcpserver import Context, MCPServer


class CachingMCPServer(MCPServer):
    list_tools_calls = 0
    read_resource_calls = 0

    async def _handle_list_tools(self, ctx, params):
        self.list_tools_calls += 1
        result = await super()._handle_list_tools(ctx, params)
        result.ttl_ms = 60_000
        result.cache_scope = "private"
        return result

    async def _handle_read_resource(self, ctx, params):
        self.read_resource_calls += 1
        return await super()._handle_read_resource(ctx, params)


server = CachingMCPServer("modern-subscriptions")
resource_version = 1


@server.resource(
    "ui://component/initial",
    name="Initial app",
    mime_type="text/html;profile=mcp-app",
)
def initial_app_resource() -> str:
    return "<html>initial</html>"


@server.tool(
    name="initial_app",
    meta={"ui": {"resourceUri": "ui://component/initial"}},
)
def initial_app() -> str:
    return "rendered"


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


@server.tool()
async def add_dynamic_app(ctx: Context) -> str:
    if "ui://component/dynamic" not in {
        str(resource.uri) for resource in await ctx.mcp_server.list_resources()
    }:

        @ctx.mcp_server.resource(
            "ui://component/dynamic",
            name="Dynamic app",
            mime_type="text/html;profile=mcp-app",
        )
        def dynamic_app_resource() -> str:
            return f"<html>version {resource_version}</html>"

        @ctx.mcp_server.tool(
            name="dynamic_app",
            meta={"ui": {"resourceUri": "ui://component/dynamic"}},
        )
        def dynamic_app() -> str:
            return "rendered"

    await ctx.notify_resources_changed()
    await ctx.notify_tools_changed()
    return "added"


@server.tool()
async def update_dynamic_app(ctx: Context) -> str:
    global resource_version
    resource_version += 1
    await ctx.notify_resource_updated("ui://component/dynamic")
    return "updated"


@server.tool()
def read_resource_call_count() -> int:
    return server.read_resource_calls


if __name__ == "__main__":
    server.run(
        transport="streamable-http",
        host="127.0.0.1",
        port=int(os.environ["FAST_AGENT_TEST_HTTP_PORT"]),
    )
