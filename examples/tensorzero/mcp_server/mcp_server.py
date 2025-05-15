import uvicorn
from mcp.server.fastmcp.server import FastMCP
from starlette.applications import Starlette
from starlette.routing import Mount

SERVER_PATH = "t0-example-server"


mcp_instance = FastMCP(name="t0-example-server")
mcp_instance.settings.message_path = f"/{SERVER_PATH}/messages/"
mcp_instance.settings.sse_path = f"/{SERVER_PATH}/sse"


@mcp_instance.tool()
def example_tool(input_text: str) -> str:
    """Example tool that reverses the text of a given string."""
    reversed_text = input_text[::-1]
    return reversed_text


app = Starlette(
    routes=[
        Mount("/", app=mcp_instance.sse_app()),
    ]
)

if __name__ == "__main__":
    print(f"Starting minimal MCP server ({mcp_instance.name}) on http://127.0.0.1:8000")
    print(f" -> SSE endpoint: {mcp_instance.settings.sse_path}")
    print(f" -> Message endpoint: {mcp_instance.settings.message_path}")
    uvicorn.run(app, host="127.0.0.1", port=8000)
