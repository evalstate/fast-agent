import uvicorn
from fastmcp import FastMCP
from starlette.applications import Starlette
from starlette.routing import Mount

SERVER_PATH = "t0-example-server"
MCP_PATH = f"/{SERVER_PATH}/mcp"


mcp_instance = FastMCP(name="t0-example-server")


@mcp_instance.tool()
def example_tool(input_text: str) -> str:
    """Example tool that reverses the text of a given string."""
    reversed_text = input_text[::-1]
    return reversed_text


mcp_app = mcp_instance.http_app(path=MCP_PATH, transport="http")
app = Starlette(
    routes=[Mount("/", app=mcp_app)],
    lifespan=mcp_app.lifespan,
)

if __name__ == "__main__":
    print(f"Starting minimal MCP server ({mcp_instance.name}) on http://127.0.0.1:8000")
    print(f" -> HTTP endpoint: {MCP_PATH}")
    uvicorn.run(app, host="127.0.0.1", port=8000)
