from typing import Annotated

from mcp.server.mcpserver import ListRoots, MCPServer, Resolve
from mcp_types import ListRootsResult


def request_roots() -> ListRoots:
    return ListRoots()


server = MCPServer("MCP Root Tester")


@server.tool()
def show_roots(roots: Annotated[ListRootsResult, Resolve(request_roots)]) -> str:
    return roots.model_dump_json()


if __name__ == "__main__":
    server.run(transport="stdio")
