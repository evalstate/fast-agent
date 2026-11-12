"""Exercise SDK HTTP authentication with FastMCP's local OAuth simulator."""

import httpx2
import pytest
from fastmcp import FastMCP
from fastmcp.server.auth import AccessToken
from fastmcp.server.auth.providers.in_memory import InMemoryOAuthProvider
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from fast_agent.mcp.helpers.content_helpers import get_text


@pytest.mark.integration
@pytest.mark.asyncio
async def test_oauth_simulator_protects_tool_round_trip() -> None:
    provider = InMemoryOAuthProvider(base_url="http://localhost", required_scopes=["access"])
    provider.access_tokens["local-token"] = AccessToken(
        token="local-token", client_id="local-client", scopes=["access"]
    )
    server = FastMCP("auth-simulator", auth=provider)

    @server.tool
    def echo(message: str) -> str:
        return message

    app = server.http_app(transport="http")
    async with app.router.lifespan_context(app):
        async with httpx2.AsyncClient(
            transport=httpx2.ASGITransport(app=app), base_url="http://localhost"
        ) as client:
            for headers in ({}, {"Authorization": "Bearer invalid-token"}):
                response = await client.post("/mcp", json={}, headers=headers)
                assert response.status_code == 401
                assert "Bearer" in response.headers["www-authenticate"]

            client.headers["Authorization"] = "Bearer local-token"
            async with streamable_http_client("http://localhost/mcp", http_client=client) as (
                read_stream,
                write_stream,
            ):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    tools = await session.list_tools()
                    assert "echo" in {tool.name for tool in tools.tools}
                    result = await session.call_tool("echo", {"message": "authenticated"})
                    assert not result.is_error
                    assert get_text(result.content[0]) == "authenticated"
