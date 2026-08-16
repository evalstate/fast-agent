from __future__ import annotations

from typing import TYPE_CHECKING

import httpx2 as httpx
import pytest
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from fast_agent.config import MCPServerSettings, MCPSettings, Settings
from fast_agent.mcp.client_callback_runtime import MCPClientCallbackRuntime
from fast_agent.mcp.mcp_connection_manager import MCPConnectionManager
from fast_agent.mcp_server_registry import ServerRegistry

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


class _BearerAuth(httpx.Auth):
    async def async_auth_flow(
        self, request: httpx.Request
    ) -> AsyncGenerator[httpx.Request, httpx.Response]:
        request.headers["Authorization"] = "Bearer test-token"
        yield request


def _build_app() -> tuple[FastAPI, list[str | None]]:
    app = FastAPI()
    initialize_auth_headers: list[str | None] = []

    @app.get("/mcp")
    async def get_mcp() -> Response:
        return Response(status_code=405)

    @app.delete("/mcp")
    async def delete_mcp() -> Response:
        return Response(status_code=204)

    @app.post("/mcp")
    async def post_mcp(request: Request) -> Response:
        payload = await request.json()
        method = payload.get("method")
        authorization = request.headers.get("authorization")

        if method == "initialize":
            initialize_auth_headers.append(authorization)

        if authorization != "Bearer test-token":
            return JSONResponse(
                {"error": "unauthorized"},
                status_code=401,
                headers={"WWW-Authenticate": 'Bearer realm="test"'},
            )

        if method == "initialize":
            return JSONResponse(
                {
                    "jsonrpc": "2.0",
                    "id": payload.get("id"),
                    "result": {
                        "protocolVersion": "2025-11-25",
                        "capabilities": {"tools": {"listChanged": True}},
                        "serverInfo": {"name": "auth-test", "version": "1.0.0"},
                    },
                },
                headers={"mcp-session-id": "session-1"},
            )

        if method == "notifications/initialized":
            return Response(status_code=202)

        if method == "tools/list":
            return JSONResponse(
                {
                    "jsonrpc": "2.0",
                    "id": payload.get("id"),
                    "result": {
                        "tools": [
                            {
                                "name": "echo",
                                "description": "echo tool",
                                "inputSchema": {"type": "object"},
                            }
                        ]
                    },
                }
            )

        return JSONResponse(
            {
                "jsonrpc": "2.0",
                "id": payload.get("id"),
                "error": {"code": -32601, "message": f"Unknown method: {method}"},
            },
            status_code=200,
        )

    return app, initialize_auth_headers


@pytest.mark.integration
@pytest.mark.asyncio
async def test_http_mcp_auto_escalates_to_oauth_on_401(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, initialize_auth_headers = _build_app()
    async_client = httpx.AsyncClient

    def _client_factory(
        headers: dict[str, str] | None = None,
        timeout: httpx.Timeout | None = None,
        auth: httpx.Auth | None = None,
        follow_redirects: bool = True,
        event_hooks: dict[str, list] | None = None,
    ) -> httpx.AsyncClient:
        return async_client(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
            headers=headers,
            timeout=timeout,
            auth=auth,
            follow_redirects=follow_redirects,
            event_hooks=event_hooks,
        )

    monkeypatch.setattr(
        "fast_agent.mcp.client_gateway.httpx2.AsyncClient",
        _client_factory,
    )
    monkeypatch.setattr(
        "fast_agent.mcp.client_gateway.build_oauth_provider",
        lambda *_args, **_kwargs: _BearerAuth(),
    )

    settings = Settings(
        mcp=MCPSettings(
            servers={
                "authsrv": MCPServerSettings(
                    transport="http",
                    url="http://testserver/mcp",
                )
            }
        )
    )
    registry = ServerRegistry(config=settings)
    server_config = registry.get_server_config("authsrv")
    assert server_config is not None

    async with MCPConnectionManager(registry) as manager:
        server_conn = await manager.get_server(
            "authsrv",
            callback_runtime=MCPClientCallbackRuntime(
                server_name="authsrv",
                server_config=server_config,
            ),
            startup_timeout_seconds=5.0,
        )

        assert server_conn.is_healthy() is True
        assert manager._oauth_required_servers == {"authsrv"}
        assert initialize_auth_headers == [None, "Bearer test-token"]

        assert server_conn.client is not None
        tools = await server_conn.client.list_tools()
        assert [tool.name for tool in tools.tools] == ["echo"]
