"""Exercise gateway-owned clients through the SDK, replacing only network I/O."""

import json
from collections.abc import Callable
from functools import partial

import anyio
import httpx2
import pytest
from mcp.shared.message import SessionMessage
from mcp.types import JSONRPCError, JSONRPCRequest, JSONRPCResponse

from fast_agent.config import MCPServerSettings
from fast_agent.mcp.client_gateway import MCPClientHooks, _create_transport


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [307, 308])
@pytest.mark.parametrize(
    "location",
    [
        "/canonical",
        "https://other.example/mcp",
        "http://service.example/mcp",
        "https://service.example:8443/mcp",
    ],
)
async def test_gateway_redirects_preserve_transport_and_credentials(
    monkeypatch: pytest.MonkeyPatch, status: int, location: str
) -> None:
    requests: list[httpx2.Request] = []
    responses: list[httpx2.Response] = []
    credentials = {"Authorization": "Bearer secret", "X-HF-Authorization": "Bearer hf-secret"}
    same_origin = location == "/canonical"

    def server(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        assert request.url.host == "service.example"
        assert request.url.scheme == "https"
        assert request.url.port is None
        for name, value in credentials.items():
            assert request.headers[name] == value
        if request.method == "DELETE":
            assert request.headers["mcp-session-id"] == "test-session"
            if request.url.path == "/mcp":
                response = httpx2.Response(status, headers={"location": location})
            else:
                response = httpx2.Response(204)
        else:
            payload = json.loads(request.content)
            if payload["method"] == "tools/list" and request.url.path == "/mcp":
                response = httpx2.Response(status, headers={"location": location})
            else:
                if payload["method"] != "initialize":
                    assert request.headers["mcp-session-id"] == "test-session"
                response = httpx2.Response(
                    200,
                    headers={"mcp-session-id": "test-session"},
                    json={"jsonrpc": "2.0", "id": payload["id"], "result": {}},
                )
        responses.append(response)
        return response

    clients: list[httpx2.AsyncClient] = []
    client_factory = partial(httpx2.AsyncClient, transport=httpx2.MockTransport(server))

    # Retain the real client: the gateway still owns construction and closing.
    def create_client(
        *,
        headers: dict[str, str],
        auth: httpx2.Auth | None,
        timeout: httpx2.Timeout | None,
        event_hooks: dict[str, list[Callable]] | None,
    ) -> httpx2.AsyncClient:
        client = client_factory(
            headers=headers, auth=auth, timeout=timeout, event_hooks=event_hooks
        )
        clients.append(client)
        return client

    monkeypatch.setattr("fast_agent.mcp.client_gateway.httpx2.AsyncClient", create_client)
    config = MCPServerSettings(url="https://service.example/mcp", headers=credentials)
    with anyio.fail_after(5):
        async with _create_transport(
            server_name="redirect-test",
            config=config,
            oauth_mode="disabled",
            oauth_active=False,
            hooks=MCPClientHooks(),
        ) as (reader, writer):
            # Establish a session, reject/follow a redirect, then reuse that session.
            for request_id, method in enumerate(["initialize", "tools/list", "ping"]):
                await writer.send(
                    SessionMessage(JSONRPCRequest(jsonrpc="2.0", id=request_id, method=method))
                )
                reply = await reader.receive()
                assert isinstance(reply, SessionMessage)
                assert isinstance(reply.message, (JSONRPCError, JSONRPCResponse))
                assert reply.message.id == request_id
                if method == "tools/list" and not same_origin:
                    assert isinstance(reply.message, JSONRPCError)
                    assert "not followed" in reply.message.error.message
                else:
                    assert isinstance(reply.message, JSONRPCResponse)
                    assert reply.message.result == {}

    assert clients[0].is_closed
    assert all(response.is_closed for response in responses)
    assert [request.url.path for request in requests if request.method == "DELETE"] == (
        ["/mcp", "/canonical"] if same_origin else ["/mcp"]
    )
    redirected_posts = [
        request
        for request in requests
        if request.method == "POST" and request.url.path == "/canonical"
    ]
    assert len(redirected_posts) == (1 if same_origin else 0)
    if redirected_posts:
        assert redirected_posts[0].content == requests[1].content
