import json
from collections.abc import Awaitable, Callable

import httpx2
import pytest
from mcp.client import Client
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.exceptions import MCPError
from mcp_types import INVALID_REQUEST, PromptReference

from fast_agent.core.exceptions import ServerSessionTerminatedError
from fast_agent.mcp.client_callback_runtime import MCPClientCallbackRuntime
from fast_agent.mcp.client_connection import MCPClientConnection


class StatefulStreamableHTTPSimulator:
    def __init__(self, *, terminal_message: str | None = None) -> None:
        self.terminal_message = terminal_message
        self.session_id: str | None = None

    async def __call__(self, request: httpx2.Request) -> httpx2.Response:
        if request.method == "GET":
            return httpx2.Response(405)
        if request.method == "DELETE":
            return httpx2.Response(200)

        payload = json.loads(request.content)
        assert isinstance(payload, dict)
        method = payload.get("method")
        if method == "initialize":
            self.session_id = "session-1"
            return httpx2.Response(
                200,
                headers={
                    "content-type": "application/json",
                    "mcp-session-id": self.session_id,
                },
                json={
                    "jsonrpc": "2.0",
                    "id": payload["id"],
                    "result": {
                        "protocolVersion": "2025-06-18",
                        "capabilities": {},
                        "serverInfo": {"name": "simulator", "version": "1"},
                    },
                },
            )
        if method == "notifications/initialized":
            return httpx2.Response(202)
        if self.terminal_message is not None:
            return httpx2.Response(
                400,
                headers={"content-type": "application/json"},
                json={
                    "jsonrpc": "2.0",
                    "id": payload["id"],
                    "error": {
                        "code": INVALID_REQUEST,
                        "message": self.terminal_message,
                    },
                },
            )

        assert self.session_id is not None
        assert request.headers["mcp-session-id"] == self.session_id
        return httpx2.Response(404)


RequestOperation = Callable[[MCPClientConnection], Awaitable[object]]


def _operations() -> list[object]:
    return [
        pytest.param(lambda connection: connection.list_tools(), id="list_tools"),
        pytest.param(lambda connection: connection.list_prompts(), id="list_prompts"),
        pytest.param(lambda connection: connection.list_resources(), id="list_resources"),
        pytest.param(
            lambda connection: connection.list_resource_templates(),
            id="list_resource_templates",
        ),
        pytest.param(lambda connection: connection.call_tool("tool"), id="call_tool"),
        pytest.param(
            lambda connection: connection.read_resource("file:///resource"),
            id="read_resource",
        ),
        pytest.param(lambda connection: connection.get_prompt("prompt"), id="get_prompt"),
        pytest.param(
            lambda connection: connection.complete(
                PromptReference(type="ref/prompt", name="prompt"),
                {"name": "argument", "value": ""},
            ),
            id="complete",
        ),
        pytest.param(
            lambda connection: connection.read_directory("file:///directory"),
            id="read_directory",
        ),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", _operations())
async def test_legacy_requests_translate_terminated_streamable_http_session(
    operation: RequestOperation,
) -> None:
    simulator = StatefulStreamableHTTPSimulator()
    http_client = httpx2.AsyncClient(transport=httpx2.MockTransport(simulator))
    transport = streamable_http_client("https://example.test/mcp", http_client=http_client)
    callbacks = MCPClientCallbackRuntime(server_name="test-server", server_config=None)

    async with MCPClientConnection(
        transport,
        callbacks,
        protocol_mode="legacy",
        cache=False,
    ) as connection:
        with pytest.raises(ServerSessionTerminatedError, match="test-server"):
            await operation(connection)

    await http_client.aclose()


@pytest.mark.asyncio
async def test_modern_request_does_not_translate_session_terminated_error() -> None:
    simulator = StatefulStreamableHTTPSimulator(terminal_message="Session terminated")
    http_client = httpx2.AsyncClient(transport=httpx2.MockTransport(simulator))
    transport = streamable_http_client("https://example.test/mcp", http_client=http_client)
    callbacks = MCPClientCallbackRuntime(server_name="test-server", server_config=None)

    async with MCPClientConnection(
        transport,
        callbacks,
        protocol_mode="modern",
        cache=False,
    ) as connection:
        with pytest.raises(MCPError) as exc_info:
            await connection.list_tools()

    await http_client.aclose()
    assert exc_info.value.code == INVALID_REQUEST
    assert exc_info.value.message == "Session terminated"


@pytest.mark.asyncio
async def test_legacy_request_requires_exact_session_terminated_message() -> None:
    simulator = StatefulStreamableHTTPSimulator(terminal_message="Invalid request")
    http_client = httpx2.AsyncClient(transport=httpx2.MockTransport(simulator))
    transport = streamable_http_client("https://example.test/mcp", http_client=http_client)
    callbacks = MCPClientCallbackRuntime(server_name="test-server", server_config=None)

    async with MCPClientConnection(
        transport,
        callbacks,
        protocol_mode="legacy",
        cache=False,
    ) as connection:
        with pytest.raises(MCPError) as exc_info:
            await connection.list_tools()

    await http_client.aclose()
    assert exc_info.value.code == INVALID_REQUEST
    assert exc_info.value.message == "Invalid request"


@pytest.mark.asyncio
async def test_sdk_reports_stateful_404_as_session_terminated_invalid_request() -> None:
    simulator = StatefulStreamableHTTPSimulator()
    http_client = httpx2.AsyncClient(transport=httpx2.MockTransport(simulator))
    transport = streamable_http_client("https://example.test/mcp", http_client=http_client)

    async with Client(transport, mode="legacy", cache=None) as client:
        with pytest.raises(MCPError) as exc_info:
            await client.list_tools()

    await http_client.aclose()
    assert exc_info.value.code == INVALID_REQUEST
    assert exc_info.value.message == "Session terminated"
