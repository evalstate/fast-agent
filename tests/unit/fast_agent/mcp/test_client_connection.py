import json
import warnings
from collections.abc import Awaitable, Callable
from typing import cast

import httpx2
import pytest
from anyio import Event, create_task_group, sleep_forever
from mcp.client import Client
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.exceptions import MCPDeprecationWarning, MCPError
from mcp_types import (
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
    BlobResourceContents,
    EmbeddedResource,
    PromptReference,
    TextContent,
    TextResourceContents,
)

from fast_agent.config import MCPServerSettings
from fast_agent.core.exceptions import ServerSessionTerminatedError
from fast_agent.mcp.client_callback_runtime import MCPClientCallbackRuntime
from fast_agent.mcp.client_connection import MCPClientConnection
from fast_agent.mcp.skills_extension import GetSkillResult, ListSkillsResult
from fast_agent.mcp.tool_result_metadata import url_elicitation_required_payload
from fast_agent.mcp.uri_security import is_file_uri


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
        if method == "server/discover":
            return httpx2.Response(
                200,
                headers={"content-type": "application/json"},
                json={
                    "jsonrpc": "2.0",
                    "id": payload["id"],
                    "result": {
                        "supportedVersions": ["2026-07-28"],
                        "capabilities": {},
                        "resultType": "complete",
                    },
                },
            )
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


class LegacyOnlyStreamableHTTPSimulator(StatefulStreamableHTTPSimulator):
    def __init__(self) -> None:
        super().__init__()
        self.methods: list[str] = []

    async def __call__(self, request: httpx2.Request) -> httpx2.Response:
        if request.method == "POST":
            payload = json.loads(request.content)
            assert isinstance(payload, dict)
            method = payload.get("method")
            if isinstance(method, str):
                self.methods.append(method)
            if method == "server/discover":
                return httpx2.Response(
                    200,
                    headers={"content-type": "application/json"},
                    json={
                        "jsonrpc": "2.0",
                        "id": payload["id"],
                        "error": {
                            "code": METHOD_NOT_FOUND,
                            "message": "Method not found",
                        },
                    },
                )
        return await super().__call__(request)


class BlockingDiscoverStreamableHTTPSimulator(StatefulStreamableHTTPSimulator):
    def __init__(self) -> None:
        super().__init__()
        self.discover_started = Event()

    async def __call__(self, request: httpx2.Request) -> httpx2.Response:
        if request.method == "POST":
            payload = json.loads(request.content)
            assert isinstance(payload, dict)
            if payload.get("method") == "server/discover":
                self.discover_started.set()
                await sleep_forever()
        return await super().__call__(request)


class SkillsResponseStreamableHTTPSimulator(StatefulStreamableHTTPSimulator):
    async def __call__(self, request: httpx2.Request) -> httpx2.Response:
        if request.method != "POST":
            return await super().__call__(request)

        payload = json.loads(request.content)
        assert isinstance(payload, dict)
        method = payload["method"]
        if method in {"initialize", "notifications/initialized"}:
            return await super().__call__(request)

        assert self.session_id is not None
        assert request.headers["mcp-session-id"] == self.session_id
        params = payload["params"]
        assert isinstance(params, dict)
        skill = {
            "uri": "skill://demo/SKILL.md",
            "frontmatter": {"name": "demo", "description": "Demo skill"},
            "resources": [
                {
                    "uri": "skill://demo/SKILL.md",
                    "digest": "sha256:abc",
                }
            ],
        }
        if method == "skills/list":
            assert params["cursor"] == "page-1"
            result = {
                "resultType": "complete",
                "cacheScope": "public",
                "ttlMs": 30_000,
                "nextCursor": "page-2",
                "skills": [skill],
            }
        else:
            assert method == "skills/get"
            assert params["uri"] == skill["uri"]
            result = {"resultType": "complete", "skill": skill}
        return httpx2.Response(
            200,
            headers={"content-type": "application/json"},
            json={"jsonrpc": "2.0", "id": payload["id"], "result": result},
        )


class MixedResponseStreamableHTTPSimulator:
    def __init__(self) -> None:
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
        if method == "server/discover":
            return httpx2.Response(
                200,
                headers={"content-type": "application/json; charset=utf-8"},
                json={
                    "jsonrpc": "2.0",
                    "id": payload["id"],
                    "result": {
                        "supportedVersions": ["2026-07-28"],
                        "capabilities": {"tools": {"listChanged": False}},
                        "resultType": "complete",
                    },
                },
            )
        if method == "tools/list":
            return httpx2.Response(
                200,
                headers={"content-type": "application/json; charset=utf-8"},
                json={
                    "jsonrpc": "2.0",
                    "id": payload["id"],
                    "result": {
                        "resultType": "complete",
                        "cacheScope": "private",
                        "ttlMs": 0,
                        "tools": [
                            {"name": "whoami", "inputSchema": {"type": "object"}},
                            {
                                "name": "generate_image",
                                "inputSchema": {"type": "object"},
                            },
                        ],
                    },
                },
            )

        assert request.headers["accept"] == "application/json, text/event-stream"
        params = payload["params"]
        assert isinstance(params, dict)
        assert method == "tools/call"
        result = {
            "jsonrpc": "2.0",
            "id": payload["id"],
            "result": {
                "resultType": "complete",
                "content": [
                    {
                        "type": "text",
                        "text": "evalstate" if params["name"] == "whoami" else "generated image",
                    }
                ],
            },
        }
        if params["name"] == "whoami":
            return httpx2.Response(
                200,
                headers={"content-type": "application/json; charset=utf-8"},
                json=result,
            )
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream; charset=utf-8"},
            text=f"event: message\ndata: {json.dumps(result)}\n\n",
        )


class AttachmentResponseSimulator:
    def __init__(self, uri: str, *, on_result: Callable[[], None] | None = None) -> None:
        self.uri = uri
        self.on_result = on_result
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

        assert self.session_id is not None, method
        assert request.headers["mcp-session-id"] == self.session_id
        if self.on_result is not None:
            self.on_result()
        return httpx2.Response(
            200,
            headers={"content-type": "application/json"},
            json={
                "jsonrpc": "2.0",
                "id": payload["id"],
                "result": self._result_for(method),
            },
        )

    def _result_for(self, method: object) -> dict[str, object]:
        if method == "tools/list":
            return {"tools": []}
        resource_link = {
            "type": "resource_link",
            "name": "attachment",
            "uri": self.uri,
            "mimeType": "text/plain",
        }
        if method == "tools/call":
            return {
                "content": [
                    resource_link,
                    {
                        "type": "resource",
                        "resource": {
                            "uri": self.uri,
                            "mimeType": "text/plain",
                            "text": "inline tool text",
                        },
                    },
                ]
            }
        if method == "prompts/get":
            return {
                "messages": [
                    {"role": "user", "content": resource_link},
                    {
                        "role": "user",
                        "content": {
                            "type": "resource",
                            "resource": {
                                "uri": self.uri,
                                "mimeType": "text/plain",
                                "text": "inline prompt text",
                            },
                        },
                    },
                ]
            }
        assert method == "resources/read"
        return {
            "contents": [
                {
                    "uri": self.uri,
                    "mimeType": "text/plain",
                    "text": "inline resource text",
                },
                {
                    "uri": self.uri,
                    "mimeType": "application/octet-stream",
                    "blob": "Ynl0ZXM=",
                },
            ]
        }


RequestOperation = Callable[[MCPClientConnection], Awaitable[object]]
FILE_URI_FORMS = (
    "file:///x",
    "file:/x",
    "file:x",
    "FILE:///x",
    "file://localhost/x",
    "file://[",
)


def test_is_file_uri_fails_closed_for_malformed_file_uri() -> None:
    assert is_file_uri("file://[")
    assert not is_file_uri("http://[")


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
        pytest.param(lambda connection: connection.list_skills(), id="list_skills"),
        pytest.param(
            lambda connection: connection.get_skill("skill://demo/SKILL.md"),
            id="get_skill",
        ),
    ]


@pytest.mark.asyncio
async def test_legacy_ping_bypasses_deprecated_client_wrapper() -> None:
    class FakeSession:
        async def send_ping(self):
            return "pong"

    class FakeClient:
        session = FakeSession()

        async def send_ping(self):
            warnings.warn(
                "ping is removed as of 2026-07-28; the method only works under mode='legacy'.",
                MCPDeprecationWarning,
                stacklevel=2,
            )
            raise AssertionError("deprecated high-level wrapper should not be called")

    connection = object.__new__(MCPClientConnection)
    connection.client = cast("Client", FakeClient())

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = await connection.ping()

    assert result == "pong"
    assert caught == []


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
async def test_forced_modern_discovery_failure_does_not_fall_back_to_initialize() -> None:
    simulator = LegacyOnlyStreamableHTTPSimulator()
    http_client = httpx2.AsyncClient(transport=httpx2.MockTransport(simulator))
    transport = streamable_http_client("https://example.test/mcp", http_client=http_client)
    callbacks = MCPClientCallbackRuntime(server_name="test-server", server_config=None)

    with pytest.raises(MCPError) as exc_info:
        async with MCPClientConnection(
            transport,
            callbacks,
            protocol_mode="modern",
            cache=False,
        ):
            pass

    await http_client.aclose()
    assert exc_info.value.code == METHOD_NOT_FOUND
    assert simulator.methods == ["server/discover"]


@pytest.mark.asyncio
async def test_forced_modern_discovery_cancellation_closes_client() -> None:
    simulator = BlockingDiscoverStreamableHTTPSimulator()
    http_client = httpx2.AsyncClient(transport=httpx2.MockTransport(simulator))
    transport = streamable_http_client("https://example.test/mcp", http_client=http_client)
    callbacks = MCPClientCallbackRuntime(server_name="test-server", server_config=None)
    connection = MCPClientConnection(
        transport,
        callbacks,
        protocol_mode="modern",
        cache=False,
    )

    async def connect() -> None:
        async with connection:
            pass

    async with create_task_group() as task_group:
        task_group.start_soon(connect)
        await simulator.discover_started.wait()
        task_group.cancel_scope.cancel()

    with pytest.raises(RuntimeError, match="async context manager"):
        _ = connection.session
    await http_client.aclose()


@pytest.mark.asyncio
async def test_skills_extension_requests_and_parses_results() -> None:
    simulator = SkillsResponseStreamableHTTPSimulator()
    http_client = httpx2.AsyncClient(transport=httpx2.MockTransport(simulator))
    transport = streamable_http_client("https://example.test/mcp", http_client=http_client)
    callbacks = MCPClientCallbackRuntime(server_name="test-server", server_config=None)

    async with MCPClientConnection(
        transport,
        callbacks,
        cache=False,
        protocol_mode="legacy",
    ) as connection:
        listed = await connection.list_skills(cursor="page-1")
        skill = await connection.get_skill("skill://demo/SKILL.md")

    await http_client.aclose()

    assert isinstance(listed, ListSkillsResult)
    assert listed.next_cursor == "page-2"
    assert listed.ttl_ms == 30_000
    assert listed.cache_scope == "public"
    assert listed.skills[0].resources is not None
    assert listed.skills[0].resources[0].digest == "sha256:abc"
    assert isinstance(skill, GetSkillResult)
    assert skill.skill.frontmatter["name"] == "demo"


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


@pytest.mark.asyncio
async def test_streamable_http_accepts_json_and_sse_responses_on_one_endpoint() -> None:
    simulator = MixedResponseStreamableHTTPSimulator()
    http_client = httpx2.AsyncClient(transport=httpx2.MockTransport(simulator))
    transport = streamable_http_client("https://example.test/mcp", http_client=http_client)
    callbacks = MCPClientCallbackRuntime(server_name="test-server", server_config=None)

    async with MCPClientConnection(
        transport,
        callbacks,
        protocol_mode="modern",
        cache=False,
    ) as connection:
        assert connection.server_capabilities.tools is not None
        assert connection.server_capabilities.tools.list_changed is False
        whoami = await connection.call_tool("whoami")
        image = await connection.call_tool("generate_image")

    await http_client.aclose()

    assert isinstance(whoami.content[0], TextContent)
    assert whoami.content[0].text == "evalstate"
    assert isinstance(image.content[0], TextContent)
    assert image.content[0].text == "generated image"


@pytest.mark.asyncio
@pytest.mark.parametrize("uri", FILE_URI_FORMS)
@pytest.mark.parametrize(
    "server_config",
    [
        None,
        MCPServerSettings(
            name="remote-http",
            transport="http",
            url="https://example.test/mcp",
        ),
        MCPServerSettings(
            name="remote-sse",
            transport="sse",
            url="https://example.test/sse",
        ),
        MCPServerSettings(name="local", transport="stdio", command="server"),
    ],
)
async def test_mcp_attachment_ingress_blocks_file_uris_for_every_transport(
    uri: str,
    server_config: MCPServerSettings | None,
) -> None:
    simulator = AttachmentResponseSimulator(uri)
    http_client = httpx2.AsyncClient(transport=httpx2.MockTransport(simulator))
    transport = streamable_http_client("https://example.test/mcp", http_client=http_client)
    callbacks = MCPClientCallbackRuntime(server_name="test-server", server_config=server_config)

    async with MCPClientConnection(
        transport,
        callbacks,
        protocol_mode="legacy",
        cache=False,
    ) as connection:
        tool_result = await connection.call_tool("attachment")
        prompt_result = await connection.get_prompt("attachment")
        read_result = await connection.read_resource(uri)

    await http_client.aclose()

    tool_link, tool_embedded = tool_result.content
    prompt_link = prompt_result.messages[0].content
    prompt_embedded = prompt_result.messages[1].content
    read_text, read_blob = read_result.contents

    assert isinstance(tool_link, TextContent)
    assert tool_link.text == "[Local file attachment from a remote MCP server was blocked.]"
    assert isinstance(prompt_link, TextContent)
    assert prompt_link.text == "[Local file attachment from a remote MCP server was blocked.]"

    assert isinstance(tool_embedded, EmbeddedResource)
    assert isinstance(tool_embedded.resource, TextResourceContents)
    assert tool_embedded.resource.text == "inline tool text"
    assert str(tool_embedded.resource.uri) == "urn:fast-agent:remote-mcp-inline"

    assert isinstance(prompt_embedded, EmbeddedResource)
    assert isinstance(prompt_embedded.resource, TextResourceContents)
    assert prompt_embedded.resource.text == "inline prompt text"
    assert str(prompt_embedded.resource.uri) == "urn:fast-agent:remote-mcp-inline"

    assert isinstance(read_text, TextResourceContents)
    assert read_text.text == "inline resource text"
    assert str(read_text.uri) == "urn:fast-agent:remote-mcp-inline"
    assert isinstance(read_blob, BlobResourceContents)
    assert read_blob.blob == "Ynl0ZXM="
    assert str(read_blob.uri) == "urn:fast-agent:remote-mcp-inline"


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ("prompts/get", "resources/read"))
async def test_sanitized_interactive_result_keeps_pending_url_elicitation(
    method: str,
) -> None:
    callbacks = MCPClientCallbackRuntime(server_name="test-server", server_config=None)

    def queue_url_elicitation() -> None:
        callbacks.queue_url_elicitation(
            message="Authenticate to continue",
            url="https://example.test/auth",
            elicitation_id="auth-1",
        )

    simulator = AttachmentResponseSimulator(
        "file:///secret",
        on_result=queue_url_elicitation,
    )
    http_client = httpx2.AsyncClient(transport=httpx2.MockTransport(simulator))
    transport = streamable_http_client("https://example.test/mcp", http_client=http_client)

    async with MCPClientConnection(
        transport,
        callbacks,
        protocol_mode="legacy",
        cache=False,
    ) as connection:
        if method == "prompts/get":
            result = await connection.get_prompt("attachment")
            assert isinstance(result.messages[0].content, TextContent)
        else:
            result = await connection.read_resource("file:///secret")
            assert str(result.contents[0].uri) == "urn:fast-agent:remote-mcp-inline"

    await http_client.aclose()

    payload = url_elicitation_required_payload(result)
    assert payload is not None
    assert payload.request_method == method
    assert [(item.message, item.url, item.elicitation_id) for item in payload.elicitations] == [
        ("Authenticate to continue", "https://example.test/auth", "auth-1")
    ]
