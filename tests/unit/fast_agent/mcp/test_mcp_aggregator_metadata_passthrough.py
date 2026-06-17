from __future__ import annotations

from datetime import timedelta
from types import SimpleNamespace
from typing import Any, cast

import pytest
from mcp.types import CallToolRequest, CallToolResult, ClientRequest, TextContent

from fast_agent.llm.fastagent_llm import _mcp_metadata_var
from fast_agent.mcp.mcp_agent_client_session import MCPAgentClientSession
from fast_agent.mcp.mcp_aggregator import MCPAggregator


class _RecordingSession:
    def __init__(self) -> None:
        self.last_kwargs: dict[str, Any] | None = None

    async def call_tool(self, **kwargs: Any) -> Any:
        self.last_kwargs = dict(kwargs)
        return "ok-call"

    async def read_resource(self, **kwargs: Any) -> Any:
        self.last_kwargs = dict(kwargs)
        return "ok-read"


class _FakeConnectionManager:
    def __init__(self, session: _RecordingSession) -> None:
        self._session = session

    async def get_server(self, server_name: str, client_session_factory) -> SimpleNamespace:
        del server_name, client_session_factory
        return SimpleNamespace(session=self._session)


class _RawCallToolSession(MCPAgentClientSession):
    def __init__(self) -> None:
        self.last_request: ClientRequest | None = None
        self.last_timeout = None
        self.last_progress_callback = None

    async def send_request(
        self,
        request,
        result_type,
        request_read_timeout_seconds=None,
        metadata=None,
        progress_callback=None,
    ):
        del result_type, metadata
        self.last_request = request
        self.last_timeout = request_read_timeout_seconds
        self.last_progress_callback = progress_callback
        return CallToolResult(content=[TextContent(type="text", text="legacy result")])


@pytest.mark.asyncio
async def test_client_session_call_tool_uses_raw_request_path_with_meta() -> None:
    session = _RawCallToolSession()
    metadata = {"trace": {"id": "abc"}}

    result = await session.call_tool(
        name="legacy_tool",
        arguments={"value": 1},
        read_timeout_seconds=timedelta(seconds=3),
        meta=metadata,
    )

    assert result.content == [TextContent(type="text", text="legacy result")]
    assert session.last_timeout == timedelta(seconds=3)
    assert session.last_request is not None
    request = cast("CallToolRequest", session.last_request.root)
    assert request.method == "tools/call"
    assert request.params.name == "legacy_tool"
    assert request.params.arguments == {"value": 1}
    assert request.params.meta is not None
    assert request.params.meta.model_dump(exclude_none=True) == metadata


@pytest.mark.asyncio
async def test_execute_on_server_uses_meta_for_call_tool() -> None:
    session = _RecordingSession()
    aggregator = MCPAggregator(server_names=[], connection_persistence=True, context=None)
    setattr(aggregator, "_persistent_connection_manager", _FakeConnectionManager(session))

    metadata = {
        "io.modelcontextprotocol/session": {
            "sessionId": "sess-123",
            "state": "token",
        }
    }
    token = _mcp_metadata_var.set(metadata)
    try:
        result = await aggregator._execute_on_server(
            server_name="demo",
            operation_type="tools/call",
            operation_name="echo",
            method_name="call_tool",
            method_args={"name": "echo", "arguments": {}},
        )
    finally:
        _mcp_metadata_var.reset(token)

    assert result == "ok-call"
    assert session.last_kwargs is not None
    assert session.last_kwargs.get("meta") == metadata
    assert "_meta" not in session.last_kwargs


@pytest.mark.asyncio
async def test_execute_on_server_uses_meta_for_read_resource() -> None:
    session = _RecordingSession()
    aggregator = MCPAggregator(server_names=[], connection_persistence=True, context=None)
    setattr(aggregator, "_persistent_connection_manager", _FakeConnectionManager(session))

    metadata = {
        "io.modelcontextprotocol/session": {
            "sessionId": "sess-123",
            "state": "token",
        }
    }
    token = _mcp_metadata_var.set(metadata)
    try:
        result = await aggregator._execute_on_server(
            server_name="demo",
            operation_type="resources/read",
            operation_name="file://demo.txt",
            method_name="read_resource",
            method_args={"uri": "file://demo.txt"},
        )
    finally:
        _mcp_metadata_var.reset(token)

    assert result == "ok-read"
    assert session.last_kwargs is not None
    assert session.last_kwargs.get("meta") == metadata
    assert "_meta" not in session.last_kwargs
