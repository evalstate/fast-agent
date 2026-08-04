from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest

from fast_agent.llm.fastagent_llm import _mcp_metadata_var
from fast_agent.mcp.mcp_aggregator import MCPAggregator

if TYPE_CHECKING:
    from fast_agent.mcp.mcp_connection_manager import MCPConnectionManager


class _RecordingSession:
    def __init__(self) -> None:
        self.last_kwargs: dict[str, Any] | None = None

    async def call_tool(self, **kwargs: Any) -> Any:
        self.last_kwargs = dict(kwargs)
        return "ok-call"

    async def read_resource(self, **kwargs: Any) -> Any:
        self.last_kwargs = dict(kwargs)
        return "ok-read"

    async def get_prompt(self, **kwargs: Any) -> Any:
        self.last_kwargs = dict(kwargs)
        return "ok-prompt"


class _FakeConnectionManager:
    def __init__(self, session: _RecordingSession) -> None:
        self._session = session

    async def get_server(self, server_name: str, callback_runtime) -> SimpleNamespace:
        del server_name, callback_runtime
        return SimpleNamespace(client=self._session)


@pytest.mark.asyncio
async def test_execute_on_server_uses_meta_for_call_tool() -> None:
    session = _RecordingSession()
    aggregator = MCPAggregator(server_names=[], connection_persistence=True, context=None)
    aggregator._persistent_connection_manager = cast(
        "MCPConnectionManager", _FakeConnectionManager(session)
    )

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
    aggregator._persistent_connection_manager = cast(
        "MCPConnectionManager", _FakeConnectionManager(session)
    )

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
