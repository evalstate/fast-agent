from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from fast_agent.llm.fastagent_llm import _mcp_metadata_var
from fast_agent.mcp.mcp_aggregator import MCPAggregator


class _RecordingSession:
    def __init__(self) -> None:
        self.last_kwargs: dict[str, Any] | None = None

    async def call_tool(self, **kwargs: Any) -> str:
        self.last_kwargs = dict(kwargs)
        return "ok-call"

    async def read_resource(self, **kwargs: Any) -> str:
        self.last_kwargs = dict(kwargs)
        return "ok-read"


class _FakeConnectionManager:
    def __init__(self, session: _RecordingSession) -> None:
        self._session = session

    async def get_server(self, server_name: str, client_session_factory) -> SimpleNamespace:
        del server_name, client_session_factory
        return SimpleNamespace(session=self._session)


@pytest.mark.asyncio
async def test_execute_on_server_uses_meta_for_call_tool() -> None:
    session = _RecordingSession()
    aggregator = MCPAggregator(server_names=[], connection_persistence=True, context=None)
    setattr(aggregator, "_persistent_connection_manager", _FakeConnectionManager(session))

    metadata = {"mcp/session": {"id": "sess-123"}}
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
async def test_execute_on_server_keeps__meta_for_read_resource() -> None:
    session = _RecordingSession()
    aggregator = MCPAggregator(server_names=[], connection_persistence=True, context=None)
    setattr(aggregator, "_persistent_connection_manager", _FakeConnectionManager(session))

    metadata = {"mcp/session": {"id": "sess-123"}}
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
    assert session.last_kwargs.get("_meta") == metadata
    assert "meta" not in session.last_kwargs
