from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from mcp.shared.exceptions import McpError
from mcp.types import CallToolResult, ErrorData, TextContent

from fast_agent.llm.fastagent_llm import _mcp_metadata_var
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


class _RejectingSession(_RecordingSession):
    def __init__(self) -> None:
        super().__init__()
        self.experimental_session_cookie: dict[str, Any] | None = {
            "sessionId": "sess-rejected"
        }

    @property
    def experimental_session_id(self) -> str | None:
        cookie = self.experimental_session_cookie
        if isinstance(cookie, dict):
            session_id = cookie.get("sessionId")
            if isinstance(session_id, str) and session_id:
                return session_id
        return None

    def set_experimental_session_cookie(self, cookie: dict[str, Any] | None) -> None:
        self.experimental_session_cookie = dict(cookie) if isinstance(cookie, dict) else None

    async def call_tool(self, **kwargs: Any) -> str:
        self.last_kwargs = dict(kwargs)
        raise McpError(
            ErrorData(
                code=-32043,
                message="Session not found",
            )
        )


class _InvalidationRecorder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str | None]] = []

    def mark_cookie_invalidated(
        self,
        server_name: str,
        *,
        session_id: str,
        reason: str | None = None,
    ) -> bool:
        self.calls.append((server_name, session_id, reason))
        return True


class _ToolErrorResultSession(_RecordingSession):
    def __init__(self) -> None:
        super().__init__()
        self.experimental_session_cookie: dict[str, Any] | None = {
            "sessionId": "sess-tool-error"
        }

    @property
    def experimental_session_id(self) -> str | None:
        cookie = self.experimental_session_cookie
        if isinstance(cookie, dict):
            session_id = cookie.get("sessionId")
            if isinstance(session_id, str) and session_id:
                return session_id
        return None

    def set_experimental_session_cookie(self, cookie: dict[str, Any] | None) -> None:
        self.experimental_session_cookie = dict(cookie) if isinstance(cookie, dict) else None

    async def call_tool(self, **kwargs: Any) -> CallToolResult:
        self.last_kwargs = dict(kwargs)
        return CallToolResult(
            isError=True,
            content=[
                TextContent(
                    type="text",
                    text="Session not found",
                )
            ],
        )


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
async def test_execute_on_server_keeps__meta_for_read_resource() -> None:
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
    assert session.last_kwargs.get("_meta") == metadata
    assert "meta" not in session.last_kwargs


@pytest.mark.asyncio
async def test_execute_on_server_marks_rejected_experimental_cookie_invalid() -> None:
    session = _RejectingSession()
    aggregator = MCPAggregator(server_names=[], connection_persistence=True, context=None)
    setattr(aggregator, "_persistent_connection_manager", _FakeConnectionManager(session))
    recorder = _InvalidationRecorder()
    aggregator.experimental_sessions = recorder  # type: ignore[assignment]

    result = await aggregator._execute_on_server(
        server_name="demo",
        operation_type="tools/call",
        operation_name="notebook_read",
        method_name="call_tool",
        method_args={"name": "notebook_read", "arguments": {}},
        error_factory=lambda message: message,
    )

    assert "Failed to call_tool 'notebook_read' on server 'demo'" in result
    assert session.experimental_session_cookie is None
    assert recorder.calls == [
        (
            "demo",
            "sess-rejected",
            "Session not found",
        )
    ]


@pytest.mark.asyncio
async def test_execute_on_server_marks_rejected_cookie_from_tool_error_result() -> None:
    session = _ToolErrorResultSession()
    aggregator = MCPAggregator(server_names=[], connection_persistence=True, context=None)
    setattr(aggregator, "_persistent_connection_manager", _FakeConnectionManager(session))
    recorder = _InvalidationRecorder()
    aggregator.experimental_sessions = recorder  # type: ignore[assignment]

    result = await aggregator._execute_on_server(
        server_name="demo",
        operation_type="tools/call",
        operation_name="notebook_status",
        method_name="call_tool",
        method_args={"name": "notebook_status", "arguments": {}},
    )

    assert isinstance(result, CallToolResult)
    assert result.isError is True
    assert session.experimental_session_cookie is None
    assert recorder.calls == [
        (
            "demo",
            "sess-tool-error",
            "Session not found",
        )
    ]
