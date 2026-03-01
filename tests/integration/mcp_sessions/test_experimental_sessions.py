from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest
from mcp.types import CallToolResult, TextContent

from fast_agent.config import MCPServerSettings
from fast_agent.context import Context
from fast_agent.core.logging.logger import LoggingConfig
from fast_agent.core.logging.transport import AsyncEventBus
from fast_agent.mcp.mcp_agent_client_session import MCPAgentClientSession
from fast_agent.mcp.mcp_aggregator import MCPAggregator
from fast_agent.mcp_server_registry import ServerRegistry


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]



def _server_settings(server_name: str, *, script: str = "session_server.py") -> MCPServerSettings:
    repo_root = _repo_root()
    server_script = repo_root / "examples" / "experimental" / "mcp_sessions" / script
    return MCPServerSettings(
        name=server_name,
        transport="stdio",
        command=sys.executable,
        args=[str(server_script)],
        cwd=str(repo_root),
    )



def _build_context(server_name: str, *, script: str = "session_server.py") -> Context:
    registry = ServerRegistry()
    registry.registry = {server_name: _server_settings(server_name, script=script)}
    return Context(server_registry=registry)



def _tool_text(result: CallToolResult) -> str:
    parts = [
        item.text
        for item in result.content
        if isinstance(item, TextContent) and item.text.strip()
    ]
    return "\n".join(parts)


async def _shutdown_logging_bus() -> None:
    await LoggingConfig.shutdown()
    await AsyncEventBus.get().stop()
    await asyncio.sleep(0.05)
    AsyncEventBus.reset()


async def _live_session(aggregator: MCPAggregator, server_name: str) -> MCPAgentClientSession:
    manager = aggregator._require_connection_manager()  # noqa: SLF001
    server_conn = await manager.get_server(
        server_name,
        client_session_factory=aggregator._create_session_factory(server_name),  # noqa: SLF001
    )
    session = server_conn.session
    if isinstance(session, MCPAgentClientSession):
        return session
    raise RuntimeError("Expected MCPAgentClientSession")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_experimental_session_auto_create_and_cookie_echo() -> None:
    server_name = "experimental_sessions"
    context = _build_context(server_name)
    aggregator = MCPAggregator(
        server_names=[server_name],
        connection_persistence=True,
        context=context,
        name="integration-agent",
    )

    try:
        async with aggregator:
            status_before = (await aggregator.collect_server_status())[server_name]

            assert status_before.experimental_session_supported is True
            assert status_before.experimental_session_features == ["create", "delete"]

            cookie_before = status_before.session_cookie
            assert isinstance(cookie_before, dict)

            session_id = cookie_before.get("sessionId")
            assert isinstance(session_id, str)
            assert session_id

            state_before = cookie_before.get("state")
            assert isinstance(state_before, str)
            assert state_before

            result = await aggregator.call_tool(
                "session_probe",
                {"note": "integration"},
            )
            rendered = _tool_text(result)
            assert f"id={session_id}" in rendered

            status_after = (await aggregator.collect_server_status())[server_name]
            cookie_after = status_after.session_cookie
            assert isinstance(cookie_after, dict)
            assert cookie_after.get("sessionId") == session_id

            state_after = cookie_after.get("state")
            assert isinstance(state_after, str)
            assert state_after
            assert state_after != state_before
    finally:
        await _shutdown_logging_bus()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_experimental_session_delete_then_recreate() -> None:
    server_name = "experimental_sessions"
    context = _build_context(server_name)
    aggregator = MCPAggregator(
        server_names=[server_name],
        connection_persistence=True,
        context=context,
        name="integration-agent",
    )

    try:
        async with aggregator:
            status_initial = (await aggregator.collect_server_status())[server_name]
            initial_cookie = status_initial.session_cookie
            assert isinstance(initial_cookie, dict)

            initial_id = initial_cookie.get("sessionId")
            assert isinstance(initial_id, str)
            assert initial_id

            live_session = await _live_session(aggregator, server_name)
            deleted = await live_session.experimental_session_delete()
            assert deleted is True

            status_deleted = (await aggregator.collect_server_status())[server_name]
            assert status_deleted.session_cookie is None

            post_delete = await aggregator.call_tool(
                "session_probe",
                {"note": "after-delete"},
            )
            assert post_delete.isError is True
            assert "Session not found" in _tool_text(post_delete)

            recreated = await live_session.experimental_session_create()
            assert isinstance(recreated, dict)

            recreated_id = recreated.get("sessionId")
            assert isinstance(recreated_id, str)
            assert recreated_id
            assert recreated_id != initial_id

            result = await aggregator.call_tool(
                "session_probe",
                {"note": "after-recreate"},
            )
            assert "session active" in _tool_text(result)
    finally:
        await _shutdown_logging_bus()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_experimental_session_cookie_omits_state_when_server_does_not_send_it() -> None:
    server_name = "experimental_sessions_notebook"
    context = _build_context(server_name, script="notebook_server.py")
    aggregator = MCPAggregator(
        server_names=[server_name],
        connection_persistence=True,
        context=context,
        name="integration-agent",
    )

    try:
        async with aggregator:
            status_before = (await aggregator.collect_server_status())[server_name]
            cookie_before = status_before.session_cookie
            assert isinstance(cookie_before, dict)
            assert isinstance(cookie_before.get("sessionId"), str)
            assert "state" not in cookie_before

            result = await aggregator.call_tool(
                "notebook_status",
                {},
            )
            assert "Session" in _tool_text(result)

            status_after = (await aggregator.collect_server_status())[server_name]
            cookie_after = status_after.session_cookie
            assert isinstance(cookie_after, dict)
            assert cookie_after.get("sessionId") == cookie_before.get("sessionId")
            assert "state" not in cookie_after
    finally:
        await _shutdown_logging_bus()
