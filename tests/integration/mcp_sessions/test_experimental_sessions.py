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
from fast_agent.mcp.mcp_aggregator import MCPAggregator
from fast_agent.mcp_server_registry import ServerRegistry


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _server_settings(server_name: str) -> MCPServerSettings:
    repo_root = _repo_root()
    server_script = repo_root / "examples" / "experimental" / "mcp_sessions" / "session_server.py"
    return MCPServerSettings(
        name=server_name,
        transport="stdio",
        command=sys.executable,
        args=[str(server_script)],
        cwd=str(repo_root),
    )


def _build_context(server_name: str) -> Context:
    registry = ServerRegistry()
    registry.registry = {server_name: _server_settings(server_name)}
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
            assert status_before.experimental_session_features == ["create", "delete", "list"]

            cookie_before = status_before.session_cookie
            assert isinstance(cookie_before, dict)

            cookie_id = cookie_before.get("id")
            assert isinstance(cookie_id, str)
            assert cookie_id

            expected_title = f"fast-agent · {server_name}"
            assert status_before.session_title == expected_title

            result = await aggregator.call_tool(
                "session_probe",
                {"action": "status", "note": "integration"},
            )
            rendered = _tool_text(result)
            assert f"id={cookie_id}" in rendered

            status_after = (await aggregator.collect_server_status())[server_name]
            cookie_after = status_after.session_cookie
            assert isinstance(cookie_after, dict)
            assert cookie_after.get("id") == cookie_id
            assert status_after.session_title == expected_title
    finally:
        await _shutdown_logging_bus()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_experimental_session_revocation_then_reestablishes_cookie() -> None:
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

            initial_id = initial_cookie.get("id")
            assert isinstance(initial_id, str)
            assert initial_id

            revoke_result = await aggregator.call_tool(
                "session_probe",
                {"action": "revoke", "note": "integration"},
            )
            assert "session revoked" in _tool_text(revoke_result)

            status_revoked = (await aggregator.collect_server_status())[server_name]
            assert status_revoked.session_cookie is None
            assert status_revoked.session_title is None

            _ = await aggregator.call_tool(
                "session_probe",
                {"action": "status", "note": "integration"},
            )

            status_reestablished = (await aggregator.collect_server_status())[server_name]
            new_cookie = status_reestablished.session_cookie
            assert isinstance(new_cookie, dict)

            new_id = new_cookie.get("id")
            assert isinstance(new_id, str)
            assert new_id
            assert new_id != initial_id

            assert status_reestablished.experimental_session_supported is True
            assert status_reestablished.experimental_session_features == ["create", "delete", "list"]
    finally:
        await _shutdown_logging_bus()
