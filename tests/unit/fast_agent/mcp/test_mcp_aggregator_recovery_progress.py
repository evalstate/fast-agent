import ast
import asyncio
import inspect
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, cast

import pytest
import pytest_asyncio

from fast_agent.config import MCPServerSettings
from fast_agent.context import Context
from fast_agent.core.exceptions import ServerSessionTerminatedError
from fast_agent.core.logging.events import Event
from fast_agent.core.logging.listeners import convert_log_event
from fast_agent.core.logging.transport import AsyncEventBus
from fast_agent.event_progress import ProgressAction, ProgressEvent
from fast_agent.mcp import mcp_aggregator as mcp_aggregator_module
from fast_agent.mcp.mcp_aggregator import MCPAggregator
from fast_agent.mcp_server_registry import ServerRegistry

if TYPE_CHECKING:
    from fast_agent.mcp.mcp_connection_manager import MCPConnectionManager


class _RecordingTransport:
    def __init__(self) -> None:
        self.events: list[Event] = []

    async def send_event(self, event: Event) -> None:
        self.events.append(event)


@pytest_asyncio.fixture
async def recovery_events() -> AsyncIterator[list[Event]]:
    AsyncEventBus.reset()
    transport = _RecordingTransport()
    bus = AsyncEventBus.get(transport=transport)
    await bus.start()
    yield transport.events
    await bus.stop()
    AsyncEventBus.reset()


def _context(*, reconnect_on_disconnect: bool = False) -> Context:
    registry = ServerRegistry()
    registry.registry = {
        "alpha": MCPServerSettings(
            name="alpha",
            reconnect_on_disconnect=reconnect_on_disconnect,
        )
    }
    return Context(server_registry=registry)


async def _progress(events: list[Event]) -> list[ProgressEvent]:
    await asyncio.sleep(0)
    return [progress for event in events if (progress := convert_log_event(event)) is not None]


@pytest.mark.asyncio
async def test_auth_escalation_failure_emits_progress_without_changing_result(
    recovery_events: list[Event],
) -> None:
    class _FailingManager:
        async def reconnect_server(self, server_name, callback_runtime, trigger_oauth=None):
            del server_name, callback_runtime, trigger_oauth
            raise RuntimeError("OAuth callback timed out")

    aggregator = MCPAggregator(
        server_names=["alpha"],
        connection_persistence=True,
        context=_context(),
        name="assistant",
    )
    aggregator._persistent_connection_manager = cast("MCPConnectionManager", _FailingManager())

    async def try_execute(client) -> None:
        raise AssertionError(client)

    recovery = await aggregator._handle_auth_challenge(
        "alpha",
        try_execute,
        lambda message: message,
        RuntimeError("401 Unauthorized"),
    )

    assert recovery.result == "OAuth callback timed out"
    assert recovery.success is False
    progress = await _progress(recovery_events)
    assert [(event.action, event.details) for event in progress] == [
        (
            ProgressAction.CONNECTING,
            "alpha - authorization required; reconnecting with OAuth",
        ),
        (
            ProgressAction.FATAL_ERROR,
            "alpha - OAuth reconnect failed: OAuth callback timed out",
        ),
    ]
    assert all(event.agent_name == "assistant" for event in progress)
    assert all(event.server_name == "alpha" for event in progress)


@pytest.mark.asyncio
async def test_connection_recovery_emits_started_and_succeeded(
    recovery_events: list[Event],
) -> None:
    class _Manager:
        async def reconnect_server(self, server_name, callback_runtime):
            del server_name
            return type(
                "_Connection",
                (),
                {
                    "client": object(),
                    "negotiation": "adopt",
                    "_callback_runtime": callback_runtime,
                },
            )()

    aggregator = MCPAggregator(
        server_names=["alpha"],
        connection_persistence=True,
        context=_context(),
        name="assistant",
    )
    aggregator._persistent_connection_manager = cast("MCPConnectionManager", _Manager())

    async def try_execute(client) -> str:
        assert client is not None
        return "ok"

    recovery = await aggregator._handle_connection_error("alpha", try_execute, None)

    assert recovery.result == "ok"
    assert recovery.success is True
    progress = await _progress(recovery_events)
    assert [(event.action, event.details) for event in progress] == [
        (ProgressAction.CONNECTING, "alpha - reconnecting"),
        (ProgressAction.READY, "alpha - reconnected"),
    ]


@pytest.mark.asyncio
async def test_session_terminated_with_reconnect_disabled_emits_failure(
    recovery_events: list[Event],
) -> None:
    aggregator = MCPAggregator(
        server_names=["alpha"],
        connection_persistence=True,
        context=_context(),
        name="assistant",
    )
    original_error = ServerSessionTerminatedError("alpha")

    async def try_execute(client) -> None:
        raise AssertionError(client)

    recovery = await aggregator._handle_session_terminated(
        "alpha",
        try_execute,
        lambda message: message,
        original_error,
    )

    assert recovery.result == ("MCP server alpha session terminated - reconnection not enabled")
    assert recovery.success is False
    progress = await _progress(recovery_events)
    assert [(event.action, event.details) for event in progress] == [
        (
            ProgressAction.FATAL_ERROR,
            "alpha - session terminated; reconnect disabled (enable reconnect_on_disconnect)",
        )
    ]


@pytest.mark.asyncio
async def test_session_terminated_reconnects_replays_and_records_success(
    recovery_events: list[Event],
) -> None:
    class _Manager:
        async def reconnect_server(self, server_name, callback_runtime):
            del server_name
            return type(
                "_Connection",
                (),
                {
                    "client": object(),
                    "negotiation": "adopt",
                    "_callback_runtime": callback_runtime,
                },
            )()

    aggregator = MCPAggregator(
        server_names=["alpha"],
        connection_persistence=True,
        context=_context(reconnect_on_disconnect=True),
        name="assistant",
    )
    aggregator._persistent_connection_manager = cast("MCPConnectionManager", _Manager())
    replay_count = 0

    async def try_execute(client) -> str:
        nonlocal replay_count
        assert client is not None
        replay_count += 1
        return "ok"

    recovery = await aggregator._handle_session_terminated(
        "alpha",
        try_execute,
        None,
        ServerSessionTerminatedError("alpha"),
    )

    assert recovery.result == "ok"
    assert recovery.success is True
    assert replay_count == 1
    assert aggregator._server_stats["alpha"].reconnect_count == 1
    progress = await _progress(recovery_events)
    assert [(event.action, event.details) for event in progress] == [
        (ProgressAction.CONNECTING, "alpha - session terminated; reconnecting"),
        (ProgressAction.READY, "alpha - reconnected"),
    ]


@pytest.mark.asyncio
async def test_session_reconnect_retry_exhaustion_emits_terminal_failure(
    recovery_events: list[Event],
) -> None:
    class _Manager:
        async def reconnect_server(self, server_name, callback_runtime):
            del server_name
            return type(
                "_Connection",
                (),
                {
                    "client": object(),
                    "negotiation": "adopt",
                    "_callback_runtime": callback_runtime,
                },
            )()

    aggregator = MCPAggregator(
        server_names=["alpha"],
        connection_persistence=True,
        context=_context(reconnect_on_disconnect=True),
        name="assistant",
    )
    aggregator._persistent_connection_manager = cast("MCPConnectionManager", _Manager())

    async def try_execute(client) -> None:
        assert client is not None
        raise ServerSessionTerminatedError("alpha")

    recovery = await aggregator._handle_session_terminated(
        "alpha",
        try_execute,
        lambda message: message,
        ServerSessionTerminatedError("alpha"),
    )

    assert recovery.success is False
    assert isinstance(recovery.result, str)
    assert "even after reconnection" in recovery.result
    progress = await _progress(recovery_events)
    assert [(event.action, event.details) for event in progress] == [
        (ProgressAction.CONNECTING, "alpha - session terminated; reconnecting"),
        (
            ProgressAction.FATAL_ERROR,
            "alpha - session terminated after reconnect; retries exhausted",
        ),
    ]


def test_mcp_aggregator_has_no_direct_console_dependency() -> None:
    tree = ast.parse(inspect.getsource(mcp_aggregator_module))

    assert not any(
        isinstance(node, ast.ImportFrom)
        and node.module == "fast_agent.ui"
        and any(alias.name == "console" for alias in node.names)
        for node in ast.walk(tree)
    )
    assert "console.console.print" not in inspect.getsource(mcp_aggregator_module)
