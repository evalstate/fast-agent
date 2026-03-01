from __future__ import annotations

from types import SimpleNamespace

import pytest
from anyio import Lock

from fast_agent.config import MCPServerSettings
from fast_agent.context import Context
from fast_agent.mcp.mcp_aggregator import MCPAggregator
from fast_agent.mcp_server_registry import ServerRegistry


def _build_context(configs: dict[str, MCPServerSettings]) -> Context:
    registry = ServerRegistry()
    registry.registry = configs
    return Context(server_registry=registry)


class _SessionStub:
    def __init__(self) -> None:
        self.client_info = SimpleNamespace(name="fast-agent-mcp", version="1.0.0")
        self.effective_elicitation_mode = "none"
        self.experimental_session_supported = True
        self.experimental_session_features = ("create", "delete")
        self.experimental_session_cookie = {
            "sessionId": "sess-cookie-1",
            "state": "state-1",
        }
        self.experimental_session_title = None


class _ServerConnStub:
    def __init__(self, config: MCPServerSettings) -> None:
        self.server_implementation = SimpleNamespace(name="demo-server", version="0.1.0")
        self.server_capabilities = None
        self.client_capabilities = {"experimental": {"experimental/sessions": {}}}
        self.session = _SessionStub()
        self._initialized_event = SimpleNamespace(is_set=lambda: True)
        self._error_message = None
        self.server_instructions_available = False
        self.server_instructions_enabled = True
        self.server_instructions = None
        self.server_config = config
        self.session_id = "local"
        self._get_session_id_cb = None
        self.transport_metrics = None
        self._ping_ok_count = 0
        self._ping_fail_count = 0
        self._ping_consecutive_failures = 0
        self._ping_last_ok_at = None
        self._ping_last_fail_at = None
        self._ping_last_error = None

    def is_healthy(self) -> bool:
        return True

    def build_ping_activity_buckets(self, _bucket_seconds: int, bucket_count: int) -> list[str]:
        return ["none"] * bucket_count


class _ManagerStub:
    def __init__(self, server_conn: _ServerConnStub) -> None:
        self._lock = Lock()
        self.running_servers = {"demo": server_conn}


@pytest.mark.asyncio
async def test_collect_server_status_includes_experimental_session_cookie() -> None:
    config = MCPServerSettings(name="demo", transport="stdio", command="echo")
    context = _build_context({"demo": config})

    aggregator = MCPAggregator(
        server_names=["demo"],
        connection_persistence=True,
        context=context,
    )
    aggregator.initialized = True

    server_conn = _ServerConnStub(config)
    manager = _ManagerStub(server_conn)
    setattr(aggregator, "_persistent_connection_manager", manager)

    status_map = await aggregator.collect_server_status()
    status = status_map["demo"]

    assert status.experimental_session_supported is True
    assert status.experimental_session_features == ["create", "delete"]
    assert status.session_cookie == {
        "sessionId": "sess-cookie-1",
        "state": "state-1",
    }
    assert status.session_title is None
    assert status.session_id == "local"
