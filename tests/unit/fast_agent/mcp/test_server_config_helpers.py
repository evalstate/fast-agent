from __future__ import annotations

from dataclasses import dataclass

from fast_agent.config import MCPServerSettings
from fast_agent.mcp.helpers.server_config_helpers import get_server_config
from fast_agent.mcp.mcp_agent_client_session import MCPAgentClientSession


@dataclass(frozen=True, slots=True)
class _RequestContext:
    session: object


def _session_with_config(server_config: MCPServerSettings | None) -> MCPAgentClientSession:
    session = object.__new__(MCPAgentClientSession)
    session.server_config = server_config
    return session


def test_get_server_config_accepts_mcp_agent_client_session() -> None:
    server_config = MCPServerSettings(name="docs")
    session = _session_with_config(server_config)

    assert get_server_config(session) is server_config


def test_get_server_config_accepts_request_context_session() -> None:
    server_config = MCPServerSettings(name="docs")
    context = _RequestContext(session=_session_with_config(server_config))

    assert get_server_config(context) is server_config


def test_get_server_config_ignores_unsupported_objects() -> None:
    assert get_server_config(object()) is None
    assert get_server_config(_RequestContext(session=object())) is None
