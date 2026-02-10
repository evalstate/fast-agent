from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest

from fast_agent.acp.server.agent_acp_server import AgentACPServer
from fast_agent.core.fastagent import AgentInstance

if TYPE_CHECKING:
    from fast_agent.core.agent_app import AgentApp
    from fast_agent.interfaces import AgentProtocol


class _Agent:
    acp_commands: dict[str, object] = {}
    instruction = ""


class _App:
    def __init__(self) -> None:
        self._attached = ["local"]

    def _agent(self, _name: str):
        return _Agent()

    def agent_names(self):
        return ["main"]

    async def list_prompts(self, namespace: str | None, agent_name: str | None = None):
        del namespace, agent_name
        return {}

    async def list_attached_mcp_servers(self, _agent_name: str) -> list[str]:
        return list(self._attached)

    async def list_configured_detached_mcp_servers(self, _agent_name: str) -> list[str]:
        return ["docs"]


@pytest.mark.asyncio
async def test_initialize_session_wires_mcp_callbacks_into_slash_handler() -> None:
    app = _App()
    instance = AgentInstance(
        app=cast("AgentApp", app),
        agents={"main": cast("AgentProtocol", _Agent())},
        registry_version=0,
    )

    async def create_instance() -> AgentInstance:
        return instance

    async def dispose_instance(_instance: AgentInstance) -> None:
        return None

    server = AgentACPServer(
        primary_instance=instance,
        create_instance=create_instance,
        dispose_instance=dispose_instance,
        instance_scope="shared",
        server_name="test",
        permissions_enabled=False,
        list_attached_mcp_servers_callback=app.list_attached_mcp_servers,
        list_configured_detached_mcp_servers_callback=app.list_configured_detached_mcp_servers,
    )

    session_state, _ = await server._initialize_session_state(
        "s-1",
        cwd=str(Path.cwd()),
        mcp_servers=[],
    )

    assert session_state.slash_handler is not None
    listed = await session_state.slash_handler.execute_command("mcp", "list")
    assert "Attached MCP servers" in listed
    assert "local" in listed
