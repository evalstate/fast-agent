from typing import TYPE_CHECKING, cast

import pytest

from fast_agent.acp.slash_commands import SlashCommandHandler
from fast_agent.core.fastagent import AgentInstance
from fast_agent.mcp.mcp_aggregator import MCPAttachResult, MCPDetachResult

if TYPE_CHECKING:
    from fast_agent.core.agent_app import AgentApp
    from fast_agent.interfaces import AgentProtocol


class _Agent:
    acp_commands = {}


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

    async def attach_mcp_server(self, _agent_name, server_name, server_config=None, options=None):
        del server_config, options
        self._attached.append(server_name)
        return MCPAttachResult(
            server_name=server_name,
            transport="stdio",
            attached=True,
            already_attached=False,
            tools_added=[f"{server_name}.echo"],
            prompts_added=[],
            warnings=[],
        )

    async def detach_mcp_server(self, _agent_name, server_name):
        if server_name in self._attached:
            self._attached.remove(server_name)
            return MCPDetachResult(
                server_name=server_name,
                detached=True,
                tools_removed=[f"{server_name}.echo"],
                prompts_removed=[],
            )
        return MCPDetachResult(
            server_name=server_name,
            detached=False,
            tools_removed=[],
            prompts_removed=[],
        )


@pytest.mark.asyncio
async def test_slash_command_mcp_list_connect_disconnect() -> None:
    app = _App()
    instance = AgentInstance(
        app=cast("AgentApp", app),
        agents={"main": cast("AgentProtocol", _Agent())},
        registry_version=0,
    )
    handler = SlashCommandHandler(
        session_id="s1",
        instance=instance,
        primary_agent_name="main",
        attach_mcp_server_callback=app.attach_mcp_server,
        detach_mcp_server_callback=app.detach_mcp_server,
        list_attached_mcp_servers_callback=app.list_attached_mcp_servers,
        list_configured_detached_mcp_servers_callback=app.list_configured_detached_mcp_servers,
    )

    listed = await handler.execute_command("mcp", "list")
    assert "Attached MCP servers" in listed

    connected = await handler.execute_command("mcp", "connect npx demo-server --name demo")
    assert "Connected MCP server 'demo'" in connected

    disconnected = await handler.execute_command("mcp", "disconnect demo")
    assert "Disconnected MCP server 'demo'" in disconnected
