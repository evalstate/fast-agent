from typing import TYPE_CHECKING, cast

import pytest
from acp.schema import ToolCallProgress, ToolCallStart

from fast_agent.acp.slash_commands import SlashCommandHandler
from fast_agent.core.fastagent import AgentInstance
from fast_agent.mcp.experimental_session_client import SessionJarEntry
from fast_agent.mcp.mcp_aggregator import MCPAttachResult, MCPDetachResult
from fast_agent.mcp.oauth_client import OAuthEvent

if TYPE_CHECKING:
    from fast_agent.acp.acp_context import ACPContext
    from fast_agent.core.agent_app import AgentApp
    from fast_agent.interfaces import AgentProtocol


class _Agent:
    acp_commands = {}

    class _SessionClient:
        async def list_jar(self):
            return [
                SessionJarEntry(
                    server_name="local",
                    server_identity="demo-server",
                    target="cmd:python demo.py",
                    cookie={"id": "sess-1"},
                    cookies=(
                        {
                            "id": "sess-1",
                            "title": "Demo",
                            "expiry": None,
                            "updatedAt": "2026-02-23T10:00:00Z",
                            "active": True,
                        },
                    ),
                    last_used_id="sess-1",
                    title="Demo",
                    supported=True,
                    features=("create", "list"),
                    connected=True,
                )
            ]

        async def resolve_server_name(self, server_identifier: str | None):
            del server_identifier
            return "local"

        async def list_sessions(self, server_identifier: str | None):
            del server_identifier
            return "local", [{"id": "sess-1"}]

        async def list_server_cookies(self, server_identifier: str | None):
            del server_identifier
            return "local", "demo-server", "sess-1", [
                {
                    "id": "sess-1",
                    "title": "Demo",
                    "expiry": None,
                    "updatedAt": "2026-02-23T10:00:00Z",
                    "active": True,
                }
            ]

        async def create_session(self, server_identifier: str | None, *, title: str | None = None):
            del server_identifier, title
            return "local", {"id": "sess-created"}

        async def resume_session(self, server_identifier: str | None, *, session_id: str):
            del server_identifier
            return "local", {"id": session_id}

        async def clear_cookie(self, server_identifier: str | None):
            del server_identifier
            return "local"

        async def clear_all_cookies(self):
            return ["local"]

    class _Aggregator:
        def __init__(self) -> None:
            self.experimental_sessions = _Agent._SessionClient()

    def __init__(self) -> None:
        self.aggregator = _Agent._Aggregator()


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
        del server_config
        if options and options.oauth_event_handler is not None:
            await options.oauth_event_handler(
                OAuthEvent(
                    event_type="authorization_url",
                    server_name=server_name,
                    url="https://auth.example.com/authorize?session=1",
                )
            )
            await options.oauth_event_handler(
                OAuthEvent(
                    event_type="wait_start",
                    server_name=server_name,
                )
            )
            await options.oauth_event_handler(
                OAuthEvent(
                    event_type="wait_end",
                    server_name=server_name,
                )
            )
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


class _FakeACPContext:
    def __init__(self) -> None:
        self.updates: list[object] = []

    async def send_session_update(self, update: object) -> None:
        self.updates.append(update)

    async def invalidate_instruction_cache(
        self, agent_name: str | None, new_instruction: str | None
    ) -> None:
        del agent_name, new_instruction

    async def send_available_commands_update(self) -> None:
        return None


@pytest.mark.asyncio
async def test_slash_command_mcp_list_connect_reconnect_disconnect() -> None:
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

    reconnected = await handler.execute_command("mcp", "reconnect demo")
    assert "Reconnected MCP server 'demo'" in reconnected

    disconnected = await handler.execute_command("mcp", "disconnect demo")
    assert "Disconnected MCP server 'demo'" in disconnected


@pytest.mark.asyncio
async def test_slash_command_mcp_connect_sends_acp_progress_updates() -> None:
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
    acp_context = _FakeACPContext()
    handler.set_acp_context(cast("ACPContext", acp_context))

    connected = await handler.execute_command("mcp", "connect npx demo-server --name demo")
    assert "Connected MCP server 'demo'" in connected
    assert len(acp_context.updates) >= 2
    assert any("auth.example.com" in str(update) for update in acp_context.updates)
    assert any(isinstance(update, ToolCallStart) for update in acp_context.updates)
    assert any(isinstance(update, ToolCallProgress) for update in acp_context.updates)
    assert any(
        "Waiting for OAuth callback" in str(update) and "auth.example.com" in str(update)
        for update in acp_context.updates
    )
    assert any("Stop/Cancel" in str(update) for update in acp_context.updates)
    assert any("fast-agent auth login" in str(update) for update in acp_context.updates)
    assert any(
        isinstance(update, ToolCallProgress) and getattr(update, "status", None) == "completed"
        for update in acp_context.updates
    )
    assert any("Connected MCP server 'demo'" in str(update) for update in acp_context.updates)


@pytest.mark.asyncio
async def test_slash_command_mcp_session_jar() -> None:
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

    rendered = await handler.execute_command("mcp", "session jar")
    assert "[ 1]" in rendered
    assert "connected" in rendered
    assert "cookies: 1" in rendered
    assert "▶ sess-1" in rendered
