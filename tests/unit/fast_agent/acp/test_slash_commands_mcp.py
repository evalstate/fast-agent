import os
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest
from acp.schema import ToolCallProgress, ToolCallStart

from fast_agent.acp.slash.handlers import mcp as mcp_handler_module
from fast_agent.acp.slash_commands import SlashCommandHandler
from fast_agent.commands.mcp_command_intents import MCP_TOP_LEVEL_ACTIONS
from fast_agent.commands.results import CommandOutcome
from fast_agent.core.fastagent import AgentInstance
from fast_agent.mcp.connect_targets import parse_connect_command_text
from fast_agent.mcp.mcp_aggregator import MCPAttachResult, MCPDetachResult
from fast_agent.mcp.oauth_client import OAuthEvent

if TYPE_CHECKING:
    from fast_agent.acp.acp_context import ACPContext
    from fast_agent.core.agent_app import AgentApp
    from fast_agent.interfaces import AgentProtocol


class _Agent:
    acp_commands = {}
    instruction = ""

    def __init__(self) -> None:
        self.config = SimpleNamespace(default=False, model=None)


def test_acp_mcp_handlers_cover_shared_top_level_actions() -> None:
    assert set(mcp_handler_module._MCP_COMMAND_HANDLERS) | {"connect"} == set(
        MCP_TOP_LEVEL_ACTIONS
    )


def test_mcp_connect_tool_call_title_preserves_quoted_target_token() -> None:
    request = parse_connect_command_text('"C:\\Program Files\\Tool\\tool.exe" --flag')

    title = mcp_handler_module._connect_tool_call_title(request)

    assert title == "Connect MCP target 'C:\\Program Files\\Tool\\tool.exe'"


def test_rewrite_connect_progress_ignores_blank_authorization_url() -> None:
    result = mcp_handler_module._rewrite_connect_progress_message(
        cast("SlashCommandHandler", SimpleNamespace(_acp_context=None)),
        message="Open this link to authorize:   ",
        oauth_authorization_url=None,
    )

    assert result.oauth_authorization_url is None
    assert result.message == "Open this link to authorize:   "


def test_summarize_connect_outcome_prefers_structured_metadata() -> None:
    outcome = CommandOutcome()
    outcome.add_message(
        "Attached runtime endpoint.",
        metadata={
            "mcp_connect_status": "connected",
            "mcp_connect_details": "Structured connection details.",
        },
    )

    summary = mcp_handler_module._summarize_connect_outcome(outcome)

    assert summary.has_error is False
    assert summary.completion_details == "Structured connection details."


def test_summarize_connect_outcome_matches_already_attached_case_insensitively() -> None:
    outcome = CommandOutcome()
    outcome.add_message("MCP SERVER IS ALREADY ATTACHED.")

    summary = mcp_handler_module._summarize_connect_outcome(outcome)

    assert summary.has_error is False
    assert summary.completion_details == "MCP SERVER IS ALREADY ATTACHED."


class _App:
    def __init__(self) -> None:
        self._attached = ["local"]
        self.attached_configs: list[object | None] = []

    def _agent(self, _name: str):
        return _Agent()

    def visible_agent_names(self, *, force_include: str | None = None):
        del force_include
        return ["main"]

    def registered_agent_names(self):
        return ["main"]

    def registered_agents(self):
        return {"main": _Agent()}

    def resolve_target_agent_name(self, agent_name: str | None = None):
        return agent_name or "main"

    async def list_prompts(self, namespace: str | None, agent_name: str | None = None):
        del namespace, agent_name
        return {}

    async def list_attached_mcp_servers(self, _agent_name: str) -> list[str]:
        return list(self._attached)

    async def list_configured_detached_mcp_servers(self, _agent_name: str) -> list[str]:
        return ["docs"]

    async def attach_mcp_server(self, _agent_name, server_name, server_config=None, options=None):
        self.attached_configs.append(server_config)
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


class _FailingMcpApp(_App):
    async def list_attached_mcp_servers(self, _agent_name: str) -> list[str]:
        raise AssertionError("app MCP list bypassed callback")

    async def list_configured_detached_mcp_servers(self, _agent_name: str) -> list[str]:
        raise AssertionError("app MCP detached list bypassed callback")

    async def attach_mcp_server(self, _agent_name, server_name, server_config=None, options=None):
        raise AssertionError("app MCP attach bypassed callback")

    async def detach_mcp_server(self, _agent_name, server_name):
        raise AssertionError("app MCP detach bypassed callback")


class _FakeACPContext:
    def __init__(self) -> None:
        self.updates: list[object] = []
        self.session_cwd = None
        self.session_store_scope = "workspace"
        self.session_store_cwd = None

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

    listed = await handler.execute_command("mcp", " LIST ")
    assert "Attached MCP servers" in listed

    invalid_list = await handler.execute_command("mcp", "list demo")
    assert "Usage: /mcp list" in invalid_list

    connected = await handler.execute_command("mcp", "CONNECT --name demo npx demo-server")
    assert "Connected MCP server 'demo'" in connected

    reconnected = await handler.execute_command("mcp", "ReConnect demo")
    assert "Reconnected MCP server 'demo'" in reconnected

    disconnected = await handler.execute_command("mcp", "DISCONNECT demo")
    assert "Disconnected MCP server 'demo'" in disconnected


@pytest.mark.asyncio
async def test_slash_command_mcp_uses_callbacks_not_instance_app() -> None:
    callback_manager = _App()
    instance = AgentInstance(
        app=cast("AgentApp", _FailingMcpApp()),
        agents={"main": cast("AgentProtocol", _Agent())},
        registry_version=0,
    )
    handler = SlashCommandHandler(
        session_id="s1",
        instance=instance,
        primary_agent_name="main",
        attach_mcp_server_callback=callback_manager.attach_mcp_server,
        detach_mcp_server_callback=callback_manager.detach_mcp_server,
        list_attached_mcp_servers_callback=callback_manager.list_attached_mcp_servers,
        list_configured_detached_mcp_servers_callback=(
            callback_manager.list_configured_detached_mcp_servers
        ),
    )

    listed = await handler.execute_command("mcp", "list")
    connected = await handler.execute_command("mcp", "connect --name demo npx demo-server")
    reconnected = await handler.execute_command("mcp", "reconnect demo")
    disconnected = await handler.execute_command("mcp", "disconnect demo")

    assert "Attached MCP servers" in listed
    assert "Connected MCP server 'demo'" in connected
    assert "Reconnected MCP server 'demo'" in reconnected
    assert "Disconnected MCP server 'demo'" in disconnected


@pytest.mark.asyncio
async def test_slash_command_mcp_rejects_extra_reconnect_disconnect_args() -> None:
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

    reconnected = await handler.execute_command("mcp", "reconnect demo extra")
    disconnected = await handler.execute_command("mcp", "disconnect demo extra")

    assert "Usage: /mcp reconnect <server_name>" in reconnected
    assert "Usage: /mcp disconnect <server_name>" in disconnected
    assert app._attached == ["local"]


@pytest.mark.asyncio
async def test_slash_command_mcp_reports_split_errors() -> None:
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

    rendered = await handler.execute_command("mcp", 'reconnect "unterminated')

    assert "Invalid arguments: No closing quotation" in rendered


@pytest.mark.asyncio
async def test_slash_command_mcp_rejects_empty_reconnect_disconnect_server_name() -> None:
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

    reconnected = await handler.execute_command("mcp", 'reconnect ""')
    disconnected = await handler.execute_command("mcp", 'disconnect ""')

    assert "Usage: /mcp reconnect <server_name>" in reconnected
    assert "Usage: /mcp disconnect <server_name>" in disconnected
    assert app._attached == ["local"]


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

    connected = await handler.execute_command("mcp", "connect --name demo npx demo-server")
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
async def test_slash_command_mcp_connect_redacts_auth_in_acp_updates() -> None:
    original_value = os.environ.get("MCP_TOKEN")
    os.environ["MCP_TOKEN"] = "secret-token-value"
    app = _App()
    try:
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

        connected = await handler.execute_command(
            "mcp",
            "connect https://example.com/mcp --name demo --auth $MCP_TOKEN",
        )

        assert "Connected MCP server 'demo'" in connected
        rendered_updates = "\n".join(str(update) for update in acp_context.updates)
        assert "secret-token-value" not in rendered_updates
        assert "--auth" in rendered_updates
        assert "[REDACTED]" in rendered_updates
    finally:
        if original_value is None:
            os.environ.pop("MCP_TOKEN", None)
        else:
            os.environ["MCP_TOKEN"] = original_value


@pytest.mark.asyncio
async def test_slash_command_mcp_connect_preserves_quoted_target_arguments() -> None:
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

    connected = await handler.execute_command(
        "mcp",
        'connect --name docs demo-server --root "My Folder"',
    )

    assert "Connected MCP server 'docs'" in connected
    assert app.attached_configs
    server_config = app.attached_configs[-1]
    assert getattr(server_config, "command", None) == "demo-server"
    assert getattr(server_config, "args", None) == ["--root", "My Folder"]


@pytest.mark.asyncio
async def test_slash_command_mcp_connect_preserves_quoted_windows_path() -> None:
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

    connected = await handler.execute_command(
        "mcp",
        'connect --name docs "C:\\Program Files\\Tool\\tool.exe" --flag',
    )

    assert "Connected MCP server 'docs'" in connected
    assert app.attached_configs
    server_config = app.attached_configs[-1]
    assert getattr(server_config, "command", None) == "C:\\Program Files\\Tool\\tool.exe"
    assert getattr(server_config, "args", None) == ["--flag"]
