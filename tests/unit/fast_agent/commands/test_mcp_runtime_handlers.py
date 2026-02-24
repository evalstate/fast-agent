import os
from typing import cast

import pytest

from fast_agent.commands.context import CommandContext
from fast_agent.commands.handlers import mcp_runtime
from fast_agent.commands.results import CommandMessage
from fast_agent.mcp.experimental_session_client import SessionJarEntry
from fast_agent.mcp.mcp_aggregator import MCPAttachResult, MCPDetachResult
from fast_agent.mcp.oauth_client import OAuthEvent


class _IO:
    async def emit(self, message: CommandMessage) -> None:
        del message

    async def prompt_text(self, prompt: str, *, default=None, allow_empty=True):
        del prompt, default, allow_empty
        return None

    async def prompt_selection(self, prompt: str, *, options, allow_cancel=False, default=None):
        del prompt, options, allow_cancel, default
        return None

    async def prompt_argument(self, arg_name: str, *, description=None, required=True):
        del arg_name, description, required
        return None

    async def display_history_turn(self, agent_name, turn, *, turn_index=None, total_turns=None):
        del agent_name, turn, turn_index, total_turns

    async def display_history_overview(self, agent_name, history, usage=None):
        del agent_name, history, usage

    async def display_usage_report(self, agents):
        del agents

    async def display_system_prompt(self, agent_name, system_prompt, *, server_count=0):
        del agent_name, system_prompt, server_count


class _Provider:
    def _agent(self, name: str):
        del name
        return object()

    def agent_names(self):
        return ["main"]

    async def list_prompts(self, namespace: str | None, agent_name: str | None = None):
        del namespace, agent_name
        return {}


class _SessionClientStub:
    async def list_jar(self):
        return [
            SessionJarEntry(
                server_name="demo",
                server_identity="demo-server",
                cookie={"id": "sess-123", "data": {"title": "Demo"}},
                cookies=(
                    {
                        "id": "sess-123",
                        "title": "Demo",
                        "expiry": "2026-02-23T12:34:56Z",
                        "updatedAt": "2026-02-23T10:00:00Z",
                        "active": True,
                    },
                    {
                        "id": "sess-abc",
                        "title": None,
                        "expiry": None,
                        "updatedAt": "2026-02-20T10:00:00Z",
                        "active": False,
                    },
                ),
                last_used_id="sess-123",
                title="Demo",
                supported=True,
                features=("create", "list", "delete"),
                connected=True,
            )
        ]

    async def resolve_server_name(self, server_identifier: str | None):
        del server_identifier
        return "demo"

    async def list_sessions(self, server_identifier: str | None):
        del server_identifier
        return "demo", [
            {"id": "sess-123", "data": {"title": "Demo"}, "expiry": "2026-02-23T12:34:56Z"},
            {"id": "sess-abc"},
        ]

    async def list_server_cookies(self, server_identifier: str | None):
        del server_identifier
        return "demo", "demo-server", "sess-123", [
            {
                "id": "sess-123",
                "title": "Demo",
                "expiry": "2026-02-23T12:34:56Z",
                "updatedAt": "2026-02-23T10:00:00Z",
                "active": True,
            },
            {
                "id": "sess-abc",
                "title": None,
                "expiry": None,
                "updatedAt": "2026-02-20T10:00:00Z",
                "active": False,
            },
        ]

    async def create_session(self, server_identifier: str | None, *, title: str | None = None):
        del server_identifier
        return "demo", {"id": "sess-created", "data": {"title": title or "Demo"}}

    async def resume_session(self, server_identifier: str | None, *, session_id: str):
        del server_identifier
        return "demo", {"id": session_id}

    async def clear_cookie(self, server_identifier: str | None):
        del server_identifier
        return "demo"

    async def clear_all_cookies(self):
        return ["demo"]


class _SessionAgent:
    def __init__(self) -> None:
        self.aggregator = type("_Aggregator", (), {"experimental_sessions": _SessionClientStub()})()


class _SessionProvider(_Provider):
    def _agent(self, name: str):
        del name
        return _SessionAgent()


class _InvalidatedSessionClientStub(_SessionClientStub):
    async def list_server_cookies(self, server_identifier: str | None):
        del server_identifier
        return "demo", "demo-server", None, [
            {
                "id": "sess-invalid",
                "title": "Old Session",
                "expiry": None,
                "updatedAt": "2026-02-23T10:00:00Z",
                "active": False,
                "invalidated": True,
            }
        ]


class _InvalidatedSessionAgent:
    def __init__(self) -> None:
        self.aggregator = type(
            "_Aggregator",
            (),
            {"experimental_sessions": _InvalidatedSessionClientStub()},
        )()


class _InvalidatedSessionProvider(_Provider):
    def _agent(self, name: str):
        del name
        return _InvalidatedSessionAgent()


class _Manager:
    def __init__(self) -> None:
        self.attached = ["local"]
        self.last_config = None
        self.last_options = None

    async def attach_mcp_server(self, agent_name, server_name, server_config=None, options=None):
        del agent_name
        self.last_config = server_config
        self.last_options = options
        self.attached.append(server_name)
        return MCPAttachResult(
            server_name=server_name,
            transport="stdio",
            attached=True,
            already_attached=False,
            tools_added=[f"{server_name}.echo"],
            prompts_added=[f"{server_name}.prompt"],
            warnings=[],
        )

    async def detach_mcp_server(self, agent_name, server_name):
        del agent_name
        if server_name in self.attached:
            self.attached.remove(server_name)
            return MCPDetachResult(
                server_name=server_name,
                detached=True,
                tools_removed=[f"{server_name}.echo"],
                prompts_removed=[f"{server_name}.prompt"],
            )
        return MCPDetachResult(
            server_name=server_name,
            detached=False,
            tools_removed=[],
            prompts_removed=[],
        )

    async def list_attached_mcp_servers(self, agent_name):
        del agent_name
        return list(self.attached)

    async def list_configured_detached_mcp_servers(self, agent_name):
        del agent_name
        return ["docs"]


class _AlreadyAttachedManager(_Manager):
    async def attach_mcp_server(self, agent_name, server_name, server_config=None, options=None):
        del agent_name
        self.last_config = server_config
        self.last_options = options
        return MCPAttachResult(
            server_name=server_name,
            transport="stdio",
            attached=True,
            already_attached=True,
            tools_added=[],
            prompts_added=[],
            warnings=[],
            tools_total=2,
            prompts_total=4,
        )


class _OAuthEventManager(_Manager):
    async def attach_mcp_server(self, agent_name, server_name, server_config=None, options=None):
        if options and options.oauth_event_handler is not None:
            await options.oauth_event_handler(
                OAuthEvent(
                    event_type="authorization_url",
                    server_name=server_name,
                    url="https://auth.example.com/authorize?code=demo",
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
        return await super().attach_mcp_server(
            agent_name,
            server_name,
            server_config=server_config,
            options=options,
        )


class _OAuthFailureManager(_Manager):
    async def attach_mcp_server(self, agent_name, server_name, server_config=None, options=None):
        del agent_name, server_name, server_config, options
        raise RuntimeError(
            "OAuth local callback server unavailable and paste fallback is disabled "
            "for this connection mode."
        )


class _OAuthRegistration404Manager(_Manager):
    async def attach_mcp_server(self, agent_name, server_name, server_config=None, options=None):
        del agent_name, server_name, server_config, options
        raise RuntimeError(
            "OAuthRegistrationError: Registration failed: 404 404 page not found "
            "for URL: https://api.githubcopilot.com/mcp/"
        )


class _Always404Manager(_Manager):
    def __init__(self) -> None:
        super().__init__()
        self.url_attempts: list[str] = []

    async def attach_mcp_server(self, agent_name, server_name, server_config=None, options=None):
        del agent_name, server_name, options
        if server_config is not None and getattr(server_config, "url", None):
            self.url_attempts.append(server_config.url)
            raise RuntimeError(f"HTTP Error: 404 Not Found for URL: {server_config.url}")
        raise RuntimeError("expected URL server config")


@pytest.mark.parametrize("raw_timeout", ["nan", "inf", "-inf", "0", "-1"])
def test_parse_connect_input_rejects_non_finite_or_non_positive_timeout(
    raw_timeout: str,
) -> None:
    with pytest.raises(ValueError, match="--timeout"):
        mcp_runtime.parse_connect_input(f"npx demo-server --timeout {raw_timeout}")


def test_parse_connect_input_resolves_auth_env_reference(monkeypatch) -> None:
    monkeypatch.setenv("DEMO_TOKEN", "token-from-env")

    parsed = mcp_runtime.parse_connect_input("https://example.com/api --auth ${DEMO_TOKEN}")

    assert parsed.auth_token == "token-from-env"


def test_parse_connect_input_resolves_simple_auth_env_reference(monkeypatch) -> None:
    monkeypatch.setenv("DEMO_TOKEN", "token-from-env")

    parsed = mcp_runtime.parse_connect_input("https://example.com/api --auth $DEMO_TOKEN")

    assert parsed.auth_token == "token-from-env"


def test_parse_connect_input_resolves_auth_env_reference_with_default(monkeypatch) -> None:
    monkeypatch.delenv("MISSING_TOKEN", raising=False)

    parsed = mcp_runtime.parse_connect_input(
        "https://example.com/api --auth ${MISSING_TOKEN:default-token}"
    )

    assert parsed.auth_token == "default-token"


def test_parse_connect_input_normalizes_bearer_prefix() -> None:
    parsed = mcp_runtime.parse_connect_input("https://example.com/api --auth 'Bearer token-from-cli'")

    assert parsed.auth_token == "token-from-cli"


def test_parse_connect_input_normalizes_bearer_prefix_before_env_resolution() -> None:
    original_token = os.environ.get("DEMO_TOKEN")
    os.environ["DEMO_TOKEN"] = "token-from-env"
    try:
        parsed = mcp_runtime.parse_connect_input("https://example.com/api --auth 'Bearer $DEMO_TOKEN'")
    finally:
        if original_token is None:
            os.environ.pop("DEMO_TOKEN", None)
        else:
            os.environ["DEMO_TOKEN"] = original_token

    assert parsed.auth_token == "token-from-env"


def test_parse_connect_input_rejects_missing_auth_env_reference(monkeypatch) -> None:
    monkeypatch.delenv("MISSING_TOKEN", raising=False)

    with pytest.raises(ValueError, match="Environment variable 'MISSING_TOKEN' is not set"):
        mcp_runtime.parse_connect_input("https://example.com/api --auth ${MISSING_TOKEN}")


@pytest.mark.asyncio
async def test_handle_mcp_connect_and_disconnect() -> None:
    manager = _Manager()
    ctx = CommandContext(agent_provider=_Provider(), current_agent_name="main", io=_IO())

    connect_outcome = await mcp_runtime.handle_mcp_connect(
        ctx,
        manager=cast("mcp_runtime.McpRuntimeManager", manager),
        agent_name="main",
        target_text="npx demo-server --name demo",
    )
    connect_text = "\n".join(str(message.text) for message in connect_outcome.messages)
    assert "Connected MCP server" in connect_text
    assert "Added 1 tool and 1 prompt." in connect_text
    assert "demo.echo" not in connect_text
    assert "demo.prompt" not in connect_text

    disconnect_outcome = await mcp_runtime.handle_mcp_disconnect(
        ctx,
        manager=cast("mcp_runtime.McpRuntimeManager", manager),
        agent_name="main",
        server_name="demo",
    )
    disconnect_text = "\n".join(str(message.text) for message in disconnect_outcome.messages)
    assert "Disconnected MCP server" in disconnect_text
    assert "Removed 1 tool and 1 prompt." in disconnect_text
    assert "demo.echo" not in disconnect_text
    assert "demo.prompt" not in disconnect_text


@pytest.mark.asyncio
async def test_handle_mcp_list_reports_attached_and_detached() -> None:
    manager = _Manager()
    ctx = CommandContext(agent_provider=_Provider(), current_agent_name="main", io=_IO())

    outcome = await mcp_runtime.handle_mcp_list(
        ctx,
        manager=cast("mcp_runtime.McpRuntimeManager", manager),
        agent_name="main",
    )

    message_text = "\n".join(str(message.text) for message in outcome.messages)
    assert "Attached MCP servers" in message_text
    assert "Configured but detached" in message_text


@pytest.mark.asyncio
async def test_handle_mcp_connect_scoped_package_uses_npx_command() -> None:
    manager = _Manager()
    ctx = CommandContext(agent_provider=_Provider(), current_agent_name="main", io=_IO())

    outcome = await mcp_runtime.handle_mcp_connect(
        ctx,
        manager=cast("mcp_runtime.McpRuntimeManager", manager),
        agent_name="main",
        target_text="@modelcontextprotocol/server-everything",
    )

    assert any("Connected MCP server" in str(msg.text) for msg in outcome.messages)
    assert manager.last_config is not None
    assert manager.last_config.command == "npx"
    assert manager.last_config.args == ["@modelcontextprotocol/server-everything"]


@pytest.mark.asyncio
async def test_handle_mcp_connect_configured_name_uses_existing_registry_entry() -> None:
    manager = _Manager()
    ctx = CommandContext(agent_provider=_Provider(), current_agent_name="main", io=_IO())

    outcome = await mcp_runtime.handle_mcp_connect(
        ctx,
        manager=cast("mcp_runtime.McpRuntimeManager", manager),
        agent_name="main",
        target_text="docs",
    )

    assert any("Connected MCP server 'docs' (configured)." in str(msg.text) for msg in outcome.messages)
    assert manager.last_config is None


@pytest.mark.asyncio
async def test_handle_mcp_connect_scoped_package_with_args_infers_server_name() -> None:
    manager = _Manager()
    ctx = CommandContext(agent_provider=_Provider(), current_agent_name="main", io=_IO())

    outcome = await mcp_runtime.handle_mcp_connect(
        ctx,
        manager=cast("mcp_runtime.McpRuntimeManager", manager),
        agent_name="main",
        target_text="@modelcontextprotocol/server-filesystem .",
    )

    message_text = "\n".join(str(msg.text) for msg in outcome.messages)
    assert "server-filesystem" in message_text
    assert manager.last_config is not None
    assert manager.last_config.command == "npx"
    assert manager.last_config.args == ["@modelcontextprotocol/server-filesystem", "."]


@pytest.mark.asyncio
async def test_handle_mcp_connect_preserves_quoted_target_arguments() -> None:
    manager = _Manager()
    ctx = CommandContext(agent_provider=_Provider(), current_agent_name="main", io=_IO())

    outcome = await mcp_runtime.handle_mcp_connect(
        ctx,
        manager=cast("mcp_runtime.McpRuntimeManager", manager),
        agent_name="main",
        target_text='demo-server --root "My Folder" --name demo',
    )

    assert any("Connected MCP server" in str(msg.text) for msg in outcome.messages)
    assert manager.last_config is not None
    assert manager.last_config.command == "demo-server"
    assert manager.last_config.args == ["--root", "My Folder"]


@pytest.mark.asyncio
async def test_handle_mcp_connect_reports_already_attached() -> None:
    manager = _AlreadyAttachedManager()
    ctx = CommandContext(agent_provider=_Provider(), current_agent_name="main", io=_IO())

    outcome = await mcp_runtime.handle_mcp_connect(
        ctx,
        manager=cast("mcp_runtime.McpRuntimeManager", manager),
        agent_name="main",
        target_text="@modelcontextprotocol/server-filesystem .",
    )

    message_text = "\n".join(str(msg.text) for msg in outcome.messages)
    assert "already attached" in message_text.lower()


@pytest.mark.asyncio
async def test_handle_mcp_connect_with_reconnect_reports_reconnected() -> None:
    manager = _AlreadyAttachedManager()
    ctx = CommandContext(agent_provider=_Provider(), current_agent_name="main", io=_IO())

    outcome = await mcp_runtime.handle_mcp_connect(
        ctx,
        manager=cast("mcp_runtime.McpRuntimeManager", manager),
        agent_name="main",
        target_text="@modelcontextprotocol/server-filesystem . --reconnect",
    )

    message_text = "\n".join(str(msg.text) for msg in outcome.messages)
    assert "reconnected mcp server" in message_text.lower()
    assert "refreshed 2 tools and 4 prompts (0 new)." in message_text.lower()
    assert "already attached" not in message_text.lower()


@pytest.mark.asyncio
async def test_handle_mcp_connect_url_uses_cli_url_parsing_for_auth_headers() -> None:
    manager = _Manager()
    ctx = CommandContext(agent_provider=_Provider(), current_agent_name="main", io=_IO())

    outcome = await mcp_runtime.handle_mcp_connect(
        ctx,
        manager=cast("mcp_runtime.McpRuntimeManager", manager),
        agent_name="main",
        target_text="https://example.com/api --auth token123",
    )

    assert any("Connected MCP server" in str(msg.text) for msg in outcome.messages)
    assert manager.last_config is not None
    assert manager.last_config.transport == "http"
    assert manager.last_config.url == "https://example.com/api/mcp"
    assert manager.last_config.headers == {"Authorization": "Bearer token123"}


@pytest.mark.asyncio
async def test_handle_mcp_connect_url_auto_appends_mcp_suffix() -> None:
    manager = _Manager()
    ctx = CommandContext(agent_provider=_Provider(), current_agent_name="main", io=_IO())

    outcome = await mcp_runtime.handle_mcp_connect(
        ctx,
        manager=cast("mcp_runtime.McpRuntimeManager", manager),
        agent_name="main",
        target_text="https://example.com/api",
    )

    assert any("Connected MCP server" in str(msg.text) for msg in outcome.messages)
    assert manager.last_config is not None
    assert manager.last_config.url == "https://example.com/api/mcp"


@pytest.mark.asyncio
async def test_handle_mcp_connect_url_with_query_preserves_explicit_endpoint() -> None:
    manager = _Always404Manager()
    ctx = CommandContext(agent_provider=_Provider(), current_agent_name="main", io=_IO())

    outcome = await mcp_runtime.handle_mcp_connect(
        ctx,
        manager=cast("mcp_runtime.McpRuntimeManager", manager),
        agent_name="main",
        target_text="https://example.com/api?version=1",
    )

    assert any(msg.channel == "error" for msg in outcome.messages)
    assert manager.url_attempts == ["https://example.com/api?version=1"]


@pytest.mark.asyncio
async def test_handle_mcp_connect_hf_url_adds_hf_auth_from_env(monkeypatch) -> None:
    manager = _Manager()
    ctx = CommandContext(agent_provider=_Provider(), current_agent_name="main", io=_IO())

    monkeypatch.setenv("HF_TOKEN", "hf_test_token")
    outcome = await mcp_runtime.handle_mcp_connect(
        ctx,
        manager=cast("mcp_runtime.McpRuntimeManager", manager),
        agent_name="main",
        target_text="https://demo.hf.space",
    )

    assert any("Connected MCP server" in str(msg.text) for msg in outcome.messages)
    assert manager.last_config is not None
    assert manager.last_config.headers is not None
    assert manager.last_config.headers.get("Authorization") == "Bearer hf_test_token"
    assert manager.last_config.headers.get("X-HF-Authorization") == "Bearer hf_test_token"


@pytest.mark.asyncio
async def test_handle_mcp_connect_emits_oauth_progress_and_final_link() -> None:
    manager = _OAuthEventManager()
    ctx = CommandContext(agent_provider=_Provider(), current_agent_name="main", io=_IO())
    progress: list[str] = []

    async def _capture_progress(message: str) -> None:
        progress.append(message)

    outcome = await mcp_runtime.handle_mcp_connect(
        ctx,
        manager=cast("mcp_runtime.McpRuntimeManager", manager),
        agent_name="main",
        target_text="npx demo-server --name demo",
        on_progress=_capture_progress,
    )

    assert any("Open this link to authorize:" in item for item in progress)
    assert any("startup timer paused" in item.lower() for item in progress)
    assert any("OAuth authorization link:" in str(msg.text) for msg in outcome.messages)
    assert manager.last_options is not None
    assert manager.last_options.allow_oauth_paste_fallback is False


@pytest.mark.asyncio
async def test_handle_mcp_connect_enables_oauth_paste_fallback_without_progress_hooks() -> None:
    manager = _Manager()
    ctx = CommandContext(agent_provider=_Provider(), current_agent_name="main", io=_IO())

    await mcp_runtime.handle_mcp_connect(
        ctx,
        manager=cast("mcp_runtime.McpRuntimeManager", manager),
        agent_name="main",
        target_text="npx demo-server --name demo",
    )

    assert manager.last_options is not None
    assert manager.last_options.allow_oauth_paste_fallback is True


@pytest.mark.asyncio
async def test_handle_mcp_connect_oauth_failure_adds_noninteractive_recovery_guidance() -> None:
    manager = _OAuthFailureManager()
    ctx = CommandContext(agent_provider=_Provider(), current_agent_name="main", io=_IO())

    progress_updates: list[str] = []

    async def _capture_progress(message: str) -> None:
        progress_updates.append(message)

    outcome = await mcp_runtime.handle_mcp_connect(
        ctx,
        manager=cast("mcp_runtime.McpRuntimeManager", manager),
        agent_name="main",
        target_text="https://example.com",
        on_progress=_capture_progress,
    )

    message_text = "\n".join(str(msg.text) for msg in outcome.messages)
    assert "Failed to connect MCP server" in message_text
    assert "fast-agent auth login" in message_text
    assert "Stop/Cancel" in message_text
    assert any("Failed to connect MCP server" in item for item in progress_updates)


@pytest.mark.asyncio
async def test_handle_mcp_connect_oauth_registration_404_adds_guidance() -> None:
    manager = _OAuthRegistration404Manager()
    ctx = CommandContext(agent_provider=_Provider(), current_agent_name="main", io=_IO())

    outcome = await mcp_runtime.handle_mcp_connect(
        ctx,
        manager=cast("mcp_runtime.McpRuntimeManager", manager),
        agent_name="main",
        target_text="https://api.githubcopilot.com/mcp/",
    )

    message_text = "\n".join(str(msg.text) for msg in outcome.messages)
    assert "Failed to connect MCP server" in message_text
    assert "registration returned HTTP 404" in message_text
    assert "--client-metadata-url" in message_text
    assert "--auth <token>" in message_text
    assert "GitHub Copilot MCP" in message_text


@pytest.mark.asyncio
async def test_handle_mcp_connect_defaults_url_oauth_timeout_to_30_seconds() -> None:
    manager = _Manager()
    ctx = CommandContext(agent_provider=_Provider(), current_agent_name="main", io=_IO())

    await mcp_runtime.handle_mcp_connect(
        ctx,
        manager=cast("mcp_runtime.McpRuntimeManager", manager),
        agent_name="main",
        target_text="https://example.com",
    )

    assert manager.last_options is not None
    assert manager.last_options.startup_timeout_seconds == 30.0


@pytest.mark.asyncio
async def test_handle_mcp_connect_defaults_url_no_oauth_timeout_to_10_seconds() -> None:
    manager = _Manager()
    ctx = CommandContext(agent_provider=_Provider(), current_agent_name="main", io=_IO())

    await mcp_runtime.handle_mcp_connect(
        ctx,
        manager=cast("mcp_runtime.McpRuntimeManager", manager),
        agent_name="main",
        target_text="https://example.com --no-oauth",
    )

    assert manager.last_options is not None
    assert manager.last_options.startup_timeout_seconds == 10.0


@pytest.mark.asyncio
async def test_handle_mcp_session_jar_renders_compact_rows() -> None:
    ctx = CommandContext(agent_provider=_SessionProvider(), current_agent_name="main", io=_IO())

    outcome = await mcp_runtime.handle_mcp_session(
        ctx,
        agent_name="main",
        action="jar",
        server_identity=None,
        session_id=None,
        title=None,
        clear_all=False,
    )

    message_text = "\n".join(str(msg.text) for msg in outcome.messages)
    assert "[ 1]" in message_text
    assert "demo-server" in message_text
    assert "connected" in message_text
    assert "active: sess-123" in message_text
    assert "v2" in message_text
    assert "cookies:" in message_text


@pytest.mark.asyncio
async def test_handle_mcp_session_list_marks_active_session() -> None:
    ctx = CommandContext(agent_provider=_SessionProvider(), current_agent_name="main", io=_IO())

    outcome = await mcp_runtime.handle_mcp_session(
        ctx,
        agent_name="main",
        action="list",
        server_identity="demo-server",
        session_id=None,
        title=None,
        clear_all=False,
    )

    message_text = "\n".join(str(msg.text) for msg in outcome.messages)
    assert "MCP sessions:" in message_text
    assert "[ 1]" in message_text
    assert "[ 2]" in message_text
    assert "▶ sess-123" in message_text
    assert "• sess-abc" in message_text
    assert "(23/02/26 10:00 → 23/02/26 12:34)" in message_text
    assert "identity: demo-server" in message_text
    assert "cookies: 2" in message_text


@pytest.mark.asyncio
async def test_handle_mcp_session_new_and_clear_all() -> None:
    ctx = CommandContext(agent_provider=_SessionProvider(), current_agent_name="main", io=_IO())

    new_outcome = await mcp_runtime.handle_mcp_session(
        ctx,
        agent_name="main",
        action="new",
        server_identity="demo-server",
        session_id=None,
        title="Demo Run",
        clear_all=False,
    )
    clear_outcome = await mcp_runtime.handle_mcp_session(
        ctx,
        agent_name="main",
        action="clear",
        server_identity=None,
        session_id=None,
        title=None,
        clear_all=True,
    )

    new_text = "\n".join(str(msg.text) for msg in new_outcome.messages)
    clear_text = "\n".join(str(msg.text) for msg in clear_outcome.messages)

    assert "Created experimental session" in new_text
    assert "sess-created" in new_text
    assert "Cleared experimental session cookies" in clear_text


@pytest.mark.asyncio
async def test_handle_mcp_session_list_marks_invalidated_session() -> None:
    ctx = CommandContext(
        agent_provider=_InvalidatedSessionProvider(),
        current_agent_name="main",
        io=_IO(),
    )

    outcome = await mcp_runtime.handle_mcp_session(
        ctx,
        agent_name="main",
        action="list",
        server_identity="demo-server",
        session_id=None,
        title=None,
        clear_all=False,
    )

    message_text = "\n".join(str(msg.text) for msg in outcome.messages)
    assert "○ sess-invalid" in message_text
    assert "invalid" in message_text
