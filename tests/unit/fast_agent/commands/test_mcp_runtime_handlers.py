import os
from typing import cast

import pytest

from fast_agent.commands.context import CommandContext
from fast_agent.commands.handlers import mcp_runtime
from fast_agent.commands.results import CommandMessage
from fast_agent.config import MCPServerSettings, MCPSettings, Settings
from fast_agent.mcp.connect_targets import parse_connect_command_text
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

    async def prompt_model_selection(
        self,
        *,
        initial_provider=None,
        default_model=None,
    ):
        del initial_provider, default_model
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


def _request(text: str):
    return parse_connect_command_text(text)


def test_mcp_resource_change_summary_pluralizes_resources() -> None:
    summary = mcp_runtime._format_removed_summary(
        mcp_runtime._McpResourceCounts(tools=1, prompts=2)
    )

    assert summary.plain == "Removed 1 tool and 2 prompts."


def test_mcp_attach_counts_exposes_added_and_refreshed_resource_counts() -> None:
    counts = mcp_runtime._McpAttachCounts(
        tools_added_count=1,
        prompts_added_count=2,
        tools_refreshed_count=3,
        prompts_refreshed_count=4,
    )

    assert counts.added == mcp_runtime._McpResourceCounts(tools=1, prompts=2)
    assert counts.refreshed == mcp_runtime._McpResourceCounts(tools=3, prompts=4)


def test_mcp_attach_counts_rejects_bool_totals() -> None:
    counts = mcp_runtime._mcp_attach_counts(
        MCPAttachResult(
            server_name="demo",
            transport="stdio",
            attached=True,
            already_attached=False,
            tools_added=["demo.echo", "demo.read"],
            prompts_added=["demo.prompt"],
            warnings=[],
            tools_total=True,
            prompts_total=False,
        )
    )

    assert counts.refreshed == mcp_runtime._McpResourceCounts(tools=2, prompts=1)


def test_connect_failure_classifier_requires_explicit_oauth_timeout() -> None:
    timeout = mcp_runtime._classify_connect_failure(
        "OAuth callback wait timed out after 120 seconds"
    )
    assert timeout.oauth_related is True
    assert timeout.oauth_timeout is True

    non_oauth_timeout = mcp_runtime._classify_connect_failure(
        "Startup timed out after 10.0s (non-OAuth startup budget)"
    )
    assert non_oauth_timeout.oauth_related is False
    assert non_oauth_timeout.oauth_timeout is False

    lifetime_policy = mcp_runtime._classify_connect_failure(
        "OAuth token lifetime policy rejected by remote server"
    )
    assert lifetime_policy.oauth_related is True
    assert lifetime_policy.oauth_timeout is False

    mixed_case_timeout = mcp_runtime._classify_connect_failure(
        "OAUTH CALLBACK WAIT TIMED OUT AFTER 120 SECONDS"
    )
    assert mixed_case_timeout.oauth_related is True
    assert mixed_case_timeout.oauth_timeout is True


class _Provider:
    def _agent(self, name: str):
        del name
        return object()

    def resolve_target_agent_name(self, agent_name: str | None = None):
        return agent_name or "main"

    def visible_agent_names(self, *, force_include: str | None = None):
        del force_include
        return ["main"]

    def registered_agent_names(self):
        return ["main"]

    def registered_agents(self):
        return {"main": object()}

    async def list_prompts(self, namespace: str | None, agent_name: str | None = None):
        del namespace, agent_name
        return {}


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


class _StartupTimeoutManager(_Manager):
    async def attach_mcp_server(self, agent_name, server_name, server_config=None, options=None):
        del agent_name, server_name, server_config, options
        raise RuntimeError(
            "MCP Server: 'desktop-commander': Startup timed out after 10.0s "
            "(non-OAuth startup budget)\n\n"
            "Try increasing --timeout or verify server/network startup."
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
def test_parse_connect_request_rejects_non_finite_or_non_positive_timeout(
    raw_timeout: str,
) -> None:
    with pytest.raises(ValueError, match="--timeout"):
        parse_connect_command_text(f"--timeout {raw_timeout} npx demo-server")


def test_runtime_resolves_auth_env_reference(monkeypatch) -> None:
    monkeypatch.setenv("DEMO_TOKEN", "token-from-env")

    parsed = mcp_runtime._resolve_request_auth(
        parse_connect_command_text("https://example.com/api --auth ${DEMO_TOKEN}")
    )

    assert parsed.options.auth_token == "token-from-env"


def test_runtime_resolves_simple_auth_env_reference(monkeypatch) -> None:
    monkeypatch.setenv("DEMO_TOKEN", "token-from-env")

    parsed = mcp_runtime._resolve_request_auth(
        parse_connect_command_text("https://example.com/api --auth $DEMO_TOKEN")
    )

    assert parsed.options.auth_token == "token-from-env"


def test_runtime_resolves_auth_env_reference_with_default(monkeypatch) -> None:
    monkeypatch.delenv("MISSING_TOKEN", raising=False)

    parsed = mcp_runtime._resolve_request_auth(
        parse_connect_command_text("https://example.com/api --auth ${MISSING_TOKEN:default-token}")
    )

    assert parsed.options.auth_token == "default-token"


def test_describe_server_config_source_normalizes_url_values() -> None:
    assert (
        mcp_runtime._describe_server_config_source({"url": " https://example.test/mcp "})
        == "https://example.test/mcp"
    )


def test_describe_server_config_source_omits_blank_values() -> None:
    assert mcp_runtime._describe_server_config_source({"url": "   ", "command": "   "}) is None


def test_describe_server_config_source_falls_back_to_command() -> None:
    assert (
        mcp_runtime._describe_server_config_source(
            {"url": "   ", "command": " python ", "args": ["server.py", "--root", "My Dir"]}
        )
        == "python server.py --root 'My Dir'"
    )


def test_runtime_normalizes_bearer_prefix() -> None:
    parsed = mcp_runtime._resolve_request_auth(
        parse_connect_command_text("https://example.com/api --auth 'Bearer token-from-cli'")
    )

    assert parsed.options.auth_token == "token-from-cli"
    assert mcp_runtime._resolve_auth_token_value(" bEaReR token-from-cli ") == "token-from-cli"


def test_runtime_normalizes_spaced_bearer_prefix() -> None:
    assert mcp_runtime._resolve_auth_token_value(" Bearer   token-from-cli ") == "token-from-cli"


def test_runtime_normalizes_bearer_prefix_before_env_resolution() -> None:
    original_token = os.environ.get("DEMO_TOKEN")
    os.environ["DEMO_TOKEN"] = "token-from-env"
    try:
        parsed = mcp_runtime._resolve_request_auth(
            parse_connect_command_text("https://example.com/api --auth 'Bearer $DEMO_TOKEN'")
        )
    finally:
        if original_token is None:
            os.environ.pop("DEMO_TOKEN", None)
        else:
            os.environ["DEMO_TOKEN"] = original_token

    assert parsed.options.auth_token == "token-from-env"


def test_runtime_rejects_missing_auth_env_reference(monkeypatch) -> None:
    monkeypatch.delenv("MISSING_TOKEN", raising=False)

    with pytest.raises(ValueError, match="Environment variable 'MISSING_TOKEN' is not set"):
        mcp_runtime._resolve_request_auth(
            parse_connect_command_text("https://example.com/api --auth ${MISSING_TOKEN}")
        )


@pytest.mark.asyncio
async def test_handle_mcp_connect_and_disconnect() -> None:
    manager = _Manager()
    ctx = CommandContext(agent_provider=_Provider(), current_agent_name="main", io=_IO())

    connect_outcome = await mcp_runtime.handle_mcp_connect(
        ctx,
        manager=cast("mcp_runtime.McpRuntimeManager", manager),
        agent_name="main",
        request=_request("--name demo npx demo-server"),
    )
    connect_text = "\n".join(str(message.text) for message in connect_outcome.messages)
    assert "Connected MCP server" in connect_text
    assert connect_outcome.messages[0].metadata["mcp_connect_status"] == "connected"
    assert (
        connect_outcome.messages[0].metadata["mcp_connect_details"]
        == "Connected MCP server 'demo' (npx)."
    )
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
async def test_handle_mcp_reconnect_attached_server() -> None:
    manager = _AlreadyAttachedManager()
    manager.attached.append("demo")
    ctx = CommandContext(agent_provider=_Provider(), current_agent_name="main", io=_IO())

    outcome = await mcp_runtime.handle_mcp_reconnect(
        ctx,
        manager=cast("mcp_runtime.McpRuntimeManager", manager),
        agent_name="main",
        server_name="demo",
    )

    message_text = "\n".join(str(message.text) for message in outcome.messages)
    assert "Reconnected MCP server 'demo'." in message_text
    assert outcome.messages[0].metadata["mcp_connect_status"] == "reconnected"
    assert "Refreshed 2 tools and 4 prompts (0 new)." in message_text
    assert manager.last_options is not None
    assert manager.last_options.force_reconnect is True


@pytest.mark.asyncio
async def test_handle_mcp_reconnect_requires_attached_server() -> None:
    manager = _Manager()
    ctx = CommandContext(agent_provider=_Provider(), current_agent_name="main", io=_IO())

    outcome = await mcp_runtime.handle_mcp_reconnect(
        ctx,
        manager=cast("mcp_runtime.McpRuntimeManager", manager),
        agent_name="main",
        server_name="missing",
    )

    message_text = "\n".join(str(message.text) for message in outcome.messages)
    assert "is not currently attached" in message_text


@pytest.mark.asyncio
async def test_handle_mcp_list_reports_attached_and_detached() -> None:
    manager = _Manager()

    outcome = await mcp_runtime.handle_mcp_list(
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
        request=_request("@modelcontextprotocol/server-everything"),
    )

    assert any("Connected MCP server" in str(msg.text) for msg in outcome.messages)
    assert manager.last_config is not None
    assert manager.last_config.command == "npx"
    assert manager.last_config.args == ["@modelcontextprotocol/server-everything"]


@pytest.mark.asyncio
async def test_handle_mcp_connect_configured_name_uses_existing_registry_entry() -> None:
    manager = _Manager()
    progress_updates: list[str] = []
    ctx = CommandContext(
        agent_provider=_Provider(),
        current_agent_name="main",
        io=_IO(),
        settings=Settings(
            mcp=MCPSettings(
                servers={
                    "docs": MCPServerSettings(
                        name="docs",
                        transport="http",
                        url="https://docs.example.com/mcp",
                    )
                }
            )
        ),
    )

    async def _capture_progress(message: str) -> None:
        progress_updates.append(message)

    outcome = await mcp_runtime.handle_mcp_connect(
        ctx,
        manager=cast("mcp_runtime.McpRuntimeManager", manager),
        agent_name="main",
        request=_request("docs"),
        on_progress=_capture_progress,
    )

    assert any(
        "Connected MCP server 'docs' from configuration: https://docs.example.com/mcp."
        in str(msg.text)
        for msg in outcome.messages
    )
    assert any("Connecting MCP server 'docs' from config file" in item for item in progress_updates)
    assert manager.last_config is None


@pytest.mark.asyncio
async def test_handle_mcp_connect_scoped_package_with_args_infers_server_name() -> None:
    manager = _Manager()
    ctx = CommandContext(agent_provider=_Provider(), current_agent_name="main", io=_IO())

    outcome = await mcp_runtime.handle_mcp_connect(
        ctx,
        manager=cast("mcp_runtime.McpRuntimeManager", manager),
        agent_name="main",
        request=_request("@modelcontextprotocol/server-filesystem ."),
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
        request=_request('--name demo demo-server --root "My Folder"'),
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
        request=_request("@modelcontextprotocol/server-filesystem ."),
    )

    message_text = "\n".join(str(msg.text) for msg in outcome.messages)
    assert "already attached" in message_text.lower()
    assert outcome.messages[0].metadata["mcp_connect_status"] == "already_attached"


@pytest.mark.asyncio
async def test_handle_mcp_connect_with_reconnect_reports_reconnected() -> None:
    manager = _AlreadyAttachedManager()
    ctx = CommandContext(agent_provider=_Provider(), current_agent_name="main", io=_IO())

    outcome = await mcp_runtime.handle_mcp_connect(
        ctx,
        manager=cast("mcp_runtime.McpRuntimeManager", manager),
        agent_name="main",
        request=_request("--reconnect @modelcontextprotocol/server-filesystem ."),
    )

    message_text = "\n".join(str(msg.text) for msg in outcome.messages)
    assert "reconnected mcp server" in message_text.lower()
    assert outcome.messages[0].metadata["mcp_connect_status"] == "reconnected"
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
        request=_request("https://example.com/api --auth token123"),
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
        request=_request("https://example.com/api"),
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
        request=_request("https://example.com/api?version=1"),
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
        request=_request("https://demo.hf.space"),
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
        request=_request("--name demo npx demo-server"),
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
        request=_request("--name demo npx demo-server"),
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
        request=_request("https://example.com"),
        on_progress=_capture_progress,
    )

    message_text = "\n".join(str(msg.text) for msg in outcome.messages)
    assert "Failed to connect MCP server" in message_text
    assert "fast-agent auth login" in message_text
    assert "Stop/Cancel" in message_text
    assert any("Failed to connect MCP server" in item for item in progress_updates)


@pytest.mark.asyncio
async def test_emit_connect_progress_ignores_callback_failures() -> None:
    async def failing_progress(_message: str) -> None:
        raise RuntimeError("progress sink failed")

    await mcp_runtime._emit_connect_progress(failing_progress, "Connecting MCP server")


@pytest.mark.asyncio
async def test_handle_mcp_connect_oauth_registration_404_adds_guidance() -> None:
    manager = _OAuthRegistration404Manager()
    ctx = CommandContext(agent_provider=_Provider(), current_agent_name="main", io=_IO())

    outcome = await mcp_runtime.handle_mcp_connect(
        ctx,
        manager=cast("mcp_runtime.McpRuntimeManager", manager),
        agent_name="main",
        request=_request("https://api.githubcopilot.com/mcp/"),
    )

    message_text = "\n".join(str(msg.text) for msg in outcome.messages)
    assert "Failed to connect MCP server" in message_text
    assert "registration returned HTTP 404" in message_text
    assert "--client-metadata-url" in message_text
    assert "--auth <token>" in message_text
    assert "GitHub Copilot MCP" in message_text


@pytest.mark.asyncio
async def test_handle_mcp_connect_non_oauth_timeout_does_not_add_oauth_guidance() -> None:
    manager = _StartupTimeoutManager()
    ctx = CommandContext(agent_provider=_Provider(), current_agent_name="main", io=_IO())

    async def _capture_progress(_message: str) -> None:
        return None

    outcome = await mcp_runtime.handle_mcp_connect(
        ctx,
        manager=cast("mcp_runtime.McpRuntimeManager", manager),
        agent_name="main",
        request=_request("@wonderwhy-er/desktop-commander@latest"),
        on_progress=_capture_progress,
    )

    message_text = "\n".join(str(msg.text) for msg in outcome.messages)
    assert "Failed to connect MCP server" in message_text
    assert "fast-agent auth login" not in message_text
    assert "OAuth could not be completed in this connection mode" not in message_text


@pytest.mark.asyncio
async def test_handle_mcp_connect_defaults_url_oauth_timeout_to_30_seconds() -> None:
    manager = _Manager()
    ctx = CommandContext(agent_provider=_Provider(), current_agent_name="main", io=_IO())

    await mcp_runtime.handle_mcp_connect(
        ctx,
        manager=cast("mcp_runtime.McpRuntimeManager", manager),
        agent_name="main",
        request=_request("https://example.com"),
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
        request=_request("https://example.com --no-oauth"),
    )

    assert manager.last_options is not None
    assert manager.last_options.startup_timeout_seconds == 10.0
