"""Diagnostic contracts at the production transport boundary; no external services."""

import asyncio
import json
from dataclasses import replace
from unittest.mock import AsyncMock, MagicMock

import pytest
from mcp_types import (
    Implementation,
    ListPromptsResult,
    ListResourcesResult,
    ListResourceTemplatesResult,
    ListToolsResult,
    ServerCapabilities,
    Tool,
    ToolsCapability,
)
from typer.testing import CliRunner

from fast_agent.cli.commands.mcp import DiagnosticCallbacks, app, diagnose_server
from fast_agent.config import MCPServerSettings, Settings
from fast_agent.mcp.client_connection import MCPClientConnection


@pytest.fixture
def settings() -> Settings:
    return Settings.model_validate({"mcp": {"servers": {"test": {"command": "unused"}}}})


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    client = MagicMock(spec=MCPClientConnection)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    client.protocol_version = "2025-11-25"
    client.discover_result = None
    client.server_info = Implementation(name="fixture", version="1")
    client.instructions = None
    client.server_capabilities = ServerCapabilities(tools=ToolsCapability())
    client.list_tools = AsyncMock(return_value=ListToolsResult(tools=[]))
    client.list_prompts = AsyncMock(return_value=ListPromptsResult(prompts=[]))
    client.list_resources = AsyncMock(return_value=ListResourcesResult(resources=[]))
    client.list_resource_templates = AsyncMock(
        return_value=ListResourceTemplatesResult(resource_templates=[])
    )
    monkeypatch.setattr(
        "fast_agent.mcp.mcp_connection_manager.create_client_connection",
        lambda **kwargs: client,
    )
    return client


@pytest.mark.asyncio
async def test_discovery_pagination_and_cleanup(settings: Settings, client: MagicMock) -> None:
    client.list_tools.side_effect = [
        ListToolsResult(tools=[Tool(name="one", input_schema={})], next_cursor="next"),
        ListToolsResult(tools=[]),
    ]
    report = await diagnose_server(settings, "test", 1)
    assert report.ok
    assert report.protocol == "2025-11-25"
    assert report.phases["tools"].count == 1
    assert report.phases["tools"].pages == 2
    assert report.phases["tools"].cache == "refresh"
    assert report.phases["prompts"].state == "unsupported"
    assert client.list_tools.await_args_list[1].kwargs["cursor"] == "next"
    client.list_prompts.assert_not_called()
    client.call_tool.assert_not_called()
    client.__aexit__.assert_awaited_once()
    assert all(phase.seconds >= 0 for phase in report.phases.values())


@pytest.mark.asyncio
async def test_repeated_cursor_is_failure(settings: Settings, client: MagicMock) -> None:
    client.list_tools.return_value = ListToolsResult(tools=[], next_cursor="secret")
    report = await diagnose_server(settings, "test", 1)
    assert not report.ok
    assert report.phases["tools"].pages == 2
    assert "secret" not in str(report)
    client.__aexit__.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["connection", "tools"])
async def test_timeout_releases_client(settings: Settings, client: MagicMock, phase: str) -> None:
    async def block(*args, **kwargs):
        await asyncio.sleep(10)

    if phase == "connection":
        client.__aenter__.side_effect = block
    else:
        client.list_tools.side_effect = block
    report = await asyncio.wait_for(diagnose_server(settings, "test", 0.3), 1)
    assert not report.ok
    assert report.phases[phase].state in {"timeout", "error"}
    assert report.phases["cleanup"].state == "ok"
    if phase == "tools":
        client.__aexit__.assert_awaited_once()


@pytest.mark.asyncio
async def test_errors_redacted(settings: Settings, client: MagicMock) -> None:
    client.list_tools.side_effect = RuntimeError("https://user:password@host/?token=SECRET")
    report = await diagnose_server(settings, "test", 1)
    assert not report.ok
    assert "SECRET" not in str(report)
    assert "password" not in str(report)
    assert "fast-agent check" in (report.phases["tools"].error or "")


def test_cloned_callbacks_cannot_sample() -> None:
    callbacks = DiagnosticCallbacks(
        server_name="test", server_config=MCPServerSettings(command="x")
    )
    for runtime in [callbacks, replace(callbacks)]:
        assert runtime.sampling_callback is None
        assert runtime.sampling_capabilities is None
        assert runtime.elicitation_callback is None


@pytest.mark.parametrize("timeout", ["0", "-1", "nan", "inf"])
def test_invalid_timeout(timeout: str) -> None:
    result = CliRunner().invoke(app, ["diagnose", "test", "--timeout", timeout])
    assert result.exit_code == 2


def test_json_unknown_server(monkeypatch: pytest.MonkeyPatch, settings: Settings) -> None:
    monkeypatch.setattr("fast_agent.cli.commands.mcp.get_settings", lambda _: settings)
    result = CliRunner().invoke(app, ["diagnose", "missing", "--json"])
    assert result.exit_code == 1
    assert json.loads(result.stdout)["phases"]["configuration"]["state"] == "error"


def test_root_registration() -> None:
    from fast_agent.cli.main import app as root_app

    result = CliRunner().invoke(root_app, ["mcp", "diagnose", "--help"])
    assert result.exit_code == 0
    assert "--timeout" in result.stdout


@pytest.mark.asyncio
async def test_all_advertised_lists(settings: Settings, client: MagicMock) -> None:
    client.server_capabilities = ServerCapabilities.model_validate(
        {"tools": {}, "prompts": {}, "resources": {}}
    )
    report = await diagnose_server(settings, "test", 1)
    assert report.ok
    for method in [
        client.list_tools,
        client.list_prompts,
        client.list_resources,
        client.list_resource_templates,
    ]:
        method.assert_awaited_once_with(cursor=None, cache_mode="refresh")
    client.call_tool.assert_not_called()
    client.read_resource.assert_not_called()
    client.get_prompt.assert_not_called()


def test_json_success(
    monkeypatch: pytest.MonkeyPatch, settings: Settings, client: MagicMock
) -> None:
    monkeypatch.setattr("fast_agent.cli.commands.mcp.get_settings", lambda _: settings)
    result = CliRunner().invoke(app, ["diagnose", "test", "--json"])
    assert result.exit_code == 0
    assert json.loads(result.stdout)["ok"] is True
    assert result.stderr == ""


def test_real_stdio_failure_is_redacted(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    settings = Settings.model_validate(
        {
            "mcp": {
                "servers": {
                    "test": {
                        "command": sys.executable,
                        "args": [
                            "-c",
                            "import sys; print('TOKEN_SECRET', file=sys.stderr); sys.exit(1)",
                        ],
                    }
                }
            }
        }
    )
    monkeypatch.setattr("fast_agent.cli.commands.mcp.get_settings", lambda _: settings)
    result = CliRunner().invoke(app, ["diagnose", "test", "--json", "--timeout", "1"])
    assert result.exit_code == 1
    assert json.loads(result.stdout)["ok"] is False
    assert "TOKEN_SECRET" not in result.output


@pytest.mark.parametrize("json_output", [False, True])
def test_cli_reports_safe_production_details(
    monkeypatch: pytest.MonkeyPatch, settings: Settings, client: MagicMock, json_output: bool
) -> None:
    from fast_agent.core.exceptions import ServerInitializationError

    error = ServerInitializationError(
        "Server failed",
        "Recent stderr from stdio server:\nModuleNotFoundError: missing_widget\n"
        "Authorization: Bearer sensitive\nhttps://user:sensitive@[broken/?token=sensitive\n"
        "\x1b[31m[bold]unsafe markup",
        server_name="test",
    )
    error.__cause__ = ValueError("Invalid discovery response")
    client.list_tools.side_effect = error
    monkeypatch.setattr("fast_agent.cli.commands.mcp.get_settings", lambda _: settings)
    result = CliRunner().invoke(app, ["diagnose", "test", *(["--json"] if json_output else [])])
    assert result.exit_code == 1
    text = json.loads(result.stdout)["phases"]["tools"]["error"] if json_output else result.stdout
    assert "ServerInitializationError" in text
    assert "missing_widget" in text
    assert "ValueError: Invalid discovery response" in text
    assert "sensitive" not in result.output
    assert "[bold]" not in text
    assert "\x1b" not in text
    client.__aexit__.assert_awaited_once()


@pytest.mark.asyncio
async def test_timeout_cancels_discovery_before_cleanup(
    settings: Settings, client: MagicMock
) -> None:
    cancelled = asyncio.Event()

    async def blocked_discovery(*, cursor: str | None, cache_mode: str) -> ListToolsResult:
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()
        return ListToolsResult(tools=[])

    client.list_tools.side_effect = blocked_discovery
    report = await asyncio.wait_for(diagnose_server(settings, "test", 0.5), 2)
    assert cancelled.is_set()
    assert report.phases["tools"].state == "timeout"
    assert "TimeoutError" in (report.phases["tools"].error or "")
    assert report.phases["cleanup"].state == "ok"
    assert "prompts" not in report.phases
    client.__aexit__.assert_awaited_once()
