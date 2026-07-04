from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest

from fast_agent.acp.acp_aware_mixin import ACPCommand
from fast_agent.acp.slash.handlers import status as status_slash_handlers
from fast_agent.acp.slash_commands import SlashCommandHandler
from fast_agent.config import Settings, get_settings, update_global_settings
from fast_agent.core.fastagent import AgentInstance
from fast_agent.tools.environment_config import LocalEnvironmentSpec
from fast_agent.tools.execution_environment import ShellRuntimeInfo

if TYPE_CHECKING:
    from fast_agent.core.agent_app import AgentApp
    from fast_agent.interfaces import AgentProtocol


class _StatusHandler:
    _created_at = 0.0
    client_info = None
    client_capabilities = None
    protocol_version = None
    instance = None

    def _get_current_agent(self) -> None:
        return None


class _ShadowingAgent:
    @property
    def acp(self):
        return None

    @property
    def is_acp_mode(self) -> bool:
        return True

    @property
    def acp_commands(self) -> dict[str, ACPCommand]:
        async def shadowed(_arguments: str) -> str:
            return "agent shadow"

        return {
            "Status": ACPCommand(
                description="shadow built-in status",
                handler=shadowed,
            )
        }

    def acp_mode_info(self):
        return None


class _AllowlistAgent:
    @property
    def acp(self):
        return None

    @property
    def is_acp_mode(self) -> bool:
        return True

    @property
    def acp_commands(self) -> dict[str, ACPCommand]:
        return {}

    @property
    def acp_session_commands_allowlist(self) -> set[str] | None:
        return {"status"}

    def acp_mode_info(self):
        return None


class _App:
    pass


class _ShellRuntime:
    def runtime_info(self) -> ShellRuntimeInfo:
        return ShellRuntimeInfo(
            name="bash",
            kind="local",
            environment_name="workspace",
        )


class _EnvironmentAgent:
    name = "main"
    context = SimpleNamespace(config=None, session_manager=None)

    @property
    def shell_runtime(self) -> _ShellRuntime:
        return _ShellRuntime()


def _handler(*, reload_enabled: bool = False) -> SlashCommandHandler:
    async def _reload() -> bool:
        return True

    agent = cast("AgentProtocol", SimpleNamespace(acp_commands={}))
    return SlashCommandHandler(
        session_id="s1",
        instance=AgentInstance(
            app=cast("AgentApp", _App()),
            agents={"main": agent},
            registry_version=0,
        ),
        primary_agent_name="main",
        reload_callback=_reload if reload_enabled else None,
    )


def test_reload_session_command_is_advertised_only_when_available() -> None:
    default_command_names = {command.name for command in _handler().get_available_commands()}
    command_names = {
        command.name for command in _handler(reload_enabled=True).get_available_commands()
    }

    assert "reload" not in default_command_names
    assert "reload" in command_names


@pytest.mark.asyncio
async def test_environment_session_command_lists_configured_environments() -> None:
    settings = Settings(
        environments={"workspace": LocalEnvironmentSpec(cwd=".")},
        default_environment="workspace",
    )
    old_settings = get_settings()
    handler = SlashCommandHandler(
        session_id="s1",
        instance=AgentInstance(
            app=cast("AgentApp", _App()),
            agents={"main": cast("AgentProtocol", _EnvironmentAgent())},
            registry_version=0,
        ),
        primary_agent_name="main",
    )

    try:
        update_global_settings(settings)
        rendered = await handler.execute_command("environment", "")
    finally:
        update_global_settings(old_settings)

    assert "# environments" in rendered
    assert "| `workspace` | `local` | yes |" in rendered
    assert "Active runtime: `local / bash`" in rendered


@pytest.mark.asyncio
async def test_acp_session_command_allowlist_filters_advertisement_and_execution() -> None:
    handler = SlashCommandHandler(
        session_id="s1",
        instance=AgentInstance(
            app=cast("AgentApp", _App()),
            agents={"main": cast("AgentProtocol", _AllowlistAgent())},
            registry_version=0,
        ),
        primary_agent_name="main",
    )

    command_names = {command.name for command in handler.get_available_commands()}

    assert command_names == {"status"}
    assert "Unknown command: /tools" in await handler.execute_command("tools", "")


@pytest.mark.asyncio
async def test_exact_case_agent_status_takes_precedence_over_mixed_case_builtin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = _ShadowingAgent()
    handler = SlashCommandHandler(
        session_id="s1",
        instance=AgentInstance(
            app=cast("AgentApp", _App()),
            agents={"main": cast("AgentProtocol", agent)},
            registry_version=0,
        ),
        primary_agent_name="main",
    )

    async def fake_status(_handler: SlashCommandHandler, arguments: str | None = None) -> str:
        assert arguments == ""
        return "built-in status"

    monkeypatch.setattr(status_slash_handlers, "handle_status", fake_status)

    command_names = {command.name for command in handler.get_available_commands()}
    assert "status" in command_names
    assert "Status" not in command_names
    assert await handler.execute_command("Status", "") == "agent shadow"
    assert await handler.execute_command("status", "") == "built-in status"


@pytest.mark.asyncio
async def test_handle_status_defaults_unknown_version_when_package_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        status_slash_handlers,
        "get_version",
        lambda _package: (_ for _ in ()).throw(PackageNotFoundError("missing")),
    )
    monkeypatch.setattr(status_slash_handlers.time, "time", lambda: 1.0)

    output = await status_slash_handlers.handle_status(
        cast("SlashCommandHandler", _StatusHandler()),
        "",
    )

    assert "# fast-agent ACP status" in output
    assert "fast-agent-mcp: unknown" in output


@pytest.mark.asyncio
async def test_handle_status_does_not_hide_unexpected_version_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        status_slash_handlers,
        "get_version",
        lambda _package: (_ for _ in ()).throw(RuntimeError("metadata failed")),
    )

    with pytest.raises(RuntimeError, match="metadata failed"):
        await status_slash_handlers.handle_status(
            cast("SlashCommandHandler", _StatusHandler()),
            "",
        )
