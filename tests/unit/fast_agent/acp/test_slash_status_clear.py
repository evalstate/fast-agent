from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest

from fast_agent.acp.acp_aware_mixin import ACPCommand
from fast_agent.acp.slash import dispatch as slash_dispatch
from fast_agent.acp.slash.handlers import status as status_slash_handlers
from fast_agent.acp.slash_commands import SlashCommandHandler
from fast_agent.core.fastagent import AgentInstance

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


class _App:
    pass


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


def test_advertised_session_commands_are_dispatch_routable() -> None:
    command_names = {command.name for command in _handler().get_available_commands()}

    assert command_names <= slash_dispatch.routed_command_names()


def test_reload_session_command_is_dispatch_routable_when_advertised() -> None:
    command_names = {command.name for command in _handler(reload_enabled=True).get_available_commands()}

    assert "reload" in command_names
    assert command_names <= slash_dispatch.routed_command_names()


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

    async def fake_execute(_handler: object, command_name: str, arguments: str) -> str:
        assert command_name == "status"
        assert arguments == ""
        return "built-in status"

    monkeypatch.setattr(slash_dispatch, "execute", fake_execute)

    command_names = {command.name for command in handler.get_available_commands()}
    assert "status" in command_names
    assert "Status" not in command_names
    assert await handler.execute_command("Status", "") == "agent shadow"


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
