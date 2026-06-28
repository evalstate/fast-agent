from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.command_actions import PluginCommandActionResult, PluginCommandActionSpec
from fast_agent.commands.results import CommandOutcome
from fast_agent.ui.command_payloads import (
    CommandPayload,
    CommandsCommand,
    McpListCommand,
    ModelVerbosityCommand,
    SkillsCommand,
    is_command_payload,
)
from fast_agent.ui.interactive import command_dispatch
from fast_agent.ui.prompt import parse_special_input

if TYPE_CHECKING:
    from fast_agent.core.agent_app import AgentApp
    from fast_agent.ui.interactive_prompt import InteractivePrompt


class _Agent:
    def __init__(self, config: AgentConfig) -> None:
        self.name = config.name
        self.config = config


class _Provider:
    plugin_command_base_path = None

    def __init__(
        self,
        *,
        agent_commands: dict[str, PluginCommandActionSpec] | None = None,
        plugin_commands: dict[str, PluginCommandActionSpec] | None = None,
    ) -> None:
        self.agent = _Agent(AgentConfig(name="default", commands=agent_commands))
        self.plugin_commands = plugin_commands

    def get_agent(self, name: str) -> _Agent | None:
        return self.agent if name == "default" else None

    def attach_mcp_server(self, *args: object, **kwargs: object) -> None:
        del args, kwargs

    def detach_mcp_server(self, *args: object, **kwargs: object) -> None:
        del args, kwargs

    def list_attached_mcp_servers(self) -> list[str]:
        return []

    def list_configured_detached_mcp_servers(self) -> list[str]:
        return []


async def _dispatch_raw(
    raw_input: str,
    provider: _Provider,
) -> command_dispatch.DispatchResult:
    payload = parse_special_input(raw_input)
    assert is_command_payload(payload)
    return await command_dispatch.dispatch_command_payload(
        cast("InteractivePrompt", object()),
        payload,
        prompt_provider=cast("AgentApp", provider),
        agent="default",
        available_agents=["default"],
        available_agents_set={"default"},
        merge_pinned_agents=lambda names: names,
    )


@pytest.fixture
def patched_context(monkeypatch: pytest.MonkeyPatch) -> list[CommandOutcome]:
    emitted: list[CommandOutcome] = []
    context = SimpleNamespace(settings=None)
    monkeypatch.setattr(
        command_dispatch,
        "build_command_context",
        lambda provider, agent, **_kwargs: context,
    )

    async def collect_outcome(_context: object, outcome: CommandOutcome) -> None:
        emitted.append(outcome)

    monkeypatch.setattr(command_dispatch, "emit_command_outcome", collect_outcome)
    return emitted


def _route_with_handler(payload_type: type[CommandPayload], handler: Any) -> object:
    route = command_dispatch._COMMAND_OUTCOME_ROUTE_BY_PAYLOAD_TYPE[payload_type]
    return replace(route, handler=handler)


@pytest.mark.asyncio
async def test_parse_and_dispatch_use_route_registry_for_plain_outcome_commands(
    monkeypatch: pytest.MonkeyPatch,
    patched_context: list[CommandOutcome],
) -> None:
    calls: list[tuple[str, dict[str, object | None]]] = []

    async def fake_commands(
        _context: object,
        *,
        agent_name: str,
        argument: str | None,
    ) -> CommandOutcome:
        calls.append(("commands", {"agent": agent_name, "argument": argument}))
        return CommandOutcome()

    async def fake_skills(
        _context: object,
        *,
        agent_name: str,
        action: str | None,
        argument: str | None,
    ) -> CommandOutcome:
        calls.append(
            ("skills", {"agent": agent_name, "action": action, "argument": argument})
        )
        return CommandOutcome()

    async def fake_mcp_list(*, manager: object, agent_name: str) -> CommandOutcome:
        calls.append(("mcp-list", {"manager": manager, "agent": agent_name}))
        return CommandOutcome()

    async def fake_model_verbosity(
        _context: object,
        *,
        agent_name: str,
        value: str | None,
    ) -> CommandOutcome:
        calls.append(("model-verbosity", {"agent": agent_name, "value": value}))
        return CommandOutcome()

    monkeypatch.setitem(
        command_dispatch._COMMAND_OUTCOME_ROUTE_BY_PAYLOAD_TYPE,
        CommandsCommand,
        _route_with_handler(CommandsCommand, fake_commands),
    )
    monkeypatch.setitem(
        command_dispatch._COMMAND_OUTCOME_ROUTE_BY_PAYLOAD_TYPE,
        SkillsCommand,
        _route_with_handler(SkillsCommand, fake_skills),
    )
    monkeypatch.setitem(
        command_dispatch._COMMAND_OUTCOME_ROUTE_BY_PAYLOAD_TYPE,
        McpListCommand,
        _route_with_handler(McpListCommand, fake_mcp_list),
    )
    monkeypatch.setitem(
        command_dispatch._COMMAND_OUTCOME_ROUTE_BY_PAYLOAD_TYPE,
        ModelVerbosityCommand,
        _route_with_handler(ModelVerbosityCommand, fake_model_verbosity),
    )

    provider = _Provider()

    assert (await _dispatch_raw("/commands tools", provider)).handled is True
    assert (await _dispatch_raw("/skills search docker", provider)).handled is True
    assert (await _dispatch_raw("/mcp list", provider)).handled is True
    assert (await _dispatch_raw("/model verbosity high", provider)).handled is True

    assert calls == [
        ("commands", {"agent": "default", "argument": "tools"}),
        ("skills", {"agent": "default", "action": "search", "argument": "docker"}),
        ("mcp-list", {"manager": provider, "agent": "default"}),
        ("model-verbosity", {"agent": "default", "value": "high"}),
    ]
    assert len(patched_context) == 4


@pytest.mark.asyncio
async def test_unknown_command_reports_after_plugin_fallback_misses(
    monkeypatch: pytest.MonkeyPatch,
    patched_context: list[CommandOutcome],
) -> None:
    printed: list[tuple[str, str]] = []
    monkeypatch.setattr(
        command_dispatch,
        "_print_styled",
        lambda message, style: printed.append((message, style)),
    )

    result = await _dispatch_raw("/does-not-exist", _Provider())

    assert result.handled is True
    assert printed == [("Command not found: /does-not-exist", "red")]
    assert patched_context == []


@pytest.mark.asyncio
async def test_plugin_fallback_handles_unknown_payload_before_unknown_report(
    monkeypatch: pytest.MonkeyPatch,
    patched_context: list[CommandOutcome],
) -> None:
    printed: list[tuple[str, str]] = []
    executed: list[tuple[str, str]] = []
    monkeypatch.setattr(
        command_dispatch,
        "_print_styled",
        lambda message, style: printed.append((message, style)),
    )

    async def fake_execute_plugin_command_action(
        *,
        command_name: str,
        spec: PluginCommandActionSpec,
        base_path: object,
        plugin_context: Any,
    ) -> PluginCommandActionResult:
        del spec, base_path
        executed.append((command_name, plugin_context.arguments))
        return PluginCommandActionResult(message="handled by plugin")

    monkeypatch.setattr(
        command_dispatch,
        "_execute_plugin_command_action",
        fake_execute_plugin_command_action,
    )

    provider = _Provider(
        agent_commands={
            "review": PluginCommandActionSpec(
                name="review",
                description="Review",
                handler="commands.py:review",
            )
        }
    )
    result = await _dispatch_raw("/review latest response", provider)

    assert result.handled is True
    assert executed == [("review", "latest response")]
    assert printed == []
    assert len(patched_context) == 1
