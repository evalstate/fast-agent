"""PTC selection contracts across CLI, model strings, config and model defaults."""

from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

import pytest
from typer.testing import CliRunner

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.mcp_agent import McpAgent
from fast_agent.cli.commands import go as go_command
from fast_agent.cli.runtime.agent_setup import _apply_runtime_context_overrides
from fast_agent.config import ProgrammaticToolCallingSettings, Settings
from fast_agent.context import Context
from fast_agent.core.direct_factory import get_model_factory
from fast_agent.core.exceptions import ModelConfigError
from fast_agent.llm.model_factory import ModelFactory
from fast_agent.mcp_server_registry import ServerRegistry
from fast_agent.ptc import PTC_BATCHING_INSTRUCTION, PTC_TOOL_NAME

if TYPE_CHECKING:
    from fast_agent.cli.runtime.run_request import AgentRunRequest


@pytest.mark.parametrize(
    ("value", "enabled"),
    [("on", True), ("true", True), ("off", False), ("false", False), ("ON", True)],
)
def test_model_string_ptc_toggle(value: str, enabled: bool) -> None:
    resolved = ModelFactory.resolve_model_spec(f"generic.test?ptc={value}")
    assert resolved.model_config.ptc is enabled
    assert resolved.resolve_ptc_enabled() is enabled
    assert resolved.wire_model_name == "test"


@pytest.mark.parametrize("value", ["auto", "maybe", ""])
def test_model_string_rejects_invalid_ptc(value: str) -> None:
    with pytest.raises(ModelConfigError, match="Invalid ptc query value"):
        ModelFactory.parse_model_string(f"generic.test?ptc={value}")


def test_explicit_false_overrides_preset_ptc_default() -> None:
    resolved = ModelFactory.resolve_model_spec(
        "test-ptc?ptc=off", presets={"test-ptc": "generic.test?ptc=on"}
    )
    assert resolved.resolve_ptc_enabled() is False


@pytest.mark.parametrize(
    ("model", "enabled"),
    [
        ("glm53flash", True),
        ("zaiglm53flash", True),
        ("zai.glm-5.3-flash", True),
        ("hf.zai-org/GLM-5.3-Flash:zai-org", True),
        ("glm53", False),
        ("astra", True),
        ("sol", True),
        ("terra", True),
        ("luna", True),
        ("responses.gpt-6.1-sol?context=1m", True),
        ("gpt6sol", False),
        ("luna56", False),
        ("generic.unknown", False),
    ],
)
def test_model_catalog_ptc_defaults(model: str, enabled: bool) -> None:
    assert ModelFactory.resolve_model_spec(model).resolve_ptc_enabled() is enabled


@pytest.mark.asyncio
@pytest.mark.parametrize("cli", [None, False, True])
@pytest.mark.parametrize("query", [None, False, True])
@pytest.mark.parametrize("configured", [None, False, True])
async def test_ptc_precedence_reaches_tools_and_instructions(
    cli: bool | None, query: bool | None, configured: bool | None
) -> None:
    model = "glm53flash"
    if query is not None:
        model += f"?ptc={str(query).lower()}"
    # GLM-5.3-Flash is on unless overridden; every explicit false must survive.
    enabled = next(value for value in (cli, query, configured, True) if value is not None)
    context = Context(
        config=Settings(ptc=ProgrammaticToolCallingSettings(enabled=configured)),
        ptc_enabled=cli,
        server_registry=ServerRegistry(),
    )
    agent = McpAgent(AgentConfig("ptc-policy", model=model, skills=[]), context=context)
    try:
        await agent.initialize()
        assert (
            PTC_TOOL_NAME in {tool.name for tool in (await agent.list_tools()).tools}
        ) is enabled
        assert (PTC_BATCHING_INSTRUCTION in agent.instruction) is enabled
    finally:
        await agent.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [False, True])
async def test_cli_override_is_applied_even_when_false(enabled: bool, monkeypatch) -> None:
    context = Context(config=Settings(ptc=ProgrammaticToolCallingSettings()))
    fast = SimpleNamespace(app=SimpleNamespace(initialize=AsyncMock(), context=context))
    requests: list[AgentRunRequest] = []
    monkeypatch.setattr(go_command, "run_request", requests.append)
    result = CliRunner().invoke(
        go_command.app, ["--message", "hi", "--ptc" if enabled else "--no-ptc"]
    )
    assert result.exit_code == 0, result.output
    await _apply_runtime_context_overrides(fast, requests[0])
    fast.app.initialize.assert_awaited_once()
    assert context.ptc_enabled is enabled
    assert context.config is not None
    assert context.config.ptc.enabled is None


@pytest.mark.asyncio
async def test_attached_model_changes_ptc_surface_and_guidance() -> None:
    agent = McpAgent(
        AgentConfig("ptc-switch", model="passthrough", skills=[]),
        context=Context(config=Settings(), server_registry=ServerRegistry()),
    )
    try:
        await agent.initialize()
        for enabled in (True, False, True):
            await agent.attach_llm(
                ModelFactory.create_factory(f"passthrough?ptc={str(enabled).lower()}")
            )
            tools = (await agent.list_tools()).tools
            assert (PTC_TOOL_NAME in {tool.name for tool in tools}) is enabled
            assert (PTC_BATCHING_INSTRUCTION in agent.instruction) is enabled
            # A runtime created while enabled must not keep serving the code tool.
            assert (agent.resolve_stream_tool_metadata(PTC_TOOL_NAME) is not None) is enabled
    finally:
        await agent.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model", "configured", "outcome"),
    [
        ("glm53flash", None, "direct"),
        ("glm53flash?ptc=on", None, "error"),
        ("glm53flash", True, "error"),
    ],
)
async def test_model_default_falls_back_in_unsupported_environments(
    model: str, configured: bool | None, outcome: str
) -> None:
    from unittest.mock import Mock

    from acp import AgentSideConnection

    from fast_agent.acp.acp_context import ACPContext
    from fast_agent.acp.terminal_runtime import ACPTerminalRuntime
    from fast_agent.core.exceptions import AgentConfigError

    connection = Mock(spec=AgentSideConnection)
    acp = ACPContext(connection=connection, session_id="ptc")
    acp.set_terminal_runtime(ACPTerminalRuntime(connection, "ptc", "test"))
    agent = McpAgent(
        AgentConfig("ptc-acp", model=model, servers=[], skills=[]),
        context=Context(
            config=Settings(ptc=ProgrammaticToolCallingSettings(enabled=configured)),
            acp=acp,
            server_registry=ServerRegistry(),
        ),
    )
    try:
        if outcome == "error":
            with pytest.raises(AgentConfigError, match="local POSIX"):
                await agent.list_tools()
        else:
            tools = {tool.name for tool in (await agent.list_tools()).tools}
            assert PTC_TOOL_NAME not in tools
            assert PTC_BATCHING_INSTRUCTION not in agent.instruction
    finally:
        await agent.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("configured", [None, False])
@pytest.mark.parametrize(
    ("model", "default_model", "cli_model", "env_model", "enabled"),
    [
        ("$system.fast", None, None, None, True),
        (None, "$system.fast", None, None, True),
        (None, None, None, "$system.fast", True),
        (None, "$system.default", "$system.fast", None, True),
        ("$system.default", None, "$system.fast", None, True),
        ("$system.fast", None, "passthrough?ptc=off", None, True),
        (None, "$system.fast", "passthrough?ptc=off", None, False),
        ("$missing.reference", "$system.fast", None, None, True),
    ],
)
async def test_initial_ptc_policy_uses_effective_model_selection(
    monkeypatch, configured, model, default_model, cli_model, env_model, enabled
) -> None:
    if env_model is None:
        monkeypatch.delenv("FAST_AGENT_MODEL", raising=False)
    else:
        monkeypatch.setenv("FAST_AGENT_MODEL", env_model)
    context = Context(
        config=Settings(
            default_model=default_model,
            cli_model_override=cli_model,
            model_references={
                "system": {
                    "fast": "passthrough?ptc=on",
                    "default": "passthrough?ptc=off",
                }
            },
            ptc=ProgrammaticToolCallingSettings(enabled=configured),
        ),
        server_registry=ServerRegistry(),
    )
    agent = McpAgent(AgentConfig("ptc-reference", model=model, skills=[]), context=context)
    try:
        await agent.initialize()
        assert (PTC_BATCHING_INSTRUCTION in agent.instruction) is enabled
        assert (
            PTC_TOOL_NAME in {tool.name for tool in (await agent.list_tools()).tools}
        ) is enabled
        await agent.attach_llm(get_model_factory(context, model=model))
        assert (PTC_BATCHING_INSTRUCTION in agent.instruction) is enabled
        assert (
            PTC_TOOL_NAME in {tool.name for tool in (await agent.list_tools()).tools}
        ) is enabled
    finally:
        await agent.shutdown()
