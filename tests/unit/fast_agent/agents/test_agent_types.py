"""
Unit tests for agent types and their interactions with the interactive prompt.
"""

from dataclasses import dataclass
from typing import Any, cast

import pytest

from fast_agent.agents import McpAgent
from fast_agent.agents.agent_types import AgentConfig, AgentType
from fast_agent.agents.smart_agent import (
    SmartAgent,
    _apply_runtime_mcp_connections,
    _resolve_default_agent_name,
    _run_mcp_connect_call,
)
from fast_agent.core.exceptions import AgentConfigError
from fast_agent.types import RequestParams


def test_agent_type_default():
    """Test that agent_type defaults to AgentType.BASIC.value"""
    agent = McpAgent(config=AgentConfig(name="test_agent"))
    assert agent.agent_type == AgentType.BASIC


def test_agent_type_smart_enum():
    assert AgentType.SMART.value == "smart"


def test_smart_agent_type():
    agent = SmartAgent(config=AgentConfig(name="smart_agent"))
    assert agent.agent_type == AgentType.SMART


def test_instruction_propagates_to_default_request_params():
    """
    Test that AgentConfig.instruction is propagated to
    default_request_params.systemPrompt when both are provided.

    This reproduces the bug where the instruction is lost when
    a user provides their own default_request_params.
    """
    # Create RequestParams with custom settings but no systemPrompt
    request_params = RequestParams(
        model="sonnet",
        temperature=0.7,
        maxTokens=32768
    )

    # Verify systemPrompt is not set initially
    assert not hasattr(request_params, 'systemPrompt') or request_params.systemPrompt is None

    # Create AgentConfig with both instruction and default_request_params
    instruction = "You are a helpful assistant specialized in testing."
    config = AgentConfig(
        name="my_agent",
        instruction=instruction,
        default_request_params=request_params,
        model="sonnet"
    )

    # The instruction should be propagated to default_request_params.systemPrompt
    assert config.default_request_params is not None
    assert config.default_request_params.systemPrompt == instruction, (
        f"Expected systemPrompt to be '{instruction}', "
        f"but got {config.default_request_params.systemPrompt}"
    )


def test_instruction_takes_precedence_over_systemPrompt():
    """
    Test that AgentConfig.instruction takes precedence over
    default_request_params.systemPrompt when both are provided.

    This ensures that the explicit instruction parameter on AgentConfig
    overrides any systemPrompt already set in the RequestParams.
    """
    # Create RequestParams with a systemPrompt already set
    original_system_prompt = "You are a generic assistant from RequestParams."
    request_params = RequestParams(
        model="sonnet",
        temperature=0.7,
        maxTokens=32768,
        systemPrompt=original_system_prompt
    )

    # Verify systemPrompt is set initially
    assert request_params.systemPrompt == original_system_prompt

    # Create AgentConfig with BOTH instruction AND default_request_params with systemPrompt
    instruction = "You are a specialized assistant from AgentConfig instruction."
    config = AgentConfig(
        name="my_agent",
        instruction=instruction,
        default_request_params=request_params,
        model="sonnet"
    )

    # The AgentConfig.instruction should take precedence over systemPrompt in RequestParams
    assert config.default_request_params is not None
    assert config.default_request_params.systemPrompt == instruction, (
        f"Expected AgentConfig.instruction ('{instruction}') to override "
        f"RequestParams.systemPrompt ('{original_system_prompt}'), "
        f"but got {config.default_request_params.systemPrompt}"
    )


@dataclass
class _FakeAttachResult:
    tools_added: list[str]
    prompts_added: list[str]


class _FakeMcpAgent:
    def __init__(self, *, default: bool = False, fail: bool = False) -> None:
        self.config = AgentConfig(name="fake", default=default)
        self._fail = fail
        self.attached: list[str] = []

    async def attach_mcp_server(self, *, server_name: str, server_config=None, options=None):
        del server_config, options
        if self._fail:
            raise RuntimeError("boom")
        self.attached.append(server_name)
        return _FakeAttachResult(tools_added=[], prompts_added=[])

    async def detach_mcp_server(self, server_name: str):
        detached = server_name in self.attached
        if detached:
            self.attached.remove(server_name)

        @dataclass
        class _DetachResult:
            detached: bool
            tools_removed: list[str]
            prompts_removed: list[str]

        return _DetachResult(detached=detached, tools_removed=[], prompts_removed=[])

    def list_attached_mcp_servers(self) -> list[str]:
        return list(self.attached)


class _FakeSmartToolAgent(_FakeMcpAgent):
    def __init__(self) -> None:
        super().__init__(default=True)
        self.name = "smart"
        self.context = None


def test_resolve_default_agent_name_prefers_non_tool_default() -> None:
    tool_default = _FakeMcpAgent(default=True)
    non_tool_default = _FakeMcpAgent(default=True)
    agents = {
        "tool": tool_default,
        "main": non_tool_default,
    }

    resolved = _resolve_default_agent_name(
        cast("Any", agents),
        tool_only_agents={"tool"},
    )
    assert resolved == "main"


@pytest.mark.asyncio
async def test_apply_runtime_mcp_connections_attaches_servers() -> None:
    agent = _FakeMcpAgent(default=True)
    summary = await _apply_runtime_mcp_connections(
        context=None,
        agents_map=cast("Any", {"main": agent}),
        target_agent_name="main",
        mcp_connect=["npx demo-server --name demo"],
    )

    assert summary.connected == ["demo"]
    assert summary.warnings == []
    assert agent.attached == ["demo"]


@pytest.mark.asyncio
async def test_apply_runtime_mcp_connections_raises_on_connect_error() -> None:
    failing = _FakeMcpAgent(default=True, fail=True)
    with pytest.raises(AgentConfigError, match="Failed to connect MCP server"):
        await _apply_runtime_mcp_connections(
            context=None,
            agents_map=cast("Any", {"main": failing}),
            target_agent_name="main",
            mcp_connect=["npx demo-server --name demo"],
        )


@pytest.mark.asyncio
async def test_run_mcp_connect_call_returns_connect_summary() -> None:
    agent = _FakeSmartToolAgent()
    result = await _run_mcp_connect_call(agent, "npx demo-server --name demo")

    assert "Connected MCP server 'demo'" in result
    assert agent.attached == ["demo"]
