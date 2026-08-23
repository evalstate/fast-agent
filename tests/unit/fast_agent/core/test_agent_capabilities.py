import pytest

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.tool_agent import ToolAgent
from fast_agent.core.agent_capabilities import (
    AgentCapabilityMode,
    cycle_agent_capability_mode,
    resolve_agent_capability_mode,
    set_agent_capability_mode,
)
from fast_agent.core.exceptions import AgentConfigError
from fast_agent.core.harness_tools import set_harness_tools
from fast_agent.tools.function_tool_loader import build_default_function_tool


def test_agent_capability_mode_cycles_all_four_states() -> None:
    agent = ToolAgent(AgentConfig("dev"))

    assert resolve_agent_capability_mode(agent) is AgentCapabilityMode.STANDARD
    assert cycle_agent_capability_mode(agent) is AgentCapabilityMode.DELEGATE
    assert cycle_agent_capability_mode(agent) is AgentCapabilityMode.ORCHESTRATE
    assert cycle_agent_capability_mode(agent) is AgentCapabilityMode.HARNESS_ONLY
    assert cycle_agent_capability_mode(agent) is AgentCapabilityMode.STANDARD


def test_agent_capability_mode_cycles_harness_only_to_standard() -> None:
    agent = ToolAgent(AgentConfig("dev", harness_tools=True))
    set_harness_tools(agent)

    assert resolve_agent_capability_mode(agent) is AgentCapabilityMode.HARNESS_ONLY
    assert cycle_agent_capability_mode(agent) is AgentCapabilityMode.STANDARD


@pytest.mark.parametrize("mode", AgentCapabilityMode)
def test_set_agent_capability_mode_applies_exact_mode(mode: AgentCapabilityMode) -> None:
    agent = ToolAgent(AgentConfig("dev"))

    assert set_agent_capability_mode(agent, mode) is mode
    assert resolve_agent_capability_mode(agent) is mode


def test_agent_capability_mode_respects_explicit_subagent_disable() -> None:
    agent = ToolAgent(AgentConfig("dev", subagents=False))

    with pytest.raises(AgentConfigError, match="disabled by configuration"):
        cycle_agent_capability_mode(agent)

    assert resolve_agent_capability_mode(agent) is AgentCapabilityMode.STANDARD
    assert agent.config.subagent_activation_source == "configuration"


def test_agent_capability_mode_preserves_user_subagent_tool() -> None:
    def subagent() -> str:
        return "custom"

    custom_tool = build_default_function_tool(subagent)
    agent = ToolAgent(AgentConfig("dev"), tools=[custom_tool])

    with pytest.raises(AgentConfigError, match="built-in subagent tool is unavailable"):
        cycle_agent_capability_mode(agent)

    assert agent._execution_tools["subagent"] is custom_tool


def test_agent_capability_mode_is_unavailable_to_tool_only_agents() -> None:
    agent = ToolAgent(AgentConfig("dev", tool_only=True))

    assert cycle_agent_capability_mode(agent) is AgentCapabilityMode.STANDARD
