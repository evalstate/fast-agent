import pytest

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.tool_agent import ToolAgent
from fast_agent.core.fastagent import FastAgent, RunSettings
from fast_agent.core.subagent_policy import (
    SubagentRuntimePolicy,
    apply_subagent_runtime_policy,
)


def test_explicit_card_disable_wins_over_cli_enable() -> None:
    config = AgentConfig("agent", subagents=False)

    apply_subagent_runtime_policy(
        config,
        SubagentRuntimePolicy(enabled=True, model="playback"),
    )

    assert config.subagents is False
    assert config.subagent_model is None
    assert config.subagent_activation_source == "configuration"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("policy", "installed", "model"),
    [
        (SubagentRuntimePolicy(enabled=False), False, None),
        (SubagentRuntimePolicy(enabled=True, model="playback"), True, "playback"),
    ],
)
async def test_refresh_reapplies_cli_subagent_policy(
    policy: SubagentRuntimePolicy,
    installed: bool,
    model: str | None,
) -> None:
    fast = FastAgent("TestAgent", parse_cli_args=False)
    runtime = fast._create_run_runtime(
        RunSettings(
            quiet_mode=True,
            cli_model_override=None,
            no_home_mode=False,
            server_mode=False,
            transport=None,
            is_acp_server_mode=True,
            reload_enabled=True,
            subagent_policy=policy,
        )
    )
    agent = ToolAgent(
        AgentConfig(
            "agent",
            instruction="fast-agent-subagents",
        )
    )

    await fast._finalize_updated_agents({"agent": agent}, runtime)

    tools = await agent.list_tools()
    assert ("subagent" in {tool.name for tool in tools.tools}) is installed
    assert agent.config.subagent_model == model
