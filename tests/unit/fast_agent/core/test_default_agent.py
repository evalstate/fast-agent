from fast_agent.agents.agent_types import AgentConfig
from fast_agent.core.default_agent import resolve_default_agent_name


def test_resolve_default_agent_name_prefers_explicit_non_tool_default() -> None:
    agents = {
        "tool": {"config": AgentConfig("tool", default=True), "tool_only": True},
        "main": {"config": AgentConfig("main", default=True), "tool_only": False},
        "other": {"config": AgentConfig("other", default=False), "tool_only": False},
    }

    assert (
        resolve_default_agent_name(
            agents,
            is_default=lambda _name, agent_data: (
                isinstance(config := agent_data.get("config"), AgentConfig) and config.default
            ),
            is_tool_only=lambda _name, agent_data: bool(agent_data.get("tool_only", False)),
        )
        == "main"
    )


def test_resolve_default_agent_name_falls_back_to_first_agent_when_all_tool_only() -> None:
    agents = {
        "tool": {"config": AgentConfig("tool", default=False), "tool_only": True},
        "other": {"config": AgentConfig("other", default=False), "tool_only": True},
    }

    assert (
        resolve_default_agent_name(
            agents,
            is_default=lambda _name, agent_data: (
                isinstance(config := agent_data.get("config"), AgentConfig) and config.default
            ),
            is_tool_only=lambda _name, agent_data: bool(agent_data.get("tool_only", False)),
        )
        == "tool"
    )
