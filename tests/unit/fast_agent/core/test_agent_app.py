from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

from fast_agent.core.agent_app import AgentApp

if TYPE_CHECKING:
    from fast_agent.interfaces import AgentProtocol


class _Agent:
    def __init__(self, name: str, *, default: bool = False) -> None:
        self.name = name
        self.config = SimpleNamespace(default=default)


def test_get_default_agent_name_prefers_explicit_non_tool_default() -> None:
    app = AgentApp(
        agents={
            "tool": cast("AgentProtocol", _Agent("tool", default=True)),
            "main": cast("AgentProtocol", _Agent("main", default=True)),
            "other": cast("AgentProtocol", _Agent("other")),
        },
        tool_only_agents={"tool"},
    )

    assert app.get_default_agent_name() == "main"
    assert app._agent(None).name == "main"


def test_get_default_agent_name_falls_back_to_first_non_tool_agent() -> None:
    app = AgentApp(
        agents={
            "tool": cast("AgentProtocol", _Agent("tool")),
            "main": cast("AgentProtocol", _Agent("main")),
        },
        tool_only_agents={"tool"},
    )

    assert app.get_default_agent_name() == "main"
    assert app._agent(None).name == "main"


def test_visible_agent_names_can_include_targeted_tool_only_agent() -> None:
    app = AgentApp(
        agents={
            "tool": cast("AgentProtocol", _Agent("tool")),
            "main": cast("AgentProtocol", _Agent("main")),
        },
        tool_only_agents={"tool"},
    )

    assert app.visible_agent_names(force_include="tool") == ["tool", "main"]


def test_resolve_target_agent_name_prefers_explicit_name_over_default() -> None:
    app = AgentApp(
        agents={
            "main": cast("AgentProtocol", _Agent("main", default=True)),
            "other": cast("AgentProtocol", _Agent("other")),
        }
    )

    assert app.resolve_target_agent_name("other") == "other"
    assert app.resolve_target_agent_name() == "main"


def test_registered_agent_names_include_tool_only_agents() -> None:
    app = AgentApp(
        agents={
            "tool": cast("AgentProtocol", _Agent("tool")),
            "main": cast("AgentProtocol", _Agent("main")),
        },
        tool_only_agents={"tool"},
    )

    assert app.registered_agent_names() == ["tool", "main"]
