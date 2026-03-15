from fast_agent.core.default_agent import resolve_default_agent_name


def test_resolve_default_agent_name_prefers_explicit_non_tool_default() -> None:
    agents = {
        "tool": {"config": type("Config", (), {"default": True})(), "tool_only": True},
        "main": {"config": type("Config", (), {"default": True})(), "tool_only": False},
        "other": {"config": type("Config", (), {"default": False})(), "tool_only": False},
    }

    assert (
        resolve_default_agent_name(
            agents,
            is_default=lambda _name, agent_data: bool(
                getattr(agent_data.get("config"), "default", False)
            ),
            is_tool_only=lambda _name, agent_data: bool(agent_data.get("tool_only", False)),
        )
        == "main"
    )


def test_resolve_default_agent_name_falls_back_to_first_agent_when_all_tool_only() -> None:
    agents = {
        "tool": {"config": type("Config", (), {"default": False})(), "tool_only": True},
        "other": {"config": type("Config", (), {"default": False})(), "tool_only": True},
    }

    assert (
        resolve_default_agent_name(
            agents,
            is_default=lambda _name, agent_data: bool(
                getattr(agent_data.get("config"), "default", False)
            ),
            is_tool_only=lambda _name, agent_data: bool(agent_data.get("tool_only", False)),
        )
        == "tool"
    )
