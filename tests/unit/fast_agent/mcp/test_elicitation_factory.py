from fast_agent.agents.agent_types import AgentConfig
from fast_agent.config import MCPElicitationSettings, MCPServerSettings, Settings
from fast_agent.mcp.elicitation_factory import (
    resolve_elicitation_handler,
    resolve_global_elicitation_mode,
)
from fast_agent.mcp.elicitation_handlers import (
    auto_cancel_elicitation_handler,
    forms_elicitation_handler,
)


def test_server_elicitation_config_overrides_global_mode() -> None:
    agent_config = AgentConfig(name="demo")
    app_config = {"elicitation": {"mode": "auto-cancel"}}
    server_config = MCPServerSettings(
        name="docs",
        elicitation=MCPElicitationSettings(mode="forms"),
    )

    handler = resolve_elicitation_handler(agent_config, app_config, server_config)

    assert handler is forms_elicitation_handler


def test_server_elicitation_dict_can_disable_capability() -> None:
    agent_config = AgentConfig(name="demo")
    app_config = {"elicitation": {"mode": "forms"}}
    server_config = {"elicitation": {"mode": "none"}}

    handler = resolve_elicitation_handler(agent_config, app_config, server_config)

    assert handler is None


def test_global_elicitation_object_can_auto_cancel() -> None:
    agent_config = AgentConfig(name="demo")
    app_config = Settings.model_validate(
        {"elicitation": MCPElicitationSettings(mode="auto-cancel")}
    )

    handler = resolve_elicitation_handler(agent_config, app_config)

    assert handler is auto_cancel_elicitation_handler
    assert resolve_global_elicitation_mode(app_config) == "auto-cancel"


def test_global_elicitation_mode_status_normalizes_padded_text() -> None:
    assert (
        resolve_global_elicitation_mode({"elicitation": {"mode": " Auto-Cancel "}}) == "auto-cancel"
    )


def test_missing_elicitation_config_defaults_to_forms() -> None:
    agent_config = AgentConfig(name="demo")

    handler = resolve_elicitation_handler(agent_config, {})

    assert handler is forms_elicitation_handler
    assert resolve_global_elicitation_mode({}) is None


def test_unknown_elicitation_mode_falls_back_to_forms() -> None:
    agent_config = AgentConfig(name="demo")
    app_config = {"elicitation": {"mode": "unexpected"}}

    handler = resolve_elicitation_handler(agent_config, app_config)

    assert handler is forms_elicitation_handler
