from fast_agent.commands.command_catalog import command_action_names, get_command_spec


def test_command_action_names_for_models() -> None:
    assert command_action_names("models") == ("doctor", "aliases", "catalog", "help")


def test_get_command_spec_returns_expected_default_action() -> None:
    spec = get_command_spec("skills")

    assert spec is not None
    assert spec.default_action == "list"


def test_command_action_names_for_skills_include_discovery_actions() -> None:
    actions = command_action_names("skills")

    assert "available" in actions
    assert "search" in actions
    assert "help" in actions
