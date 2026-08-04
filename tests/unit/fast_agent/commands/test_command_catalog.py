from fast_agent.commands.command_catalog import command_action_names, get_command_spec


def test_command_catalog_uses_canonical_surface() -> None:
    assert command_action_names("models") == ()
    assert command_action_names("cards") == ()
    assert command_action_names("packs") == (
        "list",
        "add",
        "remove",
        "readme",
        "update",
        "publish",
        "registry",
        "help",
    )
    assert command_action_names("agent") == ("status", "list", "use", "tool")
    assert command_action_names("card") == ("show", "load")


def test_get_command_spec_returns_expected_default_action() -> None:
    spec = get_command_spec("skills")

    assert spec is not None
    assert spec.default_action == "list"

    model_spec = get_command_spec("model")
    assert model_spec is not None
    assert model_spec.default_action == "status"


def test_command_action_names_for_skills_include_discovery_actions() -> None:
    actions = command_action_names("skills")

    assert "available" in actions
    assert "search" in actions
    assert "help" in actions
