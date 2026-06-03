from fast_agent.commands.command_catalog import (
    command_action_names,
    command_action_tokens,
    command_usage_lines,
    format_unknown_command_action,
    get_command_action_spec,
    get_command_spec,
    normalize_command_action,
)


def test_command_action_names_for_models() -> None:
    assert command_action_names("models") == ("doctor", "references", "catalog", "help")


def test_model_catalog_aliases_share_all_option_metadata() -> None:
    model_catalog = get_command_action_spec("model", "catalog")
    models_catalog = get_command_action_spec("models", "catalog")

    assert model_catalog is not None
    assert models_catalog is not None
    assert model_catalog.usage == "/model catalog <provider> [--all]"
    assert models_catalog.usage == "/models catalog <provider> [--all]"
    assert [option.name for option in model_catalog.options] == ["--all"]
    assert [option.name for option in models_catalog.options] == ["--all"]


def test_get_command_spec_returns_expected_default_action() -> None:
    spec = get_command_spec("skills")

    assert spec is not None
    assert spec.default_action == "list"


def test_command_catalog_lookups_use_shared_token_normalization() -> None:
    assert get_command_spec(" SKILLS ") is get_command_spec("skills")
    assert get_command_action_spec(" CARDS ", " SHOW ") is get_command_action_spec(
        "cards",
        "readme",
    )
    assert command_action_tokens(" CARDS ", " SHOW ") == ("readme", "show", "cat")


def test_command_action_names_for_skills_include_discovery_actions() -> None:
    actions = command_action_names("skills")

    assert "available" in actions
    assert "search" in actions
    assert "help" in actions


def test_cards_readme_aliases_are_catalogued() -> None:
    readme_action = get_command_action_spec("cards", "readme")
    show_action = get_command_action_spec("cards", "show")
    cat_action = get_command_action_spec("cards", "cat")

    assert readme_action is not None
    assert show_action is not None
    assert cat_action is not None
    assert show_action is readme_action
    assert cat_action is readme_action
    assert show_action.action == "readme"
    assert cat_action.action == "readme"


def test_plugins_actions_are_catalogued() -> None:
    actions = command_action_names("plugins")

    assert actions == ("list", "available", "add", "remove", "update", "registry", "help")
    assert normalize_command_action("plugins", " MARKETPLACE ") == "available"
    assert normalize_command_action("plugins", "install") == "add"
    assert normalize_command_action("plugins", "source") == "registry"


def test_command_action_tokens_return_canonical_action_and_aliases() -> None:
    assert command_action_tokens("plugins", "available") == (
        "available",
        "marketplace",
        "browse",
    )
    assert command_action_tokens("plugins", "marketplace") == (
        "available",
        "marketplace",
        "browse",
    )
    assert command_action_tokens("cards", "show") == ("readme", "show", "cat")
    assert command_action_tokens("plugins", "registry") == ("registry", "source")
    assert command_action_tokens("unknown", "registry") == ()


def test_command_usage_lines_render_catalog_usage_and_examples() -> None:
    lines = command_usage_lines("plugins")

    assert lines[0] == "Usage: /plugins [list|available|add|remove|update|registry|help] [args]"
    assert "- /plugins available" in lines
    assert "- /plugins update all --yes" in lines


def test_unknown_command_action_message_uses_catalog_actions_and_suggestions() -> None:
    message = format_unknown_command_action("cards", "showw")

    assert message.startswith(
        "Unknown /cards action: showw. "
        "Use list/add/remove/readme/update/publish/registry/help."
    )
    assert "Did you mean: `readme`" in message


def test_unknown_command_action_message_omits_empty_suggestions() -> None:
    message = format_unknown_command_action("cards", "zzzzzz")

    assert "Did you mean:" not in message


def test_normalize_command_action_uses_catalog_default_and_aliases() -> None:
    assert normalize_command_action("cards", None) == "list"
    assert normalize_command_action("cards", " SHOW ") == "readme"
    assert normalize_command_action("cards", "unexpected") == "unexpected"
    assert normalize_command_action("unknown", "Action") == "action"
