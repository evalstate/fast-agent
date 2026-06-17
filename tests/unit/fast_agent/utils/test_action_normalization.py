from __future__ import annotations

from fast_agent.utils.action_normalization import (
    enabled_disabled_label,
    is_cancel_action,
    is_help_flag,
    normalize_action_alias,
    normalize_action_token,
    on_off_label,
    parse_boolean_alias,
    split_action_arguments,
    split_first_token,
)


def test_normalize_action_token_strips_and_lowercases_values() -> None:
    assert normalize_action_token(" Install ") == "install"
    assert normalize_action_token(None) == ""


def test_normalize_action_alias_uses_default_and_aliases() -> None:
    aliases = {
        "": "list",
        "install": "add",
        "rm": "remove",
    }

    assert normalize_action_alias(None, aliases) == "list"
    assert normalize_action_alias(" install ", aliases) == "add"
    assert normalize_action_alias("RM", aliases) == "remove"
    assert normalize_action_alias("custom", aliases) == "custom"


def test_normalize_action_alias_uses_default_for_blank_action() -> None:
    aliases = {"list": "show"}

    assert normalize_action_alias("   ", aliases, default="list") == "show"


def test_normalize_action_alias_normalizes_alias_keys() -> None:
    aliases = {
        " Install ": "add",
        "RM": "remove",
    }

    assert normalize_action_alias("install", aliases) == "add"
    assert normalize_action_alias(" rm ", aliases) == "remove"


def test_is_help_flag_matches_help_aliases() -> None:
    assert is_help_flag("help")
    assert is_help_flag(" --help ")
    assert is_help_flag("-H")
    assert not is_help_flag("list")


def test_is_cancel_action_matches_prompt_cancel_aliases() -> None:
    assert is_cancel_action("q")
    assert is_cancel_action(" QUIT ")
    assert is_cancel_action("Exit")
    assert not is_cancel_action("remove")


def test_parse_boolean_alias_matches_common_on_off_values() -> None:
    assert parse_boolean_alias(" ON ") is True
    assert parse_boolean_alias("enabled") is True
    assert parse_boolean_alias("1") is True
    assert parse_boolean_alias(" OFF ") is False
    assert parse_boolean_alias("disabled") is False
    assert parse_boolean_alias("0") is False
    assert parse_boolean_alias("auto") is None


def test_parse_boolean_alias_can_exclude_numeric_aliases() -> None:
    assert parse_boolean_alias("1", numeric=False) is None
    assert parse_boolean_alias("0", numeric=False) is None
    assert parse_boolean_alias("on", numeric=False) is True
    assert parse_boolean_alias("off", numeric=False) is False


def test_on_off_label_formats_boolean_state() -> None:
    assert on_off_label(True) == "on"
    assert on_off_label(False) == "off"


def test_enabled_disabled_label_formats_boolean_state() -> None:
    assert enabled_disabled_label(True) == "enabled"
    assert enabled_disabled_label(False) == "disabled"


def test_split_action_arguments_returns_action_and_remainder() -> None:
    assert split_action_arguments("show alpha beta") == ("show", "alpha beta")
    assert split_action_arguments("  show   alpha beta  ") == ("show", "alpha beta")
    assert split_action_arguments("show\talpha beta") == ("show", "alpha beta")
    assert split_action_arguments("show\nalpha beta") == ("show", "alpha beta")
    assert split_action_arguments("show") == ("show", "")


def test_split_first_token_returns_token_and_remainder() -> None:
    assert split_first_token("show alpha beta") == ("show", "alpha beta")
    assert split_first_token("show   alpha beta  ") == ("show", "alpha beta")
    assert split_first_token("show\talpha beta") == ("show", "alpha beta")
    assert split_first_token("show\nalpha beta") == ("show", "alpha beta")
    assert split_first_token(None, default_token="list") == ("list", "")


def test_split_action_arguments_uses_default_for_blank_input() -> None:
    assert split_action_arguments(None, default_action="list") == ("list", "")
    assert split_action_arguments("   ", default_action=None) == (None, "")
