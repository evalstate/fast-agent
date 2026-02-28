from __future__ import annotations

import pytest

from fast_agent.commands.command_catalog import (
    command_action_names,
    parse_smart_operation,
    smart_operation_help_entries,
    tool_exposed_operations,
)


def test_tool_exposed_operations_include_expected_commands() -> None:
    operations = tool_exposed_operations()

    assert operations == (
        "skills.list",
        "skills.add",
        "skills.remove",
        "skills.update",
        "skills.registry",
        "cards.list",
        "cards.add",
        "cards.remove",
        "cards.update",
        "cards.publish",
        "cards.registry",
        "models.doctor",
        "models.aliases",
        "models.catalog",
        "check",
    )


def test_parse_smart_operation_for_check() -> None:
    command, action = parse_smart_operation("check")

    assert command == "check"
    assert action == "run"


def test_parse_smart_operation_rejects_unknown_value() -> None:
    with pytest.raises(ValueError, match="Unknown smart command operation"):
        parse_smart_operation("skills.invalid")


def test_command_action_names_for_models() -> None:
    assert command_action_names("models") == ("doctor", "aliases", "catalog")


def test_smart_operation_help_entries_cover_all_tool_operations() -> None:
    operations = tool_exposed_operations()
    entries = dict(smart_operation_help_entries())

    assert tuple(entries.keys()) == operations
    assert entries["check"] == "Run fast-agent check diagnostics"
