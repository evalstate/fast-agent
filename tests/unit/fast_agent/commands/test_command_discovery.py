from __future__ import annotations

from fast_agent.commands.command_discovery import (
    parse_commands_discovery_arguments,
    render_command_detail_markdown,
    render_commands_index_markdown,
    render_commands_json,
)


def test_parse_commands_discovery_arguments_supports_json_and_name() -> None:
    request = parse_commands_discovery_arguments("skills --json")

    assert request.command_name == "skills"
    assert request.as_json is True


def test_render_command_detail_markdown_contains_registry_action() -> None:
    rendered = render_command_detail_markdown("skills")

    assert rendered is not None
    assert "`registry`" in rendered
    assert "/skills registry [<number|url|path>]" in rendered


def test_render_commands_json_detail_has_schema_version() -> None:
    rendered = render_commands_json(command_name="cards")

    assert '"schema_version": "1"' in rendered
    assert '"kind": "command_detail"' in rendered


def test_render_commands_index_markdown_has_tree_actions() -> None:
    rendered = render_commands_index_markdown()

    assert "Command map:" in rendered
    assert "- `/skills`" in rendered
    assert "  - list, available, search, add, remove, update, registry, help" in rendered
