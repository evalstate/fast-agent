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
    assert request.action_name is None
    assert request.as_json is True


def test_parse_commands_discovery_arguments_supports_action_name() -> None:
    request = parse_commands_discovery_arguments("skills add --json")

    assert request.command_name == "skills"
    assert request.action_name == "add"
    assert request.as_json is True


def test_render_command_detail_markdown_contains_registry_action() -> None:
    rendered = render_command_detail_markdown("skills")

    assert rendered is not None
    assert "`registry`" in rendered
    assert "/skills registry [<number|url|path|mcp-server>]" in rendered


def test_render_commands_json_detail_has_schema_version() -> None:
    rendered = render_commands_json(command_name="cards")

    assert '"schema_version": "1"' in rendered
    assert '"kind": "command_detail"' in rendered


def test_render_command_action_detail_markdown_contains_options() -> None:
    rendered = render_command_detail_markdown("cards", "publish")

    assert rendered is not None
    assert "# commands cards publish" in rendered
    assert "`--no-push`" in rendered
    assert "`--message text`, `-m`" in rendered


def test_render_commands_json_action_detail_has_schema_version() -> None:
    rendered = render_commands_json(command_name="skills", action_name="add")

    assert '"schema_version": "1"' in rendered
    assert '"kind": "command_action_detail"' in rendered
    assert '"/skills add [<number|name|github-url|path>] [--registry url] [--skills-dir path]"' in rendered
    assert '"name": "--skills-dir"' in rendered


def test_render_command_detail_markdown_session_includes_export_options() -> None:
    rendered = render_command_detail_markdown("session")

    assert rendered is not None
    assert "`--output path`" in rendered
    assert "file path, not a directory path" in rendered
    assert "`--help`, `-h`" in rendered


def test_render_commands_json_session_includes_export_behavior() -> None:
    rendered = render_commands_json(command_name="session")

    assert '"name": "export"' in rendered
    assert '"name": "--output"' in rendered
    assert '"Default format: codex."' in rendered


def test_render_commands_index_markdown_has_tree_actions() -> None:
    rendered = render_commands_index_markdown()

    assert "Command map:" in rendered
    assert "- `/skills`" in rendered
    assert "  - list, available, search, add, remove, update, registry, help" in rendered
