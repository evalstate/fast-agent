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
    assert "`target` (`number|url|path|mcp-server`)" in rendered


def test_render_commands_json_detail_has_schema_version() -> None:
    rendered = render_commands_json(command_name="packs")

    assert '"schema_version": "1"' in rendered
    assert '"kind": "command_detail"' in rendered


def test_render_command_action_detail_markdown_contains_options() -> None:
    rendered = render_command_detail_markdown("packs", "publish")

    assert rendered is not None
    assert "# commands packs publish" in rendered
    assert "`--no-push`" in rendered
    assert "`--message text`, `-m`" in rendered


def test_render_commands_json_action_detail_has_schema_version() -> None:
    rendered = render_commands_json(command_name="skills", action_name="add")

    assert '"schema_version": "1"' in rendered
    assert '"kind": "command_action_detail"' in rendered
    assert (
        '"/skills add <number|name|github-url|path> '
        '[--registry url|path|mcp-server] [--skills-dir path]"' in rendered
    )
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
    assert "- `/packs`" in rendered
    assert "- `/cards`" not in rendered
    assert "- `/models`" not in rendered
    assert "  - list, available, search, add, remove, update, registry, help" in rendered


def test_render_mcp_action_detail_contains_executable_contract() -> None:
    rendered = render_command_detail_markdown("mcp", "attach")

    assert rendered is not None
    assert "Usage: `/mcp attach <server-name>`" in rendered
    assert "`server_name` (`server-name`)" in rendered
    assert "Run /mcp list" in rendered


def test_render_mcp_connect_detail_contains_model_restrictions() -> None:
    rendered = render_command_detail_markdown("mcp", "connect", model_facing=True)

    assert rendered is not None
    assert "`--no-oauth`" in rendered
    assert "--oauth" not in rendered
    assert "`--protocol auto|modern|legacy`" in rendered
    assert "Interactive OAuth" in rendered
    assert "stdio targets require shell access" in rendered


def test_render_mcp_connect_detail_keeps_interactive_oauth_for_shared_surfaces() -> None:
    rendered = render_command_detail_markdown("mcp", "connect")

    assert rendered is not None
    assert "`--oauth`" in rendered
    assert "model-facing commands" not in rendered


def test_filtered_discovery_includes_status_and_scopes_unknown_suggestions() -> None:
    command_names = {"commands", "mcp", "status"}

    index = render_commands_index_markdown(command_names=command_names)
    unknown = render_commands_json(command_name="missing", command_names=command_names)

    assert "- `/status`" in index
    assert '"suggestions": [' in unknown
    assert '"commands"' in unknown
    assert '"mcp"' in unknown
    assert '"status"' in unknown
    assert '"packs"' not in unknown


def test_model_skills_discovery_exposes_registry_browse_without_ambiguous_add() -> None:
    available = render_command_detail_markdown(
        "skills",
        "available",
        model_facing=True,
    )
    add = render_commands_json(
        command_name="skills",
        action_name="add",
        model_facing=True,
    )

    assert available is not None
    assert "`--registry url|path|mcp-server`, `-r`" in available
    assert "`--compact`" in available
    assert "`--json`" in available
    assert "--full" not in available
    assert "default to compact output" in available
    assert '"/skills add <number|name|github-url|path>' in add
    assert '"required": true' in add
    assert "A selector is required for model-facing installation." in add
