from __future__ import annotations

import json
from typing import Any, cast

import pytest

from fast_agent.commands import command_discovery
from fast_agent.commands.command_discovery import (
    parse_commands_discovery_arguments,
    render_command_detail_markdown,
    render_commands_index_markdown,
    render_commands_json,
    render_direct_command_help,
)
from fast_agent.commands.mcp_command_intents import (
    MCP_TOP_LEVEL_ACTION_DESCRIPTIONS,
    MCP_TOP_LEVEL_ACTIONS,
)
from fast_agent.commands.session_summaries import FULL_SESSION_USAGE
from fast_agent.commands.shared_command_intents import SESSION_COMMAND_COMPLETION_DESCRIPTIONS
from fast_agent.ui.prompt import parser as prompt_parser


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


def test_parse_commands_discovery_arguments_normalizes_tokens() -> None:
    request = parse_commands_discovery_arguments("  CARDS   SHOW   --JSON  ")

    assert request.command_name == "cards"
    assert request.action_name == "show"
    assert request.as_json is True


def test_parse_commands_discovery_arguments_rejects_empty_positional() -> None:
    with pytest.raises(ValueError, match=r"Usage: /commands"):
        parse_commands_discovery_arguments('skills ""')


def test_parse_commands_discovery_arguments_reports_split_errors() -> None:
    with pytest.raises(ValueError, match=r"Invalid /commands arguments:"):
        parse_commands_discovery_arguments('skills "unterminated')


def test_render_command_detail_markdown_contains_registry_action() -> None:
    rendered = render_command_detail_markdown("skills")

    assert rendered is not None
    assert "`registry`" in rendered
    assert "/skills registry [<number|url|path|mcp-server>]" in rendered


def test_render_command_detail_markdown_emits_one_top_level_usage_line() -> None:
    rendered = render_command_detail_markdown("skills")

    assert rendered is not None
    assert rendered.count("Usage:") == 1
    assert "Usage: `/skills" in rendered


def test_render_commands_json_detail_has_schema_version() -> None:
    rendered = render_commands_json(command_name="cards")

    assert '"schema_version": "1"' in rendered
    assert '"kind": "command_detail"' in rendered


def test_render_commands_json_index_uses_structured_actions() -> None:
    rendered = render_commands_json()

    payload = json.loads(rendered)
    commands = payload["commands"]
    skills = next(item for item in commands if item["name"] == "skills")
    mcp = next(item for item in commands if item["name"] == "mcp")

    assert payload["kind"] == "command_index"
    assert skills["actions"][0] == {"name": "list", "summary": ""}
    assert all(isinstance(action, dict) for action in skills["actions"])
    assert mcp["actions"][0]["summary"] == MCP_TOP_LEVEL_ACTION_DESCRIPTIONS["list"]
    assert all(isinstance(action, dict) for action in mcp["actions"])


def test_render_commands_json_index_has_unique_command_names() -> None:
    payload = json.loads(render_commands_json())
    names = [command["name"] for command in payload["commands"]]

    assert len(names) == len(set(names))


def test_render_commands_json_normalizes_command_name_filter() -> None:
    payload = json.loads(render_commands_json(command_names=[" SKILLS ", "   "]))

    assert [command["name"] for command in payload["commands"]] == ["skills"]


@pytest.mark.parametrize(
    "command_name",
    [
        "agent",
        "attach",
        "card",
        "connect",
        "fast",
        "history",
        "load",
        "mcpstatus",
        "prompt",
        "reload",
        "resume",
        "save",
    ],
)
def test_render_command_detail_markdown_covers_parser_only_commands(command_name: str) -> None:
    rendered = render_command_detail_markdown(command_name)

    assert rendered is not None
    assert f"# commands {command_name}" in rendered


def test_command_discovery_covers_parser_command_surface() -> None:
    ignored = {"exit", "help", "load_history", "save_history", "stop"}
    parser_commands = (
        set(prompt_parser._COMMAND_PARSERS)
        | set(prompt_parser._SIMPLE_SLASH_FACTORIES)
        | set(prompt_parser._SLASH_ACTION_FACTORIES)
        | set(prompt_parser._SLASH_ALIAS_PARSERS)
    )

    assert parser_commands - ignored <= set(command_discovery.command_discovery_names())


def test_command_discovery_mcp_actions_match_shared_parser_surface() -> None:
    rendered = render_commands_json()

    payload = json.loads(rendered)
    mcp = next(item for item in payload["commands"] if item["name"] == "mcp")

    assert [action["name"] for action in mcp["actions"]] == list(MCP_TOP_LEVEL_ACTIONS)


def test_command_discovery_session_action_summaries_use_shared_descriptions() -> None:
    payload = json.loads(render_commands_json(command_name="session"))
    actions = {
        action["name"]: action["summary"]
        for action in payload["command"]["actions"]
        if action["name"] in SESSION_COMMAND_COMPLETION_DESCRIPTIONS
        and action["name"] != "export"
    }

    assert actions == {
        name: SESSION_COMMAND_COMPLETION_DESCRIPTIONS[name]
        for name in actions
    }


def test_command_discovery_prompt_actions_match_parser_subcommands() -> None:
    payload = json.loads(render_commands_json(command_name="prompt"))
    action_names = [action["name"] for action in payload["command"]["actions"]]

    assert action_names == ["load"]
    assert "/prompts" in payload["command"]["examples"]


def test_render_discovery_json_adds_standard_envelope() -> None:
    rendered = command_discovery._render_discovery_json("sample", value=1)

    assert json.loads(rendered) == {
        "schema_version": "1",
        "kind": "sample",
        "value": 1,
    }
    assert rendered.startswith('{\n  "kind": "sample",')


def test_render_command_action_detail_markdown_contains_options() -> None:
    rendered = render_command_detail_markdown("cards", "publish")

    assert rendered is not None
    assert "# commands cards publish" in rendered
    assert "`--no-push`" in rendered
    assert "`--message text`, `-m`" in rendered


def test_render_model_fast_usage_includes_flex_value() -> None:
    rendered = render_command_detail_markdown("model", "fast")

    assert rendered is not None
    assert "Usage: `/model fast [on|off|flex|status]`" in rendered


def test_render_model_references_usage_includes_list_subaction() -> None:
    rendered = render_command_detail_markdown("model", "references")

    assert rendered is not None
    assert "/model references [list]" in rendered


def test_command_discovery_labels_escape_backticks_in_code_spans() -> None:
    argument_label = command_discovery._render_argument_label(
        {
            "name": "path`name",
            "summary": "",
            "value_name": "file`path",
            "required": True,
        }
    )
    option_label = command_discovery._render_option_label(
        {
            "name": "--flag`name",
            "summary": "",
            "value_name": "value`name",
            "aliases": ["-f`"],
        }
    )

    assert argument_label == "`` path`name `` (`` file`path ``)"
    assert option_label == "`` --flag`name value`name ``, `` -f` ``"


def test_command_discovery_metadata_renderers_skip_empty_labels() -> None:
    lines: list[str] = []

    command_discovery._render_argument_metadata(
        lines,
        [
            {"name": "", "summary": "ignored", "value_name": None, "required": True},
            {"name": "path", "summary": "file path", "value_name": "file", "required": True},
        ],
        indent="  ",
    )
    command_discovery._render_option_metadata(
        lines,
        [
            {"name": "", "summary": "ignored", "value_name": None, "aliases": []},
            {"name": "--force", "summary": "", "value_name": None, "aliases": ["-f"]},
        ],
        indent="  ",
    )

    assert lines == [
        "  - arguments:",
        "    - `path` (`file`) — file path",
        "  - options:",
        "    - `--force`, `-f`",
    ]


def test_command_discovery_metadata_renderers_skip_empty_sections() -> None:
    lines: list[str] = []

    command_discovery._render_argument_metadata(
        lines,
        [
            {"name": "", "summary": "ignored", "value_name": None, "required": True},
        ],
        indent="",
    )
    command_discovery._render_option_metadata(
        lines,
        [
            {"name": "   ", "summary": "ignored", "value_name": None, "aliases": []},
        ],
        indent="",
    )

    assert lines == []


def test_command_discovery_metadata_renderers_skip_structured_summaries() -> None:
    lines: list[str] = []
    argument_metadata = cast(
        "list[command_discovery.ActionArgumentPayload]",
        [
            {
                "name": "path",
                "summary": cast("Any", {"text": "structured"}),
                "value_name": "file",
                "required": True,
            },
        ],
    )
    option_metadata = cast(
        "list[command_discovery.ActionOptionPayload]",
        [
            {
                "name": "--force",
                "summary": cast("Any", ["structured"]),
                "value_name": None,
                "aliases": [],
            },
        ],
    )

    command_discovery._render_argument_metadata(
        lines,
        argument_metadata,
        indent="",
    )
    command_discovery._render_option_metadata(
        lines,
        option_metadata,
        indent="",
    )

    assert lines == [
        "- arguments:",
        "  - `path` (`file`)",
        "- options:",
        "  - `--force`",
    ]


def test_command_discovery_labels_trim_names_and_skip_blank_aliases() -> None:
    assert (
        command_discovery._render_argument_label(
            {
                "name": " path ",
                "summary": "",
                "value_name": " file ",
                "required": True,
            }
        )
        == "`path` (`file`)"
    )
    assert (
        command_discovery._render_option_label(
            {
                "name": " --output ",
                "summary": "",
                "value_name": " path ",
                "aliases": ["  ", " -o "],
            }
        )
        == "`--output path`, `-o`"
    )


def test_command_discovery_action_list_item_formats_aliases_and_summary() -> None:
    assert (
        command_discovery._render_action_list_item(
            {
                "name": " publish ",
                "summary": " publish cards ",
                "aliases": ["  ", " push "],
            }
        )
        == "- `publish` — publish cards (aliases: push)"
    )
    assert (
        command_discovery._render_action_list_item(
            {
                "name": "list",
                "summary": "",
            }
        )
        == "- `list`"
    )
    assert command_discovery._render_action_list_item({"name": "", "summary": ""}) is None


def test_command_discovery_action_list_item_escapes_summary_and_aliases() -> None:
    rendered = command_discovery._render_action_list_item(
        {
            "name": "publish",
            "summary": "Use [docs](bad) and *care*",
            "aliases": ["push_now"],
        }
    )

    assert rendered == "- `publish` — Use \\[docs\\](bad) and \\*care\\* (aliases: push\\_now)"


def test_command_discovery_action_payload_matching_normalizes_names_and_aliases() -> None:
    action: command_discovery.ActionPayload = {
        "name": "publish",
        "summary": "",
        "aliases": ["push", "deploy"],
    }

    assert command_discovery._action_payload_matches(action, "publish") is True
    assert command_discovery._action_payload_matches(action, "push") is True
    assert command_discovery._action_payload_matches(action, "deploy") is True
    assert command_discovery._action_payload_matches(action, "missing") is False
    assert command_discovery._action_payload_matches(action, "") is False


def test_command_discovery_metadata_renderers_escape_prose() -> None:
    lines: list[str] = []

    command_discovery._render_argument_metadata(
        lines,
        [
            {
                "name": "path",
                "summary": "Use [docs](bad) and *care*",
                "value_name": None,
                "required": True,
            },
        ],
        indent="",
    )
    command_discovery._render_action_metadata(
        lines,
        {
            "name": "publish",
            "summary": "",
            "notes": ["Default [docs](bad) and *care*."],
        },
        indent="",
    )

    assert "  - `path` — Use \\[docs\\](bad) and \\*care\\*" in lines
    assert "  - Default \\[docs\\](bad) and \\*care\\*." in lines


def test_command_discovery_action_metadata_normalizes_optional_lines() -> None:
    lines: list[str] = []

    command_discovery._render_action_metadata(
        lines,
        {
            "name": "publish",
            "summary": "",
            "usage": "  /cards publish  ",
            "notes": ["   ", "  Use the selected registry.  "],
            "examples": ["   ", "  /cards publish  "],
        },
        indent="",
    )

    assert lines == [
        "- usage: `/cards publish`",
        "- notes:",
        "  - Use the selected registry.",
        "- example: `/cards publish`",
    ]


def test_render_command_action_detail_markdown_skips_blank_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detail: command_discovery.CommandDetailEntry = {
        "name": "demo",
        "summary": "Demo command",
        "usage": "/demo",
        "actions": [{"name": "run", "summary": "Run demo", "usage": "   "}],
        "examples": [],
    }

    monkeypatch.setattr(command_discovery, "_build_command_detail", lambda _name: detail)

    rendered = render_command_detail_markdown("demo", "run")

    assert rendered is not None
    assert "Usage:" not in rendered


def test_command_discovery_metadata_summary_text_normalizes_values() -> None:
    assert command_discovery._metadata_summary_text("  details  ") == "details"
    assert command_discovery._metadata_summary_text(123) == "123"
    assert command_discovery._metadata_summary_text(12.5) == "12.5"
    assert command_discovery._metadata_summary_text("   ") is None
    assert command_discovery._metadata_summary_text(None) is None
    assert command_discovery._metadata_summary_text(True) is None
    assert command_discovery._metadata_summary_text(float("inf")) is None
    assert command_discovery._metadata_summary_text(float("nan")) is None
    assert command_discovery._metadata_summary_text({"text": "details"}) is None
    assert command_discovery._metadata_summary_text(["details"]) is None


def test_render_command_action_detail_markdown_emits_one_usage_line() -> None:
    rendered = render_command_detail_markdown("cards", "publish")

    assert rendered is not None
    assert rendered.count("Usage:") == 1
    assert "- usage:" not in rendered


def test_usage_without_label_normalizes_blank_usage() -> None:
    assert command_discovery._usage_without_label("Usage:   ") == ""
    assert command_discovery._usage_without_label("Usage: /session list") == "/session list"


def test_render_command_action_detail_markdown_uses_default_summary_for_blank_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detail: command_discovery.CommandDetailEntry = {
        "name": "demo",
        "summary": "Demo command",
        "usage": "/demo",
        "actions": [{"name": "run", "summary": "   "}],
        "examples": [],
    }

    monkeypatch.setattr(command_discovery, "_build_command_detail", lambda _name: detail)

    rendered = render_command_detail_markdown("demo", "run")

    assert rendered is not None
    assert "`/demo` action" in rendered


def test_render_commands_json_action_detail_has_schema_version() -> None:
    rendered = render_commands_json(command_name="skills", action_name="add")

    assert '"schema_version": "1"' in rendered
    assert '"kind": "command_action_detail"' in rendered
    assert '"/skills add [<number|name|github-url|path>] [--registry url] [--skills-dir path]"' in rendered
    assert '"name": "--skills-dir"' in rendered


def test_render_command_detail_markdown_session_includes_export_options() -> None:
    rendered = render_command_detail_markdown("session")

    assert rendered is not None
    assert f"Usage: `{FULL_SESSION_USAGE.removeprefix('Usage: ')}`" in rendered
    assert "`--output path`" in rendered
    assert "file path, not a directory path" in rendered
    assert "`--help`, `-h`" in rendered


def test_render_command_detail_markdown_session_accepts_clear_alias() -> None:
    rendered = render_command_detail_markdown("session", "clear")

    assert rendered is not None
    assert "# commands session delete" in rendered
    assert "Usage: `/session delete <id|number|all>`" in rendered


def test_render_commands_json_session_includes_export_behavior() -> None:
    rendered = render_commands_json(command_name="session")

    assert '"name": "export"' in rendered
    assert '"name": "--output"' in rendered
    assert '"Default format: codex."' in rendered


def test_render_direct_command_help_accepts_help_flags() -> None:
    rendered = render_direct_command_help("skills", "--help")

    assert rendered is not None
    assert "# commands skills" in rendered


def test_render_direct_command_help_accepts_action_help_alias() -> None:
    rendered = render_direct_command_help("cards", "publish help")

    assert rendered is not None
    assert "# commands cards publish" in rendered


def test_render_direct_command_help_ignores_split_errors() -> None:
    assert render_direct_command_help("cards", 'publish "unterminated') is None


def test_render_commands_index_markdown_has_tree_actions() -> None:
    rendered = render_commands_index_markdown()

    assert "Command map:" in rendered
    assert "- `/skills`" in rendered
    assert "  - list, available, search, add, remove, update, registry, help" in rendered


def test_render_commands_index_markdown_normalizes_command_name_filter() -> None:
    rendered = render_commands_index_markdown(command_names=[" SKILLS ", "   "])

    assert "- `/skills`" in rendered
    assert "- `/cards`" not in rendered


def test_render_commands_index_markdown_includes_plugins_catalog() -> None:
    rendered = render_commands_index_markdown()

    assert "- `/plugins`" in rendered
    assert "  - list, available, add, remove, update, registry, help" in rendered


def test_render_plugins_detail_uses_catalog() -> None:
    rendered = render_command_detail_markdown("plugins")

    assert rendered is not None
    assert "Usage: `/plugins [list|available|add|remove|update|registry|help] [args]`" in rendered
    assert "`available`" in rendered
