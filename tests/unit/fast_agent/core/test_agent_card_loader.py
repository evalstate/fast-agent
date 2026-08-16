"""Tests for agent_card_loader module."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import yaml

from fast_agent.agents.workflow.agents_as_tools_agent import HistoryMergeTarget, HistorySource
from fast_agent.core.agent_card_loader import (
    _agents_as_tools_options,
    _load_markdown_card,
    _resolve_name,
    dump_agent_to_string,
    load_agent_cards,
)
from fast_agent.core.exceptions import AgentConfigError
from fast_agent.tools.function_tool_config import FunctionToolSpec

if TYPE_CHECKING:
    from pathlib import Path


class TestResolveName:
    """Tests for _resolve_name function."""

    def test_name_with_spaces_replaced_by_underscores(self, tmp_path: Path) -> None:
        """Agent names with spaces should have them replaced with underscores."""
        dummy_path = tmp_path / "test.md"
        result = _resolve_name("cat card", dummy_path)
        assert result == "cat_card"

    def test_name_with_multiple_spaces(self, tmp_path: Path) -> None:
        """Multiple spaces should each be replaced with underscores."""
        dummy_path = tmp_path / "test.md"
        result = _resolve_name("my cool agent", dummy_path)
        assert result == "my_cool_agent"

    def test_name_without_spaces_unchanged(self, tmp_path: Path) -> None:
        """Names without spaces should remain unchanged."""
        dummy_path = tmp_path / "test.md"
        result = _resolve_name("my_agent", dummy_path)
        assert result == "my_agent"

    def test_name_from_path_stem_with_spaces(self, tmp_path: Path) -> None:
        """When name is None, path stem with spaces should be converted."""
        dummy_path = tmp_path / "cat card.md"
        result = _resolve_name(None, dummy_path)
        assert result == "cat_card"

    def test_name_from_path_stem_without_spaces(self, tmp_path: Path) -> None:
        """When name is None, path stem without spaces should be unchanged."""
        dummy_path = tmp_path / "my_agent.md"
        result = _resolve_name(None, dummy_path)
        assert result == "my_agent"

    def test_name_stripped_before_space_replacement(self, tmp_path: Path) -> None:
        """Name should be stripped of leading/trailing whitespace."""
        dummy_path = tmp_path / "test.md"
        result = _resolve_name("  cat card  ", dummy_path)
        assert result == "cat_card"

    def test_empty_name_raises_error(self, tmp_path: Path) -> None:
        """Empty string name should raise AgentConfigError."""
        dummy_path = tmp_path / "test.md"
        with pytest.raises(AgentConfigError):
            _resolve_name("", dummy_path)

    def test_whitespace_only_name_raises_error(self, tmp_path: Path) -> None:
        """Whitespace-only name should raise AgentConfigError."""
        dummy_path = tmp_path / "test.md"
        with pytest.raises(AgentConfigError):
            _resolve_name("   ", dummy_path)


def test_load_agent_card_normalizes_deprecated_smart_type(tmp_path: Path) -> None:
    card_path = tmp_path / "removed.md"
    card_path.write_text("---\ntype: smart\nname: removed\n---\n", encoding="utf-8")

    with pytest.warns(UserWarning, match="type 'smart' is deprecated"):
        loaded = load_agent_cards(card_path)[0]

    assert loaded.agent_data["type"] == "basic"
    assert loaded.agent_data["config"].subagents is True
    assert loaded.agent_data["config"].harness_tools is True
    dumped = dump_agent_to_string("removed", loaded.agent_data, as_yaml=True)
    assert "type: agent" in dumped
    assert "type: smart" not in dumped


def test_load_deprecated_smart_type_preserves_explicit_overrides(tmp_path: Path) -> None:
    card_path = tmp_path / "overrides.yaml"
    card_path.write_text(
        "type: smart\nname: overrides\nsubagents: false\nharness_tools: false\n",
        encoding="utf-8",
    )

    with pytest.warns(UserWarning, match="type 'smart' is deprecated"):
        config = load_agent_cards(card_path)[0].agent_data["config"]

    assert config.subagents is False
    assert config.harness_tools is False


def test_local_cards_resolve_nested_environment_and_preserve_file_directives(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    card_path = tmp_path / "local.md"
    card_path.write_text(
        "\n".join(
            [
                "---",
                'name: "${LOCAL_AGENT_NAME}"',
                "mcp_connect:",
                '  - target: "@example/server"',
                "    headers:",
                '      Authorization: "Bearer ${LOCAL_CARD_TOKEN}"',
                "---",
                "{{file:instructions.md}}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("LOCAL_AGENT_NAME", "local_agent")
    monkeypatch.setenv("LOCAL_CARD_TOKEN", "token")

    config = load_agent_cards(card_path)[0].agent_data["config"]

    assert config.name == "local_agent"
    assert config.mcp_connect[0].headers == {"Authorization": "Bearer token"}
    assert config.instruction == "{{file:instructions.md}}"


def test_local_cards_preserve_messages_paths(tmp_path: Path) -> None:
    history_path = tmp_path / "history.json"
    history_path.write_text("[]", encoding="utf-8")
    card_path = tmp_path / "local.yaml"
    card_path.write_text(
        "name: local_agent\nmessages: history.json\n",
        encoding="utf-8",
    )

    loaded = load_agent_cards(card_path)[0]

    assert loaded.message_files == [history_path.resolve()]


def test_load_agent_card_parses_mcp_connect_entries(tmp_path: Path) -> None:
    card_path = tmp_path / "mcp_agent.yaml"
    card_path.write_text(
        "\n".join(
            [
                "name: mcp_agent",
                "mcp_connect:",
                '  - target: "https://demo.hf.space"',
                '  - target: "  @foo/bar  "',
                '    name: "  foo_bar  "',
                "    headers:",
                '      Authorization: "Bearer abc"',
                "    auth:",
                "      oauth: false",
            ]
        ),
        encoding="utf-8",
    )

    loaded = load_agent_cards(card_path)
    assert len(loaded) == 1

    config = loaded[0].agent_data["config"]
    assert len(config.mcp_connect) == 2
    assert config.mcp_connect[0].target == "https://demo.hf.space"
    assert config.mcp_connect[0].name is None
    assert config.mcp_connect[0].headers is None
    assert config.mcp_connect[0].auth is None
    assert config.mcp_connect[1].target == "@foo/bar"
    assert config.mcp_connect[1].name == "foo_bar"
    assert config.mcp_connect[1].headers == {"Authorization": "Bearer abc"}
    assert config.mcp_connect[1].auth == {"oauth": False}


def test_mcp_connect_mapping_roundtrip_preserves_canonical_form(tmp_path: Path) -> None:
    card_path = tmp_path / "mcp_agent.yaml"
    card_path.write_text(
        "\n".join(
            [
                "name: mcp_agent",
                "mcp_connect:",
                "  docs:",
                '    target: "https://demo.hf.space"',
                "    protocol_mode: modern",
            ]
        ),
        encoding="utf-8",
    )

    loaded = load_agent_cards(card_path)[0]
    config = loaded.agent_data["config"]
    assert config.mcp_connect_source_form == "mapping"
    assert len(config.mcp_connect) == 1
    assert config.mcp_connect[0].name == "docs"
    assert config.mcp_connect[0].protocol_mode == "modern"

    dumped = dump_agent_to_string("mcp_agent", loaded.agent_data, as_yaml=True)
    payload = yaml.safe_load(dumped)
    assert payload["mcp_connect"] == {
        "docs": {
            "target": "https://demo.hf.space",
            "protocol_mode": "modern",
        }
    }

    roundtripped_path = tmp_path / "roundtripped.yaml"
    roundtripped_path.write_text(dumped, encoding="utf-8")
    roundtripped = load_agent_cards(roundtripped_path)[0].agent_data["config"]
    assert roundtripped.mcp_connect_source_form == "mapping"
    assert roundtripped.mcp_connect == config.mcp_connect


def test_mcp_connect_list_roundtrip_preserves_compatibility_form(tmp_path: Path) -> None:
    card_path = tmp_path / "mcp_agent.yaml"
    card_path.write_text(
        "name: mcp_agent\nmcp_connect:\n  - target: '@foo/bar'\n    protocol_mode: legacy\n",
        encoding="utf-8",
    )

    loaded = load_agent_cards(card_path)[0]
    dumped = dump_agent_to_string("mcp_agent", loaded.agent_data, as_yaml=True)
    payload = yaml.safe_load(dumped)

    assert isinstance(payload["mcp_connect"], list)
    assert payload["mcp_connect"][0]["protocol_mode"] == "legacy"


@pytest.mark.parametrize("process_field", ["command", "args", "env", "cwd"])
def test_mcp_connect_rejects_untrusted_process_fields(
    tmp_path: Path,
    process_field: str,
) -> None:
    card_path = tmp_path / "untrusted.yaml"
    card_path.write_text(
        "\n".join(
            [
                "name: untrusted",
                "mcp_connect:",
                "  docs:",
                "    target: '@foo/bar'",
                f"    {process_field}: untrusted",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(AgentConfigError, match=rf"unsupported keys: {process_field}"):
        load_agent_cards(card_path)


def test_load_agent_card_normalizes_padded_instruction(tmp_path: Path) -> None:
    card_path = tmp_path / "agent.yaml"
    card_path.write_text(
        "\n".join(
            [
                "name: test_agent",
                "instruction: '  Be helpful.  '",
            ]
        ),
        encoding="utf-8",
    )

    loaded = load_agent_cards(card_path)

    assert loaded[0].agent_data["config"].instruction == "Be helpful."


@pytest.mark.parametrize(
    ("suffix", "content"),
    [
        (
            ".md",
            "\n".join(
                [
                    "---",
                    "name: markdown_agent",
                    "subagents: true",
                    "subagent_model: '  passthrough  '",
                    "harness_tools: true",
                    "---",
                    "Be helpful.",
                ]
            ),
        ),
        (
            ".yaml",
            "\n".join(
                [
                    "name: yaml_agent",
                    "subagents: false",
                    "subagent_model: passthrough",
                    "harness_tools: true",
                ]
            ),
        ),
    ],
)
def test_load_agent_card_parses_subagent_controls(
    tmp_path: Path,
    suffix: str,
    content: str,
) -> None:
    card_path = tmp_path / f"agent{suffix}"
    card_path.write_text(content, encoding="utf-8")

    config = load_agent_cards(card_path)[0].agent_data["config"]

    assert config.subagents is (suffix == ".md")
    assert config.subagent_model == "passthrough"
    assert config.harness_tools is True


def test_load_agent_card_rejects_harness_tools_on_tool_only_agent(tmp_path: Path) -> None:
    card_path = tmp_path / "agent.yaml"
    card_path.write_text(
        "name: tool\ntool_only: true\nharness_tools: true\n",
        encoding="utf-8",
    )

    with pytest.raises(AgentConfigError, match="cannot be enabled on a tool-only agent"):
        load_agent_cards(card_path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("subagents", "not-a-bool", "subagents.*boolean"),
        ("subagents", "1", "subagents.*boolean"),
        ("subagents", "null", "subagents.*boolean"),
        ("subagent_model", "''", "subagent_model.*non-empty string"),
        ("subagent_model", "'   '", "subagent_model.*non-empty string"),
        ("harness_tools", "not-a-bool", "harness_tools.*boolean"),
        ("subagent_model", "null", "subagent_model.*non-empty string"),
    ],
)
def test_load_agent_card_rejects_invalid_subagent_controls(
    tmp_path: Path,
    field: str,
    value: str,
    message: str,
) -> None:
    card_path = tmp_path / "agent.yaml"
    card_path.write_text(f"name: agent\n{field}: {value}\n", encoding="utf-8")

    with pytest.raises(AgentConfigError, match=message):
        load_agent_cards(card_path)


def test_load_agent_card_accepts_declared_variables_metadata(tmp_path: Path) -> None:
    card_path = tmp_path / "classifier.md"
    card_path.write_text(
        "\n".join(
            [
                "---",
                "name: classifier",
                "model: passthrough",
                "variables:",
                "  policy: ''",
                "---",
                "",
                "Policy:",
                "{{policy}}",
            ]
        ),
        encoding="utf-8",
    )

    loaded = load_agent_cards(card_path)

    config = loaded[0].agent_data["config"]
    assert config.name == "classifier"
    assert "{{policy}}" in config.instruction


def test_load_agent_card_normalizes_markdown_body_markers(tmp_path: Path) -> None:
    card_path = tmp_path / "agent.md"
    card_path.write_text(
        "\n".join(
            [
                "   ---   ",
                "name: markdown_agent",
                "---",
                "",
                "   ---SYSTEM   ",
                "   Be helpful.   ",
            ]
        ),
        encoding="utf-8",
    )

    loaded = load_agent_cards(card_path)

    assert loaded[0].agent_data["config"].instruction == "Be helpful."


def test_load_agent_card_rejects_skill_manifest_with_clear_error(tmp_path: Path) -> None:
    skill_path = tmp_path / "SKILL.md"
    skill_path.write_text(
        "\n".join(
            [
                "---",
                "name: sample-skill",
                "description: Skill description",
                "metadata:",
                "  source: test",
                "---",
                "Skill body",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(AgentConfigError) as exc_info:
        load_agent_cards(skill_path)

    message = str(exc_info.value)
    assert "Agent Skill manifest, not an AgentCard" in message
    assert "read_text_file/read_skill" in message


def test_load_agent_card_reports_markdown_decode_errors(tmp_path: Path) -> None:
    card_path = tmp_path / "agent.md"
    card_path.write_bytes(b"---\nname: bad\n---\n\xff")

    with pytest.raises(AgentConfigError, match="Failed to parse frontmatter"):
        _load_markdown_card(card_path)


def test_load_agent_card_rejects_boolean_schema_version(tmp_path: Path) -> None:
    card_path = tmp_path / "agent.yaml"
    card_path.write_text(
        "\n".join(
            [
                "schema_version: true",
                "name: bool_schema_agent",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(AgentConfigError, match="schema_version.*integer"):
        load_agent_cards(card_path)


@pytest.mark.parametrize(
    ("option", "message"),
    [
        ("max_parallel", "integer"),
        ("max_display_instances", "integer"),
        ("child_timeout_sec", "number"),
    ],
)
def test_agents_as_tools_options_rejects_boolean_numeric_options(
    tmp_path: Path,
    option: str,
    message: str,
) -> None:
    with pytest.raises(AgentConfigError, match=message):
        _agents_as_tools_options({option: True}, tmp_path / "agent.yaml")


def test_load_agent_card_parses_provider_managed_mcp_connect_entries(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("STRIPE_TOKEN", "secret-token")
    card_path = tmp_path / "provider_mcp_agent.yaml"
    card_path.write_text(
        "\n".join(
            [
                "name: provider_mcp_agent",
                "mcp_connect:",
                "  - target: https://mcp.stripe.com",
                "    name: stripe",
                "    description: Stripe official MCP",
                "    management: provider",
                "    access_token: ${STRIPE_TOKEN}",
                "    defer_loading: true",
            ]
        ),
        encoding="utf-8",
    )

    loaded = load_agent_cards(card_path)
    config = loaded[0].agent_data["config"]

    assert len(config.mcp_connect) == 1
    entry = config.mcp_connect[0]
    assert entry.target == "https://mcp.stripe.com"
    assert entry.name == "stripe"
    assert entry.description == "Stripe official MCP"
    assert entry.management == "provider"
    assert entry.access_token == "secret-token"
    assert entry.defer_loading is True


def test_load_agent_card_parses_provider_managed_connector_entries(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("DROPBOX_TOKEN", "secret-token")
    card_path = tmp_path / "provider_connector_agent.yaml"
    card_path.write_text(
        "\n".join(
            [
                "name: provider_connector_agent",
                "mcp_connect:",
                "  - name: dropbox",
                "    management: provider",
                "    connector_id: connector_dropbox",
                "    access_token: ${DROPBOX_TOKEN}",
                "    defer_loading: true",
            ]
        ),
        encoding="utf-8",
    )

    loaded = load_agent_cards(card_path)
    config = loaded[0].agent_data["config"]

    assert len(config.mcp_connect) == 1
    entry = config.mcp_connect[0]
    assert entry.target is None
    assert entry.name == "dropbox"
    assert entry.management == "provider"
    assert entry.connector_id == "connector_dropbox"
    assert entry.access_token == "secret-token"
    assert entry.defer_loading is True


def test_dump_agent_card_preserves_mcp_connect_auth_fields(tmp_path: Path) -> None:
    card_path = tmp_path / "mcp_agent.yaml"
    card_path.write_text(
        "\n".join(
            [
                "name: mcp_agent",
                "mcp_connect:",
                '  - target: "https://demo.hf.space"',
                '    name: "demo"',
                "    headers:",
                '      Authorization: "Bearer abc"',
                "    auth:",
                "      oauth: false",
            ]
        ),
        encoding="utf-8",
    )

    loaded = load_agent_cards(card_path)
    dumped = dump_agent_to_string("mcp_agent", loaded[0].agent_data, as_yaml=True)

    assert "mcp_connect:" in dumped
    assert "headers:" in dumped
    assert "auth:" in dumped
    assert "Authorization: Bearer abc" in dumped


def test_dump_agent_card_serializes_agents_as_tools_enum_options(tmp_path: Path) -> None:
    card_path = tmp_path / "orchestrator.yaml"
    card_path.write_text(
        "\n".join(
            [
                "name: orchestrator",
                "type: agent",
                "agents:",
                "  - child",
            ]
        ),
        encoding="utf-8",
    )
    loaded = load_agent_cards(card_path)
    loaded[0].agent_data["agents_as_tools_options"] = {
        "history_source": HistorySource.CHILD,
        "history_merge_target": HistoryMergeTarget.ORCHESTRATOR,
    }

    dumped = dump_agent_to_string("orchestrator", loaded[0].agent_data, as_yaml=True)

    assert "history_source: child" in dumped
    assert "history_merge_target: orchestrator" in dumped


def test_dump_agent_card_preserves_provider_mcp_connect_fields(tmp_path: Path) -> None:
    card_path = tmp_path / "provider_mcp_agent.yaml"
    card_path.write_text(
        "\n".join(
            [
                "name: provider_mcp_agent",
                "mcp_connect:",
                "  - target: https://mcp.stripe.com",
                "    name: stripe",
                "    description: Stripe official MCP",
                "    management: provider",
                "    access_token: token-123",
                "    defer_loading: true",
            ]
        ),
        encoding="utf-8",
    )

    loaded = load_agent_cards(card_path)
    dumped = dump_agent_to_string("provider_mcp_agent", loaded[0].agent_data, as_yaml=True)

    assert "management: provider" in dumped
    assert "access_token: token-123" in dumped
    assert "defer_loading: true" in dumped


def test_dump_agent_card_preserves_provider_connector_fields(tmp_path: Path) -> None:
    card_path = tmp_path / "provider_connector_agent.yaml"
    card_path.write_text(
        "\n".join(
            [
                "name: provider_connector_agent",
                "mcp_connect:",
                "  - name: dropbox",
                "    management: provider",
                "    connector_id: connector_dropbox",
                "    access_token: token-123",
                "    defer_loading: true",
            ]
        ),
        encoding="utf-8",
    )

    loaded = load_agent_cards(card_path)
    dumped = dump_agent_to_string("provider_connector_agent", loaded[0].agent_data, as_yaml=True)

    assert "connector_id: connector_dropbox" in dumped
    assert "management: provider" in dumped
    assert "access_token: token-123" in dumped
    assert "defer_loading: true" in dumped


def test_load_agent_card_rejects_mcp_connect_unknown_keys(tmp_path: Path) -> None:
    card_path = tmp_path / "bad_mcp.yaml"
    card_path.write_text(
        "\n".join(
            [
                "name: bad_mcp",
                "mcp_connect:",
                '  - target: "@foo/bar"',
                '    alias: "foo"',
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(AgentConfigError, match="mcp_connect\\[0\\]"):
        load_agent_cards(card_path)


def test_load_agent_card_rejects_mcp_connect_missing_target(tmp_path: Path) -> None:
    card_path = tmp_path / "bad_mcp_target.yaml"
    card_path.write_text(
        "\n".join(
            [
                "name: bad_mcp",
                "mcp_connect:",
                "  - name: test",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(AgentConfigError, match="mcp_connect\\[0\\]\\.target"):
        load_agent_cards(card_path)


def test_load_agent_card_parses_tool_input_schema(tmp_path: Path) -> None:
    card_path = tmp_path / "schema_agent.yaml"
    card_path.write_text(
        "\n".join(
            [
                "name: schema_agent",
                "tool_input_schema:",
                "  type: object",
                "  properties:",
                "    query:",
                "      type: string",
                '      description: "What to investigate"',
                "  required:",
                "    - query",
            ]
        ),
        encoding="utf-8",
    )

    loaded = load_agent_cards(card_path)
    config = loaded[0].agent_data["config"]
    assert config.tool_input_schema == {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to investigate",
            },
        },
        "required": ["query"],
    }


def test_load_agent_card_parses_structured_function_tool_metadata(tmp_path: Path) -> None:
    card_path = tmp_path / "code_agent.yaml"
    card_path.write_text(
        "\n".join(
            [
                "name: code_agent",
                "function_tools:",
                "  - entrypoint: tools.py:run_query",
                "    variant: code",
                "    code_arg: code",
                "    language: python",
            ]
        ),
        encoding="utf-8",
    )

    loaded = load_agent_cards(card_path)
    config = loaded[0].agent_data["config"]

    assert config.function_tools is not None
    spec = config.function_tools[0]
    assert isinstance(spec, FunctionToolSpec)
    assert spec.entrypoint == "tools.py:run_query"
    assert spec.variant == "code"
    assert spec.code_arg == "code"
    assert spec.language == "python"


def test_dump_agent_card_preserves_structured_function_tool_metadata(tmp_path: Path) -> None:
    card_path = tmp_path / "code_agent.yaml"
    card_path.write_text(
        "\n".join(
            [
                "name: code_agent",
                "function_tools:",
                "  - entrypoint: tools.py:run_query",
                "    variant: code",
                "    language: python",
            ]
        ),
        encoding="utf-8",
    )

    loaded = load_agent_cards(card_path)
    dumped = dump_agent_to_string("code_agent", loaded[0].agent_data, as_yaml=True)

    assert "function_tools:" in dumped
    assert "entrypoint: tools.py:run_query" in dumped
    assert "variant: code" in dumped
    assert "language: python" in dumped


def test_load_agent_card_rejects_invalid_tool_input_schema(tmp_path: Path) -> None:
    card_path = tmp_path / "bad_schema_agent.yaml"
    card_path.write_text(
        "\n".join(
            [
                "name: bad_schema_agent",
                "tool_input_schema:",
                "  type: array",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(AgentConfigError, match="tool_input_schema"):
        load_agent_cards(card_path)


def test_load_agent_card_warns_when_required_property_description_missing(tmp_path: Path) -> None:
    card_path = tmp_path / "warn_schema_agent.yaml"
    card_path.write_text(
        "\n".join(
            [
                "name: warn_schema_agent",
                "tool_input_schema:",
                "  type: object",
                "  properties:",
                "    query:",
                "      type: string",
                "  required:",
                "    - query",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.warns(UserWarning, match="required property 'query'"):
        load_agent_cards(card_path)


def test_dump_agent_card_preserves_tool_input_schema(tmp_path: Path) -> None:
    card_path = tmp_path / "schema_agent.yaml"
    card_path.write_text(
        "\n".join(
            [
                "name: schema_agent",
                "tool_input_schema:",
                "  type: object",
                "  properties:",
                "    query:",
                "      type: string",
                '      description: "What to investigate"',
                "  required:",
                "    - query",
            ]
        ),
        encoding="utf-8",
    )

    loaded = load_agent_cards(card_path)
    dumped = dump_agent_to_string("schema_agent", loaded[0].agent_data, as_yaml=True)

    assert "tool_input_schema:" in dumped
    assert "query:" in dumped
    assert "required:" in dumped


def test_load_agent_card_parses_plugin_command_actions(tmp_path: Path) -> None:
    card_path = tmp_path / "agent.yaml"
    card_path.write_text(
        "\n".join(
            [
                "name: command_agent",
                "commands:",
                "  draft-next:",
                "    description: Draft the next user message",
                '    input_hint: "[format]"',
                '    handler: "commands.py:draft_next"',
                '    key: "c-x d"',
            ]
        ),
        encoding="utf-8",
    )

    loaded = load_agent_cards(card_path)
    commands = loaded[0].agent_data["config"].commands

    assert commands is not None
    assert commands["draft-next"].description == "Draft the next user message"
    assert commands["draft-next"].handler == "commands.py:draft_next"
    assert commands["draft-next"].input_hint == "[format]"
    assert commands["draft-next"].key == "c-x d"


def test_dump_agent_card_preserves_plugin_command_actions(tmp_path: Path) -> None:
    card_path = tmp_path / "agent.yaml"
    card_path.write_text(
        "\n".join(
            [
                "name: command_agent",
                "commands:",
                "  review-last:",
                "    description: Review the last response",
                '    handler: "commands.py:review_last"',
            ]
        ),
        encoding="utf-8",
    )

    loaded = load_agent_cards(card_path)
    dumped = dump_agent_to_string("command_agent", loaded[0].agent_data, as_yaml=True)

    assert "commands:" in dumped
    assert "review-last:" in dumped
    assert "description: Review the last response" in dumped
    assert "handler: commands.py:review_last" in dumped
