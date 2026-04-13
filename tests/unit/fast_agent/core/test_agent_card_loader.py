"""Tests for agent_card_loader module."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from fast_agent.core.agent_card_loader import _resolve_name, dump_agent_to_string, load_agent_cards
from fast_agent.core.exceptions import AgentConfigError

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


def test_load_agent_card_parses_mcp_connect_entries(tmp_path: Path) -> None:
    card_path = tmp_path / "mcp_agent.yaml"
    card_path.write_text(
        "\n".join(
            [
                "name: mcp_agent",
                "mcp_connect:",
                '  - target: "https://demo.hf.space"',
                '  - target: "@foo/bar"',
                '    name: "foo_bar"',
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
                '      type: string',
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
    assert getattr(spec, "entrypoint") == "tools.py:run_query"
    assert getattr(spec, "variant") == "code"
    assert getattr(spec, "code_arg") == "code"
    assert getattr(spec, "language") == "python"


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
                '      type: string',
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
