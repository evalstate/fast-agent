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
