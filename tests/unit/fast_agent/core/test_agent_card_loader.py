"""Tests for agent_card_loader module."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from fast_agent.core.agent_card_loader import _resolve_name
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
