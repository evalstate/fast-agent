"""Tests for message_styles bottom metadata formatting with highlight jump."""

from fast_agent.ui.message_styles import (
    A3MessageStyle,
    ClassicMessageStyle,
    _format_bottom_metadata,
    _format_bottom_metadata_compact,
    _render_items_normal,
    _render_items_with_jump,
)


class TestFormatBottomMetadataCompact:
    """Tests for the compact (' • ' separator) formatter."""

    def test_all_items_fit_no_highlight(self) -> None:
        """When all items fit and no highlight, render normally."""
        items = ["aaa", "bbb", "ccc"]
        result = _format_bottom_metadata_compact(items, None, "bold", max_width=50)
        assert result.plain == "aaa • bbb • ccc"

    def test_all_items_fit_with_highlight(self) -> None:
        """When all items fit with highlight, render normally with highlight applied."""
        items = ["aaa", "bbb", "ccc"]
        result = _format_bottom_metadata_compact(items, 1, "bold", max_width=50)
        assert result.plain == "aaa • bbb • ccc"

    def test_truncation_no_highlight(self) -> None:
        """When items don't fit and no highlight, truncate with ellipsis."""
        items = ["aaa", "bbb", "ccc"]
        # Width 10: "aaa • bbb" = 9 chars, so ccc won't fit
        result = _format_bottom_metadata_compact(items, None, "bold", max_width=10)
        assert "…" in result.plain
        assert "ccc" not in result.plain

    def test_truncation_highlight_visible(self) -> None:
        """When highlight is visible even with truncation, render normally."""
        items = ["aaa", "bbb", "ccc"]
        # Highlight index 0 (aaa) - it's visible
        result = _format_bottom_metadata_compact(items, 0, "bold", max_width=10)
        assert "aaa" in result.plain

    def test_truncation_highlight_not_visible_shows_jump(self) -> None:
        """When highlighted item would be truncated, show jump indicator and item."""
        items = ["aaa", "bbb", "ccc", "ddd", "eee"]
        # Highlight index 4 (eee) - won't be visible normally in width 15
        result = _format_bottom_metadata_compact(items, 4, "bold", max_width=20)
        # Should show jump indicator and the highlighted item
        assert "▶" in result.plain
        assert "eee" in result.plain

    def test_highlight_at_end_of_long_list(self) -> None:
        """Highlight at the end of a long list should show jump."""
        items = ["item1", "item2", "item3", "item4", "item5", "highlighted"]
        # Small width forces truncation
        result = _format_bottom_metadata_compact(items, 5, "bold", max_width=30)
        assert "▶" in result.plain
        assert "highlighted" in result.plain

    def test_truncation_with_wide_chars_forces_jump(self) -> None:
        """Wide characters should be measured by display width, not len()."""
        items = ["a", "🐍🐍", "bb", "cc"]
        result = _format_bottom_metadata_compact(items, 2, "bold", max_width=11)
        assert "▶" in result.plain
        assert "bb" in result.plain


class TestFormatBottomMetadata:
    """Tests for the classic (' | ' separator) formatter."""

    def test_all_items_fit_no_highlight(self) -> None:
        """When all items fit and no highlight, render normally."""
        items = ["aaa", "bbb", "ccc"]
        result = _format_bottom_metadata(items, None, "bold", max_width=50)
        assert result.plain == "aaa | bbb | ccc"

    def test_truncation_highlight_not_visible_shows_jump(self) -> None:
        """When highlighted item would be truncated, show jump indicator."""
        items = ["aaa", "bbb", "ccc", "ddd", "eee"]
        result = _format_bottom_metadata(items, 4, "bold", max_width=20)
        assert "▶" in result.plain
        assert "eee" in result.plain


class TestRenderItemsNormal:
    """Tests for normal item rendering."""

    def test_empty_list(self) -> None:
        """Empty list returns empty Text."""
        result = _render_items_normal([], None, "bold", "dim", " • ", max_width=50)
        assert result.plain == ""

    def test_single_item(self) -> None:
        """Single item renders without separator."""
        result = _render_items_normal(["single"], None, "bold", "dim", " • ", max_width=50)
        assert result.plain == "single"

    def test_ellipsis_when_nothing_fits(self) -> None:
        """Very small width shows just ellipsis."""
        result = _render_items_normal(["toolong"], None, "bold", "dim", " • ", max_width=3)
        assert result.plain == "…"


class TestRenderItemsWithJump:
    """Tests for jump rendering."""

    def test_jump_to_highlighted(self) -> None:
        """Jump indicator shown before highlighted item."""
        items = ["a", "b", "c", "d", "highlighted"]
        result = _render_items_with_jump(items, 4, "bold", "dim", " • ", max_width=30)
        assert " ▶ " in result.plain
        assert "highlighted" in result.plain

    def test_jump_with_no_space_for_prefix(self) -> None:
        """When no space for prefix, show just jump + highlight."""
        items = ["a", "b", "c", "highlighted"]
        # Very small width - just enough for jump + highlight
        result = _render_items_with_jump(items, 3, "bold", "dim", " • ", max_width=16)
        assert "highlighted" in result.plain

    def test_jump_preserves_some_prefix_items(self) -> None:
        """Prefix items that fit before jump are shown."""
        items = ["a", "b", "c", "d", "e", "highlighted"]
        result = _render_items_with_jump(items, 5, "bold", "dim", " • ", max_width=30)
        # Should have some prefix, jump, then highlighted
        assert "▶" in result.plain
        assert "highlighted" in result.plain
        # At least 'a' should be visible
        assert "a" in result.plain

    def test_jump_skips_indicator_for_first_item(self) -> None:
        """Jump indicator is omitted when highlighting the first item."""
        items = ["highlighted", "b", "c"]
        result = _render_items_with_jump(items, 0, "bold", "dim", " • ", max_width=20)
        assert "▶" not in result.plain
        assert "highlighted" in result.plain


class TestA3MessageStyle:
    """Tests for A3MessageStyle."""

    def test_bottom_metadata_with_jump(self) -> None:
        """A3 style integrates the jump functionality."""
        style = A3MessageStyle()
        items = ["tool1", "tool2", "tool3", "tool4", "highlighted"]
        result = style.bottom_metadata_line(
            items=items,
            highlight_index=4,
            highlight_color="bold",
            max_item_length=None,
            width=40,
        )
        assert result is not None
        # Should contain the prefix and highlighted item
        assert "highlighted" in result.plain

    def test_shell_exit_line_with_detail(self) -> None:
        """A3 shell exit lines include optional compact detail text."""
        style = A3MessageStyle()
        result = style.shell_exit_line(
            exit_code=0,
            width=80,
            detail="(no output) id: call_…123456",
        )
        assert "exit code 0" in result.plain
        assert "(no output)" in result.plain
        assert "id: call_…123456" in result.plain


class TestClassicMessageStyle:
    """Tests for ClassicMessageStyle."""

    def test_bottom_metadata_with_jump(self) -> None:
        """Classic style integrates the jump functionality."""
        style = ClassicMessageStyle()
        items = ["tool1", "tool2", "tool3", "tool4", "highlighted"]
        result = style.bottom_metadata_line(
            items=items,
            highlight_index=4,
            highlight_color="bold",
            max_item_length=None,
            width=50,
        )
        assert result is not None
        assert "highlighted" in result.plain

    def test_shell_exit_line_with_detail(self) -> None:
        """Classic shell exit lines include optional compact detail text."""
        style = ClassicMessageStyle()
        result = style.shell_exit_line(
            exit_code=0,
            width=80,
            detail="(no output) id: call_…123456",
        )
        assert "exit code 0" in result.plain
        assert "(no output)" in result.plain
        assert "id: call_…123456" in result.plain
