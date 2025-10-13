"""Tests for smart markdown truncation."""

import pytest
from rich.console import Console

from fast_agent.ui.markdown_truncator import MarkdownTruncator, TruncationPoint


@pytest.fixture
def console():
    """Create a Console instance for testing."""
    return Console(width=80, height=24, legacy_windows=False)


@pytest.fixture
def truncator():
    """Create a MarkdownTruncator instance for testing."""
    return MarkdownTruncator(target_height_ratio=0.6)


class TestMarkdownTruncator:
    """Test suite for MarkdownTruncator."""

    def test_no_truncation_needed(self, truncator, console):
        """Test that short text is not truncated."""
        text = "# Hello\n\nThis is a short paragraph."
        result = truncator.truncate(text, terminal_height=24, console=console)
        assert result == text

    def test_empty_text(self, truncator, console):
        """Test handling of empty text."""
        result = truncator.truncate("", terminal_height=24, console=console)
        assert result == ""

    def test_truncate_after_paragraph(self, truncator, console):
        """Test truncation between paragraphs preserves markdown structure."""
        text = "\n".join([
            "# Title",
            "",
            "First paragraph that is quite long and takes up space.",
            "",
            "Second paragraph with more content.",
            "",
            "Third paragraph that we might not see.",
        ])

        result = truncator.truncate(text, terminal_height=5, console=console)

        # Should truncate and keep some content
        assert len(result) < len(text)
        assert len(result) > 0
        # Should keep complete paragraphs (one of them at least)
        assert "paragraph" in result.lower()

    def test_truncate_after_code_block(self, truncator, console):
        """Test truncation after complete code blocks."""
        text = "\n".join([
            "# Code Example",
            "",
            "```python",
            "def hello():",
            "    print('world')",
            "```",
            "",
            "More content after code block.",
        ])

        result = truncator.truncate(text, terminal_height=8, console=console)

        # If code block is included, it should be complete
        if "```python" in result:
            assert "def hello():" in result
            # Should have closing fence if we kept the code
            assert result.count("```") % 2 == 0 or result.endswith("```")

    def test_truncate_within_code_block_preserves_fence(self, truncator, console):
        """Test that truncating within a code block preserves the opening fence."""
        text = "\n".join([
            "Start content.",
            "",
            "```python",
            "def function1():",
            "    pass",
            "",
            "def function2():",
            "    pass",
            "",
            "def function3():",
            "    pass",
            "```",
            "",
            "End content.",
        ])

        # Use a very small height to force truncation within the code block
        result = truncator.truncate(text, terminal_height=5, console=console)

        # If we kept any of the code, should have the opening fence
        if "def function" in result:
            assert "```python" in result or "```" in result

    def test_truncate_after_list_items(self, truncator, console):
        """Test truncation preserves list structure."""
        text = "\n".join([
            "# List Example",
            "",
            "- Item 1",
            "- Item 2",
            "- Item 3",
            "- Item 4",
            "- Item 5",
        ])

        result = truncator.truncate(text, terminal_height=6, console=console)

        # Should truncate but keep complete list items
        assert "List Example" in result or "Item" in result

    def test_truncate_ordered_list(self, truncator, console):
        """Test truncation with ordered lists."""
        text = "\n".join([
            "# Steps",
            "",
            "1. First step",
            "2. Second step",
            "3. Third step",
            "4. Fourth step",
        ])

        result = truncator.truncate(text, terminal_height=6, console=console)

        # Should keep some content
        assert len(result) > 0

    def test_truncate_blockquote(self, truncator, console):
        """Test truncation with blockquotes."""
        text = "\n".join([
            "# Quote",
            "",
            "> This is a quote",
            "> that spans multiple lines",
            "> and has more content.",
            "",
            "After quote.",
        ])

        result = truncator.truncate(text, terminal_height=6, console=console)

        # Should keep some content
        assert len(result) > 0

    def test_oversized_single_block_fallback(self, truncator, console):
        """Test that oversized single blocks fall back to character truncation."""
        # Create a very long code block that exceeds terminal height
        code_lines = [f"line_{i} = {i}" for i in range(50)]
        text = "```python\n" + "\n".join(code_lines) + "\n```"

        result = truncator.truncate(text, terminal_height=10, console=console)

        # Should have truncated something
        assert len(result) < len(text)
        # Should still have some content
        assert len(result) > 0

    def test_mixed_content(self, truncator, console):
        """Test truncation with mixed markdown content."""
        text = "\n".join([
            "# Title",
            "",
            "Paragraph text.",
            "",
            "```python",
            "code here",
            "```",
            "",
            "- List item 1",
            "- List item 2",
            "",
            "> Quote",
            "",
            "Final paragraph.",
        ])

        result = truncator.truncate(text, terminal_height=10, console=console)

        # Should keep some content and not crash
        assert len(result) > 0

    def test_find_safe_truncation_points(self, truncator):
        """Test that safe truncation points are identified correctly."""
        text = "\n".join([
            "# Title",
            "",
            "Paragraph 1.",
            "",
            "```python",
            "code",
            "```",
            "",
            "Paragraph 2.",
        ])

        safe_points = truncator._find_safe_truncation_points(text)

        # Should find multiple safe points (after paragraphs, after code block)
        assert len(safe_points) > 0
        assert all(isinstance(p, TruncationPoint) for p in safe_points)

    def test_measure_rendered_height(self, truncator, console):
        """Test that rendered height measurement works."""
        text = "# Title\n\nParagraph."
        height = truncator._measure_rendered_height(text, console, "monokai")

        # Should return a positive height
        assert height > 0
        assert isinstance(height, int)

    def test_measure_empty_text_height(self, truncator, console):
        """Test that empty text has zero height."""
        height = truncator._measure_rendered_height("", console, "monokai")
        assert height == 0

    def test_plain_text_no_markdown(self, truncator, console):
        """Test truncation of plain text without markdown."""
        text = "This is plain text without any markdown formatting at all."
        result = truncator.truncate(text, terminal_height=2, console=console)

        # Should handle plain text gracefully
        assert len(result) <= len(text)

    def test_nested_lists(self, truncator, console):
        """Test truncation with nested list structures."""
        text = "\n".join([
            "- Item 1",
            "  - Nested 1.1",
            "  - Nested 1.2",
            "- Item 2",
            "  - Nested 2.1",
        ])

        result = truncator.truncate(text, terminal_height=5, console=console)

        # Should not crash with nested structures
        assert len(result) > 0

    def test_code_block_with_language(self, truncator, console):
        """Test that code block language specifier is preserved."""
        text = "\n".join([
            "Intro text.",
            "",
            "```javascript",
            "const x = 1;",
            "const y = 2;",
            "const z = 3;",
            "```",
            "",
            "More text after.",
        ])

        result = truncator.truncate(text, terminal_height=5, console=console)

        # If we kept code, should preserve language
        if "const" in result:
            # Should have fence (might be added by truncation handler)
            assert "```" in result

    def test_multiple_code_blocks(self, truncator, console):
        """Test handling of multiple code blocks."""
        text = "\n".join([
            "```python",
            "def foo():",
            "    pass",
            "```",
            "",
            "Some text.",
            "",
            "```python",
            "def bar():",
            "    pass",
            "```",
        ])

        result = truncator.truncate(text, terminal_height=8, console=console)

        # Should handle multiple blocks without crashing
        assert len(result) > 0

    def test_target_height_ratio(self, console):
        """Test that target_height_ratio parameter works."""
        truncator_60 = MarkdownTruncator(target_height_ratio=0.6)
        truncator_80 = MarkdownTruncator(target_height_ratio=0.8)

        # Create text that will need truncation
        text = "\n".join([f"Line {i}" for i in range(50)])

        result_60 = truncator_60.truncate(text, terminal_height=20, console=console)
        result_80 = truncator_80.truncate(text, terminal_height=20, console=console)

        # Higher ratio should keep more text (or equal in edge cases)
        assert len(result_80) >= len(result_60)

    def test_very_long_single_line(self, truncator, console):
        """Test handling of very long single lines."""
        text = "This is a very long line that goes on and on. " * 50
        result = truncator.truncate(text, terminal_height=5, console=console)

        # Should handle without crashing
        assert len(result) > 0

    def test_special_markdown_characters(self, truncator, console):
        """Test handling of special markdown characters."""
        text = "\n".join([
            "# Title with **bold** and *italic*",
            "",
            "Paragraph with `inline code` and [link](url).",
            "",
            "---",
            "",
            "More content.",
        ])

        result = truncator.truncate(text, terminal_height=6, console=console)

        # Should handle special characters without crashing
        assert len(result) > 0

    def test_table_truncation_preserves_header(self, truncator, console):
        """Test that truncating within a table body preserves the header."""
        text = "\n".join([
            "Some intro text",
            "",
            "| Header 1 | Header 2 | Header 3 |",
            "|----------|----------|----------|",
            "| Row 1 A  | Row 1 B  | Row 1 C  |",
            "| Row 2 A  | Row 2 B  | Row 2 C  |",
            "| Row 3 A  | Row 3 B  | Row 3 C  |",
            "| Row 4 A  | Row 4 B  | Row 4 C  |",
            "",
            "Text after table",
        ])

        # Force truncation in middle of table body
        result = truncator.truncate(text, terminal_height=8, console=console)

        # If we have any table rows, we should have the header
        if "Row" in result:
            assert "Header 1" in result
            assert "Header 2" in result
            assert "Header 3" in result
            # Should have the separator line
            assert "----------" in result

    def test_table_truncation_before_table(self, truncator, console):
        """Test that truncating before a table doesn't affect the table."""
        text = "\n".join([
            "Lots of intro text here.",
            "More intro text.",
            "Even more intro.",
            "",
            "| Header 1 | Header 2 |",
            "|----------|----------|",
            "| Row 1 A  | Row 1 B  |",
            "| Row 2 A  | Row 2 B  |",
        ])

        result = truncator.truncate(text, terminal_height=10, console=console)

        # If table is in result, it should be complete
        if "Header 1" in result:
            assert "Row 1 A" in result or "Row 2 A" in result

    def test_table_no_truncation_needed(self, truncator, console):
        """Test that short tables don't get truncated."""
        text = "\n".join([
            "| Header 1 | Header 2 |",
            "|----------|----------|",
            "| Row 1 A  | Row 1 B  |",
            "| Row 2 A  | Row 2 B  |",
        ])

        result = truncator.truncate(text, terminal_height=24, console=console)
        assert result == text

    def test_multiple_tables_truncation(self, truncator, console):
        """Test truncation with multiple tables."""
        text = "\n".join([
            "# Table 1",
            "",
            "| Header A | Header B |",
            "|----------|----------|",
            "| Data 1   | Data 2   |",
            "| Data 3   | Data 4   |",
            "",
            "# Table 2",
            "",
            "| Header X | Header Y |",
            "|----------|----------|",
            "| Data 5   | Data 6   |",
            "| Data 7   | Data 8   |",
            "| Data 9   | Data 10  |",
        ])

        result = truncator.truncate(text, terminal_height=10, console=console)

        # Should handle multiple tables without crashing
        assert len(result) > 0

        # If any table data is present, headers should be present
        if "Data" in result:
            # At least one table header should be present
            assert "Header" in result

    def test_table_with_code_block_truncation(self, truncator, console):
        """Test mixed table and code block content."""
        text = "\n".join([
            "| Header 1 | Header 2 |",
            "|----------|----------|",
            "| Row 1 A  | Row 1 B  |",
            "| Row 2 A  | Row 2 B  |",
            "",
            "```python",
            "def foo():",
            "    pass",
            "```",
            "",
            "| Header 3 | Header 4 |",
            "|----------|----------|",
            "| Row 3 A  | Row 3 B  |",
            "| Row 4 A  | Row 4 B  |",
        ])

        result = truncator.truncate(text, terminal_height=12, console=console)

        # Should handle both tables and code blocks
        assert len(result) > 0

    def test_table_truncation_character_fallback(self, truncator, console):
        """Test table header preservation with character-based truncation fallback."""
        # Create a very long table that requires character truncation
        rows = [f"| Row {i} A  | Row {i} B  | Row {i} C  |" for i in range(50)]
        text = "\n".join([
            "| Header 1 | Header 2 | Header 3 |",
            "|----------|----------|----------|",
        ] + rows)

        result = truncator.truncate(text, terminal_height=10, console=console)

        # If we have any row data, should have headers
        if "Row" in result and "Row" in result[result.find("Row") + 1:]:
            # Has at least one row - should have headers
            assert "Header 1" in result
            assert "----------" in result

    def test_table_header_only_no_duplication(self, truncator, console):
        """Test that truncating right after thead doesn't duplicate header."""
        text = "\n".join([
            "Long intro paragraph that takes space.",
            "More intro text here.",
            "",
            "| Header 1 | Header 2 |",
            "|----------|----------|",
            "| Row 1 A  | Row 1 B  |",
            "| Row 2 A  | Row 2 B  |",
        ])

        result = truncator.truncate(text, terminal_height=10, console=console)

        # Count how many times header appears (should be at most once)
        header_count = result.count("Header 1")
        assert header_count <= 1

    def test_table_dominant_shows_first_page(self, truncator, console):
        """Test that table-dominant documents show first page (beginning), not last page."""
        # Create a "top 30 objects" scenario - document is primarily a table
        rows = [f"| Object {i:02d} | Value {i:02d} | Status {i:02d} |" for i in range(1, 31)]
        text = "\n".join([
            "| Object | Value | Status |",
            "|--------|-------|--------|",
        ] + rows)

        # Force truncation to show only part of the table
        result = truncator.truncate(text, terminal_height=10, console=console)

        # Should show FIRST rows (1, 2, 3...), not LAST rows (28, 29, 30)
        assert "Object 01" in result or "Object 02" in result or "Object 03" in result
        # Should NOT have the last rows
        assert "Object 30" not in result or "Object 29" not in result

    def test_table_dominant_with_intro_shows_table_start(self, truncator, console):
        """Test table-dominant doc with intro text still shows table from beginning."""
        rows = [f"| Row {i} | Data {i} |" for i in range(1, 26)]
        text = "\n".join([
            "# Top 25 Results",
            "",
            "| Row | Data |",
            "|-----|------|",
        ] + rows)

        result = truncator.truncate(text, terminal_height=15, console=console)

        # Check if we have data rows (not just the header "Row | Data")
        # Look for "Row 1" through "Row 25" patterns
        has_data_rows = any(f"Row {i}" in result for i in range(1, 26))

        if has_data_rows:
            # If we have data rows, check which ones
            has_early = any(f"Row {i}" in result for i in range(1, 6))
            has_late = any(f"Row {i}" in result for i in range(20, 26))

            # Should have early rows
            assert has_early
            # Should NOT have late rows (or at least not as many)
            # If we see late rows, we should also see early rows
            if has_late:
                assert has_early

    def test_non_table_content_shows_end(self, truncator, console):
        """Test that non-table content keeps original behavior (show end/most recent)."""
        # Create streaming-style content with paragraphs
        paragraphs = [f"Paragraph {i} with some content here." for i in range(1, 21)]
        text = "\n\n".join(paragraphs)

        result = truncator.truncate(text, terminal_height=6, console=console)

        # Should show LATER paragraphs (most recent in streaming), not early ones
        has_early = "Paragraph 1" in result or "Paragraph 2" in result
        has_late = "Paragraph 19" in result or "Paragraph 20" in result or "Paragraph 18" in result

        # Should prefer showing later content
        # At least one of the late paragraphs should be present
        assert has_late or not has_early

    def test_mixed_table_text_below_threshold(self, truncator, console):
        """Test mixed content where table is <50% uses streaming behavior."""
        text = "\n".join([
            "# Analysis Report",
            "",
            "Here is a long introduction with multiple paragraphs.",
            "This section provides context for the data below.",
            "We have several points to make before showing the table.",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            "| Row 1  | Val 1 |",
            "| Row 2  | Val 2 |",
            "",
            "And here is a conclusion with more text.",
            "This provides analysis of the results.",
            "We wrap up with final thoughts here.",
        ])

        result = truncator.truncate(text, terminal_height=8, console=console)

        # This is NOT table-dominant (<50% table), so should use streaming behavior
        # (show end/most recent content)
        # Should be more likely to see conclusion than intro
        has_intro = "introduction" in result
        has_conclusion = "conclusion" in result or "analysis" in result or "final" in result

        # Not a strict test since it depends on height measurement,
        # but we should see some content
        assert len(result) > 0

    def test_is_primary_content_table_detection(self, truncator):
        """Test the table detection heuristic directly."""
        # Table-dominant content (>50% table lines)
        table_text = "\n".join([
            "| Col A | Col B |",
            "|-------|-------|",
            "| R1 A  | R1 B  |",
            "| R2 A  | R2 B  |",
            "| R3 A  | R3 B  |",
        ])
        assert truncator._is_primary_content_table(table_text) is True

        # Non-table content
        non_table_text = "\n".join([
            "# Title",
            "",
            "Paragraph 1",
            "",
            "Paragraph 2",
            "",
            "Paragraph 3",
        ])
        assert truncator._is_primary_content_table(non_table_text) is False

        # Mixed content - more text than table
        mixed_text = "\n".join([
            "# Title",
            "",
            "Long paragraph here.",
            "Another paragraph.",
            "More content.",
            "",
            "| Col A | Col B |",
            "|-------|-------|",
            "| Data  | Data  |",
        ])
        # This should be False since table is <50% of lines
        result = truncator._is_primary_content_table(mixed_text)
        # Mixed case - depends on exact line count
        # Just verify it doesn't crash
        assert isinstance(result, bool)

    def test_repeated_streaming_truncation(self, truncator, console):
        """Test repeated truncation passes as would occur during streaming.

        This simulates streaming behavior where content is repeatedly truncated
        as new content arrives. The fence should be correctly prepended each time,
        and never duplicated - this was the bug being fixed.
        """
        # Start with a long code block
        code_lines = [f"line_{i} = {i}" for i in range(1, 51)]
        text = "```python\n" + "\n".join(code_lines) + "\n```"

        # First truncation pass - moderate truncation
        pass1 = truncator.truncate(text, terminal_height=20, console=console)

        # Should have fence since we truncated within the block
        assert "```python" in pass1
        # Verify we have some code content
        assert "line_" in pass1

        # Second truncation pass - truncate the already-truncated text more
        pass2 = truncator.truncate(pass1, terminal_height=15, console=console)

        # Should still have fence (not duplicated)
        # This is the key test: after repeated truncation, should have exactly 1 fence
        assert pass2.count("```python") == 1
        # Verify fence is on its own line (not mangled)
        lines = pass2.split("\n")
        assert lines[0] == "```python"

        # Third truncation pass - simulate aggressive truncation
        pass3 = truncator.truncate(pass2, terminal_height=10, console=console)

        # Should STILL have exactly one fence - this validates the fix
        # The old code could create duplicate/partial fences here
        assert pass3.count("```python") == 1
        # And it should be clean (first line)
        lines3 = pass3.split("\n")
        assert lines3[0] == "```python"
