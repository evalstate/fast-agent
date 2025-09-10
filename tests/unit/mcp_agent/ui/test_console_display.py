"""Unit tests for console_display module, specifically _prepare_markdown_content."""

from fast_agent.ui.console_display import _prepare_markdown_content


class TestPrepareMarkdownContent:
    """Test the _prepare_markdown_content function."""

    def test_none_input(self):
        """Test that None input returns None unchanged."""
        result = _prepare_markdown_content(None)
        assert result is None

    def test_none_input_with_escape_false(self):
        """Test that None input returns None even when escape_xml is False."""
        result = _prepare_markdown_content(None, escape_xml=False)
        assert result is None

    def test_empty_string(self):
        """Test that empty string doesn't crash and returns empty string."""
        result = _prepare_markdown_content("")
        assert result == ""

    def test_empty_string_with_escape_false(self):
        """Test that empty string returns empty when escape_xml is False."""
        result = _prepare_markdown_content("", escape_xml=False)
        assert result == ""

    def test_escape_xml_false_returns_unchanged(self):
        """Test that escape_xml=False returns content unchanged."""
        content = "<tag>content & 'quotes' \"double\"</tag>"
        result = _prepare_markdown_content(content, escape_xml=False)
        assert result == content

    def test_non_string_input(self):
        """Test that non-string inputs are returned unchanged."""
        # Test with integer
        result = _prepare_markdown_content(123)
        assert result == 123

        # Test with list
        result = _prepare_markdown_content([1, 2, 3])
        assert result == [1, 2, 3]

        # Test with dict
        test_dict = {"key": "value"}
        result = _prepare_markdown_content(test_dict)
        assert result == test_dict

    def test_basic_html_escaping(self):
        """Test that HTML characters are properly escaped outside code blocks."""
        content = "This has <tag> and & and > and < and \" and ' characters"
        result = _prepare_markdown_content(content)
        expected = (
            "This has &lt;tag&gt; and &amp; and &gt; and &lt; and &quot; and &#39; characters"
        )
        assert result == expected

    def test_preserves_fenced_code_blocks(self):
        """Test that content inside fenced code blocks is not escaped."""
        content = """Before code
```python
def func():
    return "<tag>" & 'value'
```
After code with <tag>"""
        result = _prepare_markdown_content(content)

        # Check that code block content is preserved
        assert "def func():" in result
        assert "return \"<tag>\" & 'value'" in result

        # Check that content outside code blocks is escaped
        assert "After code with &lt;tag&gt;" in result

    def test_preserves_inline_code(self):
        """Test that content inside inline code is not escaped."""
        content = "Use `<tag>` and `x & y` in code, but escape <tag> outside"
        result = _prepare_markdown_content(content)

        # Inline code should be preserved
        assert "`<tag>`" in result
        assert "`x & y`" in result

        # Outside content should be escaped
        assert "but escape &lt;tag&gt; outside" in result

    def test_multiple_code_blocks(self):
        """Test handling of multiple code blocks in the same content."""
        content = """First <tag>
```
<code1> & "quotes"
```
Middle <tag>
```
<code2> & 'quotes'
```
End <tag>"""
        result = _prepare_markdown_content(content)

        # Code blocks should be preserved
        assert '<code1> & "quotes"' in result
        assert "<code2> & 'quotes'" in result

        # Outside content should be escaped
        assert "First &lt;tag&gt;" in result
        assert "Middle &lt;tag&gt;" in result
        assert "End &lt;tag&gt;" in result

    def test_mixed_inline_and_fenced_code(self):
        """Test content with both inline and fenced code blocks."""
        content = """Use `<inline>` here
```
<fenced> & "code"
```
And `<more>` inline with <tag> outside"""
        result = _prepare_markdown_content(content)

        # Both types of code should be preserved
        assert "`<inline>`" in result
        assert '<fenced> & "code"' in result
        assert "`<more>`" in result

        # Outside content should be escaped
        assert "with &lt;tag&gt; outside" in result

    def test_empty_code_blocks(self):
        """Test that empty code blocks don't cause issues."""
        content = """Before
```
```
After <tag>"""
        result = _prepare_markdown_content(content)
        assert "After &lt;tag&gt;" in result

    def test_nested_backticks_not_treated_as_inline_code(self):
        """Test that triple backticks are not treated as inline code."""
        content = "This ```is not``` inline code <tag>"
        result = _prepare_markdown_content(content)
        # The content between triple backticks should be escaped
        assert "```is not``` inline code &lt;tag&gt;" in result

    def test_single_backtick_not_treated_as_code(self):
        """Test that single backtick without closing is not treated as code."""
        content = "This ` is not code <tag>"
        result = _prepare_markdown_content(content)
        assert "This ` is not code &lt;tag&gt;" in result

    def test_all_escape_characters(self):
        """Test that all defined escape characters are properly replaced."""
        content = "& < > \" '"
        result = _prepare_markdown_content(content)
        assert result == "&amp; &lt; &gt; &quot; &#39;"

    def test_preserve_newlines_and_whitespace(self):
        """Test that newlines and whitespace are preserved."""
        content = "Line 1\n  Line 2 with spaces\n\tLine 3 with tab"
        result = _prepare_markdown_content(content)
        assert "Line 1\n  Line 2 with spaces\n\tLine 3 with tab" == result

    def test_code_block_at_start(self):
        """Test code block at the very start of content."""
        content = """```
<code>
```
After <tag>"""
        result = _prepare_markdown_content(content)
        assert "<code>" in result
        assert "After &lt;tag&gt;" in result

    def test_code_block_at_end(self):
        """Test code block at the very end of content."""
        content = """Before <tag>
```
<code>
```"""
        result = _prepare_markdown_content(content)
        assert "Before &lt;tag&gt;" in result
        assert "<code>" in result

    def test_adjacent_inline_code(self):
        """Test adjacent inline code blocks.

        Note: The current regex pattern doesn't handle adjacent inline code blocks
        correctly when they're directly adjacent with no space. This is a known
        limitation but unlikely to occur in real usage.
        """
        # Test with space between inline code blocks (works correctly)
        content = "`<code1>` `<code2>` and <tag>"
        result = _prepare_markdown_content(content)
        assert "`<code1>`" in result
        assert "`<code2>`" in result
        assert "and &lt;tag&gt;" in result

        # Adjacent without space doesn't work as expected - documenting actual behavior
        content_adjacent = "`<code1>``<code2>` and <tag>"
        result_adjacent = _prepare_markdown_content(content_adjacent)
        # The regex doesn't match this pattern correctly, so content gets escaped
        assert "&lt;code1&gt;" in result_adjacent
        assert "&lt;code2&gt;" in result_adjacent

    def test_realistic_xml_content(self):
        """Test with realistic XML content that should be escaped."""
        content = """Here's an XML example:
<root>
    <child attr="value">Content & more</child>
</root>
But in code it's preserved:
```xml
<root>
    <child attr="value">Content & more</child>
</root>
```"""
        result = _prepare_markdown_content(content)

        # Outside code should be escaped
        assert "&lt;root&gt;" in result
        assert "&lt;child attr=&quot;value&quot;&gt;" in result

        # Inside code should be preserved
        assert '    <child attr="value">Content & more</child>' in result
