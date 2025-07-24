"""Unit tests for template functionality in direct_decorators."""

import re

from mcp_agent.core.direct_decorators import _apply_templates


class TestApplyTemplates:
    """Test the _apply_templates function."""

    def test_current_date_template(self):
        """Test that {{currentDate}} is replaced with a valid date."""
        text = "The current date is {{currentDate}}."
        result = _apply_templates(text)

        # Should not contain the template anymore
        assert "{{currentDate}}" not in result

        # Should contain "The current date is " followed by a date
        assert result.startswith("The current date is ")

        # Extract the date part and verify format
        date_part = result.replace("The current date is ", "").replace(".", "")

        # Should match "DD Month YYYY" pattern (e.g., "24 July 2025")
        date_pattern = r"^\d{1,2} [A-Z][a-z]+ \d{4}$"
        assert re.match(date_pattern, date_part), (
            f"Date '{date_part}' doesn't match expected format"
        )

    def test_multiple_templates(self):
        """Test multiple {{currentDate}} templates in the same text."""
        text = "Today is {{currentDate}} and tomorrow will be after {{currentDate}}."
        result = _apply_templates(text)

        # Should not contain any templates
        assert "{{currentDate}}" not in result

        # Should contain "Today is " and "tomorrow will be after"
        assert "Today is " in result
        assert " and tomorrow will be after " in result

        # Extract both date parts
        parts = result.split(" and tomorrow will be after ")
        first_date = parts[0].replace("Today is ", "")
        second_date = parts[1].rstrip(".")

        # Both should be valid dates and identical
        date_pattern = r"^\d{1,2} [A-Z][a-z]+ \d{4}$"
        assert re.match(date_pattern, first_date)
        assert re.match(date_pattern, second_date)
        assert first_date == second_date  # Same date should be used for both

    def test_no_templates(self):
        """Test that text without templates is unchanged."""
        text = "This is a normal instruction without any templates."
        result = _apply_templates(text)

        assert result == text

    def test_empty_string(self):
        """Test that empty strings are handled correctly."""
        result = _apply_templates("")
        assert result == ""

    def test_unknown_template(self):
        """Test that unknown templates are left unchanged."""
        text = "This has an {{unknownTemplate}} that should remain."
        result = _apply_templates(text)

        assert result == text
        assert "{{unknownTemplate}}" in result

    def test_malformed_template(self):
        """Test that malformed templates are left unchanged."""
        text = "This has {{incomplete and {missing} braces."
        result = _apply_templates(text)

        assert result == text

    def test_template_in_multiline_text(self):
        """Test templates in multiline text."""
        text = """# Agent Instructions
            
You are a helpful AI assistant.
The current date is {{currentDate}}.
Use this information when needed.
"""
        result = _apply_templates(text)

        # Should not contain the template
        assert "{{currentDate}}" not in result

        # Should contain the structure with a date
        assert "# Agent Instructions" in result
        assert "You are a helpful AI assistant." in result
        assert "The current date is " in result
        assert "Use this information when needed." in result

        # Extract the date from the multiline text
        lines = result.split("\n")
        date_line = [line for line in lines if "The current date is " in line][0]
        date_part = date_line.replace("The current date is ", "").replace(".", "").strip()

        # Should be a valid date
        date_pattern = r"^\d{1,2} [A-Z][a-z]+ \d{4}$"
        assert re.match(date_pattern, date_part)

    def test_url_template_pattern_recognition(self):
        """Test that URL templates are recognized by the regex pattern."""
        # Test the pattern matching without making actual requests
        text = "Instructions: {{url:https://example.com/prompt.md}} More text."

        # Check that the pattern would be found
        import re

        url_pattern = re.compile(r"\{\{url:(https?://[^}]+)\}\}")
        matches = url_pattern.findall(text)

        assert len(matches) == 1
        assert matches[0] == "https://example.com/prompt.md"

    def test_multiple_url_templates_pattern(self):
        """Test multiple URL templates in the same text."""
        text = "Start {{url:http://site1.com}} middle {{url:https://site2.com/file}} end."

        import re

        url_pattern = re.compile(r"\{\{url:(https?://[^}]+)\}\}")
        matches = url_pattern.findall(text)

        assert len(matches) == 2
        assert matches[0] == "http://site1.com"
        assert matches[1] == "https://site2.com/file"

    def test_malformed_url_templates(self):
        """Test that malformed URL templates are left unchanged."""
        text = "This has {{url:not-a-url}} and {{url:}} and {{url:ftp://invalid}} templates."

        # Since we only match http/https, ftp should be ignored
        import re

        url_pattern = re.compile(r"\{\{url:(https?://[^}]+)\}\}")
        matches = url_pattern.findall(text)

        # Should find no valid HTTP/HTTPS URLs
        assert len(matches) == 0

    def test_mixed_templates(self):
        """Test both currentDate and URL templates together."""
        text = "Date: {{currentDate}} and content: {{url:https://raw.githubusercontent.com/evalstate/fast-agent/refs/heads/main/README.md}}."

        # Test currentDate still works
        result = _apply_templates(text)
        assert "{{currentDate}}" not in result

        # URL template should be replaced with actual content
        assert "{{url:" not in result

        # Should contain date and URL content
        import re

        date_pattern = r"\d{1,2} [A-Z][a-z]+ \d{4}"
        assert re.search(date_pattern, result)
        assert "fast-agent" in result
