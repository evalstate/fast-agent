"""
Unit tests for elicitation form format validation.

Tests the FormatValidator class used in elicitation forms to verify that
Pydantic-based format validation works as expected with valid/invalid inputs.
"""

import pytest
from prompt_toolkit.document import Document
from prompt_toolkit.validation import ValidationError

from mcp_agent.human_input.elicitation_form import FormatValidator


class TestElicitationFormatValidator:
    """Test the FormatValidator used in elicitation forms with different format types."""

    def test_email_format_valid(self):
        """Test valid email addresses."""
        validator = FormatValidator("email")

        valid_emails = [
            "test@example.com",
            "user.name@domain.co.uk",
            "firstname+lastname@example.org",
            "test123@test-domain.com",
        ]

        for email in valid_emails:
            doc = Document(email)
            # Should not raise ValidationError
            validator.validate(doc)

    def test_email_format_invalid(self):
        """Test invalid email addresses."""
        validator = FormatValidator("email")

        invalid_emails = [
            "notanemail",
            "@example.com",
            "test@",
            "test..test@example.com",
            "test@.com",
        ]

        for email in invalid_emails:
            doc = Document(email)
            with pytest.raises(ValidationError) as exc_info:
                validator.validate(doc)
            assert "Invalid email format" in str(exc_info.value.message)

    def test_uri_format_valid(self):
        """Test valid URIs."""
        validator = FormatValidator("uri")

        valid_uris = [
            "https://example.com",
            "http://test.org/path",
            "ftp://files.example.com",
            "https://sub.domain.com:8080/path?query=value",
            "mailto:test@example.com",
        ]

        for uri in valid_uris:
            doc = Document(uri)
            # Should not raise ValidationError
            validator.validate(doc)

    def test_uri_format_invalid(self):
        """Test invalid URIs."""
        validator = FormatValidator("uri")

        # Test clearly invalid URIs that should definitely fail
        clearly_invalid_uris = [
            "not-a-uri-at-all",
            "://missing-scheme",
        ]

        for uri in clearly_invalid_uris:
            doc = Document(uri)
            with pytest.raises(ValidationError) as exc_info:
                validator.validate(doc)
            assert "Invalid URI format" in str(exc_info.value.message)

    def test_date_format_valid(self):
        """Test valid date formats."""
        validator = FormatValidator("date")

        valid_dates = [
            "2023-12-25",
            "2024-01-01",
            "2024-02-29",  # Leap year
            "1990-06-15",
        ]

        for date_str in valid_dates:
            doc = Document(date_str)
            # Should not raise ValidationError
            validator.validate(doc)

    def test_date_format_invalid(self):
        """Test invalid date formats."""
        validator = FormatValidator("date")

        invalid_dates = [
            "2023-13-01",  # Invalid month
            "2023-02-30",  # Invalid day for February
            "23-12-25",  # Wrong format
            "2023/12/25",  # Wrong separator
            "December 25, 2023",  # Wrong format
            "not-a-date",
        ]

        for date_str in invalid_dates:
            doc = Document(date_str)
            with pytest.raises(ValidationError) as exc_info:
                validator.validate(doc)
            assert "Invalid date" in str(exc_info.value.message)

    def test_datetime_format_valid(self):
        """Test valid datetime formats."""
        validator = FormatValidator("date-time")

        valid_datetimes = [
            "2023-12-25T10:30:00",
            "2024-01-01T00:00:00Z",
            "2024-02-29T23:59:59+00:00",
            "2023-06-15T14:30:45.123456",
        ]

        for datetime_str in valid_datetimes:
            doc = Document(datetime_str)
            # Should not raise ValidationError
            validator.validate(doc)

    def test_datetime_format_invalid(self):
        """Test invalid datetime formats."""
        validator = FormatValidator("date-time")

        # Test clearly invalid datetime formats that should definitely fail
        clearly_invalid_datetimes = [
            "2023-12-25T25:00:00",  # Invalid hour
            "2023-12-25T10:60:00",  # Invalid minute
            "not-a-datetime",
        ]

        for datetime_str in clearly_invalid_datetimes:
            doc = Document(datetime_str)
            with pytest.raises(ValidationError) as exc_info:
                validator.validate(doc)
            assert "Invalid datetime" in str(exc_info.value.message)

    def test_empty_string_is_valid(self):
        """Test that empty strings are considered valid (for optional fields)."""
        for format_type in ["email", "uri", "date", "date-time"]:
            validator = FormatValidator(format_type)
            doc = Document("")
            # Should not raise ValidationError for empty string
            validator.validate(doc)

    def test_whitespace_only_is_valid(self):
        """Test that whitespace-only strings are considered valid (stripped to empty)."""
        for format_type in ["email", "uri", "date", "date-time"]:
            validator = FormatValidator(format_type)
            doc = Document("   ")
            # Should not raise ValidationError for whitespace-only
            validator.validate(doc)
