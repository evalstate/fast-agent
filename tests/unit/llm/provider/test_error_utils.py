"""Tests for LLM provider error utilities."""

from mcp.types import TextContent

from fast_agent.constants import FAST_AGENT_ERROR_CHANNEL
from fast_agent.llm.provider.error_utils import build_stream_failure_response
from fast_agent.llm.provider_types import Provider
from fast_agent.types.llm_stop_reason import LlmStopReason


class FakeAPIError(Exception):
    """Test error with optional code and status_code attributes."""

    def __init__(
        self,
        message: str,
        code: str | None = None,
        status_code: int | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code


def _get_error_text(result) -> str:
    """Extract error channel text from result, with type assertions."""
    assert result.channels is not None
    error_content = result.channels[FAST_AGENT_ERROR_CHANNEL][0]
    assert isinstance(error_content, TextContent)
    return error_content.text


def _get_assistant_text(result) -> str:
    """Extract assistant content text from result, with type assertions."""
    assert result.content is not None
    content = result.content[0]
    assert isinstance(content, TextContent)
    return content.text


class TestBuildStreamFailureResponse:
    """Tests for build_stream_failure_response function."""

    def test_basic_error(self):
        """Basic error produces correct structure."""
        result = build_stream_failure_response(
            provider=Provider.OPENAI,
            error=Exception("connection timeout"),
            model_name="gpt-4o",
        )

        assert result.role == "assistant"
        assert result.stop_reason == LlmStopReason.ERROR
        assert result.content is not None
        assert len(result.content) == 1

        # Check error channel
        assert result.channels is not None
        assert FAST_AGENT_ERROR_CHANNEL in result.channels
        error_text = _get_error_text(result)
        assert "gpt-4o" in error_text
        assert "connection timeout" in error_text
        assert "openai" in error_text.lower()

    def test_error_with_code_and_status(self):
        """Error with code and status_code includes them in output."""
        error = FakeAPIError(
            message="rate limited",
            code="rate_limit_exceeded",
            status_code=429,
        )

        result = build_stream_failure_response(
            provider=Provider.ANTHROPIC,
            error=error,
            model_name="claude-3-sonnet",
        )

        error_text = _get_error_text(result)
        assert "(code: rate_limit_exceeded)" in error_text
        assert "(status=429)" in error_text
        assert "claude-3-sonnet" in error_text

    def test_truncates_long_messages(self):
        """Long error messages are truncated in user-facing text."""
        long_message = "x" * 500
        error = Exception(long_message)

        result = build_stream_failure_response(
            provider=Provider.OPENAI,
            error=error,
            model_name="gpt-4",
        )

        # User-facing assistant text should be truncated
        assistant_text = _get_assistant_text(result)
        # The truncation happens at 280 chars for user_summary, plus wrapper text
        assert "..." in assistant_text
        assert len(assistant_text) < 450

    def test_string_provider(self):
        """Accepts string provider for custom/unknown providers."""
        result = build_stream_failure_response(
            provider="CustomProvider",
            error=Exception("oops"),
            model_name="custom-model",
        )

        error_text = _get_error_text(result)
        assert "CustomProvider" in error_text
        assert "custom-model" in error_text

    def test_empty_model_name(self):
        """Handles empty model name gracefully."""
        result = build_stream_failure_response(
            provider=Provider.OPENAI,
            error=Exception("error"),
            model_name="",
        )

        error_text = _get_error_text(result)
        assert "openai request failed" in error_text.lower()
        # Should not have "for model ''" in output
        assert "for model" not in error_text.lower()

    def test_assistant_text_ends_with_punctuation(self):
        """Assistant text always ends properly."""
        result = build_stream_failure_response(
            provider=Provider.OPENAI,
            error=Exception("no punctuation here"),
            model_name="gpt-4",
        )

        assistant_text = _get_assistant_text(result)
        assert assistant_text.endswith("See fast-agent-error for additional details.")

    def test_error_without_message_attribute(self):
        """Falls back to str(error) when no message attribute."""
        error = ValueError("simple value error")

        result = build_stream_failure_response(
            provider=Provider.OPENAI,
            error=error,
            model_name="gpt-4",
        )

        error_text = _get_error_text(result)
        assert "simple value error" in error_text
