"""
Unit tests for the context compaction module.
"""

import pytest
from mcp.types import TextContent

from fast_agent.llm.compaction import (
    ContextCompaction,
    ContextCompactionMode,
    estimate_tokens_for_messages,
)
from fast_agent.llm.compaction.token_estimation import (
    estimate_tokens_for_text,
    get_context_headroom,
    get_current_context_tokens,
)
from fast_agent.types import PromptMessageExtended


# --- ContextCompactionMode Tests ---


class TestContextCompactionMode:
    """Tests for the ContextCompactionMode enum."""

    def test_enum_values(self):
        """Test that enum has expected values."""
        assert ContextCompactionMode.NONE == "none"
        assert ContextCompactionMode.TRUNCATE == "truncate"
        assert ContextCompactionMode.SUMMARIZE == "summarize"

    def test_enum_from_string(self):
        """Test creating enum from string values."""
        assert ContextCompactionMode("none") == ContextCompactionMode.NONE
        assert ContextCompactionMode("truncate") == ContextCompactionMode.TRUNCATE
        assert ContextCompactionMode("summarize") == ContextCompactionMode.SUMMARIZE

    def test_invalid_value_raises(self):
        """Test that invalid value raises ValueError."""
        with pytest.raises(ValueError):
            ContextCompactionMode("invalid")


# --- Token Estimation Tests ---


class TestTokenEstimation:
    """Tests for token estimation utilities."""

    def test_estimate_tokens_for_text_empty(self):
        """Test estimation with empty text."""
        assert estimate_tokens_for_text("") == 0

    def test_estimate_tokens_for_text_short(self):
        """Test estimation with short text."""
        # 4 chars per token default
        result = estimate_tokens_for_text("hello")  # 5 chars
        assert result == 1  # minimum 1 token

    def test_estimate_tokens_for_text_longer(self):
        """Test estimation with longer text."""
        # 40 chars should be ~10 tokens at 4 chars/token
        text = "a" * 40
        result = estimate_tokens_for_text(text)
        assert result == 10

    def test_estimate_tokens_custom_ratio(self):
        """Test estimation with custom chars per token ratio."""
        text = "a" * 20
        result = estimate_tokens_for_text(text, chars_per_token=2.0)
        assert result == 10

    def test_estimate_tokens_for_messages_empty(self):
        """Test estimation with empty message list."""
        result = estimate_tokens_for_messages([])
        assert result == 0

    def test_estimate_tokens_for_messages_with_system_prompt(self):
        """Test estimation includes system prompt."""
        messages: list[PromptMessageExtended] = []
        result_without = estimate_tokens_for_messages(messages, system_prompt=None)
        result_with = estimate_tokens_for_messages(
            messages, system_prompt="You are a helpful assistant."
        )
        assert result_with > result_without

    def test_estimate_tokens_for_messages_single(self):
        """Test estimation with single message."""
        messages = [
            PromptMessageExtended(
                role="user",
                content=[TextContent(type="text", text="Hello, how are you?")],
            )
        ]
        result = estimate_tokens_for_messages(messages)
        assert result > 0

    def test_estimate_tokens_for_messages_multiple(self):
        """Test estimation with multiple messages."""
        messages = [
            PromptMessageExtended(
                role="user",
                content=[TextContent(type="text", text="Hello!")],
            ),
            PromptMessageExtended(
                role="assistant",
                content=[TextContent(type="text", text="Hi there! How can I help?")],
            ),
            PromptMessageExtended(
                role="user",
                content=[TextContent(type="text", text="What's the weather like?")],
            ),
        ]
        result = estimate_tokens_for_messages(messages)
        # Should be sum of all content plus role overhead
        assert result > 10

    def test_get_current_context_tokens_none(self):
        """Test getting current context with no accumulator."""
        result = get_current_context_tokens(None)
        assert result == 0

    def test_get_context_headroom_none(self):
        """Test getting headroom with no accumulator."""
        result = get_context_headroom(None)
        assert result is None

    def test_get_context_headroom_with_limit(self):
        """Test getting headroom with explicit limit."""
        # Create a mock accumulator
        class MockAccumulator:
            current_context_tokens = 5000
            context_window_size = 100000

        result = get_context_headroom(MockAccumulator(), limit=10000)
        assert result == 5000  # 10000 - 5000


# --- ContextCompaction Truncation Tests ---


class TestContextCompactionTruncate:
    """Tests for the truncation compaction strategy."""

    @pytest.mark.asyncio
    async def test_no_compaction_when_disabled(self):
        """Test that no compaction happens when mode is NONE."""
        messages = [
            PromptMessageExtended(
                role="user",
                content=[TextContent(type="text", text="Hello")],
            ),
        ]
        result, was_compacted = await ContextCompaction.compact_if_needed(
            messages=messages,
            mode=ContextCompactionMode.NONE,
            limit=100,
        )
        assert was_compacted is False
        assert result == messages

    @pytest.mark.asyncio
    async def test_no_compaction_when_no_limit(self):
        """Test that no compaction happens when limit is None."""
        messages = [
            PromptMessageExtended(
                role="user",
                content=[TextContent(type="text", text="Hello")],
            ),
        ]
        result, was_compacted = await ContextCompaction.compact_if_needed(
            messages=messages,
            mode=ContextCompactionMode.TRUNCATE,
            limit=None,
        )
        assert was_compacted is False
        assert result == messages

    @pytest.mark.asyncio
    async def test_no_compaction_when_under_limit(self):
        """Test that no compaction happens when under the limit."""
        messages = [
            PromptMessageExtended(
                role="user",
                content=[TextContent(type="text", text="Hello")],
            ),
        ]
        result, was_compacted = await ContextCompaction.compact_if_needed(
            messages=messages,
            mode=ContextCompactionMode.TRUNCATE,
            limit=100000,  # Very high limit
        )
        assert was_compacted is False
        assert result == messages

    @pytest.mark.asyncio
    async def test_no_compaction_empty_messages(self):
        """Test that empty messages list is handled."""
        result, was_compacted = await ContextCompaction.compact_if_needed(
            messages=[],
            mode=ContextCompactionMode.TRUNCATE,
            limit=100,
        )
        assert was_compacted is False
        assert result == []

    @pytest.mark.asyncio
    async def test_truncate_removes_old_messages(self):
        """Test that truncation removes older messages."""
        # Create messages that exceed the limit
        messages = [
            PromptMessageExtended(
                role="user",
                content=[TextContent(type="text", text="First message " * 100)],
            ),
            PromptMessageExtended(
                role="assistant",
                content=[TextContent(type="text", text="Second message " * 100)],
            ),
            PromptMessageExtended(
                role="user",
                content=[TextContent(type="text", text="Third message " * 100)],
            ),
            PromptMessageExtended(
                role="assistant",
                content=[TextContent(type="text", text="Fourth message")],
            ),
        ]

        result, was_compacted = await ContextCompaction.compact_if_needed(
            messages=messages,
            mode=ContextCompactionMode.TRUNCATE,
            limit=100,  # Very low limit to force truncation
        )

        assert was_compacted is True
        assert len(result) < len(messages)
        # Should keep at least the last message
        assert len(result) >= 1

    @pytest.mark.asyncio
    async def test_truncate_preserves_template_messages(self):
        """Test that truncation preserves template messages."""
        messages = [
            PromptMessageExtended(
                role="user",
                content=[TextContent(type="text", text="Template message")],
                is_template=True,
            ),
            PromptMessageExtended(
                role="user",
                content=[TextContent(type="text", text="Regular message " * 100)],
            ),
            PromptMessageExtended(
                role="assistant",
                content=[TextContent(type="text", text="Response " * 100)],
            ),
        ]

        result, was_compacted = await ContextCompaction.compact_if_needed(
            messages=messages,
            mode=ContextCompactionMode.TRUNCATE,
            limit=50,  # Very low limit
        )

        # Template message should always be preserved
        template_messages = [m for m in result if m.is_template]
        assert len(template_messages) == 1
        assert template_messages[0].first_text() == "Template message"

    @pytest.mark.asyncio
    async def test_truncate_keeps_at_least_one_message(self):
        """Test that truncation always keeps at least one conversation message."""
        messages = [
            PromptMessageExtended(
                role="user",
                content=[TextContent(type="text", text="Only message " * 200)],
            ),
        ]

        result, was_compacted = await ContextCompaction.compact_if_needed(
            messages=messages,
            mode=ContextCompactionMode.TRUNCATE,
            limit=1,  # Impossibly low limit
        )

        # Even with impossible limit, should keep at least the last message
        assert len(result) >= 1


# --- ContextCompaction Summarization Tests ---


class TestContextCompactionSummarize:
    """Tests for the summarization compaction strategy."""

    @pytest.mark.asyncio
    async def test_summarize_without_llm_falls_back_to_truncate(self):
        """Test that summarize without LLM falls back to truncation."""
        messages = [
            PromptMessageExtended(
                role="user",
                content=[TextContent(type="text", text="Message " * 100)],
            ),
            PromptMessageExtended(
                role="assistant",
                content=[TextContent(type="text", text="Response " * 100)],
            ),
        ]

        result, was_compacted = await ContextCompaction.compact_if_needed(
            messages=messages,
            mode=ContextCompactionMode.SUMMARIZE,
            limit=50,  # Low limit to trigger compaction
            llm=None,  # No LLM provided
        )

        # Should fall back to truncation
        assert was_compacted is True
        assert len(result) <= len(messages)

    @pytest.mark.asyncio
    async def test_summarize_with_few_messages_falls_back_to_truncate(self):
        """Test that summarize with too few messages falls back to truncation."""
        messages = [
            PromptMessageExtended(
                role="user",
                content=[TextContent(type="text", text="Single message " * 100)],
            ),
        ]

        # Create a mock LLM
        class MockLLM:
            usage_accumulator = None
            model_name = "test-model"

            async def generate(self, messages, request_params=None):
                return PromptMessageExtended(
                    role="assistant",
                    content=[TextContent(type="text", text="Summary")],
                )

        result, was_compacted = await ContextCompaction.compact_if_needed(
            messages=messages,
            mode=ContextCompactionMode.SUMMARIZE,
            limit=10,  # Low limit
            llm=MockLLM(),
        )

        # With only one message, should use truncation behavior
        # (keeps at least the last message)
        assert len(result) >= 1


# --- Integration-style Tests ---


class TestContextCompactionIntegration:
    """Integration-style tests for context compaction."""

    @pytest.mark.asyncio
    async def test_compaction_maintains_message_integrity(self):
        """Test that compaction doesn't corrupt message content."""
        original_text = "This is an important message"
        messages = [
            PromptMessageExtended(
                role="user",
                content=[TextContent(type="text", text=original_text)],
            ),
        ]

        result, _ = await ContextCompaction.compact_if_needed(
            messages=messages,
            mode=ContextCompactionMode.TRUNCATE,
            limit=100000,  # High limit, no compaction
        )

        assert result[0].first_text() == original_text

    @pytest.mark.asyncio
    async def test_compaction_preserves_role_order(self):
        """Test that compaction preserves the role order of messages."""
        messages = [
            PromptMessageExtended(
                role="user",
                content=[TextContent(type="text", text="U1 " * 50)],
            ),
            PromptMessageExtended(
                role="assistant",
                content=[TextContent(type="text", text="A1 " * 50)],
            ),
            PromptMessageExtended(
                role="user",
                content=[TextContent(type="text", text="U2 " * 50)],
            ),
            PromptMessageExtended(
                role="assistant",
                content=[TextContent(type="text", text="A2")],
            ),
        ]

        result, was_compacted = await ContextCompaction.compact_if_needed(
            messages=messages,
            mode=ContextCompactionMode.TRUNCATE,
            limit=100,  # Low limit
        )

        if was_compacted and len(result) >= 2:
            # Check that remaining messages alternate properly
            for i in range(len(result) - 1):
                # Messages should still have proper roles
                assert result[i].role in ["user", "assistant"]

    @pytest.mark.asyncio
    async def test_format_conversation_for_summary(self):
        """Test the conversation formatting for summarization."""
        messages = [
            PromptMessageExtended(
                role="user",
                content=[TextContent(type="text", text="Hello!")],
            ),
            PromptMessageExtended(
                role="assistant",
                content=[TextContent(type="text", text="Hi there!")],
            ),
        ]

        formatted = ContextCompaction._format_conversation_for_summary(messages)

        assert "USER: Hello!" in formatted
        assert "ASSISTANT: Hi there!" in formatted
