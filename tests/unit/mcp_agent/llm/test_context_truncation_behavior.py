"""
Unit tests for the behavior of context truncation in AugmentedLLM.
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mcp_agent.core.prompt import Prompt
from mcp_agent.core.request_params import RequestParams
from mcp_agent.llm.augmented_llm_passthrough import PassthroughLLM


class TestContextTruncationBehavior:
    """Test cases for the context truncation behavior."""

    @pytest.fixture
    def mock_llm(self):
        """Fixture to create a mock AugmentedLLM for testing."""
        with patch('mcp_agent.llm.augmented_llm_passthrough.PassthroughLLM._apply_prompt_provider_specific', new_callable=AsyncMock) as mock_apply:
            # Configure the mock to return a summary when called for summarization
            mock_apply.return_value = Prompt.assistant("This is a summary.")

            mock_agent = MagicMock()
            mock_agent.instruction = "This is a mock system prompt."
            llm = PassthroughLLM(provider=MagicMock(), agent=mock_agent)

            llm.usage_accumulator = MagicMock()
            yield llm, mock_apply

    @pytest.mark.asyncio
    async def test_truncation_triggered_when_context_exceeds_limit(self, mock_llm):
        """Test that truncation is triggered when context exceeds the limit."""
        llm, mock_apply = mock_llm
        llm.usage_accumulator.current_context_tokens = 150
        
        # Add some history to ensure the truncation logic doesn't exit early
        llm._message_history.extend([Prompt.user("Hello")])
        llm.history.extend([{'role': 'user', 'content': 'Hello'}])

        request_params = RequestParams(max_context_length_total=100)
        
        await llm._check_and_truncate_context(request_params)

        # Verify that the summarization was called
        mock_apply.assert_called_once()
        # Verify history was cleared
        assert len(llm.history.get()) == 2 # Summary messages
        assert len(llm._message_history) == 2

    @pytest.mark.asyncio
    async def test_truncation_not_triggered_when_context_within_limit(self, mock_llm):
        """Test that truncation is not triggered when context is within the limit."""
        llm, mock_apply = mock_llm
        llm.usage_accumulator.current_context_tokens = 50

        request_params = RequestParams(max_context_length_total=100)
        
        await llm._check_and_truncate_context(request_params)

        # Verify that the summarization was not called
        mock_apply.assert_not_called()

    @pytest.mark.asyncio
    async def test_truncation_disabled_when_parameter_is_none(self, mock_llm):
        """Test that truncation is disabled when max_context_length_total is None."""
        llm, mock_apply = mock_llm
        llm.usage_accumulator.current_context_tokens = 150

        request_params = RequestParams(max_context_length_total=None)
        
        await llm._check_and_truncate_context(request_params)

        # Verify that the summarization was not called
        mock_apply.assert_not_called()

    @pytest.mark.asyncio
    async def test_history_is_cleared_and_replaced_with_summary(self, mock_llm):
        """Test that the history is cleared and replaced with the summary."""
        llm, mock_apply = mock_llm
        llm.usage_accumulator.current_context_tokens = 150
        
        # Add some history
        llm._message_history.extend([Prompt.user("Hello"), Prompt.assistant("Hi there")])
        llm.history.extend([{'role': 'user', 'content': 'Hello'}, {'role': 'assistant', 'content': 'Hi there'}])

        request_params = RequestParams(max_context_length_total=100)
        
        await llm._check_and_truncate_context(request_params)

        # Verify history is cleared and replaced with summary
        assert len(llm._message_history) == 2
        assert "summary" in llm._message_history[0].first_text()
        assert "summary" in llm.history.get()[0].first_text()

    @pytest.mark.asyncio
    async def test_truncation_skipped_if_pending_tool_calls(self, mock_llm):
        """Test that truncation is skipped if there are pending tool calls."""
        llm, mock_apply = mock_llm
        llm.usage_accumulator.current_context_tokens = 150

        # Add a message with a tool call
        tool_call_message = Prompt.assistant("")
        tool_call_message.content = [{"type": "tool_calls"}]
        llm._message_history.append(tool_call_message)

        request_params = RequestParams(max_context_length_total=100)
        
        await llm._check_and_truncate_context(request_params)

        # Verify that the summarization was not called
        mock_apply.assert_not_called()
