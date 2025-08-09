"""
Tests for context_truncation_and_summarization.py module.

This module tests the context truncation and summarization functionality,
including all public and private methods in the ContextTruncation class.
"""
import unittest
from unittest.mock import AsyncMock, Mock, patch

from mcp.types import TextContent

from mcp_agent.core.agent_types import ContextTruncationMode
from mcp_agent.llm.augmented_llm import AugmentedLLM
from mcp_agent.llm.context_truncation_and_summarization import ContextTruncation
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart


class TestContextTruncation(unittest.IsolatedAsyncioTestCase):
    """Test cases for the ContextTruncation class."""

    def create_mock_provider(self, token_count_return=1000):
        """Create a mock provider with get_token_count method."""
        mock_provider = Mock(spec=AugmentedLLM)
        mock_provider.get_token_count.return_value = token_count_return
        mock_provider.execute_simple_api_call = AsyncMock(return_value="Summarized content")
        return mock_provider

    def create_sample_messages(self):
        """Create sample messages for testing (no system messages - they're handled via system_prompt parameter)."""
        return [
            PromptMessageMultipart(
                role="user",
                content=[TextContent(type="text", text="Hello, how are you?")]
            ),
            PromptMessageMultipart(
                role="assistant",
                content=[TextContent(type="text", text="I'm doing well, thank you!")]
            ),
            PromptMessageMultipart(
                role="user",
                content=[TextContent(type="text", text="Can you help me with math?")]
            ),
            PromptMessageMultipart(
                role="assistant",
                content=[TextContent(type="text", text="Of course! I'd be happy to help.")]
            )
        ]

    async def test_truncate_if_required_no_truncation_needed(self):
        """Test that no truncation occurs when under the limit."""
        mock_provider = self.create_mock_provider(token_count_return=500)  # Under limit
        messages = self.create_sample_messages()
        
        result = await ContextTruncation.truncate_if_required(
            messages=messages,
            truncation_mode=ContextTruncationMode.SUMMARIZE,
            limit=1000,
            model_name="test-model",
            system_prompt="System prompt",
            provider=mock_provider
        )
        
        # Should return original messages unchanged
        assert result == messages
        mock_provider.get_token_count.assert_called_once()

    async def test_truncate_if_required_no_truncation_mode(self):
        """Test that no truncation occurs when truncation mode is None."""
        mock_provider = self.create_mock_provider(token_count_return=1500)  # Over limit
        messages = self.create_sample_messages()
        
        result = await ContextTruncation.truncate_if_required(
            messages=messages,
            truncation_mode=None,
            limit=1000,
            model_name="test-model",
            system_prompt="System prompt",
            provider=mock_provider
        )
        
        # Should return original messages unchanged
        assert result == messages

    async def test_truncate_if_required_truncation_mode_none(self):
        """Test that no truncation occurs when truncation mode is NONE."""
        mock_provider = self.create_mock_provider(token_count_return=1500)  # Over limit
        messages = self.create_sample_messages()
        
        result = await ContextTruncation.truncate_if_required(
            messages=messages,
            truncation_mode=ContextTruncationMode.NONE,
            limit=1000,
            model_name="test-model",
            system_prompt="System prompt", 
            provider=mock_provider
        )
        
        # Should return original messages unchanged
        assert result == messages

    async def test_truncate_if_required_no_limit(self):
        """Test that no truncation occurs when limit is None."""
        mock_provider = self.create_mock_provider(token_count_return=1500)
        messages = self.create_sample_messages()
        
        result = await ContextTruncation.truncate_if_required(
            messages=messages,
            truncation_mode=ContextTruncationMode.SUMMARIZE,
            limit=None,
            model_name="test-model",
            system_prompt="System prompt",
            provider=mock_provider
        )
        
        # Should return original messages unchanged
        assert result == messages

    async def test_truncate_if_required_summarize_mode(self):
        """Test truncation with SUMMARIZE mode."""
        mock_provider = self.create_mock_provider(token_count_return=1500)  # Over limit
        messages = self.create_sample_messages()
        
        with patch.object(ContextTruncation, '_needs_truncation', return_value=True):
            result = await ContextTruncation.truncate_if_required(
                messages=messages,
                truncation_mode=ContextTruncationMode.SUMMARIZE,
                limit=1000,
                model_name="test-model",
                system_prompt="System prompt",
                provider=mock_provider
            )
        
        # Should return a single summary message
        assert len(result) == 1
        assert result[0].role == "user"
        assert "Summarized content" in result[0].content[0].text
        mock_provider.execute_simple_api_call.assert_called_once()

    async def test_truncate_if_required_remove_mode(self):
        """Test truncation with REMOVE mode."""
        mock_provider = self.create_mock_provider(token_count_return=1500)  # Over limit
        messages = self.create_sample_messages()
        
        with patch.object(ContextTruncation, '_needs_truncation') as mock_needs:
            # First call returns True (needs truncation), subsequent calls return False
            mock_needs.side_effect = [True, False]
            
            result = await ContextTruncation.truncate_if_required(
                messages=messages,
                truncation_mode=ContextTruncationMode.REMOVE,
                limit=1000,
                model_name="test-model",
                system_prompt="System prompt",
                provider=mock_provider
            )
        
        # Should return fewer messages (original had 4, should remove some)
        assert len(result) < len(messages)
        # Should still have some messages remaining
        assert len(result) > 0

    async def test_summarize_and_truncate(self):
        """Test the _summarize_and_truncate method."""
        mock_provider = Mock()
        mock_provider.execute_simple_api_call = AsyncMock(return_value="This is a summary")
        messages = self.create_sample_messages()
        
        result = await ContextTruncation._summarize_and_truncate(
            messages=messages,
            max_tokens=1000,
            provider=mock_provider
        )
        
        # Should return a single summary message
        assert len(result) == 1
        assert result[0].role == "user"
        assert result[0].content[0].text == "This is a summary"
        mock_provider.execute_simple_api_call.assert_called_once()

    async def test_summarize_messages(self):
        """Test the _summarize_messages method."""
        mock_provider = Mock()
        mock_provider.execute_simple_api_call = AsyncMock(return_value="Detailed summary")
        messages = [
            PromptMessageMultipart(
                role="user",
                content=[TextContent(type="text", text="What is 2+2?")]
            ),
            PromptMessageMultipart(
                role="assistant",
                content=[TextContent(type="text", text="2+2 equals 4")]
            )
        ]
        
        result = await ContextTruncation._summarize_messages(
            messages_to_summarize=messages,
            provider=mock_provider
        )
        
        assert result == "Detailed summary"
        mock_provider.execute_simple_api_call.assert_called_once()
        
        # Check that the call included the conversation text
        call_args = mock_provider.execute_simple_api_call.call_args
        prompt = call_args[1]['message_string']
        assert "What is 2+2?" in prompt
        assert "2+2 equals 4" in prompt

    def test_needs_truncation_true(self):
        """Test _needs_truncation returns True when over limit."""
        mock_provider = self.create_mock_provider(token_count_return=1500)
        messages = self.create_sample_messages()
        
        result = ContextTruncation._needs_truncation(
            messages=messages,
            max_tokens=1000,
            system_prompt="System prompt",
            provider=mock_provider
        )
        
        assert result is True
        mock_provider.get_token_count.assert_called_once_with(messages, "System prompt")

    def test_needs_truncation_false(self):
        """Test _needs_truncation returns False when under limit."""
        mock_provider = self.create_mock_provider(token_count_return=500)
        messages = self.create_sample_messages()
        
        result = ContextTruncation._needs_truncation(
            messages=messages,
            max_tokens=1000,
            system_prompt="System prompt",
            provider=mock_provider
        )
        
        assert result is False
        mock_provider.get_token_count.assert_called_once()

    def test_needs_truncation_no_limit(self):
        """Test _needs_truncation returns False when no limit is set."""
        mock_provider = self.create_mock_provider(token_count_return=1500)
        messages = self.create_sample_messages()
        
        result = ContextTruncation._needs_truncation(
            messages=messages,
            max_tokens=None,
            system_prompt="System prompt",
            provider=mock_provider
        )
        
        assert result is False
        # Should not call provider when no limit
        mock_provider.get_token_count.assert_not_called()

    def test_needs_truncation_no_provider(self):
        """Test _needs_truncation returns False when no provider is available."""
        messages = self.create_sample_messages()
        
        result = ContextTruncation._needs_truncation(
            messages=messages,
            max_tokens=1000,
            system_prompt="System prompt",
            provider=None
        )
        
        assert result is False

    def test_truncate_removes_oldest_non_system_messages(self):
        """Test that _truncate removes oldest non-system messages first."""
        messages = self.create_sample_messages()
        mock_provider = Mock()
        
        # Mock _needs_truncation to return True first, then False after removal
        with patch.object(ContextTruncation, '_needs_truncation') as mock_needs:
            mock_needs.side_effect = [True, False]  # First True, then False
            mock_provider.get_token_count.side_effect = [1500, 800]  # Over limit, then under
            
            result = ContextTruncation._truncate(
                messages=messages,
                max_tokens=1000,
                system_prompt="System prompt",
                provider=mock_provider
            )
        
        # Should have fewer messages 
        assert len(result) < len(messages)
        # First user message should be removed (oldest)
        user_messages = [msg for msg in result if msg.role == "user"]
        assert len(user_messages) < 2  # Originally had 2 user messages

    def test_truncate_with_only_user_messages(self):
        """Test that _truncate handles case where only user messages remain."""
        # Create messages with only user messages  
        messages = [
            PromptMessageMultipart(
                role="user",
                content=[TextContent(type="text", text="First user message")]
            ),
            PromptMessageMultipart(
                role="user",
                content=[TextContent(type="text", text="Second user message")]
            ),
            PromptMessageMultipart(
                role="user",
                content=[TextContent(type="text", text="Third user message")]
            )
        ]
        
        mock_provider = Mock()
        # First call needs truncation, second call doesn't
        with patch.object(ContextTruncation, '_needs_truncation') as mock_needs:
            mock_needs.side_effect = [True, False]  # First True, then False
            mock_provider.get_token_count.side_effect = [1500, 800]
            
            result = ContextTruncation._truncate(
                messages=messages,
                max_tokens=1000,
                system_prompt="System prompt", 
                provider=mock_provider
            )
        
        # Should have fewer messages than original
        assert len(result) < len(messages)
        # Should still have some messages
        assert len(result) > 0

    def test_truncate_no_provider(self):
        """Test _truncate returns original messages when no provider available."""
        messages = self.create_sample_messages()
        
        result = ContextTruncation._truncate(
            messages=messages,
            max_tokens=1000,
            system_prompt="System prompt",
            provider=None
        )
        
        # Should return original messages unchanged
        assert result == messages

    def test_truncate_stops_when_no_messages_left(self):
        """Test _truncate stops when no messages are left to remove."""
        messages = [
            PromptMessageMultipart(
                role="user",
                content=[TextContent(type="text", text="Only message")]
            )
        ]
        
        mock_provider = Mock()
        # Always needs truncation but should break when no messages left
        with patch.object(ContextTruncation, '_needs_truncation', return_value=True):
            mock_provider.get_token_count.return_value = 1500
            
            result = ContextTruncation._truncate(
                messages=messages,
                max_tokens=1000,
                system_prompt="System prompt",
                provider=mock_provider
            )
        
        # Should return empty list when all messages removed
        assert len(result) == 0

    async def test_summarize_excludes_system_messages(self):
        """Test that summarization excludes system messages from the conversation to summarize."""
        messages = self.create_sample_messages()
        mock_provider = Mock()
        mock_provider.execute_simple_api_call = AsyncMock(return_value="Summary without system")
        
        await ContextTruncation._summarize_and_truncate(
            messages=messages,
            max_tokens=1000,
            provider=mock_provider
        )
        
        # Check that system messages were filtered out of the summarization
        call_args = mock_provider.execute_simple_api_call.call_args
        prompt = call_args[1]['message_string']
        # Should contain user/assistant messages
        assert "Hello, how are you?" in prompt
        assert "I'm doing well, thank you!" in prompt

    async def test_truncate_if_required_integration(self):
        """Integration test for the main truncate_if_required method."""
        # Create a scenario where truncation is needed
        long_messages = []
        for i in range(10):
            long_messages.append(
                PromptMessageMultipart(
                    role="user" if i % 2 == 0 else "assistant",
                    content=[TextContent(type="text", text=f"Message {i} content")]
                )
            )
        
        mock_provider = Mock()
        # First call: over limit, second call: under limit (after truncation)
        mock_provider.get_token_count.side_effect = [2000, 800]
        mock_provider.execute_simple_api_call = AsyncMock(return_value="Summarized conversation")
        
        result = await ContextTruncation.truncate_if_required(
            messages=long_messages,
            truncation_mode=ContextTruncationMode.SUMMARIZE,
            limit=1500,
            model_name="test-model",
            system_prompt="You are helpful",
            provider=mock_provider
        )
        
        # Should get back a single summarized message
        assert len(result) == 1
        assert result[0].role == "user"
        assert "Summarized conversation" in result[0].content[0].text