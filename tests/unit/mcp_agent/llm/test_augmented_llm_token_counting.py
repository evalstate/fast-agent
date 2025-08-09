"""
Tests for the get_token_count() method in augmented_llm.py.

This module tests the unified token counting functionality that leverages
the existing usage_tracking.py infrastructure.
"""
from unittest.mock import patch

from mcp.types import TextContent

from mcp_agent.llm.augmented_llm_passthrough import PassthroughLLM
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart


class TestGetTokenCount:
    """Test cases for the get_token_count() method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.llm = PassthroughLLM(name="test-llm", model="passthrough")

    def test_get_token_count_simple_text(self):
        """Test token counting with simple text messages."""
        # Create test messages
        messages = [
            PromptMessageMultipart(
                role="user",
                content=[TextContent(type="text", text="Hello, world!")]
            ),
            PromptMessageMultipart(
                role="assistant",
                content=[TextContent(type="text", text="Hi there!")]
            )
        ]
        
        # Test the method - should work with real implementation
        token_count = self.llm.get_token_count(messages)
        
        # Basic sanity check - should be a positive integer
        assert isinstance(token_count, int)
        assert token_count > 0

    def test_get_token_count_with_system_prompt(self):
        """Test token counting with system prompt included."""
        messages = [
            PromptMessageMultipart(
                role="user",
                content=[TextContent(type="text", text="What is 2+2?")]
            )
        ]
        system_prompt = "You are a helpful math assistant."
        
        count_without_system = self.llm.get_token_count(messages)
        count_with_system = self.llm.get_token_count(messages, system_prompt)
        
        # Count with system prompt should be higher
        assert isinstance(count_with_system, int)
        assert count_with_system > count_without_system

    def test_get_token_count_tool_use(self):
        """Test token counting with tool use representation."""
        # Use TextContent to represent tool use (since it gets converted to text anyway)
        messages = [
            PromptMessageMultipart(
                role="assistant",
                content=[TextContent(type="text", text="<tool_use>calculator(operation=add, a=2, b=2)</tool_use>")]
            )
        ]
        
        token_count = self.llm.get_token_count(messages)
        
        # Should handle tool use representation
        assert isinstance(token_count, int)
        assert token_count > 0

    def test_get_token_count_tool_result(self):
        """Test token counting with tool result representation."""
        # Use TextContent to represent tool result (since it gets converted to text anyway)
        messages = [
            PromptMessageMultipart(
                role="user",
                content=[TextContent(type="text", text="<tool_result>The result is 4</tool_result>")]
            )
        ]
        
        token_count = self.llm.get_token_count(messages)
        
        # Should handle tool result representation
        assert isinstance(token_count, int)
        assert token_count > 0

    def test_get_token_count_image_blocks(self):
        """Test token counting with image representation."""
        # Use TextContent to represent image (since it gets converted to text anyway)
        messages = [
            PromptMessageMultipart(
                role="user",
                content=[
                    TextContent(type="text", text="Describe this image"),
                    TextContent(type="text", text="[IMAGE: base64_encoded_image_data]")
                ]
            )
        ]
        
        token_count = self.llm.get_token_count(messages)
        
        # Should handle mixed content
        assert isinstance(token_count, int)
        assert token_count > 0

    def test_get_token_count_empty_messages(self):
        """Test token counting with empty messages."""
        messages = []
        
        token_count = self.llm.get_token_count(messages)
        
        # Should return a valid count (probably 0 or very low)
        assert isinstance(token_count, int)
        assert token_count >= 0

    def test_get_token_count_messages_without_content(self):
        """Test token counting with messages that have minimal content."""
        messages = [
            PromptMessageMultipart(
                role="user",
                content=[TextContent(type="text", text="")]  # Empty text content
            ),
            PromptMessageMultipart(
                role="assistant",
                content=[TextContent(type="text", text="")]  # Empty text content
            )
        ]
        
        token_count = self.llm.get_token_count(messages)
        
        # Should handle empty content gracefully
        assert isinstance(token_count, int)
        assert token_count >= 0

    def test_get_token_count_unknown_block_types(self):
        """Test token counting with unknown content representation."""
        # Use TextContent to represent unknown content (since it gets converted to text anyway)
        messages = [
            PromptMessageMultipart(
                role="user",
                content=[TextContent(type="text", text="<unknown_block>some_unknown_content</unknown_block>")]
            )
        ]
        
        token_count = self.llm.get_token_count(messages)
        
        # Should handle unknown content gracefully
        assert isinstance(token_count, int)
        assert token_count > 0

    def test_get_token_count_complex_tool_result(self):
        """Test token counting with complex tool result representation."""
        # Use TextContent to represent complex tool result
        messages = [
            PromptMessageMultipart(
                role="user",
                content=[TextContent(type="text", text="<tool_result id='tool_456'>Direct string result</tool_result>")]
            )
        ]
        
        token_count = self.llm.get_token_count(messages)
        
        # Should handle complex tool result representation
        assert isinstance(token_count, int)
        assert token_count > 0

    def test_get_token_count_multiple_messages_scaling(self):
        """Test that token count increases with more content."""
        # Single message
        single_message = [
            PromptMessageMultipart(
                role="user",
                content=[TextContent(type="text", text="Hello")]
            )
        ]
        
        # Multiple messages with more content
        multiple_messages = [
            PromptMessageMultipart(
                role="user",
                content=[TextContent(type="text", text="Hello, this is a much longer message with more content")]
            ),
            PromptMessageMultipart(
                role="assistant",
                content=[TextContent(type="text", text="This is also a longer response with more details")]
            ),
            PromptMessageMultipart(
                role="user",
                content=[TextContent(type="text", text="And here is even more content to process")]
            )
        ]
        
        single_count = self.llm.get_token_count(single_message)
        multiple_count = self.llm.get_token_count(multiple_messages)
        
        # More content should result in higher token count
        assert isinstance(single_count, int)
        assert isinstance(multiple_count, int)
        assert multiple_count > single_count

    @patch('mcp_agent.llm.augmented_llm.create_turn_usage_from_messages')
    def test_get_token_count_calls_usage_tracking(self, mock_create_turn_usage):
        """Test that get_token_count properly calls the usage tracking function."""
        # Mock the return value with proper structure
        from mcp_agent.llm.usage_tracking import create_turn_usage_from_messages
        
        # Create a real turn usage using the actual function to avoid complex mocking
        mock_create_turn_usage.return_value = create_turn_usage_from_messages(
            input_content="test content",
            output_content="",
            model="test-model",
            model_type="testing"
        )
        
        messages = [
            PromptMessageMultipart(
                role="user",
                content=[TextContent(type="text", text="test")]
            )
        ]
        
        token_count = self.llm.get_token_count(messages)
        
        # Should call the usage tracking function
        mock_create_turn_usage.assert_called_once()
        call_args = mock_create_turn_usage.call_args
        
        # Verify the parameters passed to usage tracking
        assert 'input_content' in call_args[1]
        assert 'output_content' in call_args[1]
        assert 'model' in call_args[1]
        assert call_args[1]['output_content'] == ""
        assert "user: test" in call_args[1]['input_content']
        
        # Should return the input_tokens from the created usage
        assert isinstance(token_count, int)
        assert token_count >= 0