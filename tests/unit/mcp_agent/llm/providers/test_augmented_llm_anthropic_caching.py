import unittest
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

from anthropic.types import MessageParam, ToolParam
from mcp.types import TextContent

from mcp_agent.config import AnthropicSettings, Settings
from mcp_agent.llm.providers.augmented_llm_anthropic import AnthropicAugmentedLLM
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart


class TestAnthropicCaching(unittest.TestCase):
    """Test cases for Anthropic caching functionality."""

    def setUp(self):
        """Set up test environment."""
        self.mock_context = MagicMock()
        self.mock_context.config = Settings()
        self.mock_aggregator = AsyncMock()
        self.mock_aggregator.list_tools = AsyncMock(
            return_value=MagicMock(
                tools=[
                    MagicMock(
                        name="test_tool",
                        description="Test tool",
                        inputSchema={"type": "object", "properties": {}},
                    )
                ]
            )
        )
        
    def _create_llm(self, cache_mode: str = "off") -> AnthropicAugmentedLLM:
        """Create an AnthropicAugmentedLLM instance with specified cache mode."""
        self.mock_context.config.anthropic = AnthropicSettings(
            api_key="test_key",
            cache_mode=cache_mode
        )
        
        llm = AnthropicAugmentedLLM(
            context=self.mock_context,
            aggregator=self.mock_aggregator
        )
        return llm
    
    def _extract_tools_from_args(self, arguments: Dict[str, Any]) -> List[ToolParam]:
        """Extract tools from API call arguments."""
        return arguments.get("tools", [])
    
    def _extract_system_from_args(self, arguments: Dict[str, Any]) -> Any:
        """Extract system prompt from API call arguments."""
        return arguments.get("system")
    
    def _extract_messages_from_args(self, arguments: Dict[str, Any]) -> List[MessageParam]:
        """Extract messages from API call arguments."""
        return arguments.get("messages", [])
    
    @patch("mcp_agent.llm.providers.augmented_llm_anthropic.Anthropic")
    async def test_caching_off_mode(self, mock_anthropic_class):
        """Test that no caching is applied when cache_mode is 'off'."""
        llm = self._create_llm(cache_mode="off")
        
        # Mock the Anthropic client
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client
        mock_response = MagicMock(
            content=[MagicMock(type="text", text="Test response")],
            stop_reason="end_turn",
            usage=MagicMock(input_tokens=100, output_tokens=50)
        )
        mock_client.messages.create = AsyncMock(return_value=mock_response)
        
        # Execute with mock executor
        llm.executor = MagicMock()
        llm.executor.execute = AsyncMock(return_value=[mock_response])
        
        # Create a test message
        message_param = {"role": "user", "content": [{"type": "text", "text": "Test message"}]}
        
        # Run the completion
        await llm._anthropic_completion(message_param)
        
        # Get the arguments passed to the API
        call_args = llm.executor.execute.call_args[1]
        
        # Verify no cache control blocks are present
        tools = self._extract_tools_from_args(call_args)
        if tools:
            self.assertNotIn("cache_control", tools[0])
        
        system = self._extract_system_from_args(call_args)
        if isinstance(system, list) and system:
            self.assertNotIn("cache_control", system[0])
    
    @patch("mcp_agent.llm.providers.augmented_llm_anthropic.Anthropic")
    async def test_caching_prompt_mode(self, mock_anthropic_class):
        """Test caching behavior in 'prompt' mode."""
        llm = self._create_llm(cache_mode="prompt")
        llm.instruction = "Test system prompt"
        
        # Mock the Anthropic client
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client
        mock_response = MagicMock(
            content=[MagicMock(type="text", text="Test response")],
            stop_reason="end_turn",
            usage=MagicMock(input_tokens=100, output_tokens=50)
        )
        mock_client.messages.create = AsyncMock(return_value=mock_response)
        
        # Execute with mock executor
        llm.executor = MagicMock()
        llm.executor.execute = AsyncMock(return_value=[mock_response])
        
        # Create a test message
        message_param = {"role": "user", "content": [{"type": "text", "text": "Test message"}]}
        
        # Run the completion
        await llm._anthropic_completion(message_param)
        
        # Get the arguments passed to the API
        call_args = llm.executor.execute.call_args[1]
        
        # Verify cache control is applied to tools
        tools = self._extract_tools_from_args(call_args)
        self.assertTrue(len(tools) > 0)
        self.assertIn("cache_control", tools[0])
        self.assertEqual(tools[0]["cache_control"]["type"], "ephemeral")
        
        # Verify cache control is applied to system prompt
        system = self._extract_system_from_args(call_args)
        self.assertIsInstance(system, list)
        self.assertEqual(system[0]["type"], "text")
        self.assertIn("cache_control", system[0])
        self.assertEqual(system[0]["cache_control"]["type"], "ephemeral")
    
    @patch("mcp_agent.llm.providers.augmented_llm_anthropic.Anthropic")
    async def test_caching_auto_mode(self, mock_anthropic_class):
        """Test caching behavior in 'auto' mode."""
        llm = self._create_llm(cache_mode="auto")
        llm.instruction = "Test system prompt"
        
        # Add some messages to history to test message caching
        llm.history.extend([
            {"role": "user", "content": [{"type": "text", "text": "First message"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "First response"}]},
            {"role": "user", "content": [{"type": "text", "text": "Second message"}]},
        ])
        
        # Mock the Anthropic client
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client
        mock_response = MagicMock(
            content=[MagicMock(type="text", text="Test response")],
            stop_reason="end_turn",
            usage=MagicMock(input_tokens=100, output_tokens=50)
        )
        mock_client.messages.create = AsyncMock(return_value=mock_response)
        
        # Execute with mock executor
        llm.executor = MagicMock()
        llm.executor.execute = AsyncMock(return_value=[mock_response])
        
        # Create a test message
        message_param = {"role": "user", "content": [{"type": "text", "text": "Test message"}]}
        
        # Run the completion
        await llm._anthropic_completion(message_param)
        
        # Get the arguments passed to the API
        call_args = llm.executor.execute.call_args[1]
        
        # Verify cache control is applied to tools
        tools = self._extract_tools_from_args(call_args)
        self.assertTrue(len(tools) > 0)
        self.assertIn("cache_control", tools[0])
        self.assertEqual(tools[0]["cache_control"]["type"], "ephemeral")
        
        # Verify cache control is applied to system prompt
        system = self._extract_system_from_args(call_args)
        self.assertIsInstance(system, list)
        self.assertEqual(system[0]["type"], "text")
        self.assertIn("cache_control", system[0])
        self.assertEqual(system[0]["cache_control"]["type"], "ephemeral")
        
        # Verify cache control is applied to recent messages
        messages = self._extract_messages_from_args(call_args)
        # Check that at least one of the recent messages has cache control
        cached_messages = 0
        for msg in messages[-2:]:  # Check last 2 messages
            if isinstance(msg, dict) and "content" in msg:
                for block in msg["content"]:
                    if isinstance(block, dict) and "cache_control" in block:
                        cached_messages += 1
                        self.assertEqual(block["cache_control"]["type"], "ephemeral")
        self.assertGreater(cached_messages, 0, "No messages were cached in auto mode")
    
    async def test_template_caching_prompt_mode(self):
        """Test that template messages are cached in 'prompt' mode."""
        llm = self._create_llm(cache_mode="prompt")
        
        # Create template messages
        template_messages = [
            PromptMessageMultipart(
                role="user",
                content=[TextContent(type="text", text="Template message 1")]
            ),
            PromptMessageMultipart(
                role="assistant",
                content=[TextContent(type="text", text="Template response 1")]
            ),
            PromptMessageMultipart(
                role="user",
                content=[TextContent(type="text", text="Current question")]
            )
        ]
        
        # Mock generate_messages to capture the message_param
        captured_message_param = None
        async def mock_generate_messages(message_param, request_params=None):
            nonlocal captured_message_param
            captured_message_param = message_param
            return PromptMessageMultipart(
                role="assistant",
                content=[TextContent(type="text", text="Response")]
            )
        
        llm.generate_messages = mock_generate_messages
        
        # Apply template with is_template=True
        await llm._apply_prompt_provider_specific(
            template_messages, 
            request_params=None,
            is_template=True
        )
        
        # Check that template messages in history have cache control
        history_messages = llm.history.get(include_completion_history=False)
        
        # Verify that at least one template message has cache control
        found_cache_control = False
        for msg in history_messages:
            if isinstance(msg, dict) and "content" in msg:
                for block in msg["content"]:
                    if isinstance(block, dict) and "cache_control" in block:
                        found_cache_control = True
                        self.assertEqual(block["cache_control"]["type"], "ephemeral")
        
        self.assertTrue(found_cache_control, "No cache control found in template messages")
    
    async def test_template_caching_off_mode(self):
        """Test that template messages are NOT cached in 'off' mode."""
        llm = self._create_llm(cache_mode="off")
        
        # Create template messages
        template_messages = [
            PromptMessageMultipart(
                role="user",
                content=[TextContent(type="text", text="Template message")]
            ),
            PromptMessageMultipart(
                role="user",
                content=[TextContent(type="text", text="Current question")]
            )
        ]
        
        # Mock generate_messages
        async def mock_generate_messages(message_param, request_params=None):
            return PromptMessageMultipart(
                role="assistant",
                content=[TextContent(type="text", text="Response")]
            )
        
        llm.generate_messages = mock_generate_messages
        
        # Apply template with is_template=True
        await llm._apply_prompt_provider_specific(
            template_messages, 
            request_params=None,
            is_template=True
        )
        
        # Check that template messages in history do NOT have cache control
        history_messages = llm.history.get(include_completion_history=False)
        
        # Verify that no template message has cache control
        for msg in history_messages:
            if isinstance(msg, dict) and "content" in msg:
                for block in msg["content"]:
                    if isinstance(block, dict):
                        self.assertNotIn("cache_control", block, 
                                       "Cache control found in template message when cache_mode is 'off'")


if __name__ == "__main__":
    unittest.main()