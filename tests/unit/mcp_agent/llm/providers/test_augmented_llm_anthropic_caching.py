import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from mcp.types import TextContent

from fast_agent.config import AnthropicSettings, Settings
from mcp_agent.llm.providers.augmented_llm_anthropic import AnthropicAugmentedLLM
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart


class TestAnthropicCaching(unittest.IsolatedAsyncioTestCase):
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
            api_key="test_key", cache_mode=cache_mode
        )

        llm = AnthropicAugmentedLLM(context=self.mock_context, aggregator=self.mock_aggregator)
        return llm

    @patch("mcp_agent.llm.providers.augmented_llm_anthropic.AsyncAnthropic")
    async def test_caching_off_mode(self, mock_anthropic_class):
        """Test that no caching is applied when cache_mode is 'off'."""
        llm = self._create_llm(cache_mode="off")
        llm.instruction = "Test system prompt"

        # Capture the arguments passed to the streaming API
        captured_args = None

        # Mock the Anthropic client
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        # Create a proper async context manager for the stream
        class MockStream:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return None

            def __aiter__(self):
                return iter([])

        # Capture arguments and return the mock stream
        def stream_method(**kwargs):
            nonlocal captured_args
            captured_args = kwargs
            return MockStream()

        mock_client.messages.stream = stream_method

        # Mock the _process_stream method to return a response
        # Create a usage mock that won't trigger warnings
        mock_usage = MagicMock()
        mock_usage.input_tokens = 100
        mock_usage.output_tokens = 50
        mock_usage.cache_creation_input_tokens = None
        mock_usage.cache_read_input_tokens = None
        mock_usage.trafficType = None  # Add trafficType to prevent Google genai warning

        mock_response = MagicMock(
            content=[MagicMock(type="text", text="Test response")],
            stop_reason="end_turn",
            usage=mock_usage,
        )
        llm._process_stream = AsyncMock(return_value=mock_response)

        # Create a test message
        message_param = {"role": "user", "content": [{"type": "text", "text": "Test message"}]}

        # Run the completion
        await llm._anthropic_completion(message_param)

        # Verify arguments were captured
        self.assertIsNotNone(captured_args)

        # Check that system prompt exists but has no cache_control
        system = captured_args.get("system")
        self.assertIsNotNone(system)

        # When cache_mode is "off", system should remain a string
        self.assertIsInstance(system, str)
        self.assertEqual(system, "Test system prompt")

    @patch("mcp_agent.llm.providers.augmented_llm_anthropic.AsyncAnthropic")
    async def test_caching_prompt_mode(self, mock_anthropic_class):
        """Test caching behavior in 'prompt' mode."""
        llm = self._create_llm(cache_mode="prompt")
        llm.instruction = "Test system prompt"

        # Capture the arguments passed to the streaming API
        captured_args = None

        # Mock the Anthropic client
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        # Create a proper async context manager for the stream
        class MockStream:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return None

            def __aiter__(self):
                return iter([])

        # Capture arguments and return the mock stream
        def stream_method(**kwargs):
            nonlocal captured_args
            captured_args = kwargs
            return MockStream()

        mock_client.messages.stream = stream_method

        # Mock the _process_stream method to return a response
        # Create a usage mock that won't trigger warnings
        mock_usage = MagicMock()
        mock_usage.input_tokens = 100
        mock_usage.output_tokens = 50
        mock_usage.cache_creation_input_tokens = None
        mock_usage.cache_read_input_tokens = None
        mock_usage.trafficType = None  # Add trafficType to prevent Google genai warning

        mock_response = MagicMock(
            content=[MagicMock(type="text", text="Test response")],
            stop_reason="end_turn",
            usage=mock_usage,
        )
        llm._process_stream = AsyncMock(return_value=mock_response)

        # Create a test message
        message_param = {"role": "user", "content": [{"type": "text", "text": "Test message"}]}

        # Run the completion
        await llm._anthropic_completion(message_param)

        # Verify arguments were captured
        self.assertIsNotNone(captured_args)

        # Check that system prompt has cache_control when cache_mode is "prompt"
        system = captured_args.get("system")
        self.assertIsNotNone(system)

        # When cache_mode is "prompt", system should be converted to a list with cache_control
        self.assertIsInstance(system, list)
        self.assertEqual(len(system), 1)
        self.assertEqual(system[0]["type"], "text")
        self.assertEqual(system[0]["text"], "Test system prompt")
        self.assertIn("cache_control", system[0])
        self.assertEqual(system[0]["cache_control"]["type"], "ephemeral")

        # Note: According to the code comment, tools and system are cached together
        # via the system prompt, so tools themselves don't get cache_control

    @patch("mcp_agent.llm.providers.augmented_llm_anthropic.AsyncAnthropic")
    async def test_caching_auto_mode(self, mock_anthropic_class):
        """Test caching behavior in 'auto' mode."""
        llm = self._create_llm(cache_mode="auto")
        llm.instruction = "Test system prompt"

        # Add some messages to history to test message caching
        llm.history.extend(
            [
                {"role": "user", "content": [{"type": "text", "text": "First message"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "First response"}]},
                {"role": "user", "content": [{"type": "text", "text": "Second message"}]},
            ]
        )

        # Capture the arguments passed to the streaming API
        captured_args = None

        # Mock the Anthropic client
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        # Create a proper async context manager for the stream
        class MockStream:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return None

            def __aiter__(self):
                return iter([])

        # Capture arguments and return the mock stream
        def stream_method(**kwargs):
            nonlocal captured_args
            captured_args = kwargs
            return MockStream()

        mock_client.messages.stream = stream_method

        # Mock the _process_stream method to return a response
        # Create a usage mock that won't trigger warnings
        mock_usage = MagicMock()
        mock_usage.input_tokens = 100
        mock_usage.output_tokens = 50
        mock_usage.cache_creation_input_tokens = None
        mock_usage.cache_read_input_tokens = None
        mock_usage.trafficType = None  # Add trafficType to prevent Google genai warning

        mock_response = MagicMock(
            content=[MagicMock(type="text", text="Test response")],
            stop_reason="end_turn",
            usage=mock_usage,
        )
        llm._process_stream = AsyncMock(return_value=mock_response)

        # Create a test message
        message_param = {"role": "user", "content": [{"type": "text", "text": "Test message"}]}

        # Run the completion
        await llm._anthropic_completion(message_param)

        # Verify arguments were captured
        self.assertIsNotNone(captured_args)

        # Check that system prompt has cache_control when cache_mode is "auto"
        system = captured_args.get("system")
        self.assertIsNotNone(system)

        # When cache_mode is "auto", system should be converted to a list with cache_control
        self.assertIsInstance(system, list)
        self.assertEqual(len(system), 1)
        self.assertEqual(system[0]["type"], "text")
        self.assertEqual(system[0]["text"], "Test system prompt")
        self.assertIn("cache_control", system[0])
        self.assertEqual(system[0]["cache_control"]["type"], "ephemeral")

        # In auto mode, conversation messages may have cache control if there are enough messages
        messages = captured_args.get("messages", [])
        self.assertGreater(len(messages), 0)

        # Verify we have the expected messages
        # History has 3 messages + prompt messages (if any) + the new message
        # Let's just verify we have messages and the structure is correct
        self.assertGreaterEqual(len(messages), 4)  # At least the history + new message

    @patch("mcp_agent.llm.providers.augmented_llm_anthropic.AsyncAnthropic")
    async def test_template_caching_prompt_mode(self, mock_anthropic_class):
        """Test that template messages are cached in 'prompt' mode."""
        llm = self._create_llm(cache_mode="prompt")

        # Mock the Anthropic client
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        # Create a proper async context manager for the stream
        class MockStream:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return None

            def __aiter__(self):
                return iter([])

        # Mock the stream method
        mock_client.messages.stream = lambda **kwargs: MockStream()

        # Mock the _process_stream method to return a response
        mock_usage = MagicMock()
        mock_usage.input_tokens = 100
        mock_usage.output_tokens = 50
        mock_usage.cache_creation_input_tokens = None
        mock_usage.cache_read_input_tokens = None
        mock_usage.trafficType = None

        mock_response = MagicMock(
            content=[MagicMock(type="text", text="Response")],
            stop_reason="end_turn",
            usage=mock_usage,
        )
        llm._process_stream = AsyncMock(return_value=mock_response)

        # Create template messages
        template_messages = [
            PromptMessageMultipart(
                role="user", content=[TextContent(type="text", text="Template message 1")]
            ),
            PromptMessageMultipart(
                role="assistant", content=[TextContent(type="text", text="Template response 1")]
            ),
            PromptMessageMultipart(
                role="user", content=[TextContent(type="text", text="Current question")]
            ),
        ]

        # Apply template with is_template=True
        await llm._apply_prompt_provider_specific(
            template_messages, request_params=None, tools=None, is_template=True
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

    @patch("mcp_agent.llm.providers.augmented_llm_anthropic.AsyncAnthropic")
    async def test_template_caching_off_mode(self, mock_anthropic_class):
        """Test that template messages are NOT cached in 'off' mode."""
        llm = self._create_llm(cache_mode="off")

        # Mock the Anthropic client
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        # Create a proper async context manager for the stream
        class MockStream:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return None

            def __aiter__(self):
                return iter([])

        # Mock the stream method
        mock_client.messages.stream = lambda **kwargs: MockStream()

        # Mock the _process_stream method to return a response
        mock_usage = MagicMock()
        mock_usage.input_tokens = 100
        mock_usage.output_tokens = 50
        mock_usage.cache_creation_input_tokens = None
        mock_usage.cache_read_input_tokens = None
        mock_usage.trafficType = None

        mock_response = MagicMock(
            content=[MagicMock(type="text", text="Response")],
            stop_reason="end_turn",
            usage=mock_usage,
        )
        llm._process_stream = AsyncMock(return_value=mock_response)

        # Create template messages
        template_messages = [
            PromptMessageMultipart(
                role="user", content=[TextContent(type="text", text="Template message")]
            ),
            PromptMessageMultipart(
                role="user", content=[TextContent(type="text", text="Current question")]
            ),
        ]

        # Apply template with is_template=True
        await llm._apply_prompt_provider_specific(
            template_messages, request_params=None, is_template=True
        )

        # Check that template messages in history do NOT have cache control
        history_messages = llm.history.get(include_completion_history=False)

        # Verify that no template message has cache control
        for msg in history_messages:
            if isinstance(msg, dict) and "content" in msg:
                for block in msg["content"]:
                    if isinstance(block, dict):
                        self.assertNotIn(
                            "cache_control",
                            block,
                            "Cache control found in template message when cache_mode is 'off'",
                        )


if __name__ == "__main__":
    unittest.main()
