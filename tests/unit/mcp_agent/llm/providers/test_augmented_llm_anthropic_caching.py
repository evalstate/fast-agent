import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from mcp.types import TextContent

from mcp_agent.config import AnthropicSettings, Settings
from mcp_agent.llm.providers.augmented_llm_anthropic import AnthropicAugmentedLLM
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart


class TestAnthropicCaching(unittest.IsolatedAsyncioTestCase):
    """Test cases for Anthropic caching functionality."""

    def setUp(self):
            """Set up test environment."""
            self.mock_context = MagicMock()
            self.mock_context.config = Settings()
            self.mock_aggregator = AsyncMock()
            self.mock_aggregator.list_tools.return_value = MagicMock(tools=[])


    def _create_llm(self, cache_mode: str = "off") -> AnthropicAugmentedLLM:
        """Create an AnthropicAugmentedLLM instance with specified cache mode."""
        self.mock_context.config.anthropic = AnthropicSettings(
            api_key="test_key", cache_mode=cache_mode
        )
        return AnthropicAugmentedLLM(context=self.mock_context, aggregator=self.mock_aggregator)


    def _create_mock_stream_class(self):
        """Helper to create the MockStream class for tests."""
        class MockStream:
            async def __aenter__(self):
                mock_usage = MagicMock(input_tokens=100, output_tokens=50)
                final_message = MagicMock(
                    content=[MagicMock(type="text", text="Test response")],
                    stop_reason="end_turn",
                    usage=mock_usage,
                )
                self.get_final_message = AsyncMock(return_value=final_message)
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return None

            async def __aiter__(self):
                # This creates a proper async iterator that yields nothing
                if False:
                    yield
        return MockStream

    @patch("mcp_agent.llm.providers.augmented_llm_anthropic.AsyncAnthropic")
    async def test_caching_off_mode(self, mock_anthropic_class):
        """Test that no caching is applied when cache_mode is 'off'."""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client
        
        # FIX: Correct cache_mode and remove duplicate mock setup
        llm = self._create_llm(cache_mode="off")
        llm.instruction = "Test system prompt"

        captured_args = None
        MockStream = self._create_mock_stream_class()

        def stream_method(**kwargs):
            nonlocal captured_args
            captured_args = kwargs
            return MockStream()

        mock_client.messages.stream = stream_method

        message_param = {"role": "user", "content": [{"type": "text", "text": "Test message"}]}
        await llm._anthropic_completion(message_param)

        self.assertIsNotNone(captured_args)
        system = captured_args.get("system")
        self.assertIsInstance(system, str)
        self.assertEqual(system, "Test system prompt")

    @patch("mcp_agent.llm.providers.augmented_llm_anthropic.AsyncAnthropic")
    async def test_caching_prompt_mode(self, mock_anthropic_class):
        """Test caching behavior in 'prompt' mode."""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client
        
        # FIX: Correct cache_mode and remove duplicate mock setup
        llm = self._create_llm(cache_mode="prompt")
        llm.instruction = "Test system prompt"

        captured_args = None
        MockStream = self._create_mock_stream_class()

        def stream_method(**kwargs):
            nonlocal captured_args
            captured_args = kwargs
            return MockStream()

        mock_client.messages.stream = stream_method

        message_param = {"role": "user", "content": [{"type": "text", "text": "Test message"}]}
        await llm._anthropic_completion(message_param)

        self.assertIsNotNone(captured_args)
        system = captured_args.get("system")
        self.assertIsInstance(system, list)
        self.assertEqual(system[0].get("cache_control"), {"type": "ephemeral"})

    @patch("mcp_agent.llm.providers.augmented_llm_anthropic.AsyncAnthropic")
    async def test_caching_auto_mode(self, mock_anthropic_class):
        """Test caching behavior in 'auto' mode."""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client
        
        # FIX: Remove duplicate mock setup
        llm = self._create_llm(cache_mode="auto")
        llm.instruction = "Test system prompt"
        
        llm.history.extend(
            [
                {"role": "user", "content": [{"type": "text", "text": "First message"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "First response"}]},
                {"role": "user", "content": [{"type": "text", "text": "Second message"}]},
            ]
        )

        captured_args = None
        MockStream = self._create_mock_stream_class()

        def stream_method(**kwargs):
            nonlocal captured_args
            captured_args = kwargs
            return MockStream()

        mock_client.messages.stream = stream_method
        
        message_param = {"role": "user", "content": [{"type": "text", "text": "Test message"}]}
        await llm._anthropic_completion(message_param)

        self.assertIsNotNone(captured_args)
        system = captured_args.get("system")
        self.assertIsInstance(system, list)
        self.assertEqual(system[0].get("cache_control"), {"type": "ephemeral"})


    async def test_template_caching_prompt_mode(self):
        """Test that template messages are cached in 'prompt' mode."""
        llm = self._create_llm(cache_mode="prompt")

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

        # Mock generate_messages to capture the message_param
        captured_message_param = None

        async def mock_generate_messages(message_param, request_params=None):
            nonlocal captured_message_param
            captured_message_param = message_param
            return PromptMessageMultipart(
                role="assistant", content=[TextContent(type="text", text="Response")]
            )

        llm.generate_messages = mock_generate_messages

        # Apply template with is_template=True
        await llm._apply_prompt_provider_specific(
            template_messages, request_params=None, is_template=True
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
                role="user", content=[TextContent(type="text", text="Template message")]
            ),
            PromptMessageMultipart(
                role="user", content=[TextContent(type="text", text="Current question")]
            ),
        ]

        # Mock generate_messages
        async def mock_generate_messages(message_param, request_params=None):
            return PromptMessageMultipart(
                role="assistant", content=[TextContent(type="text", text="Response")]
            )

        llm.generate_messages = mock_generate_messages

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
