"""
Unit tests for Anthropic OpenTelemetry compatibility.

Tests the compatibility layer that handles OpenTelemetry instrumentation wrapping
the stream() call and returning a coroutine that must be awaited.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fast_agent.config import AnthropicSettings, Settings
from fast_agent.context import Context
from fast_agent.llm.provider.anthropic.beta_types import Message, TextBlock, Usage
from fast_agent.llm.provider.anthropic.llm_anthropic import AnthropicLLM


class MockStreamManager:
    """Mock stream manager that simulates Anthropic's stream interface."""

    def __init__(self, final_message: Message):
        self._entered = False
        self._exited = False
        self._final_message = final_message

    async def __aenter__(self):
        self._entered = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._exited = True
        return False

    async def __aiter__(self):
        # Yield no events for simplicity
        return
        yield  # Make this a generator

    async def get_final_message(self) -> Message:
        """Return the mock final message."""
        return self._final_message


class TestOpenTelemetryCompatibility:
    """Test cases for OpenTelemetry compatibility in streaming."""

    def _create_llm(self) -> AnthropicLLM:
        """Create an AnthropicLLM instance for testing."""
        ctx = Context()
        ctx.config = Settings()
        ctx.config.anthropic = AnthropicSettings(api_key="test_key")
        llm = AnthropicLLM(context=ctx)
        return llm

    def _create_mock_message(self, text: str = "Hello from AI") -> Message:
        """Create a mock Anthropic message."""
        return Message(
            id="msg_123",
            type="message",
            role="assistant",
            content=[TextBlock(type="text", text=text)],
            model="claude-3-5-sonnet-20241022",
            stop_reason="end_turn",
            usage=Usage(input_tokens=10, output_tokens=20),
        )

    @pytest.mark.asyncio
    async def test_stream_without_opentelemetry(self):
        """
        Test streaming when OpenTelemetry is NOT installed.
        The stream() call returns a stream manager directly (not a coroutine).
        """
        llm = self._create_llm()
        final_message = self._create_mock_message()
        mock_stream_manager = MockStreamManager(final_message)

        with patch("fast_agent.llm.provider.anthropic.llm_anthropic.AsyncAnthropic") as mock_anthropic_cls:
            mock_anthropic = MagicMock()
            mock_anthropic_cls.return_value = mock_anthropic

            # Simulate non-OpenTelemetry behavior: stream() returns manager directly
            mock_anthropic.beta.messages.stream.return_value = mock_stream_manager

            # Mock _process_stream to return the final message
            with patch.object(llm, "_process_stream", new_callable=AsyncMock) as mock_process:
                mock_process.return_value = (final_message, [], [])

                from mcp.types import TextContent

                from fast_agent.mcp.prompt_message_extended import PromptMessageExtended

                message_param = {
                    "role": "user",
                    "content": [{"type": "text", "text": "Hello"}],
                }
                current_extended = PromptMessageExtended(
                    role="user", content=[TextContent(type="text", text="Hello")]
                )

                result = await llm._anthropic_completion(
                    message_param,
                    history=[],
                    current_extended=current_extended,
                )

                # Verify the stream manager was used correctly
                assert mock_stream_manager._entered, "Stream manager should have been entered"
                assert mock_stream_manager._exited, "Stream manager should have been exited"
                assert result.role == "assistant"
                # stop_reason is an enum-like object, compare the value string
                assert result.stop_reason is not None
                assert str(result.stop_reason.value) == "endTurn" or result.stop_reason.value == "end_turn"

    @pytest.mark.asyncio
    async def test_stream_with_opentelemetry(self):
        """
        Test streaming when OpenTelemetry IS installed.
        The stream() call returns a coroutine that must be awaited first.
        """
        llm = self._create_llm()
        final_message = self._create_mock_message()
        mock_stream_manager = MockStreamManager(final_message)

        async def coroutine_stream_call():
            """Simulate OpenTelemetry wrapping: return coroutine that resolves to manager."""
            return mock_stream_manager

        with patch("fast_agent.llm.provider.anthropic.llm_anthropic.AsyncAnthropic") as mock_anthropic_cls:
            mock_anthropic = MagicMock()
            mock_anthropic_cls.return_value = mock_anthropic

            # Simulate OpenTelemetry behavior: stream() returns a coroutine
            mock_anthropic.beta.messages.stream.return_value = coroutine_stream_call()

            from mcp.types import TextContent

            from fast_agent.mcp.prompt_message_extended import PromptMessageExtended

            message_param = {
                "role": "user",
                "content": [{"type": "text", "text": "Hello"}],
            }
            current_extended = PromptMessageExtended(
                role="user", content=[TextContent(type="text", text="Hello")]
            )

            # Mock the _process_stream method
            with patch.object(llm, "_process_stream", new_callable=AsyncMock) as mock_process:
                mock_process.return_value = (final_message, [], [])

                result = await llm._anthropic_completion(
                    message_param,
                    history=[],
                    current_extended=current_extended,
                )

                # Verify the stream manager was correctly awaited and used
                assert mock_stream_manager._entered, "Stream manager should have been entered"
                assert mock_stream_manager._exited, "Stream manager should have been exited"
                assert result.role == "assistant"
                # stop_reason is an enum-like object, compare the value string
                assert result.stop_reason is not None
                assert str(result.stop_reason.value) == "endTurn" or result.stop_reason.value == "end_turn"

    @pytest.mark.asyncio
    async def test_iscoroutine_detection(self):
        """
        Test that asyncio.iscoroutine() correctly detects both scenarios.
        """
        final_message = self._create_mock_message()
        
        # Test non-coroutine case
        mock_stream_manager = MockStreamManager(final_message)
        assert not asyncio.iscoroutine(mock_stream_manager)

        # Test coroutine case
        async def async_func():
            return mock_stream_manager

        coroutine_obj = async_func()
        assert asyncio.iscoroutine(coroutine_obj)
        # Clean up the coroutine
        await coroutine_obj
