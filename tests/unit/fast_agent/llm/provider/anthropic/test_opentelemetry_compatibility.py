"""
Unit tests for Anthropic OpenTelemetry compatibility.

Tests the compatibility layer that handles OpenTelemetry instrumentation wrapping
the stream() call and returning a coroutine that must be awaited.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from anthropic.types.beta import BetaMessage, BetaTextBlock, BetaUsage

from fast_agent.config import AnthropicSettings, Settings
from fast_agent.context import Context
from fast_agent.llm.provider.anthropic.llm_anthropic import AnthropicLLM


class MockStreamManager:
    """Mock stream manager that simulates Anthropic's stream interface."""

    def __init__(self, final_message: BetaMessage):
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
        if asyncio.current_task() is None:  # pragma: no cover - keeps this an async generator
            yield None

    async def get_final_message(self) -> BetaMessage:
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

    def _create_mock_message(self, text: str = "Hello from AI") -> BetaMessage:
        """Create a mock Anthropic message."""
        return BetaMessage(
            id="msg_123",
            type="message",
            role="assistant",
            content=[BetaTextBlock(type="text", text=text)],
            model="claude-3-5-sonnet-20241022",
            stop_reason="end_turn",
            usage=BetaUsage(input_tokens=10, output_tokens=20),
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

        with patch(
            "fast_agent.llm.provider.anthropic.llm_anthropic.AsyncAnthropic"
        ) as mock_anthropic_cls:
            mock_anthropic = MagicMock()
            mock_anthropic_cls.return_value = mock_anthropic

            # Simulate non-OpenTelemetry behavior: stream() returns manager directly
            mock_anthropic.beta.messages.stream.return_value = mock_stream_manager

            # Mock _process_stream to return the final message
            with patch.object(llm, "_process_stream", new_callable=AsyncMock) as mock_process:
                mock_process.return_value = (final_message, [], [])

                from mcp_types import TextContent

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
                assert (
                    str(result.stop_reason.value) == "endTurn"
                    or result.stop_reason.value == "end_turn"
                )

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

        with patch(
            "fast_agent.llm.provider.anthropic.llm_anthropic.AsyncAnthropic"
        ) as mock_anthropic_cls:
            mock_anthropic = MagicMock()
            mock_anthropic_cls.return_value = mock_anthropic

            # Simulate OpenTelemetry behavior: stream() returns a coroutine
            mock_anthropic.beta.messages.stream.return_value = coroutine_stream_call()

            from mcp_types import TextContent

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
                assert (
                    str(result.stop_reason.value) == "endTurn"
                    or result.stop_reason.value == "end_turn"
                )

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


@pytest.mark.asyncio
@pytest.mark.parametrize("with_span", [False, True])
@pytest.mark.parametrize("outcome", ["success", "error", "api_error", "cancel", "idle_timeout"])
async def test_stream_execution_contract(with_span: bool, outcome: str):
    from contextlib import nullcontext

    import httpx2
    from anthropic import APIError

    from fast_agent.llm.provider.anthropic import llm_anthropic as provider
    from fast_agent.llm.provider.streaming_timeouts import StreamIdleTimeoutError

    fixtures = TestOpenTelemetryCompatibility()
    llm = fixtures._create_llm()
    message = fixtures._create_mock_message()
    failure = (
        APIError("failed", httpx2.Request("POST", "https://example.com"), body=None)
        if outcome == "api_error"
        else RuntimeError("failed")
    )
    waiting = asyncio.Event()
    iterator_closed = asyncio.Event()

    class Stream(MockStreamManager):
        async def __aiter__(self):
            try:
                if outcome in {"error", "api_error"}:
                    raise failure
                if outcome in {"cancel", "idle_timeout"}:
                    waiting.set()
                    await asyncio.Event().wait()
                async for event in super().__aiter__():
                    yield event
            finally:
                iterator_closed.set()

    stream = Stream(message)
    client = MagicMock()
    stream_method = client.beta.messages.stream
    stream_method.return_value = stream
    span = MagicMock()
    span.is_recording.return_value = True
    # A distinct method triggers the fallback span, as an unwrapped OTel method does.
    selected_method = MagicMock(return_value=stream) if with_span else stream_method
    with (
        patch.object(provider, "_maybe_unwrap_otel_beta_stream", return_value=selected_method),
        patch.object(provider, "_start_fallback_stream_span", return_value=span),
        patch.object(provider.trace, "use_span", return_value=nullcontext()) as use_span,
        patch.object(provider.logger, "error") as log_error,
    ):
        task = asyncio.create_task(
            llm._execute_anthropic_stream(
                anthropic=client,
                arguments={},
                model=message.model,
                capture_filename=None,
                timeout_seconds=0.01 if outcome == "idle_timeout" else None,
            )
        )
        if outcome == "cancel":
            await asyncio.wait_for(waiting.wait(), timeout=1)
            task.cancel()
        if outcome == "success":
            result = await task
            assert result == (message, [], [])
        elif outcome in {"error", "api_error"}:
            with pytest.raises(type(failure)) as caught:
                await task
            assert caught.value is failure
        elif outcome == "cancel":
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            with pytest.raises(StreamIdleTimeoutError):
                await task

        assert stream._entered and stream._exited
        assert iterator_closed.is_set()
        if with_span:
            use_span.assert_called_once_with(span, end_on_exit=False)
            span.end.assert_called_once()
            if outcome in {"error", "api_error", "idle_timeout"}:
                span.record_exception.assert_called_once()
            else:
                span.record_exception.assert_not_called()
        else:
            use_span.assert_not_called()
            span.end.assert_not_called()
        if outcome == "api_error":
            assert any(
                call.args == ("Streaming APIError during Anthropic completion",)
                and call.kwargs.get("exc_info") is failure
                for call in log_error.call_args_list
            )
