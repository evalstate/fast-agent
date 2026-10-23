"""Raw Responses SSE streaming that bypasses the OpenAI SDK snapshot accumulator."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from fast_agent.llm.provider.openai.responses_events import is_responses_terminal_event
from fast_agent.llm.provider.streaming_timeouts import await_stream_start

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from openai import AsyncOpenAI


class RawResponsesStream:
    """Wrap raw Responses SSE events without the SDK's accumulator.

    The SDK's ``responses.stream(...)`` accumulator enforces invariants that some
    Responses-compatible backends do not keep: non-contiguous ``content_index``
    values (OpenResponses servers) or a ``response.id`` that changes between
    lifecycle events (Copilot). Iterating ``responses.create(..., stream=True)``
    yields the typed events directly, which is all fast-agent's own stream
    processors need; the final response comes from the terminal event.
    """

    def __init__(self, raw_stream: Any) -> None:
        self._raw_stream = raw_stream
        self._iterator = self._iterate()
        self._final_response: Any | None = None

    def __aiter__(self) -> RawResponsesStream:
        return self

    async def __anext__(self) -> Any:
        return await self._iterator.__anext__()

    async def _iterate(self):
        async for event in self._raw_stream:
            if is_responses_terminal_event(getattr(event, "type", None)):
                self._final_response = getattr(event, "response", None) or self._final_response
            yield event

    async def get_final_response(self) -> Any:
        if self._final_response is not None:
            return self._final_response

        async for _event in self:
            pass

        if self._final_response is None:
            raise RuntimeError("Streaming completed without a final response payload.")
        return self._final_response

    async def close(self) -> None:
        response = getattr(self._raw_stream, "response", None)
        if response is not None:
            await response.aclose()


@asynccontextmanager
async def open_raw_responses_stream(
    *,
    client: AsyncOpenAI,
    arguments: dict[str, Any],
    timeout_seconds: float | None,
    label: str,
) -> AsyncGenerator[RawResponsesStream]:
    raw_stream = await await_stream_start(
        client.responses.create(**arguments, stream=True),
        timeout_seconds=timeout_seconds,
        timeout_message=f"{label} stream did not start within {timeout_seconds} seconds.",
    )
    wrapped_stream = RawResponsesStream(raw_stream)
    try:
        yield wrapped_stream
    finally:
        await wrapped_stream.close()
