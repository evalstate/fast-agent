from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from fast_agent.llm.provider.openai.openresponses_streaming import OpenResponsesStreamingMixin
from fast_agent.llm.provider.openai.responses import ResponsesLLM
from fast_agent.llm.provider_types import Provider

if TYPE_CHECKING:
    from openai import AsyncOpenAI


class _OpenResponsesRawStream:
    """Wrap raw Responses SSE events without the SDK's accumulator.

    Some OpenResponses-compatible backends emit non-contiguous `content_index`
    values. The OpenAI SDK's higher-level `responses.stream(...)` accumulator
    assumes contiguous content indices and can crash before fast-agent sees the
    events. Iterating over `responses.create(..., stream=True)` yields the raw
    typed events directly, which is sufficient for fast-agent's own stream
    processors.
    """

    def __init__(self, raw_stream: Any) -> None:
        self._raw_stream = raw_stream
        self._iterator = self._iterate()
        self._final_response: Any | None = None

    def __aiter__(self) -> _OpenResponsesRawStream:
        return self

    async def __anext__(self) -> Any:
        return await self._iterator.__anext__()

    async def _iterate(self):
        async for event in self._raw_stream:
            if getattr(event, "type", None) in {
                "response.completed",
                "response.incomplete",
                "response.done",
            }:
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


class OpenResponsesLLM(OpenResponsesStreamingMixin, ResponsesLLM):
    """LLM implementation for Open Responses-compatible APIs."""

    config_section: str | None = "openresponses"

    def __init__(self, provider: Provider = Provider.OPENRESPONSES, **kwargs: Any) -> None:
        kwargs.pop("provider", None)
        super().__init__(provider=provider, **kwargs)

    @asynccontextmanager
    async def _response_sse_stream(
        self,
        *,
        client: AsyncOpenAI,
        arguments: dict[str, Any],
    ):
        raw_stream = await client.responses.create(**arguments, stream=True)
        wrapped_stream = _OpenResponsesRawStream(raw_stream)
        try:
            yield wrapped_stream
        finally:
            await wrapped_stream.close()
