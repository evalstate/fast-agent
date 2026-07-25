from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest
from openai.types.responses import ResponseTextDeltaEvent

from fast_agent.llm.provider.openai.codex_responses import CodexResponsesLLM
from fast_agent.llm.provider.openai.responses import ResponsesLLM
from fast_agent.llm.request_params import RequestParams

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from mcp import Tool


class _ClientContext:
    async def __aenter__(self) -> object:
        return object()

    async def __aexit__(self, *_args: object) -> None:
        return None


class _DelayedResponsesSseStream:
    def __init__(self) -> None:
        self.release_terminal = asyncio.Event()
        self._index = 0
        self.final_response = SimpleNamespace(
            status="completed",
            output=[
                SimpleNamespace(
                    type="message",
                    content=[SimpleNamespace(type="output_text", text="hello world")],
                )
            ],
            usage=None,
        )

    def __aiter__(self) -> _DelayedResponsesSseStream:
        return self

    async def __anext__(self) -> Any:
        if self._index == 0:
            self._index += 1
            return ResponseTextDeltaEvent(
                content_index=0,
                delta="hello ",
                item_id="msg_1",
                logprobs=[],
                output_index=0,
                sequence_number=1,
                type="response.output_text.delta",
            )
        if self._index == 1:
            self._index += 1
            await self.release_terminal.wait()
            return SimpleNamespace(
                type="response.completed",
                response=self.final_response,
            )
        raise StopAsyncIteration

    async def get_final_response(self) -> Any:
        return self.final_response


class _SimulatedSseMixin:
    sse_stream: _DelayedResponsesSseStream

    def _responses_client(self) -> _ClientContext:
        return _ClientContext()

    async def _normalize_input_files(
        self,
        client: Any,
        input_items: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        del client
        return input_items

    def _build_response_args(
        self,
        input_items: list[dict[str, Any]],
        request_params: RequestParams,
        tools: list[Tool] | None,
    ) -> dict[str, Any]:
        del tools
        return {
            "model": request_params.model,
            "input": input_items,
        }

    @asynccontextmanager
    async def _response_sse_stream(
        self,
        *,
        client: Any,
        arguments: dict[str, Any],
        timeout_seconds: float | None = None,
    ) -> AsyncIterator[_DelayedResponsesSseStream]:
        del client, arguments, timeout_seconds
        yield self.sse_stream


class _ResponsesSseHarness(_SimulatedSseMixin, ResponsesLLM):
    def __init__(self) -> None:
        ResponsesLLM.__init__(self, model="gpt-test", transport="sse")
        self.sse_stream = _DelayedResponsesSseStream()


class _CodexResponsesSseHarness(_SimulatedSseMixin, CodexResponsesLLM):
    def __init__(self) -> None:
        CodexResponsesLLM.__init__(self, model="gpt-test", transport="sse")
        self.sse_stream = _DelayedResponsesSseStream()


@pytest.mark.asyncio
@pytest.mark.parametrize("harness_type", [_ResponsesSseHarness, _CodexResponsesSseHarness])
async def test_sse_delta_reaches_listener_before_response_completes(
    harness_type: type[_ResponsesSseHarness] | type[_CodexResponsesSseHarness],
) -> None:
    harness = harness_type()
    chunk_received = asyncio.Event()
    chunks: list[str] = []

    def receive_chunk(chunk: Any) -> None:
        chunks.append(chunk.text)
        chunk_received.set()

    harness.add_stream_listener(receive_chunk)
    completion = asyncio.create_task(
        harness._responses_completion_sse(
            input_items=[
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hello"}],
                }
            ],
            request_params=RequestParams(model="gpt-test", streaming_timeout=1.0),
            tools=None,
            model_name="gpt-test",
        )
    )

    await asyncio.wait_for(chunk_received.wait(), timeout=1.0)

    assert chunks == ["hello "]
    assert not completion.done()

    harness.sse_stream.release_terminal.set()
    response, _summary, _input = await completion

    assert response is harness.sse_stream.final_response
