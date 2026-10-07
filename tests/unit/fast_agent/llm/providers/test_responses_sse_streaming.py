from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest
from openai.types.responses import (
    Response,
    ResponseCreatedEvent,
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputRefusal,
    ResponseOutputText,
    ResponseTextDeltaEvent,
)

from fast_agent.llm.provider.openai.codex_responses import CodexResponsesLLM
from fast_agent.llm.provider.openai.responses import ResponsesLLM
from fast_agent.llm.request_params import RequestParams
from fast_agent.types import LlmStopReason

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from mcp import Tool
    from openai import AsyncOpenAI


class _ClientContext:
    async def __aenter__(self) -> object:
        return object()

    async def __aexit__(self, *_args: object) -> None:
        return None


class _DelayedResponsesSseStream:
    def __init__(self, *, safety_buffering: bool = False) -> None:
        self.release_terminal = asyncio.Event()
        self._index = 0
        self.safety_buffering = safety_buffering
        self.final_response = Response(
            id="resp_1",
            created_at=0.0,
            model="gpt-test",
            object="response",
            status="completed",
            output=[],
            parallel_tool_calls=True,
            tool_choice="auto",
            tools=[],
        )
        self.completed_message = ResponseOutputMessage(
            id="msg_1",
            type="message",
            role="assistant",
            status="completed",
            content=[
                ResponseOutputText(
                    annotations=[],
                    text="hello world",
                    type="output_text",
                )
            ],
        )

    def __aiter__(self) -> _DelayedResponsesSseStream:
        return self

    async def __anext__(self) -> Any:
        # Buffered Codex streams repeat ``safety_buffering`` on every event.
        buffering = (
            {
                "safety_buffering": {
                    "use_cases": ["cyber"],
                    "reasons": ["policy-check"],
                    "retry_model": "gpt-test-fast",
                }
            }
            if self.safety_buffering
            else {}
        )
        if self.safety_buffering and self._index == 0:
            self._index += 1
            return ResponseCreatedEvent.model_validate(
                {
                    "response": self.final_response,
                    "sequence_number": 0,
                    "type": "response.created",
                    **buffering,
                }
            )
        stream_index = self._index - int(self.safety_buffering)
        if stream_index == 0:
            self._index += 1
            return ResponseTextDeltaEvent.model_validate(
                {
                    "content_index": 0,
                    "delta": "hello ",
                    "item_id": "msg_1",
                    "logprobs": [],
                    "output_index": 0,
                    "sequence_number": 1,
                    "type": "response.output_text.delta",
                    **buffering,
                }
            )
        if stream_index == 1:
            self._index += 1
            return SimpleNamespace(
                type="response.output_item.done",
                item=self.completed_message,
                item_id="msg_1",
                output_index=0,
                sequence_number=2,
            )
        if stream_index == 2:
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
    sse_calls: int = 0

    def _responses_client(self) -> AsyncOpenAI:
        return cast("AsyncOpenAI", _ClientContext())

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
        self.sse_calls += 1
        yield self.sse_stream


class _ResponsesSseHarness(_SimulatedSseMixin, ResponsesLLM):
    def __init__(self, *, safety_buffering: bool = False) -> None:
        ResponsesLLM.__init__(self, model="gpt-test", transport="sse")
        self.sse_stream = _DelayedResponsesSseStream(safety_buffering=safety_buffering)


class _CodexResponsesSseHarness(_SimulatedSseMixin, CodexResponsesLLM):
    def __init__(self, *, safety_buffering: bool = False) -> None:
        CodexResponsesLLM.__init__(self, model="gpt-test", transport="sse")
        self.sse_stream = _DelayedResponsesSseStream(safety_buffering=safety_buffering)


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

    assert harness.sse_stream.final_response.output == []
    assert response.output == [harness.sse_stream.completed_message]


@pytest.mark.asyncio
@pytest.mark.parametrize("harness_type", [_ResponsesSseHarness, _CodexResponsesSseHarness])
@pytest.mark.parametrize("refusal", [False, True])
async def test_safety_buffering_notice_reaches_listener_before_response_completes(
    harness_type: type[_ResponsesSseHarness] | type[_CodexResponsesSseHarness],
    refusal: bool,
) -> None:
    harness = harness_type(safety_buffering=True)
    if refusal:
        harness.sse_stream.completed_message.content = [
            ResponseOutputRefusal(type="refusal", refusal="I cannot help with that.")
        ]
        harness.sse_stream.final_response.output = [
            harness.sse_stream.completed_message,
            ResponseFunctionToolCall(
                type="function_call", call_id="call_1", name="unused", arguments="{}"
            ),
        ]
    chunk_received = asyncio.Event()
    chunks: list[Any] = []

    def receive_chunk(chunk: Any) -> None:
        chunks.append(chunk)
        chunk_received.set()

    harness.add_stream_listener(receive_chunk)
    completion = asyncio.create_task(
        harness._responses_completion(
            input_items=[
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hello"}],
                }
            ],
            request_params=RequestParams(model="gpt-test", streaming_timeout=1.0),
            tools=None,
        )
    )

    await asyncio.wait_for(chunk_received.wait(), timeout=1.0)

    assert chunks
    assert chunks[0].is_reasoning
    assert "Waiting for the original stream" in chunks[0].text
    assert not completion.done()
    assert harness.sse_calls == 1

    harness.sse_stream.release_terminal.set()
    response = await asyncio.wait_for(completion, timeout=1.0)
    assert sum("Waiting for the original stream" in chunk.text for chunk in chunks) == 1
    assert response.stop_reason == (LlmStopReason.SAFETY if refusal else LlmStopReason.END_TURN)
    assert response.last_text() == ("I cannot help with that." if refusal else "hello world")
    assert not response.tool_calls
    assert harness.sse_calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("harness_type", [_ResponsesSseHarness, _CodexResponsesSseHarness])
@pytest.mark.parametrize("use_history", [True, False])
async def test_explicit_end_turn_false_continues_without_synthetic_input(
    harness_type: type[_ResponsesSseHarness] | type[_CodexResponsesSseHarness],
    use_history: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fast_agent.agents.agent_types import AgentConfig
    from fast_agent.agents.tool_agent import ToolAgent
    from fast_agent.llm.provider.openai.responses import RESPONSES_DIAGNOSTICS_CHANNEL
    from fast_agent.mcp.prompt_message_extended import PromptMessageExtended

    harness = harness_type()
    requests: list[list[dict[str, Any]]] = []

    @asynccontextmanager
    async def stream_response(**kwargs: Any) -> AsyncIterator[_DelayedResponsesSseStream]:
        requests.append(kwargs["arguments"]["input"])
        stream = _DelayedResponsesSseStream()
        stream.completed_message.content = [
            ResponseOutputText(type="output_text", text=f"step-{len(requests)}", annotations=[])
        ]
        stream.completed_message.phase = "commentary" if len(requests) == 1 else "final_answer"
        stream.final_response = Response.model_validate(
            {**stream.final_response.model_dump(), "end_turn": len(requests) > 1}
        )
        stream.release_terminal.set()
        yield stream

    monkeypatch.setattr(harness, "_response_sse_stream", stream_response)
    agent = ToolAgent(AgentConfig("continuation", use_history=use_history), [])
    agent._llm = harness
    result = await agent.generate("go", RequestParams(use_history=use_history, max_iterations=2))

    assert result.last_text() == "step-2"
    assert result.stop_reason == LlmStopReason.END_TURN
    assert len(requests) == 2
    # Follow-up is the original user message plus the actual assistant item,
    # not a fabricated user "continue" prompt or tool result.
    assert [item.get("role") for item in requests[1]] == ["user", "assistant"]
    assert requests[1][-1]["phase"] == "commentary"
    assert requests[1][-1]["content"][0]["text"] == "step-1"
    if use_history:
        history = list(agent.message_history)
        assert [m.role for m in history] == ["user", "assistant", "assistant"]
        first = history[1]
        assert first.stop_reason == LlmStopReason.CONTINUE
        restored = PromptMessageExtended.model_validate_json(first.model_dump_json())
        assert restored.stop_reason == LlmStopReason.CONTINUE
        assert restored.channels
        diagnostics = restored.channels[RESPONSES_DIAGNOSTICS_CHANNEL][0]
        assert diagnostics.type == "text"
        assert json.loads(diagnostics.text)["end_turn"] is False
    else:
        assert not agent.message_history


@pytest.mark.asyncio
@pytest.mark.parametrize("harness_type", [_ResponsesSseHarness, _CodexResponsesSseHarness])
@pytest.mark.parametrize(
    ("extension", "expected"),
    [
        ({}, LlmStopReason.END_TURN),
        ({"end_turn": True}, LlmStopReason.END_TURN),
        ({"end_turn": False}, LlmStopReason.CONTINUE),
        ({"end_turn": 0}, LlmStopReason.END_TURN),
        ({"end_turn": "false"}, LlmStopReason.END_TURN),
        ({"end_turn": None}, LlmStopReason.END_TURN),
    ],
)
async def test_sse_end_turn_requires_explicit_boolean_false(
    harness_type: type[_ResponsesSseHarness] | type[_CodexResponsesSseHarness],
    extension: dict[str, object],
    expected: LlmStopReason,
) -> None:
    from fast_agent.mcp.prompt import Prompt

    harness = harness_type()
    harness.sse_stream.final_response = Response.model_validate(
        {**harness.sse_stream.final_response.model_dump(), **extension}
    )
    harness.sse_stream.release_terminal.set()

    result = await harness.generate([Prompt.user("go")])

    assert result.stop_reason == expected
    assert result.last_text() == "hello world"
    assert harness.sse_calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["refusal", "max_output_tokens", "content_filter", "unknown"])
async def test_end_turn_false_cannot_override_safety_or_incomplete_response(failure: str) -> None:
    from fast_agent.mcp.prompt import Prompt

    harness = _ResponsesSseHarness()
    payload: dict[str, Any] = {**harness.sse_stream.final_response.model_dump(), "end_turn": False}
    if failure == "refusal":
        harness.sse_stream.completed_message.content = [
            ResponseOutputRefusal(type="refusal", refusal="Cannot help.")
        ]
    else:
        payload.update(status="incomplete", incomplete_details={"reason": failure})
    # The SDK's incomplete reason Literal rejects unknown values in validation;
    # forward-compatible provider objects can nevertheless carry them.
    harness.sse_stream.final_response = Response.model_construct(**payload)
    harness.sse_stream.release_terminal.set()
    if failure != "refusal":
        harness.sse_stream.final_response.incomplete_details = cast(
            "Any", SimpleNamespace(reason=failure)
        )

    result = await harness.generate([Prompt.user("go")])

    assert result.stop_reason != LlmStopReason.CONTINUE
    if failure in ("refusal", "content_filter"):
        assert result.stop_reason == LlmStopReason.SAFETY
    elif failure == "max_output_tokens":
        assert result.stop_reason == LlmStopReason.MAX_TOKENS


@pytest.mark.asyncio
async def test_completed_event_can_request_continuation_without_redundant_status() -> None:
    from fast_agent.mcp.prompt import Prompt

    harness = _ResponsesSseHarness()
    payload = harness.sse_stream.final_response.model_dump()
    payload.pop("status")
    payload["end_turn"] = False
    harness.sse_stream.final_response = Response.model_validate(payload)
    harness.sse_stream.release_terminal.set()

    result = await harness.generate([Prompt.user("go")])

    assert result.stop_reason == LlmStopReason.CONTINUE


@pytest.mark.asyncio
@pytest.mark.parametrize("end_turn", [True, False])
async def test_explicit_end_turn_does_not_discard_actual_tool_calls(end_turn: bool) -> None:
    from fast_agent.mcp.prompt import Prompt

    harness = _ResponsesSseHarness()
    harness.sse_stream.final_response = Response.model_validate(
        {
            **harness.sse_stream.final_response.model_dump(),
            "end_turn": end_turn,
            "output": [
                {
                    "type": "function_call",
                    "id": "fc_read",
                    "call_id": "call_read",
                    "name": "read",
                    "arguments": "{}",
                    "status": "completed",
                }
            ],
        }
    )
    harness.sse_stream.release_terminal.set()

    result = await harness.generate([Prompt.user("go")])

    assert result.stop_reason == LlmStopReason.TOOL_USE
    assert result.tool_calls
    assert result.tool_calls["call_read"].params.name == "read"


@pytest.mark.asyncio
async def test_empty_content_with_explicit_continuation_is_not_retried_as_empty_response() -> None:
    from fast_agent.mcp.prompt import Prompt

    harness = _ResponsesSseHarness()
    harness.sse_stream.completed_message.content = []
    harness.sse_stream.final_response = Response.model_validate(
        {**harness.sse_stream.final_response.model_dump(), "end_turn": False}
    )
    harness.sse_stream.release_terminal.set()

    result = await harness.generate([Prompt.user("go")])

    assert result.stop_reason == LlmStopReason.CONTINUE
    assert not result.content
    assert harness.sse_calls == 1
