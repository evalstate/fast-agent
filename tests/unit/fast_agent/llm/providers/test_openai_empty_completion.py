"""OpenAI-compatible chat completions: an end_turn with no content and no tool call
(often a stream cut off mid-reasoning) is asked once more, as OpenAI Responses does.

Regression: on an HF-routed DeepSeek V4.1 Flash run three trials ended this way; the
agent loop took the empty reply as the end of its work.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest
from mcp_types import TextContent
from openai.types.chat import ChatCompletion
from openai.types.responses import Response, ResponseUsage

from fast_agent.constants import (
    FAST_AGENT_PROVIDER_ERROR,
    FAST_AGENT_RETRY,
    OPENAI_COMPACTION_ITEMS,
)
from fast_agent.context import Context
from fast_agent.core.logging.logger import Logger
from fast_agent.llm.provider.error_utils import is_empty_completion
from fast_agent.llm.provider.openai.llm_openai import (
    OpenAILLM,
    _OpenAICompletionResponse,
)
from fast_agent.llm.provider.openai.responses import ResponsesLLM, _ResponsesCompletionResult
from fast_agent.llm.request_params import RequestParams
from fast_agent.mcp.prompt import Prompt
from fast_agent.session.trace_export_atif import AtifRunSource, build_atif_trajectory
from fast_agent.types.llm_stop_reason import LlmStopReason

if TYPE_CHECKING:
    from fast_agent.core.logging.events import Event
    from fast_agent.types import PromptMessageExtended


def _empty(reasoning: str = "Let me try the near-identity pai") -> PromptMessageExtended:
    response = Prompt.assistant("", stop_reason=LlmStopReason.END_TURN)
    response.channels = {"reasoning": [TextContent(type="text", text=reasoning)]}
    return response


def _retries(response: PromptMessageExtended) -> list[dict[str, Any]]:
    block = (response.channels or {})[FAST_AGENT_RETRY][0]
    assert isinstance(block, TextContent)
    return json.loads(block.text)["retries"]


class ScriptedOpenAI(OpenAILLM):
    """Replies from a script; records every request it was sent."""

    def __init__(self, replies: list[PromptMessageExtended]) -> None:
        super().__init__(context=Context(), model="deepseek-ai/DeepSeek-V4.1-Flash")
        self.replies = replies
        self.requests: list[Any] = []

    async def _run_openai_completion_request(self, request: Any) -> Any:
        self.requests.append(request)
        return None

    async def _finalize_openai_completion(self, request: Any, completion: Any) -> Any:
        request.messages.append({"role": "assistant", "content": ""})  # as the real one does
        return self.replies[len(self.requests) - 1]


@pytest.mark.asyncio
async def test_an_empty_end_turn_is_asked_once_more_with_a_fresh_request() -> None:
    llm = ScriptedOpenAI([_empty(), Prompt.assistant("Done.")])

    response = await llm._openai_completion([{"role": "user", "content": "go"}])

    assert response.all_text() == "Done."
    (retry,) = _retries(response)
    assert retry["reason"] == "empty_completion"
    assert retry["error_type"] == "EmptyCompletionError"
    first, second = llm.requests
    assert first.messages is not second.messages
    assert [m["role"] for m in second.messages] == ["user", "assistant"]  # one reply, not two


@pytest.mark.asyncio
async def test_empty_twice_is_an_error_not_a_finished_turn() -> None:
    llm = ScriptedOpenAI([_empty(), _empty()])

    response = await llm._openai_completion([{"role": "user", "content": "go"}])

    assert len(llm.requests) == 2
    assert response.stop_reason == LlmStopReason.ERROR
    assert "no assistant content or tool calls" in response.all_text()
    assert [r["reason"] for r in _retries(response)] == ["empty_completion"]
    block = (response.channels or {})[FAST_AGENT_PROVIDER_ERROR][0]
    assert isinstance(block, TextContent)
    error = json.loads(block.text)
    assert error.pop("empty_completion_attempts")[0]["reasoning_chars"] == len(
        "Let me try the near-identity pai"
    )
    assert error == {
        "schema": "fast-agent.provider-error/v1",
        "error_type": "EmptyCompletionError",
        "status": None,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reply",
    [
        Prompt.assistant("Done.", stop_reason=LlmStopReason.END_TURN),
        Prompt.assistant("", stop_reason=LlmStopReason.MAX_TOKENS),
        Prompt.assistant("", stop_reason=LlmStopReason.CANCELLED),
    ],
)
async def test_replies_and_other_stops_are_returned_as_they_are(
    reply: PromptMessageExtended,
) -> None:
    llm = ScriptedOpenAI([reply])

    assert await llm._openai_completion([{"role": "user", "content": "go"}]) is reply
    assert len(llm.requests) == 1
    assert FAST_AGENT_RETRY not in (reply.channels or {})


def test_a_compaction_checkpoint_is_not_an_empty_completion() -> None:
    checkpoint = Prompt.assistant("", stop_reason=LlmStopReason.END_TURN)
    checkpoint.channels = {OPENAI_COMPACTION_ITEMS: [TextContent(type="text", text="{}")]}

    assert not is_empty_completion(checkpoint)
    assert is_empty_completion(_empty())


class RecordingLogger(Logger):
    def __init__(self) -> None:
        super().__init__("empty-completion-test")
        self.records: list[Event] = []

    def _emit_event(self, event: Event) -> None:
        if event.type in ("warning", "error"):
            self.records.append(event)


class RawScriptedOpenAI(OpenAILLM):
    """Script the provider boundary, keeping real normalization and retry telemetry."""

    def __init__(self, replies: list[_OpenAICompletionResponse]) -> None:
        super().__init__(context=Context(), model="test-model")
        self.replies = iter(replies)
        self.recording_logger = RecordingLogger()
        self.logger = self.recording_logger

    async def _run_openai_completion_request(self, request: Any) -> _OpenAICompletionResponse:
        return next(self.replies)


def _raw_chat_reply(
    *,
    reasoning: str = "",
    content: str | None = None,
    completion_tokens: int | None = None,
    reasoning_tokens: int | None = None,
    streamed: bool = True,
    finish_reason: str | None = "stop",
) -> _OpenAICompletionResponse:
    response = ChatCompletion.model_validate(
        {
            "id": "test",
            "created": 0,
            "model": "test-model",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": finish_reason or "stop",
                    "message": {"role": "assistant", "content": content},
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": completion_tokens,
                "total_tokens": 10 + completion_tokens,
                "completion_tokens_details": {"reasoning_tokens": reasoning_tokens},
            }
            if completion_tokens is not None
            else None,
        }
    )
    # Manual streaming can preserve an absent/unknown provider finish reason.
    if finish_reason is None:
        response.choices[0] = response.choices[0].model_copy(update={"finish_reason": None})
    return _OpenAICompletionResponse(response, [reasoning] if reasoning else [], streamed)


def _provider_error(response: PromptMessageExtended) -> dict[str, Any]:
    block = (response.channels or {})[FAST_AGENT_PROVIDER_ERROR][0]
    assert isinstance(block, TextContent)
    return json.loads(block.text)


@pytest.mark.asyncio
async def test_raw_empty_attempt_metadata_survives_a_successful_retry() -> None:
    reasoning = "private reasoning that must not enter diagnostics"
    llm = RawScriptedOpenAI(
        [
            _raw_chat_reply(reasoning=reasoning, completion_tokens=33500, reasoning_tokens=30200),
            _raw_chat_reply(content="Recovered."),
        ]
    )
    response = await llm._openai_completion([{"role": "user", "content": "go"}])

    assert response.all_text() == "Recovered."
    (retry,) = _retries(response)
    diagnostics = retry["empty_completion"]
    assert diagnostics["finish_reason"] == "stop"
    assert diagnostics["completion_tokens"] == 33500
    assert diagnostics["reasoning_tokens"] == 30200
    assert diagnostics["reasoning_chars"] == len(reasoning)
    assert diagnostics["streamed"] is True
    assert llm.recording_logger.records[0].data["data"]["empty_completion"] == diagnostics
    assert reasoning not in response.model_dump_json()
    assert FAST_AGENT_PROVIDER_ERROR not in (response.channels or {})


@pytest.mark.asyncio
async def test_raw_repeated_empty_metadata_survives_error_logs_and_atif() -> None:
    reasoning = "discarded private reasoning"
    llm = RawScriptedOpenAI(
        [
            _raw_chat_reply(reasoning=reasoning, completion_tokens=100, reasoning_tokens=15),
            _raw_chat_reply(reasoning=reasoning * 2, streamed=False, finish_reason=None),
        ]
    )
    response = await llm._openai_completion([{"role": "user", "content": "go"}])

    error = _provider_error(response)
    first, last = error["empty_completion_attempts"]
    assert first == _retries(response)[0]["empty_completion"]
    assert first["completion_tokens"] - first["reasoning_tokens"] == 85
    assert last["streamed"] is False
    assert last["finish_reason"] is None
    assert last["completion_tokens"] is None
    assert last["reasoning_tokens"] is None
    assert last["reasoning_chars"] == 2 * len(reasoning)
    assert llm.recording_logger.records[-1].data["data"]["empty_completion_attempts"] == [
        first,
        last,
    ]
    assert reasoning not in response.model_dump_json()

    trajectory = build_atif_trajectory(
        AtifRunSource(
            session_id="test",
            agent_name="test",
            model_name="test-model",
            provider="openai",
            history=[response],
            message_timestamps=(None,),
        )
    )
    extra = trajectory.steps[0].extra
    assert extra is not None
    assert extra["provider_error"] == error
    assert extra["retry"]["retries"][0]["empty_completion"] == first


class ScriptedResponses(ResponsesLLM):
    def __init__(self, replies: list[tuple[Response, list[str]]]) -> None:
        super().__init__(context=Context(), model="test-model", transport="sse")
        self.replies = iter(replies)
        self.recording_logger = RecordingLogger()
        self.logger = self.recording_logger

    async def _run_responses_transport(
        self,
        *,
        input_items: list[dict[str, Any]],
        request_params: Any,
        tools: Any,
        context: Any,
    ) -> _ResponsesCompletionResult:
        response, reasoning = next(self.replies)
        return _ResponsesCompletionResult(response, reasoning, input_items)


def _raw_responses_reply(
    *,
    content: str | None = None,
    usage: ResponseUsage | None = None,
) -> Response:
    return Response.model_validate(
        {
            "id": "test",
            "created_at": 0,
            "model": "test-model",
            "object": "response",
            "status": "completed",
            "usage": usage,
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "output": [
                {
                    "type": "message",
                    "id": "msg-test",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": content, "annotations": []}],
                }
            ]
            if content is not None
            else [],
        }
    )


@pytest.mark.asyncio
async def test_responses_empty_retry_records_usage_and_discarded_reasoning() -> None:
    usage = ResponseUsage(
        input_tokens=10,
        output_tokens=100,
        total_tokens=110,
        input_tokens_details={"cached_tokens": 0, "cache_write_tokens": 0},
        output_tokens_details={"reasoning_tokens": 85},
    )
    reasoning = " private streamed reasoning "
    llm = ScriptedResponses(
        [
            (_raw_responses_reply(usage=usage), [reasoning]),
            (_raw_responses_reply(content="Recovered."), []),
        ]
    )
    result = await llm._responses_completion([], RequestParams(model="test-model"))

    assert result.all_text() == "Recovered."
    diagnostics = _retries(result)[0]["empty_completion"]
    assert diagnostics["finish_reason"] == "completed"
    assert diagnostics["completion_tokens"] == 100
    assert diagnostics["reasoning_tokens"] == 85
    assert diagnostics["reasoning_chars"] == len(reasoning)
    assert diagnostics["streamed"] is True
    assert llm.recording_logger.records[0].data["data"]["empty_completion"] == diagnostics
    assert reasoning not in result.model_dump_json()


@pytest.mark.asyncio
async def test_responses_repeated_empty_preserves_null_usage_and_both_attempts() -> None:
    llm = ScriptedResponses(
        [
            (_raw_responses_reply(), ["discarded"]),
            (
                _raw_responses_reply(
                    usage=ResponseUsage(
                        input_tokens=10,
                        output_tokens=200,
                        total_tokens=210,
                        input_tokens_details={"cached_tokens": 0, "cache_write_tokens": 0},
                        output_tokens_details={"reasoning_tokens": 120},
                    )
                ),
                [],
            ),
        ]
    )
    result = await llm._responses_completion([], RequestParams(model="test-model"))

    assert result.stop_reason == LlmStopReason.ERROR
    error = _provider_error(result)
    first, last = error["empty_completion_attempts"]
    assert first == _retries(result)[0]["empty_completion"]
    assert first["reasoning_chars"] == len("discarded")
    assert last["reasoning_chars"] == 0
    assert first["completion_tokens"] is None
    assert last["completion_tokens"] == 200
    assert first["reasoning_tokens"] is None
    assert last["reasoning_tokens"] == 120
    assert last["streamed"] is True
    assert llm.recording_logger.records[-1].data["data"]["empty_completion_attempts"] == [
        first,
        last,
    ]
    assert "discarded" not in result.model_dump_json()
