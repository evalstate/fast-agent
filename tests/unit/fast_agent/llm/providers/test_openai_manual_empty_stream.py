from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest
from openai.types.chat import ChatCompletionMessage

from fast_agent.context import Context
from fast_agent.llm.provider.openai.llm_openai import (
    EmptyStreamError,
    OpenAILLM,
    _OpenAICompletionRequest,
)
from fast_agent.types import RequestParams

if TYPE_CHECKING:
    from openai import AsyncOpenAI

MODEL = "glm-5.2"


@dataclass
class _Delta:
    content: str | None = None
    reasoning_content: str | None = None
    reasoning: str | None = None
    tool_calls: list[object] | None = None
    role: str | None = None
    function_call: object | None = None


@dataclass
class _Choice:
    delta: _Delta
    finish_reason: str | None = None


@dataclass
class _Chunk:
    choices: list[_Choice]
    usage: object | None = None


class _Stream:
    def __init__(self, chunks: list[_Chunk]) -> None:
        self._chunks = iter(chunks)

    def __aiter__(self) -> _Stream:
        return self

    async def __anext__(self) -> _Chunk:
        try:
            return next(self._chunks)
        except StopIteration:
            raise StopAsyncIteration from None


def _completion(
    *,
    content: str | None = None,
    reasoning_content: str | None = None,
    finish_reason: str = "stop",
) -> SimpleNamespace:
    payload: dict[str, Any] = {
        "content": content,
        "role": "assistant",
    }
    if reasoning_content is not None:
        payload["reasoning_content"] = reasoning_content
    message = ChatCompletionMessage.model_validate(payload)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason=finish_reason)],
        usage=None,
    )


class _Completions:
    def __init__(self, stream: _Stream, fallback: SimpleNamespace) -> None:
        self._responses = iter((stream, fallback))
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> object:
        self.calls.append(kwargs)
        return next(self._responses)


class _Client:
    def __init__(self, completions: _Completions) -> None:
        self.chat = SimpleNamespace(completions=completions)


def _request() -> _OpenAICompletionRequest:
    return _OpenAICompletionRequest(
        params=RequestParams(model=MODEL),
        model_name=MODEL,
        messages=[],
        arguments={
            "model": MODEL,
            "messages": [],
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "chunks",
    [
        [],
        [_Chunk([_Choice(_Delta(role="assistant"), finish_reason="stop")])],
        [_Chunk([], usage=SimpleNamespace(prompt_tokens=1, completion_tokens=0))],
    ],
)
async def test_semantically_empty_manual_stream_retries_non_streaming(
    chunks: list[_Chunk],
) -> None:
    llm = OpenAILLM(context=Context(), model=MODEL)
    completions = _Completions(_Stream(chunks), _completion(content="recovered"))

    response = await llm._create_openai_streaming_response(
        cast("AsyncOpenAI", _Client(completions)),
        _request(),
        None,
    )

    assert response.response.choices[0].message.content == "recovered"
    assert response.streamed_reasoning == []
    assert len(completions.calls) == 2
    assert completions.calls[0]["stream"] is True
    assert completions.calls[1]["stream"] is False
    assert "stream_options" not in completions.calls[1]


@pytest.mark.asyncio
async def test_semantically_empty_non_streaming_fallback_raises() -> None:
    llm = OpenAILLM(context=Context(), model=MODEL)
    chunks = [_Chunk([_Choice(_Delta(role="assistant"), finish_reason="stop")])]
    completions = _Completions(_Stream(chunks), _completion(content=""))

    with pytest.raises(
        EmptyStreamError,
        match="non-streaming fallback response contained no usable completion",
    ):
        await llm._create_openai_streaming_response(
            cast("AsyncOpenAI", _Client(completions)),
            _request(),
            None,
        )


@pytest.mark.asyncio
async def test_reasoning_only_non_streaming_fallback_is_preserved() -> None:
    llm = OpenAILLM(context=Context(), model=MODEL)
    chunks = [_Chunk([_Choice(_Delta(role="assistant"), finish_reason="stop")])]
    completions = _Completions(
        _Stream(chunks),
        _completion(content="", reasoning_content="fallback reasoning"),
    )

    response = await llm._create_openai_streaming_response(
        cast("AsyncOpenAI", _Client(completions)),
        _request(),
        None,
    )

    assert response.streamed_reasoning == ["fallback reasoning"]
