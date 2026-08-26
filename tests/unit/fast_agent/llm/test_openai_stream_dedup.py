from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from fast_agent.context import Context
from fast_agent.llm.provider.openai.llm_openai import OpenAILLM
from fast_agent.llm.provider.openai.llm_zai import ZaiLLM


def test_extract_incremental_delta_with_cumulative_content() -> None:
    delta, cumulative = OpenAILLM._extract_incremental_delta("Hello, world", "")
    assert delta == "Hello, world"
    assert cumulative == "Hello, world"

    delta, cumulative = OpenAILLM._extract_incremental_delta("Hello, world!", cumulative)
    assert delta == "!"
    assert cumulative == "Hello, world!"


def test_extract_incremental_delta_with_non_cumulative_content() -> None:
    delta, cumulative = OpenAILLM._extract_incremental_delta("Part 1", "")
    assert delta == "Part 1"
    assert cumulative == "Part 1"

    delta, cumulative = OpenAILLM._extract_incremental_delta("Part 2", cumulative)
    assert delta == "Part 2"
    assert cumulative == "Part 1Part 2"


def test_reasoning_content_wins_over_blank_reasoning_field() -> None:
    assert OpenAILLM._extract_reasoning_text(" ", "actual reasoning") == "actual reasoning"
    assert OpenAILLM._extract_reasoning_text(None, " ") == " "


@dataclass
class StubFunction:
    name: str | None = None
    arguments: str | None = None


@dataclass
class StubToolCallDelta:
    index: int | None
    id: str | None = None
    type: str | None = "function"
    function: StubFunction | None = None


@dataclass
class StubDelta:
    content: str | None = None
    reasoning_content: str | None = None
    reasoning: str | None = None
    tool_calls: list[StubToolCallDelta] | None = None
    role: str | None = None
    function_call: object | None = None
    refusal: object | None = None


@dataclass
class StubChoice:
    delta: StubDelta
    finish_reason: str | None = None
    index: int = 0
    logprobs: object | None = None


@dataclass
class StubChunk:
    choices: list[StubChoice]
    usage: object | None = None


class ToolStreamRecorder:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []

    def __call__(self, event_type: str, payload: dict[str, object] | None) -> None:
        if event_type in {"start", "delta", "stop"}:
            self.events.append((event_type, payload or {}))


async def _stream_chunks(chunks: list[StubChunk]):
    for chunk in chunks:
        yield chunk


async def _raw_stream_chunks(chunks: list[object]):
    for chunk in chunks:
        yield chunk


@pytest.mark.asyncio
async def test_stream_error_chunk_raises_provider_error() -> None:
    llm = OpenAILLM(context=Context(), model="gpt-test")

    with pytest.raises(RuntimeError, match="Provider stream error: .*empty array"):
        await llm._process_stream(
            _raw_stream_chunks(
                [
                    SimpleNamespace(
                        choices=None,
                        error_message="`tools` must not be an empty array.",
                    )
                ]
            ),
            "gpt-test",
        )


@pytest.mark.asyncio
async def test_tool_streaming_survives_cumulative_content() -> None:
    context = Context()
    llm = OpenAILLM(context=context, model="moonshotai/kimi-k2-instruct-0905")
    recorder = ToolStreamRecorder()
    llm.add_tool_stream_listener(recorder)

    chunks = [
        StubChunk([StubChoice(StubDelta(content="Hello"))]),
        StubChunk(
            [
                StubChoice(
                    StubDelta(
                        content="Hello",
                        tool_calls=[
                            StubToolCallDelta(
                                index=0,
                                id="tool-1",
                                function=StubFunction(name="do_work", arguments="{"),
                            )
                        ],
                    )
                )
            ]
        ),
        StubChunk(
            [
                StubChoice(
                    StubDelta(
                        content=None,
                        tool_calls=[
                            StubToolCallDelta(
                                index=0,
                                function=StubFunction(arguments='"x": 1}'),
                            )
                        ],
                    )
                )
            ]
        ),
        StubChunk([StubChoice(StubDelta(content=None), finish_reason="tool_calls")]),
    ]

    await llm._process_stream_manual(
        _stream_chunks(chunks),
        "moonshotai/kimi-k2-instruct-0905",
    )

    event_types = [event for event, _ in recorder.events]
    assert "start" in event_types
    assert "delta" in event_types
    assert "stop" in event_types
    assert event_types.index("start") < event_types.index("stop")


@pytest.mark.asyncio
async def test_manual_stream_accumulates_interleaved_reasoning_content_and_tools() -> None:
    llm = OpenAILLM(context=Context(), model="glm-5.2")
    chunks = [
        StubChunk(
            [
                StubChoice(
                    StubDelta(
                        content="answer-",
                        reasoning_content="think-1",
                        tool_calls=[
                            StubToolCallDelta(
                                index=0,
                                id="tool-1",
                                function=StubFunction(
                                    name="do_work",
                                    arguments='{"x":',
                                ),
                            )
                        ],
                    )
                )
            ]
        ),
        StubChunk(
            [
                StubChoice(
                    StubDelta(
                        content="done",
                        reasoning_content="think-2",
                        tool_calls=[
                            StubToolCallDelta(
                                index=0,
                                function=StubFunction(arguments="1}"),
                            )
                        ],
                    )
                )
            ]
        ),
        StubChunk([StubChoice(StubDelta(), finish_reason="tool_calls")]),
    ]

    completion, reasoning = await llm._process_stream_manual(
        _stream_chunks(chunks),
        "glm-5.2",
    )

    assert completion.choices[0].message.content == "answer-done"
    assert reasoning == ["think-1", "think-2"]
    assert completion.choices[0].message.tool_calls[0].id == "tool-1"
    assert completion.choices[0].message.tool_calls[0].function.name == "do_work"
    assert completion.choices[0].message.tool_calls[0].function.arguments == '{"x":1}'


@pytest.mark.asyncio
async def test_manual_stream_normalizes_reasoning_for_generic_history() -> None:
    llm = OpenAILLM(context=Context(), model="glm-5.2")
    chunks = [
        StubChunk([StubChoice(StubDelta(reasoning_content="first."))]),
        StubChunk([StubChoice(StubDelta(reasoning_content="Second sentence"))]),
        StubChunk([StubChoice(StubDelta(content="done"), finish_reason="stop")]),
    ]

    _, reasoning = await llm._process_stream_manual(
        _stream_chunks(chunks),
        "glm-5.2",
    )

    assert reasoning == ["first.", " Second sentence"]


@pytest.mark.asyncio
async def test_zai_manual_stream_returns_unmodified_reasoning_for_history() -> None:
    llm = ZaiLLM(context=Context(), model="glm-5.3")
    chunks = [
        StubChunk([StubChoice(StubDelta(reasoning_content="first"))]),
        StubChunk([StubChoice(StubDelta(reasoning_content=" "))]),
        StubChunk([StubChoice(StubDelta(reasoning_content="Second sentence"))]),
        StubChunk([StubChoice(StubDelta(content="done"), finish_reason="stop")]),
    ]

    _, reasoning = await llm._process_stream_manual(
        _stream_chunks(chunks),
        "glm-5.3",
    )

    assert reasoning == ["first", " ", "Second sentence"]
