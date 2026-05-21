from dataclasses import dataclass
from typing import Any, cast

import pytest
from mcp.types import TextContent

from fast_agent.constants import OPENAI_REASONING_ENCRYPTED, REASONING
from fast_agent.context import Context
from fast_agent.llm.provider.openai.llm_openai import OpenAILLM
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended


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
    reasoning: str | None = None
    reasoning_details: list[dict[str, object]] | None = None
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
                                function=StubFunction(arguments="\"x\": 1}"),
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
async def test_openai_responses_chat_reasoning_adapter_streams_and_records_details() -> None:
    context = Context()
    llm = OpenAILLM(context=context, model="gpt-5.5")
    stream_chunks: list[tuple[str, bool]] = []
    llm.add_stream_listener(lambda chunk: stream_chunks.append((chunk.text, chunk.is_reasoning)))
    llm._last_chat_reasoning_details = []

    encrypted_detail: dict[str, object] = {
        "type": "reasoning.encrypted",
        "data": "enc",
        "format": "openai-responses-v1",
        "index": 0,
    }
    chunks = [
        StubChunk(
            [
                StubChoice(
                    StubDelta(
                        reasoning="summary",
                        reasoning_details=[
                            {
                                "type": "reasoning.summary",
                                "summary": "summary",
                                "format": "openai-responses-v1",
                                "index": 0,
                            }
                        ],
                    )
                )
            ]
        ),
        StubChunk([StubChoice(StubDelta(reasoning_details=[encrypted_detail]))]),
        StubChunk([StubChoice(StubDelta(content="done"), finish_reason="stop")]),
    ]

    _completion, reasoning = await llm._process_stream_manual(_stream_chunks(chunks), "gpt-5.5")

    assert reasoning == ["summary"]
    assert ("summary", True) in stream_chunks
    assert llm._last_chat_reasoning_details == [encrypted_detail]


def test_openai_responses_chat_reasoning_adapter_replays_details() -> None:
    context = Context()
    llm = OpenAILLM(context=context, model="gpt-5.5")
    encrypted_detail: dict[str, object] = {
        "type": "reasoning.encrypted",
        "data": "enc",
        "format": "openai-responses-v1",
        "index": 0,
    }
    msg = PromptMessageExtended(
        role="assistant",
        content=[TextContent(type="text", text="done")],
        channels={
            REASONING: [TextContent(type="text", text="summary")],
            OPENAI_REASONING_ENCRYPTED: [
                TextContent(
                    type="text",
                    text=(
                        '{"type":"reasoning.encrypted","data":"enc",'
                        '"format":"openai-responses-v1","index":0}'
                    ),
                )
            ],
        },
    )

    converted = llm._convert_extended_messages_to_provider([msg])

    assert len(converted) == 1
    outgoing = cast("dict[str, Any]", converted[0])
    assert outgoing["reasoning"] == "summary"
    assert outgoing["reasoning_details"] == [
        {
            "type": "reasoning.summary",
            "summary": "summary",
            "format": "openai-responses-v1",
            "index": 0,
        },
        encrypted_detail,
    ]
