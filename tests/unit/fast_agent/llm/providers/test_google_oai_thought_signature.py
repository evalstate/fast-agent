"""Gemini 3 thought signatures must survive the googleoai tool loop."""

import json
from typing import Any, cast
from unittest.mock import patch

import httpx2
import pytest
from mcp_types import CallToolRequest, CallToolRequestParams, TextContent
from openai.types.chat import ChatCompletionChunk, ChatCompletionMessage

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.tool_agent import ToolAgent
from fast_agent.config import Settings
from fast_agent.constants import OPENAI_TOOL_CALL_EXTRA_CONTENT
from fast_agent.context import Context
from fast_agent.llm.model_factory import ModelFactory
from fast_agent.llm.provider.openai.llm_openai import OpenAILLM
from fast_agent.llm.provider.openai.multipart_converter_openai import OpenAIConverter
from fast_agent.llm.provider.openai.tool_call_extra import tool_call_extra_by_id
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended
from fast_agent.mcp.prompt_serialization import from_json, to_json
from fast_agent.types import RequestParams

SIGNATURE = "CiQBcsjafE-opaque-thought-signature"


def _sse(chunks: list[dict[str, Any]]) -> bytes:
    body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
    return f"{body}data: [DONE]\n\n".encode()


def _chunk(
    delta: dict[str, Any],
    finish: str | None = None,
    usage: dict[str, int] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "gemini-3.5-flash",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    if usage:
        payload["usage"] = usage
    return payload


def _completion_chunk(
    delta: dict[str, Any],
    *,
    finish_reason: str | None = None,
) -> ChatCompletionChunk:
    return ChatCompletionChunk.model_validate(_chunk(delta, finish_reason))


async def _stream_chunks(chunks: list[ChatCompletionChunk]):
    for chunk in chunks:
        yield chunk


def _weather_request(city: str = "Paris") -> CallToolRequest:
    return CallToolRequest(
        method="tools/call",
        params=CallToolRequestParams(name="get_weather", arguments={"city": city}),
    )


def _replayed_tool_calls(message: object) -> list[dict[str, Any]]:
    payload = cast("dict[str, Any]", message)
    tool_calls = payload["tool_calls"]
    assert isinstance(tool_calls, list)
    return [cast("dict[str, Any]", tool_call) for tool_call in tool_calls]


def get_weather(city: str) -> str:
    """Return the weather for a city."""
    return f"sunny in {city}"


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model",
    ["googleoai.gemini-3.5-flash", "openai.gpt-4.1"],
)
async def test_tool_loop_replays_thought_signature(
    monkeypatch: pytest.MonkeyPatch, model: str
) -> None:
    monkeypatch.setenv("GOOGLEOAI_API_KEY", "offline-test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "offline-test-key")
    tool_reply = _sse(
        [
            _chunk(
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                            "extra_content": {"google": {"thought_signature": SIGNATURE}},
                        }
                    ],
                }
            ),
            _chunk(
                {},
                finish="tool_calls",
                usage={"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
            ),
        ]
    )
    final_reply = _sse(
        [
            _chunk({"role": "assistant", "content": "It is sunny in Paris."}),
            _chunk(
                {},
                finish="stop",
                usage={"prompt_tokens": 9, "completion_tokens": 6, "total_tokens": 15},
            ),
        ]
    )
    bodies: list[dict[str, Any]] = []

    async def respond(request: httpx2.Request) -> httpx2.Response:
        bodies.append(json.loads(await request.aread()))
        content = tool_reply if len(bodies) == 1 else final_reply
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=content)

    def offline_http_client(*args: Any, **kwargs: Any) -> httpx2.AsyncClient:
        del args, kwargs
        return httpx2.AsyncClient(transport=httpx2.MockTransport(respond))

    with patch(
        "fast_agent.llm.provider.openai.llm_openai.DefaultAioHttpClient",
        offline_http_client,
    ):
        context = Context(config=Settings())
        agent = ToolAgent(AgentConfig("repro"), [get_weather], context=context)
        agent._llm = ModelFactory.create_factory(model)(agent, context=context)
        await agent.send("What is the weather in Paris?", RequestParams(max_tokens=256))

    assert len(bodies) == 2
    replayed = [message for message in bodies[1]["messages"] if message.get("role") == "assistant"]
    replayed_signature = (
        replayed[0]["tool_calls"][0]
        .get("extra_content", {})
        .get("google", {})
        .get("thought_signature")
    )
    assert replayed_signature == SIGNATURE

    restored = from_json(to_json(agent.message_history))
    assistant = next(message for message in restored if message.tool_calls)
    replayed_call = _replayed_tool_calls(OpenAIConverter.convert_to_openai(assistant)[0])[0]
    assert replayed_call["extra_content"]["google"]["thought_signature"] == SIGNATURE


@pytest.mark.unit
@pytest.mark.asyncio
async def test_manual_stream_keeps_signature_from_a_later_chunk() -> None:
    llm = OpenAILLM(context=Context(), model="gemini-3.5-flash")
    chunks = [
        _completion_chunk(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city":'},
                    }
                ],
            }
        ),
        _completion_chunk(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "function": {"arguments": ' "Paris"}'},
                        "extra_content": {"google": {"thought_signature": SIGNATURE}},
                    }
                ]
            }
        ),
        _completion_chunk({}, finish_reason="tool_calls"),
    ]

    completion, _reasoning = await llm._process_stream_manual(
        _stream_chunks(chunks),
        "gemini-3.5-flash",
    )

    extras = tool_call_extra_by_id(completion.choices[0].message.tool_calls)
    assert extras["call_1"]["google"]["thought_signature"] == SIGNATURE


@pytest.mark.unit
@pytest.mark.asyncio
async def test_chat_completion_stream_state_keeps_signature_from_a_later_chunk() -> None:
    llm = OpenAILLM(context=Context(), model="gpt-4.1")
    chunks = [
        _completion_chunk(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city":'},
                    }
                ],
            }
        ),
        _completion_chunk(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "function": {"arguments": ' "Paris"}'},
                        "extra_content": {"google": {"thought_signature": SIGNATURE}},
                    }
                ]
            }
        ),
        _completion_chunk({}, finish_reason="tool_calls"),
    ]

    completion, _reasoning = await llm._process_stream(_stream_chunks(chunks), "gpt-4.1")

    extras = tool_call_extra_by_id(completion.choices[0].message.tool_calls)
    assert extras["call_1"]["google"]["thought_signature"] == SIGNATURE


@pytest.mark.unit
def test_nonstreaming_tool_call_extra_is_addressable_by_id() -> None:
    message = ChatCompletionMessage.model_validate(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                    "extra_content": {"google": {"thought_signature": SIGNATURE}},
                }
            ],
        }
    )

    assert (
        tool_call_extra_by_id(message.tool_calls)["call_1"]["google"]["thought_signature"]
        == SIGNATURE
    )


@pytest.mark.unit
def test_converter_replays_each_tool_call_extra_and_skips_malformed_channel() -> None:
    message = PromptMessageExtended(
        role="assistant",
        content=[],
        tool_calls={
            "call_1": _weather_request(),
            "call_2": _weather_request("Lyon"),
        },
        channels={
            OPENAI_TOOL_CALL_EXTRA_CONTENT: [
                TextContent(type="text", text="not-json"),
                TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "call_1": {"google": {"thought_signature": "sig-1"}},
                            "call_2": {"google": {"thought_signature": "sig-2"}},
                        }
                    ),
                ),
            ]
        },
    )

    converted = _replayed_tool_calls(OpenAIConverter.convert_to_openai(message)[0])
    signatures = {
        tool_call["id"]: tool_call["extra_content"]["google"]["thought_signature"]
        for tool_call in converted
    }
    assert signatures == {"call_1": "sig-1", "call_2": "sig-2"}

    plain = PromptMessageExtended(
        role="assistant",
        content=[],
        tool_calls={"call_1": _weather_request()},
    )
    plain_call = _replayed_tool_calls(OpenAIConverter.convert_to_openai(plain)[0])[0]
    assert "extra_content" not in plain_call
