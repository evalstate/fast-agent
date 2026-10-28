"""Copilot Responses over SSE tolerates lifecycle events with differing response ids."""

from __future__ import annotations

import json
import socket
from typing import TYPE_CHECKING, Any, NoReturn

import httpx2
import pytest
from openai import AsyncOpenAI
from openai.lib.streaming.responses import AsyncResponseStream

from fast_agent.config import CopilotSettings, Settings
from fast_agent.constants import FAST_AGENT_AUTH_FILE
from fast_agent.context import Context
from fast_agent.llm.provider.copilot.broker import CopilotBroker
from fast_agent.llm.provider.copilot.responses import CopilotResponsesLLM
from fast_agent.llm.provider.openai.responses import ResponsesLLM
from fast_agent.types import RequestParams

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.asyncio

_MODEL = "gpt-6-astra"


def _response(response_id: str, status: str, output: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "id": response_id,
        "object": "response",
        "created_at": 1,
        "model": _MODEL,
        "status": status,
        "output": output,
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
    }


_MESSAGE = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "status": "completed",
    "content": [{"type": "output_text", "text": "hello", "annotations": []}],
}

# Copilot does not keep response.id stable across lifecycle events.
_EVENTS: list[dict[str, Any]] = [
    {"type": "response.created", "response": _response("resp_created", "in_progress", [])},
    {"type": "response.in_progress", "response": _response("resp_created", "in_progress", [])},
    {
        "type": "response.output_item.added",
        "output_index": 0,
        "item": {**_MESSAGE, "status": "in_progress", "content": []},
    },
    {
        "type": "response.content_part.added",
        "output_index": 0,
        "item_id": "msg_1",
        "content_index": 0,
        "part": {"type": "output_text", "text": "", "annotations": []},
    },
    {
        "type": "response.output_text.delta",
        "output_index": 0,
        "item_id": "msg_1",
        "content_index": 0,
        "delta": "hello",
        "logprobs": [],
    },
    {"type": "response.output_item.done", "output_index": 0, "item": _MESSAGE},
    {
        "type": "response.completed",
        "response": _response("resp_completed", "completed", [_MESSAGE]),
    },
]


def _sse_body() -> bytes:
    return b"".join(
        f"event: {event['type']}\ndata: {json.dumps({**event, 'sequence_number': index})}\n\n".encode()
        for index, event in enumerate(_EVENTS)
    )


@pytest.fixture(autouse=True)
def sse_wire(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbid_connect(*args: object, **kwargs: object) -> NoReturn:
        raise AssertionError("Live network access is forbidden in Copilot wire tests")

    async def send(
        _transport: httpx2.AsyncHTTPTransport, _request: httpx2.Request
    ) -> httpx2.Response:
        return httpx2.Response(
            200, headers={"content-type": "text/event-stream"}, content=_sse_body()
        )

    monkeypatch.setattr(socket.socket, "connect", forbid_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", forbid_connect)
    monkeypatch.setattr(httpx2.AsyncHTTPTransport, "handle_async_request", send)


@pytest.fixture
def copilot_context(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Context:
    monkeypatch.setenv("COPILOT_GITHUB_TOKEN", "dummy-copilot-sse-token")
    monkeypatch.setenv(FAST_AGENT_AUTH_FILE, str(tmp_path / "auth.json"))
    settings = CopilotSettings()
    broker = CopilotBroker(settings)
    monkeypatch.setattr(
        "fast_agent.llm.provider.copilot.broker.get_copilot_broker", lambda _context: broker
    )
    return Context(config=Settings(copilot=settings))


async def test_copilot_sse_completes_when_lifecycle_response_id_changes(
    copilot_context: Context,
) -> None:
    llm = CopilotResponsesLLM(context=copilot_context, model=_MODEL, transport="sse")
    try:
        response, _summary, _input = await llm._responses_completion_sse(
            input_items=[{"type": "message", "role": "user", "content": "hi"}],
            request_params=RequestParams(model=_MODEL),
            tools=None,
            model_name=_MODEL,
        )
    finally:
        await llm.close()

    assert response.id == "resp_completed"
    assert response.status == "completed"
    assert response.output[0].content[0].text == "hello"


async def test_default_responses_sse_keeps_sdk_accumulator() -> None:
    llm = ResponsesLLM(model="gpt-5.4", api_key="test")
    async with AsyncOpenAI(api_key="test", base_url="https://responses.invalid/v1") as client:
        async with llm._response_sse_stream(
            client=client, arguments={"model": "gpt-5.4", "input": "hi"}
        ) as stream:
            assert isinstance(stream, AsyncResponseStream)
