from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pytest
import pytest_asyncio
from aiohttp import WSMsgType, web
from mcp.types import TextContent

from fast_agent.llm.provider.openai.codex_responses import CodexResponsesLLM
from fast_agent.llm.provider.openai.responses import RESPONSES_DIAGNOSTICS_CHANNEL
from fast_agent.llm.provider_types import Provider
from fast_agent.llm.request_params import RequestParams
from fast_agent.types import LlmStopReason

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


class _FakeResponsesClient:
    async def __aenter__(self) -> _FakeResponsesClient:
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        del exc_type, exc, tb


class _LocalWsCodexResponsesLLM(CodexResponsesLLM):
    def __init__(self, base_url: str) -> None:
        super().__init__(provider=Provider.CODEX_RESPONSES, model="gpt-5.3-codex", transport="websocket")
        self._local_base_url = base_url

    def _base_url(self) -> str | None:
        return self._local_base_url

    def _build_websocket_headers(self) -> dict[str, str]:
        return {}

    def _responses_client(self) -> Any:
        return _FakeResponsesClient()

    async def _normalize_input_files(
        self,
        client: Any,
        input_items: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        del client
        return input_items


@dataclass
class _LocalWebSocketState:
    handshake_count: int = 0
    received_request_types: list[str] = field(default_factory=list)
    received_payloads: list[dict[str, Any]] = field(default_factory=list)
    scripted_responses: list[dict[str, Any]] = field(default_factory=list)


@pytest_asyncio.fixture
async def local_responses_ws_server(
    unused_tcp_port: int,
) -> AsyncGenerator[tuple[str, _LocalWebSocketState], None]:
    state = _LocalWebSocketState()

    async def _handler(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(autoping=True)
        await ws.prepare(request)
        state.handshake_count += 1

        async for message in ws:
            if message.type != WSMsgType.TEXT:
                continue
            payload = json.loads(str(message.data))
            state.received_payloads.append(payload)
            request_type = payload.get("type")
            if isinstance(request_type, str):
                state.received_request_types.append(request_type)

            response_index = len(state.received_request_types) - 1
            if response_index < len(state.scripted_responses):
                response_payload = dict(state.scripted_responses[response_index])
                response_payload.setdefault("id", f"resp_{len(state.received_request_types)}")
            else:
                response_text = f"turn-{len(state.received_request_types)}"
                response_payload = {
                    "id": f"resp_{len(state.received_request_types)}",
                    "status": "completed",
                    "output_text": response_text,
                    "output": [
                        {
                            "type": "message",
                            "content": [{"type": "output_text", "text": response_text}],
                        }
                    ],
                    "usage": None,
                }

            output_text = response_payload.get("output_text")
            if isinstance(output_text, str) and output_text:
                await ws.send_json({"type": "response.output_text.delta", "delta": output_text})

            await ws.send_json(
                {
                    "type": "response.completed",
                    "response": response_payload,
                }
            )

        return ws

    app = web.Application()
    app.router.add_get("/v1/responses", _handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", unused_tcp_port)
    await site.start()

    base_url = f"http://127.0.0.1:{unused_tcp_port}/v1"
    try:
        yield base_url, state
    finally:
        await runner.cleanup()


def _input_message(text: str) -> dict[str, Any]:
    return {
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": text}],
    }


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_local_websocket_reuses_connection_and_continues_with_previous_response_id(
    local_responses_ws_server: tuple[str, _LocalWebSocketState],
) -> None:
    base_url, state = local_responses_ws_server
    llm = _LocalWsCodexResponsesLLM(base_url)
    params = RequestParams(model="gpt-5.3-codex")

    try:
        first_response, _, _ = await llm._responses_completion_ws(
            input_items=[_input_message("first")],
            request_params=params,
            tools=None,
            model_name="gpt-5.3-codex",
        )
        second_response, _, _ = await llm._responses_completion_ws(
            input_items=[_input_message("first"), _input_message("second")],
            request_params=params,
            tools=None,
            model_name="gpt-5.3-codex",
        )
    finally:
        await llm._ws_connections.close()

    assert getattr(first_response, "output_text", None) == "turn-1"
    assert getattr(second_response, "output_text", None) == "turn-2"
    assert state.handshake_count == 1
    assert state.received_request_types[:2] == ["response.create", "response.create"]
    assert state.received_payloads[1]["previous_response_id"] == "resp_1"
    assert state.received_payloads[1]["input"] == [_input_message("second")]


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_local_websocket_continuation_dedupes_duplicate_tool_calls(
    local_responses_ws_server: tuple[str, _LocalWebSocketState],
) -> None:
    base_url, state = local_responses_ws_server
    state.scripted_responses = [
        {
            "status": "completed",
            "output_text": "",
            "output": [
                {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_1",
                    "name": "lookup",
                    "arguments": "{}",
                }
            ],
            "usage": None,
        },
        {
            "status": "completed",
            "output_text": "",
            "output": [
                {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_1",
                    "name": "lookup",
                    "arguments": "{}",
                },
                {
                    "type": "function_call",
                    "id": "fc_2",
                    "call_id": "call_2",
                    "name": "lookup",
                    "arguments": "{}",
                },
            ],
            "usage": None,
        },
    ]

    llm = _LocalWsCodexResponsesLLM(base_url)
    params = RequestParams(model="gpt-5.3-codex")

    try:
        first = await llm._responses_completion(
            input_items=[_input_message("first")],
            request_params=params,
            tools=None,
        )
        second = await llm._responses_completion(
            input_items=[_input_message("first"), _input_message("second")],
            request_params=params,
            tools=None,
        )
    finally:
        await llm._ws_connections.close()

    assert first.stop_reason is LlmStopReason.TOOL_USE
    assert first.tool_calls is not None
    assert list(first.tool_calls.keys()) == ["call_1"]

    assert second.stop_reason is LlmStopReason.TOOL_USE
    assert second.tool_calls is not None
    assert list(second.tool_calls.keys()) == ["call_2"]

    diagnostics_channel = (second.channels or {}).get(RESPONSES_DIAGNOSTICS_CHANNEL)
    assert diagnostics_channel is not None
    diagnostics_block = diagnostics_channel[0]
    assert isinstance(diagnostics_block, TextContent)
    diagnostics = json.loads(diagnostics_block.text)
    assert diagnostics["kind"] == "duplicate_tool_calls_filtered"
    assert diagnostics["transport"] == "websocket"
    assert diagnostics["websocket_request_type"] == "response.create"
    assert diagnostics["raw_function_call_count"] == 2
    assert diagnostics["new_function_call_count"] == 1
    assert diagnostics["duplicate_count"] == 1

    assert state.handshake_count == 1
    assert state.received_request_types[:2] == ["response.create", "response.create"]
    assert state.received_payloads[1]["previous_response_id"] == "resp_1"
    assert state.received_payloads[1]["input"] == [_input_message("second")]
