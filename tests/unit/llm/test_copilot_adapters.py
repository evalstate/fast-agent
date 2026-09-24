"""Adapter contracts exercised through real parent request loops and SDK serialization."""

from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Literal
from unittest.mock import AsyncMock

import httpx2
import pytest
from anthropic import AsyncAnthropic
from mcp import Tool
from openai import AsyncOpenAI

from fast_agent.config import AnthropicSettings, CopilotSettings, OpenAISettings, Settings
from fast_agent.constants import ANTHROPIC_CITATIONS_CHANNEL, ANTHROPIC_SERVER_TOOLS_CHANNEL
from fast_agent.context import Context
from fast_agent.core.exceptions import ModelConfigError
from fast_agent.llm.model_factory import ModelFactory
from fast_agent.llm.model_overlays import load_model_overlay_registry
from fast_agent.llm.provider.copilot.broker import CopilotEndpoint
from fast_agent.llm.provider.copilot.messages import CopilotMessagesLLM
from fast_agent.llm.provider.copilot.models import get_copilot_model
from fast_agent.llm.provider.copilot.policy import request_headers
from fast_agent.llm.provider.copilot.responses import CopilotResponsesLLM
from fast_agent.llm.provider.openai.responses_websocket import (
    ResponsesWebSocketError,
    send_response_request,
)
from fast_agent.llm.provider_types import Provider
from fast_agent.llm.reasoning_effort import ReasoningEffortSetting
from fast_agent.mcp.prompt import Prompt
from fast_agent.types import RequestParams

if TYPE_CHECKING:
    from pathlib import Path


class FakeBroker:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []

    async def resolve(
        self, model_id: str, *, owner_id: str, transport: Literal["sse", "websocket"]
    ) -> CopilotEndpoint:
        self.calls.append((model_id, owner_id, transport))
        generation = len(self.calls)
        await asyncio.sleep(0)
        wire_api = get_copilot_model(model_id).wire_api
        return CopilotEndpoint(
            model_id=model_id,
            wire_api=wire_api,
            transport=transport,
            base_url=f"https://copilot-{generation}.example"
            + ("/v1" if wire_api == "messages" and generation % 2 == 0 else ""),
            headers={"authorization": f"Bearer broker-{generation}"},
        )


@pytest.fixture
def broker(monkeypatch: pytest.MonkeyPatch) -> FakeBroker:
    fake = FakeBroker()

    def shared(context):
        return fake

    monkeypatch.setattr("fast_agent.llm.provider.copilot.broker.get_copilot_broker", shared)
    for key in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "OPENAI_API_KEY", "OPENAI_ADMIN_KEY"):
        monkeypatch.setenv(key, "DO-NOT-SEND")
    monkeypatch.setenv("OPENAI_ORG_ID", "DO-NOT-SEND")
    monkeypatch.setenv("OPENAI_PROJECT_ID", "DO-NOT-SEND")
    return fake


@pytest.fixture(autouse=True)
def image_uploads(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    # Adapter tests exercise SDK serialization, never the attachment HTTP service.
    async def normalize(payload: dict[str, Any], endpoint: CopilotEndpoint) -> dict[str, Any]:
        return deepcopy(payload)

    mock = AsyncMock(side_effect=normalize)
    monkeypatch.setattr(
        "fast_agent.llm.provider.copilot.images.CopilotImageUploads.normalize", mock
    )
    return mock


@pytest.fixture
def context() -> Context:
    return Context(
        config=Settings(
            anthropic=AnthropicSettings(
                api_key="DO-NOT-SEND", base_url="https://wrong.example", reasoning=False
            ),
            openai=OpenAISettings(api_key="DO-NOT-SEND", base_url="https://wrong.example"),
        )
    )


def anthropic_events() -> bytes:
    events = [
        {
            "type": "message_start",
            "message": {
                "id": "m",
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": "claude-sonnet-5",
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 3, "output_tokens": 0},
            },
        },
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "OK"}},
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 1},
        },
        {"type": "message_stop"},
    ]
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()


def responses_events() -> bytes:
    response = {
        "id": "r",
        "object": "response",
        "created_at": 0,
        "model": "gpt-6-astra",
        "status": "completed",
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "output": [
            {
                "id": "msg",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "OK", "annotations": []}],
            }
        ],
        "usage": {
            "input_tokens": 3,
            "output_tokens": 1,
            "total_tokens": 4,
            "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    }
    events = [
        {"type": "response.created", "response": response, "sequence_number": 0},
        {"type": "response.completed", "response": response, "sequence_number": 1},
    ]
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "wire,model",
    [
        ("messages", "claude-sonnet-5"),
        ("messages", "claude-opus-4-8"),
        ("responses", "gpt-6-astra"),
    ],
)
async def test_parent_sse_loop_fresh_binding_and_payload(
    wire: str,
    model: str,
    broker: FakeBroker,
    context: Context,
    monkeypatch: pytest.MonkeyPatch,
    image_uploads: AsyncMock,
) -> None:
    requests: list[httpx2.Request] = []
    bodies: list[dict[str, Any]] = []

    async def respond(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        bodies.append(json.loads(await request.aread()))
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=anthropic_events() if wire == "messages" else responses_events(),
        )

    # Keep SDK validation and serialization live, replacing only HTTP I/O.
    def client_factory(**kwargs: Any) -> AsyncAnthropic | AsyncOpenAI:
        kwargs["http_client"] = httpx2.AsyncClient(transport=httpx2.MockTransport(respond))
        return AsyncAnthropic(**kwargs) if wire == "messages" else AsyncOpenAI(**kwargs)

    module = f"fast_agent.llm.provider.copilot.{wire}"
    monkeypatch.setattr(
        f"{module}.{'AsyncAnthropic' if wire == 'messages' else 'AsyncOpenAI'}", client_factory
    )
    llm = (
        CopilotMessagesLLM(context=context, model=model)
        if wire == "messages"
        else CopilotResponsesLLM(context=context, model=model, transport="sse")
    )
    tool = Tool(name="local_tool", input_schema={"type": "object", "properties": {}})
    uploaded_url = "https://attachments.example/uploaded.png"
    # Heterogeneous provider wire payloads at the SDK boundary.
    message: dict[str, Any] = {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "call",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "AA==",
                        },
                    }
                ],
            }
        ],
    }
    inputs = [
        {
            "type": "function_call_output",
            "call_id": "call",
            "output": [{"type": "input_image", "image_url": "data:image/png;base64,AA=="}],
        }
    ]
    original_message = deepcopy(message)
    original_inputs = deepcopy(inputs)

    async def normalize(payload: dict[str, Any], endpoint: CopilotEndpoint) -> dict[str, Any]:
        normalized = deepcopy(payload)
        if wire == "messages":
            image = normalized["messages"][-1]["content"][0]["content"][0]
            assert image["source"] == original_message["content"][0]["content"][0]["source"]
            image["source"] = {"type": "url", "url": uploaded_url}
        else:
            image = normalized["input"][0]["output"][0]
            assert image["image_url"] == "data:image/png;base64,AA=="
            image["image_url"] = uploaded_url
        return normalized

    image_uploads.side_effect = normalize
    for _ in range(2):
        params = RequestParams(model=llm.default_request_params.model, max_tokens=37)
        if isinstance(llm, CopilotMessagesLLM):
            await llm._anthropic_completion(message, params, tools=[tool])
        else:
            await llm._responses_completion(inputs, params, tools=[tool])
        assert message == original_message
        assert inputs == original_inputs
    assert image_uploads.await_count == 2
    for generation, call in enumerate(image_uploads.await_args_list, 1):
        payload, endpoint = call.args
        assert set(payload) == {"messages" if wire == "messages" else "input"}
        assert endpoint.model_id == model
        assert endpoint.wire_api == wire
        assert endpoint.headers == {"authorization": f"Bearer broker-{generation}"}
        assert endpoint.base_url == f"https://copilot-{generation}.example" + (
            "/v1" if wire == "messages" and generation % 2 == 0 else ""
        )
    assert llm.provider is Provider.COPILOT
    assert len(llm.usage_accumulator.turns) == 2
    assert all(turn.provider is Provider.COPILOT for turn in llm.usage_accumulator.turns)
    assert len(broker.calls) == len(requests) == 2
    assert broker.calls[0][1] == broker.calls[1][1]
    for index, (request, body) in enumerate(zip(requests, bodies, strict=True), 1):
        assert request.url.host == f"copilot-{index}.example"
        assert request.url.path == ("/v1/messages" if wire == "messages" else "/responses")
        assert request.headers.get_list("authorization") == [f"Bearer broker-{index}"]
        assert "x-api-key" not in request.headers
        assert "DO-NOT-SEND" not in str(request.headers)
        assert request.headers["x-initiator"] == "agent"
        assert request.headers["copilot-vision-request"] == "true"
        assert body["model"] == model
        assert body["tools"][0]["name"] == "local_tool"
        assert body["max_tokens" if wire == "messages" else "max_output_tokens"] == 37
        if wire == "messages":
            assert body["messages"][-1]["content"][0]["content"][0]["source"] == {
                "type": "url",
                "url": uploaded_url,
            }
            assert "eager_input_streaming" not in body["tools"][0]
        else:
            assert body["input"][0]["output"][0]["image_url"] == uploaded_url
            assert body["store"] is False
            assert "previous_response_id" not in body
    if isinstance(llm, CopilotResponsesLLM):
        await llm.close()


@pytest.mark.asyncio
async def test_owner_and_task_binding_isolation(broker: FakeBroker, context: Context) -> None:
    first = CopilotResponsesLLM(context=context, model="gpt-6-astra", name="same-name")
    second = CopilotResponsesLLM(context=context, model="gpt-6-astra", name="same-name")

    async def bind(llm: CopilotResponsesLLM) -> str:
        await llm._prepare_responses_client("gpt-6-astra", "sse")
        await asyncio.sleep(0)
        return llm._provider_base_url()

    urls = await asyncio.gather(bind(first), bind(second), bind(first))
    assert len(set(urls)) == 3
    assert broker.calls[0][1] != broker.calls[1][1]
    assert broker.calls[0][1] == broker.calls[2][1]


@pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh", "max"])
def test_opus_55_effort_controls(
    context: Context, effort: Literal["low", "medium", "high", "xhigh", "max"]
) -> None:
    llm = CopilotMessagesLLM(
        context=context,
        model="claude-opus-5.5",
    )
    llm.set_reasoning_effort(ReasoningEffortSetting(kind="effort", value=effort))
    args, enabled = llm._resolve_thinking_arguments("claude-opus-5.5", 128000, None)
    assert enabled
    assert "thinking" not in args
    assert args["output_config"] == {"effort": effort}


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["claude-fable-5.1", "claude-opus-5.5"])
async def test_always_on_policy_and_thinking_preservation(
    broker: FakeBroker, context: Context, model: str
) -> None:
    llm = CopilotMessagesLLM(context=context, model=model)
    await llm._prepare_anthropic_client(model)
    if model == "claude-opus-5.5":
        thinking_args, enabled = llm._resolve_thinking_arguments(model, 128000, None)
        assert enabled
        assert "thinking" not in thinking_args
        assert thinking_args["output_config"] == {"effort": "medium"}
    base = {
        "model": model,
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "private", "signature": "opaque"},
                    {
                        "type": "tool_use",
                        "id": "local-call",
                        "name": "local",
                        "input": {"file_id": "local-file", "type": "file"},
                    },
                ],
            }
        ],
        "tools": [{"name": "local", "input_schema": {"type": "object"}}],
        "thinking": {"type": "adaptive"},
    }
    arguments = llm.prepare_provider_arguments(base, RequestParams(max_tokens=123))
    assert arguments["thinking"] == base["thinking"]
    assert arguments["messages"] == base["messages"]
    assert arguments["max_tokens"] == 123
    assert "eager_input_streaming" not in arguments["tools"][0]
    for metadata in (
        {"tool_choice": {"type": "tool", "name": "local"}},
        {"extra_body": {"tool_choice": {"type": "any"}}},
    ):
        with pytest.raises(ValueError, match="tool"):
            llm.prepare_provider_arguments(base, RequestParams(metadata=metadata))
    with pytest.raises(ValueError, match="tool"):
        await llm._anthropic_completion(
            {"role": "user", "content": "structured"},
            RequestParams(max_tokens=123),
            structured_schema={"type": "object"},
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model",
    ["gpt-6-astra", "gpt-6-sol", "gpt-6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"],
)
@pytest.mark.parametrize("factory", [False, True])
async def test_gpt_defaults_to_websocket_without_sse_fallback(
    model: str, factory: bool, context: Context, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fast_agent.agents.agent_types import AgentConfig
    from fast_agent.agents.tool_agent import ToolAgent

    if factory:
        agent = ToolAgent(AgentConfig("default-transport"), context=context)
        llm = ModelFactory.create_factory(f"copilot.{model}")(agent, context=context)
        assert isinstance(llm, CopilotResponsesLLM)
    else:
        llm = CopilotResponsesLLM(context=context, model=model)
    websocket = AsyncMock(side_effect=ResponsesWebSocketError("test connection failure"))
    sse = AsyncMock(side_effect=AssertionError("Default WebSocket must not fall back to SSE"))
    monkeypatch.setattr(llm, "_responses_completion_ws", websocket)
    monkeypatch.setattr(llm, "_responses_completion_sse", sse)
    try:
        with pytest.raises(ResponsesWebSocketError, match="test connection failure"):
            await llm._responses_completion([{"role": "user", "content": "hello"}], RequestParams())
        websocket.assert_awaited_once()
        sse.assert_not_awaited()
    finally:
        await llm.close()


@pytest.mark.asyncio
async def test_websocket_context_and_stateless_replay(broker: FakeBroker, context: Context) -> None:
    llm = CopilotResponsesLLM(context=context, model="gpt-6-astra", transport="websocket")
    inputs = [
        {"type": "reasoning", "id": "reason", "encrypted_content": "opaque", "summary": []},
        {"type": "message", "role": "user", "content": "hello"},
    ]
    ws = await llm._responses_ws_context(
        input_items=inputs,
        request_params=RequestParams(max_tokens=51),
        tools=None,
        model_name="gpt-6-astra",
    )
    assert broker.calls[-1][2] == "websocket"
    assert ws.ws_url.startswith("wss://copilot-1.example")
    assert ws.ws_headers["x-initiator"] == "user"
    planner = llm._new_ws_request_planner()
    first = planner.plan(ws.arguments)
    planner.commit(ws.arguments, first, {"id": "previous"})
    followup = {
        **ws.arguments,
        "input": [*inputs, {"type": "function_call_output", "call_id": "c", "output": "done"}],
    }
    planned = planner.plan(followup)
    assert planned.arguments["input"] == followup["input"]
    assert "previous_response_id" not in planned.arguments
    assert planned.arguments["store"] is False
    await llm.close()


@pytest.mark.parametrize(
    "cls,model", [(CopilotMessagesLLM, "claude-sonnet-5"), (CopilotResponsesLLM, "gpt-6-astra")]
)
def test_reject_external_credentials_and_web_fetch(cls, model: str, context: Context) -> None:
    for overrides in (
        {"api_key": "secret"},
        {"base_url": "https://wrong"},
        {"default_headers": {"AUTHORIZATION": "secret"}},
        {"default_headers": {"Copilot-Integration-Id": "other-integration"}},
        {"extra_headers": {"COPILOT-INTEGRATION-ID": "other-integration"}},
        {"web_fetch": True},
    ):
        with pytest.raises(ValueError):
            cls(context=context, model=model, **overrides)
    llm = cls(context=context, model=model)
    assert not llm.web_search_enabled
    llm.validate_provider_credentials()


def test_semantic_headers_ignore_old_tool_results() -> None:
    assert request_headers("hello") == {"x-initiator": "user"}
    assert request_headers(
        [{"type": "function_call_output", "output": "done"}, {"role": "user", "content": "next"}]
    ) == {"x-initiator": "user"}


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses"])
async def test_request_boundary_rejects_auth_hosted_tools_and_files(
    wire: str, broker: FakeBroker, context: Context
) -> None:
    if wire == "messages":
        llm = CopilotMessagesLLM(context=context, model="claude-sonnet-5")
        await llm._prepare_anthropic_client("claude-sonnet-5")
        base = {"model": "claude-sonnet-5", "messages": []}
        hosted = {"type": "web_search_20250305", "name": "web_search"}
    else:
        llm = CopilotResponsesLLM(context=context, model="gpt-6-astra")
        await llm._prepare_responses_client("gpt-6-astra", "sse")
        base = {"model": "gpt-6-astra", "input": "hello"}
        hosted = {"type": "file_search", "vector_store_ids": ["unsupported"]}
    for metadata in (
        {"extra_headers": {"aUtHoRiZaTiOn": "external"}},
        {"extra_body": {"api_key": "external"}},
        {"extra_headers": {"x-api-key": "external"}},
        {"extra_headers": {"Copilot-Integration-Id": "other-integration"}},
        {"extra_body": {"extra_headers": {"COPILOT-INTEGRATION-ID": "other-integration"}}},
        {"tools": [hosted]},
        {"tools": [{"type": "web_search_preview"}]},
        {"extra_body": {"tools": [{"type": "mcp", "server_url": "https://wrong"}]}},
        {"model": "other-model"},
    ):
        with pytest.raises(ValueError):
            llm.prepare_provider_arguments(base, RequestParams(metadata=metadata))
    if isinstance(llm, CopilotResponsesLLM):
        for content in (
            {"type": "input_file", "file_url": "https://file"},
            {"type": "input_image", "file_id": "hosted"},
        ):
            with pytest.raises(ValueError, match="file"):
                await llm._normalize_input_files(llm._responses_client(), [{"content": [content]}])
        with pytest.raises(ValueError, match="replay"):
            llm.prepare_provider_arguments(
                base, RequestParams(metadata={"previous_response_id": "r"})
            )
        arguments = llm.prepare_provider_arguments(
            base,
            RequestParams(
                metadata={"extra_body": {"store": True}, "reasoning": {"effort": "high"}}
            ),
        )
        assert arguments["store"] is False
        assert arguments["extra_body"]["store"] is False
        assert arguments["reasoning"] == {"effort": "high"}
        await llm.close()


@pytest.mark.asyncio
async def test_websocket_reconnect_refreshes_binding(
    broker: FakeBroker, context: Context, monkeypatch: pytest.MonkeyPatch
) -> None:
    from unittest.mock import AsyncMock

    from fast_agent.llm.provider.openai.responses import ResponsesLLM

    llm = CopilotResponsesLLM(context=context, model="gpt-6-astra", transport="websocket")
    ws = await llm._responses_ws_context(
        input_items=[{"type": "function_call_output", "call_id": "c", "output": "done"}],
        request_params=RequestParams(metadata={"extra_headers": {"x-test-hint": "keep"}}),
        tools=None,
        model_name="gpt-6-astra",
    )
    # Leave the adapter and context builder live; stop at the socket acquisition boundary.
    acquire = AsyncMock()
    monkeypatch.setattr(ResponsesLLM, "_acquire_responses_ws_attempt", acquire)
    await llm._acquire_responses_ws_attempt(attempt=0, context=ws)
    first_url = ws.ws_url
    first_headers = dict(ws.ws_headers)
    await llm._acquire_responses_ws_attempt(attempt=1, context=ws)
    assert len(broker.calls) == 2
    assert all(call[2] == "websocket" for call in broker.calls)
    assert broker.calls[0][1] == broker.calls[1][1]
    assert ws.ws_url != first_url
    assert ws.ws_headers["authorization"] != first_headers["authorization"]
    assert ws.ws_headers["x-initiator"] == "agent"
    assert ws.ws_headers["x-test-hint"] == "keep"
    assert acquire.await_count == 2
    await llm.close()


@pytest.mark.asyncio
async def test_websocket_sdk_handshake_never_inherits_openai_account(
    broker: FakeBroker, context: Context, monkeypatch: pytest.MonkeyPatch
) -> None:
    class HandshakeCaptured(Exception):
        pass

    captured: list[tuple[str, dict[str, str]]] = []

    async def connect(url: str, *, additional_headers: dict[str, str], **kwargs: Any) -> None:
        captured.append((url, additional_headers))
        raise HandshakeCaptured

    monkeypatch.setattr("openai.lib._websocket._WebSocketConnect", connect)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://wrong.example?secret=DO-NOT-SEND")
    llm = CopilotResponsesLLM(context=context, model="gpt-6-astra", transport="websocket")
    ws = await llm._responses_ws_context(
        input_items=[{"role": "user", "content": "hello"}],
        request_params=RequestParams(),
        tools=None,
        model_name="gpt-6-astra",
    )
    with pytest.raises(HandshakeCaptured):
        await llm._acquire_responses_ws_attempt(attempt=0, context=ws)
    assert len(captured) == 1
    url, headers = captured[0]
    assert url == "wss://copilot-1.example/responses"
    assert "DO-NOT-SEND" not in str(headers)
    assert headers["authorization"] == "Bearer broker-1"
    assert all(isinstance(value, str) for value in headers.values())
    assert headers["x-initiator"] == "user"
    await llm.close()


@pytest.mark.parametrize(
    "content,initiator",
    [
        ([{"type": "tool_result", "content": "done"}], "agent"),
        ([{"type": "tool_result", "content": "done"}, {"type": "text", "text": "next"}], "user"),
        ([{"type": "text", "text": "next"}, {"type": "tool_result", "content": "done"}], "user"),
        ([], "user"),
    ],
)
def test_messages_initiator_mixed_content(content: list[dict[str, str]], initiator: str) -> None:
    assert request_headers([{"role": "user", "content": content}]) == {"x-initiator": initiator}


@pytest.mark.parametrize("kind", ["function_call_output", "custom_tool_call_output"])
def test_responses_tool_output_initiator(kind: str) -> None:
    assert request_headers([{"type": kind, "output": "done"}]) == {"x-initiator": "agent"}


def test_headers_ignore_arbitrary_tool_inputs() -> None:
    assert request_headers(
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_use",
                        "input": {"type": "tool_result", "content": [{"type": "image"}]},
                    }
                ],
            }
        ]
    ) == {"x-initiator": "user"}
    assert request_headers(
        [{"role": "user", "content": [{"type": "tool_result", "content": [{"type": "image"}]}]}]
    ) == {"x-initiator": "agent", "copilot-vision-request": "true"}


def test_messages_document_uploads_unsupported(context: Context) -> None:
    llm = CopilotMessagesLLM(context=context, model="claude-sonnet-5")
    assert not llm.supports_files_api()
    assert not llm.supports_document_uploads()


@pytest.mark.parametrize("cap", [128, 1024, 1025])
def test_numeric_thinking_budget_preserves_or_rejects_cap(context: Context, cap: int) -> None:
    llm = CopilotMessagesLLM(context=context, model="claude-haiku-4.5")
    if cap <= 1024:
        with pytest.raises(ValueError, match=r"Increase max_tokens.*reasoning=off"):
            llm._resolve_thinking_arguments("claude-haiku-4.5", cap, None)
    else:
        arguments, enabled = llm._resolve_thinking_arguments("claude-haiku-4.5", cap, None)
        assert enabled
        assert arguments["max_tokens"] == cap


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "wire,web_search", [("messages", False), ("responses", False), ("responses", True)]
)
async def test_factory_tool_agent_round_trip(
    wire: str,
    web_search: bool,
    broker: FakeBroker,
    context: Context,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fast_agent.agents.agent_types import AgentConfig
    from fast_agent.agents.tool_agent import ToolAgent
    from fast_agent.llm.model_factory import ModelFactory

    calls: list[str] = []
    requests: list[httpx2.Request] = []
    bodies: list[dict[str, Any]] = []

    def echo_marker(marker: str) -> str:
        """Echo a harmless marker."""
        calls.append(marker)
        return marker

    async def respond(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        bodies.append(json.loads(await request.aread()))
        data = anthropic_events() if wire == "messages" else responses_events()
        if len(requests) == 1:
            events = [
                json.loads(line[6:])
                for line in data.decode().splitlines()
                if line.startswith("data: ")
            ]
            if wire == "messages":
                events[1]["content_block"] = {
                    "type": "tool_use",
                    "id": "echo-call",
                    "name": "echo_marker",
                    "input": {},
                }
                events[2]["delta"] = {
                    "type": "input_json_delta",
                    "partial_json": '{"marker":"offline-marker"}',
                }
                events[4]["delta"]["stop_reason"] = "tool_use"
            else:
                for event in events:
                    event["response"]["output"] = [
                        {
                            "type": "function_call",
                            "id": "fc-echo",
                            "call_id": "echo-call",
                            "name": "echo_marker",
                            "arguments": '{"marker":"offline-marker"}',
                            "status": "completed",
                        }
                    ]
                    if web_search:
                        event["response"]["output"].extend(
                            [
                                {
                                    "type": "web_search_call",
                                    "id": "search-call",
                                    "status": "completed",
                                    "action": {"type": "search", "query": "public source"},
                                },
                                {
                                    "type": "message",
                                    "id": "cited-message",
                                    "role": "assistant",
                                    "status": "completed",
                                    "content": [
                                        {
                                            "type": "output_text",
                                            "text": "Source",
                                            "annotations": [
                                                {
                                                    "type": "url_citation",
                                                    "url": "https://example.com/source",
                                                    "title": "Public source",
                                                    "start_index": 0,
                                                    "end_index": 6,
                                                }
                                            ],
                                        }
                                    ],
                                },
                            ]
                        )
            data = "".join(
                f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events
            ).encode()
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=data)

    def client_factory(**kwargs: Any) -> AsyncAnthropic | AsyncOpenAI:
        kwargs["http_client"] = httpx2.AsyncClient(transport=httpx2.MockTransport(respond))
        return AsyncAnthropic(**kwargs) if wire == "messages" else AsyncOpenAI(**kwargs)

    monkeypatch.setattr(
        f"fast_agent.llm.provider.copilot.{wire}."
        + ("AsyncAnthropic" if wire == "messages" else "AsyncOpenAI"),
        client_factory,
    )
    model = "claude-fable-5.1" if wire == "messages" else "gpt-6-astra"
    agent = ToolAgent(AgentConfig("offline-copilot"), [echo_marker], context=context)
    model_spec = f"copilot.{model}?transport=sse" + ("&web_search=true" if web_search else "")
    agent._llm = ModelFactory.create_factory(model_spec)(agent, context=context)
    try:
        assert await agent.send("Echo offline-marker", RequestParams(max_tokens=128)) == "OK"
        assert calls == ["offline-marker"]
        assert len(requests) == 2
        assert [r.headers["x-initiator"] for r in requests] == ["user", "agent"]
        assert len(broker.calls) == 2
        assert broker.calls[0][1] == broker.calls[1][1]
        if wire == "messages":
            result = bodies[1]["messages"][-1]["content"][0]
            assert result["type"] == "tool_result"
            assert result["tool_use_id"] == "echo-call"
        else:
            result = bodies[1]["input"][-1]
            assert result["type"] == "function_call_output"
            assert result["call_id"] == next(
                item["call_id"] for item in bodies[1]["input"] if item["type"] == "function_call"
            )
            assert bodies[1]["store"] is False
            assert "previous_response_id" not in bodies[1]
        assert "offline-marker" in json.dumps(result)
        if web_search:
            assert all(
                {tool["type"] for tool in body["tools"]} == {"function", "web_search"}
                for body in bodies
            )
            turn = next(message for message in agent.last_turn_messages if message.tool_calls)
            assert turn.channels
            assert turn.channels[ANTHROPIC_SERVER_TOOLS_CHANNEL]
            assert turn.channels[ANTHROPIC_CITATIONS_CHANNEL]
            assert "https://example.com/source" in json.dumps(bodies[1]["input"])
    finally:
        if isinstance(agent._llm, CopilotResponsesLLM):
            await agent._llm.close()


def test_factory_thinking_off_preserves_small_cap(context: Context) -> None:
    from fast_agent.agents.agent_types import AgentConfig
    from fast_agent.agents.tool_agent import ToolAgent
    from fast_agent.llm.model_factory import ModelFactory

    agent = ToolAgent(AgentConfig("thinking-off"), context=context)
    llm = ModelFactory.create_factory("copilot.claude-haiku-4.5?reasoning=off")(agent)
    assert isinstance(llm, CopilotMessagesLLM)
    arguments, enabled = llm._resolve_thinking_arguments("claude-haiku-4.5", 128, None)
    assert not enabled
    assert arguments["max_tokens"] == 128


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses"])
async def test_broker_headers_are_the_only_credentials_serialized(
    wire: str,
    broker: FakeBroker,
    context: Context,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[httpx2.Request] = []

    async def respond(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        data = anthropic_events() if wire == "messages" else responses_events()
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=data)

    def client_factory(**kwargs: Any) -> AsyncAnthropic | AsyncOpenAI:
        kwargs["http_client"] = httpx2.AsyncClient(transport=httpx2.MockTransport(respond))
        return AsyncAnthropic(**kwargs) if wire == "messages" else AsyncOpenAI(**kwargs)

    monkeypatch.setattr(
        f"fast_agent.llm.provider.copilot.{wire}."
        + ("AsyncAnthropic" if wire == "messages" else "AsyncOpenAI"),
        client_factory,
    )
    model = "claude-sonnet-5" if wire == "messages" else "gpt-6-astra"
    headers = {"authorization": "Bearer broker-token", "x-broker-header": "trusted"}
    endpoint = CopilotEndpoint(
        model_id=model,
        wire_api=get_copilot_model(model).wire_api,
        transport="sse",
        base_url="https://copilot.example",
        headers=headers,
    )
    if wire == "messages":
        llm = CopilotMessagesLLM(context=context, model=model)
        llm._copilot_endpoint.set(endpoint)
        async with llm._initialize_anthropic_client() as client:
            stream = await client.messages.create(
                model=model,
                max_tokens=32,
                messages=[{"role": "user", "content": "hello"}],
                stream=True,
            )
            async for _ in stream:
                pass
    else:
        responses = CopilotResponsesLLM(context=context, model=model)
        responses._copilot_endpoint.set(endpoint)
        assert responses._build_websocket_headers() == headers
        async with responses._responses_client() as client:
            arguments = responses._build_response_args(
                [{"role": "user", "content": "hello"}], RequestParams(), tools=None
            )
            async with responses._response_sse_stream(client=client, arguments=arguments) as stream:
                async for _ in stream:
                    pass
            assert all(isinstance(value, str) for value in arguments["extra_headers"].values())
        await responses.close()
    assert len(requests) == 1
    serialized = requests[0].headers
    assert serialized["x-broker-header"] == "trusted"
    assert serialized.get_list("authorization") == ["Bearer broker-token"]
    assert "x-api-key" not in serialized
    assert "DO-NOT-SEND" not in str(serialized)


@pytest.mark.asyncio
async def test_responses_extra_body_matches_sse_and_websocket_wire_payload(
    broker: FakeBroker, context: Context, monkeypatch: pytest.MonkeyPatch
) -> None:
    requests: list[httpx2.Request] = []
    bodies: list[dict[str, Any]] = []

    async def respond(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        bodies.append(json.loads(await request.aread()))
        return httpx2.Response(
            200, headers={"content-type": "text/event-stream"}, content=responses_events()
        )

    def client_factory(**kwargs: Any) -> AsyncOpenAI:
        kwargs["http_client"] = httpx2.AsyncClient(transport=httpx2.MockTransport(respond))
        return AsyncOpenAI(**kwargs)

    monkeypatch.setattr("fast_agent.llm.provider.copilot.responses.AsyncOpenAI", client_factory)
    llm = CopilotResponsesLLM(context=context, model="gpt-6-astra")
    effective_input = [
        {
            "role": "user",
            "content": [{"type": "input_image", "image_url": "data:image/png;base64,AA=="}],
        },
        {"type": "function_call_output", "call_id": "c", "output": "done"},
    ]
    extra_body = {"input": effective_input, "max_output_tokens": 73, "store": True}
    params = RequestParams(max_tokens=51, metadata={"extra_body": extra_body})
    inputs = [{"role": "user", "content": "replaced input"}]
    try:
        await llm._responses_completion_sse(
            input_items=inputs, request_params=params, tools=None, model_name="gpt-6-astra"
        )
        ws = await llm._responses_ws_context(
            input_items=inputs, request_params=params, tools=None, model_name="gpt-6-astra"
        )
        socket = AsyncMock()
        await send_response_request(socket, llm._new_ws_request_planner().plan(ws.arguments))
        frame = json.loads(socket.send_str.call_args.args[0])
        assert frame.pop("type") == "response.create"
        sse_body = bodies[0]
        assert sse_body.pop("stream") is True
        assert frame == sse_body
        assert frame["input"] == effective_input
        assert frame["max_output_tokens"] == 73
        assert frame["store"] is False
        assert "extra_body" not in frame
        for headers in (requests[0].headers, ws.ws_headers):
            assert headers["x-initiator"] == "agent"
            assert headers["copilot-vision-request"] == "true"
        assert extra_body["store"] is True
    finally:
        await llm.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model", ["gpt-6-astra", "gpt-6-sol", "gpt-6-luna", "gpt-5.6-luna", "gpt-5.6-sol"]
)
async def test_verified_web_search_toggle_and_websocket_payload(
    model: str, broker: FakeBroker, context: Context
) -> None:
    assert context.config and context.config.openai
    context.config.openai.web_search.enabled = True
    llm = CopilotResponsesLLM(context=context, model=model)
    try:
        assert llm.web_search_supported
        assert not llm.web_search_enabled  # Do not inherit an OpenAI account's settings.
        llm.set_web_search_enabled(True)
        ws = await llm._responses_ws_context(
            input_items=[{"role": "user", "content": "Search the web."}],
            request_params=RequestParams(),
            tools=None,
            model_name=model,
        )
        socket = AsyncMock()
        await send_response_request(socket, llm._new_ws_request_planner().plan(ws.arguments))
        frame = json.loads(socket.send_str.call_args.args[0])
        assert frame["tools"] == [{"type": "web_search"}]
        assert "web_search_call.action.sources" in frame["include"]
        assert frame["parallel_tool_calls"] is True
        llm.set_web_search_enabled(False)
        arguments = llm._build_response_args([], RequestParams(), tools=None)
        assert "tools" not in arguments
    finally:
        await llm.close()


@pytest.mark.parametrize(
    "model",
    [
        "claude-haiku-4.5",
        "claude-sonnet-5",
        "claude-opus-5",
        "claude-fable-5",
        "claude-fable-5.1",
        "gpt-5.6-terra",
    ],
)
@pytest.mark.asyncio
async def test_unverified_web_search_rejected(
    model: str, broker: FakeBroker, context: Context
) -> None:
    if model.startswith("claude"):
        cls = CopilotMessagesLLM
        llm = cls(context=context, model=model)
        await llm._prepare_anthropic_client(model)
        base = {"model": model, "messages": []}
    else:
        cls = CopilotResponsesLLM
        llm = cls(context=context, model=model)
        await llm._prepare_responses_client(model, "sse")
        base = {"model": model, "input": []}
    try:
        assert not llm.web_search_supported
        with pytest.raises(ValueError, match="web"):
            cls(context=context, model=model, web_search=True)
        with pytest.raises(ValueError, match="web"):
            llm.set_web_search_enabled(True)
        for tool_type in ("web_search", "web_search_preview", "web_search_20250305"):
            with pytest.raises(ValueError, match="hosted"):
                llm.prepare_provider_arguments(
                    base,
                    RequestParams(
                        metadata={
                            "extra_body": {"tools": [{"type": tool_type, "name": "web_search"}]}
                        }
                    ),
                )
    finally:
        if isinstance(llm, CopilotResponsesLLM):
            await llm.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["gpt-6-astra", "claude-sonnet-5"])
async def test_copilot_service_tiers_unavailable_at_request_boundary(
    model: str, broker: FakeBroker, context: Context
) -> None:
    if model.startswith("claude"):
        llm = CopilotMessagesLLM(context=context, model=model)
        await llm._prepare_anthropic_client(model)
        base = {"model": model, "messages": []}
    else:
        llm = CopilotResponsesLLM(context=context, model=model)
        await llm._prepare_responses_client(model, "sse")
        base = {"model": model, "input": []}
    try:
        assert not llm.service_tier_supported
        assert llm.available_service_tiers == ()
        for tier in ("fast", "flex"):
            with pytest.raises((ValueError, ModelConfigError)):
                llm.set_service_tier(tier)
            with pytest.raises((ValueError, ModelConfigError)):
                if isinstance(llm, CopilotResponsesLLM):
                    llm._build_response_args([], RequestParams(service_tier=tier), tools=None)
                else:
                    llm.prepare_provider_arguments(base, RequestParams(service_tier=tier))
        for tier in ("auto", "default", "priority", "flex", "standard_only", "invalid"):
            for metadata in ({"service_tier": tier}, {"extra_body": {"service_tier": tier}}):
                with pytest.raises(ValueError, match="service_tier"):
                    llm.prepare_provider_arguments(base, RequestParams(metadata=metadata))
    finally:
        if isinstance(llm, CopilotResponsesLLM):
            await llm.close()


@pytest.mark.asyncio
async def test_messages_cache_controls_and_streamed_usage_reach_turn_accounting(
    broker: FakeBroker, context: Context, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fast_agent.core.agent_app import AgentApp

    assert context.config is not None
    context.config.copilot.cache_mode = "off"
    bodies: list[dict[str, Any]] = []

    async def respond(request: httpx2.Request) -> httpx2.Response:
        bodies.append(json.loads(await request.aread()))
        events = [
            json.loads(line[6:])
            for line in anthropic_events().decode().splitlines()
            if line.startswith("data: ")
        ]
        # Copilot's successful cache probes reported matching cumulative prompt
        # partitions at both message_start and message_delta, not token deltas.
        usage = {
            "input_tokens": 20,
            "cache_creation_input_tokens": 6000 if len(bodies) == 1 else 0,
            "cache_read_input_tokens": 0 if len(bodies) == 1 else 6000,
        }
        events[0]["message"]["model"] = "claude-opus-5"
        events[0]["message"]["usage"].update(usage)
        events[-2]["usage"].update(usage)
        data = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=data)

    def client_factory(**kwargs: Any) -> AsyncAnthropic:
        kwargs["http_client"] = httpx2.AsyncClient(transport=httpx2.MockTransport(respond))
        return AsyncAnthropic(**kwargs)

    monkeypatch.setattr("fast_agent.llm.provider.copilot.messages.AsyncAnthropic", client_factory)
    llm = CopilotMessagesLLM(context=context, model="claude-opus-5")
    cache_control = {"type": "ephemeral", "ttl": "5m"}
    params = RequestParams(max_tokens=128, metadata={"cache_control": cache_control})
    for _ in range(2):
        await llm._anthropic_completion({"role": "user", "content": "Repeat prefix"}, params)

    assert all(body["cache_control"] == cache_control for body in bodies)
    first, second = llm.usage_accumulator.turns
    assert first.provider == second.provider == Provider.COPILOT
    assert first.prompt.total == second.prompt.total == 6020
    assert first.prompt.cache_write == second.prompt.cache_read == 6000
    assert AgentApp._cached_prompt_percentage([first]) == 0
    percentage = AgentApp._cached_prompt_percentage([second])
    assert percentage is not None and percentage > 99


@pytest.fixture
def messages_requests(monkeypatch: pytest.MonkeyPatch) -> list[httpx2.Request]:
    requests: list[httpx2.Request] = []

    async def respond(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(
            200, headers={"content-type": "text/event-stream"}, content=anthropic_events()
        )

    # Keep the real SDK's validation/serialization; only replace network I/O.
    def client_factory(**kwargs: Any) -> AsyncAnthropic:
        kwargs["http_client"] = httpx2.AsyncClient(transport=httpx2.MockTransport(respond))
        return AsyncAnthropic(**kwargs)

    monkeypatch.setattr("fast_agent.llm.provider.copilot.messages.AsyncAnthropic", client_factory)
    return requests


@pytest.fixture
def cache_context() -> Context:
    return Context(
        config=Settings(
            anthropic=AnthropicSettings(
                cache_mode="off",
                cache_ttl="1h",
                api_key="DO-NOT-SEND",
                base_url="https://wrong.example",
                default_headers={"x-anthropic-only": "DO-NOT-SEND"},
                cache_diagnostics=True,
            )
        )
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("anthropic_configured", [False, True])
async def test_messages_default_cache_advances_and_retains_conversation_boundaries(
    anthropic_configured: bool,
    broker: FakeBroker,
    cache_context: Context,
    messages_requests: list[httpx2.Request],
) -> None:
    assert cache_context.config is not None
    if not anthropic_configured:
        cache_context.config.anthropic = None
    llm = CopilotMessagesLLM(
        context=cache_context, model="claude-opus-5", instruction="Stable system prompt"
    )
    model_params = llm.resolved_model.model_params
    assert model_params is not None
    marker = {"type": "ephemeral", "ttl": model_params.cache_ttl or AnthropicSettings().cache_ttl}
    tools = [Tool(name="local_tool", input_schema={"type": "object", "properties": {}})]
    history = []
    for turn in range(3):
        history.append(Prompt.user(f"Turn {turn}"))
        result = await llm.generate(history, tools=tools)
        assert result.first_text() == "OK"
        history.append(result)
        body = json.loads(messages_requests[-1].content)
        assert isinstance(body, dict)
        assert body["system"][-1]["cache_control"] == marker
        assert body["tools"][0]["name"] == "local_tool"
        assert "cache_control" not in body  # No manually supplied top-level control.
        messages = body["messages"]
        assert len(messages) == 2 * turn + 1
        marked = [
            index
            for index, message in enumerate(messages)
            if "cache_control" in message["content"][-1]
        ]
        # Keep the previous user boundary while advancing to the newest one;
        # retire older boundaries rather than accumulating them indefinitely.
        assert marked == list(range(max(0, 2 * turn - 2), 2 * turn + 1, 2))
        assert all(messages[index]["content"][-1]["cache_control"] == marker for index in marked)
        assert "diagnostics" not in body
        request = messages_requests[-1]
        assert request.url.host == f"copilot-{turn + 1}.example"
        assert request.headers["authorization"] == f"Bearer broker-{turn + 1}"
        assert "x-api-key" not in request.headers
        assert "x-anthropic-only" not in request.headers
        assert "DO-NOT-SEND" not in str(request.headers)
        assert "cache-diagnosis" not in request.headers.get("anthropic-beta", "")
    assert len(messages_requests) == len(broker.calls) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["off", "prompt"])
async def test_messages_explicit_cache_mode_controls_system_templates_and_live_turns(
    mode: Literal["off", "prompt"],
    broker: FakeBroker,
    cache_context: Context,
    messages_requests: list[httpx2.Request],
) -> None:
    assert cache_context.config is not None
    context_config = cache_context.config
    context_config.copilot = CopilotSettings(cache_mode=mode)
    # The Copilot switch, not the Anthropic account switch, owns this request.
    assert context_config.anthropic is not None
    context_config.anthropic.cache_mode = "auto"
    llm = CopilotMessagesLLM(
        context=cache_context, model="claude-opus-5", instruction="Stable system prompt"
    )
    template = Prompt.user("Reusable prompt")
    template.is_template = True
    await llm.generate([template, Prompt.assistant("Ready"), Prompt.user("Live turn")])
    body = json.loads(messages_requests[0].content)
    assert isinstance(body, dict)
    if mode == "off":
        assert "cache_control" not in json.dumps(body)
    else:
        marker = {"type": "ephemeral", "ttl": "5m"}
        assert body["system"][-1]["cache_control"] == marker
        assert body["messages"][0]["content"][-1]["cache_control"] == marker
        assert "cache_control" not in json.dumps(body["messages"][1:])
        assert "cache_control" not in body


@pytest.mark.asyncio
@pytest.mark.parametrize("resolved_ttl", [None, "5m", "1h"])
@pytest.mark.parametrize("override", [None, "5m", "1h"])
async def test_messages_cache_ttl_uses_copilot_then_resolved_model_then_fresh_defaults(
    resolved_ttl: Literal["5m", "1h"] | None,
    override: Literal["5m", "1h"] | None,
    broker: FakeBroker,
    cache_context: Context,
    messages_requests: list[httpx2.Request],
) -> None:
    assert cache_context.config is not None
    cache_context.config.copilot = CopilotSettings(cache_ttl=override)
    resolved = ModelFactory.resolve_model_spec("copilot.claude-opus-5")
    assert resolved.model_params is not None
    # Vary only resolved TTL metadata so precedence is tested independently of
    # today's catalog values, including the missing-metadata fallback.
    resolved = replace(
        resolved, model_params=resolved.model_params.model_copy(update={"cache_ttl": resolved_ttl})
    )
    llm = CopilotMessagesLLM(
        context=cache_context,
        model=resolved.wire_model_name,
        resolved_model_spec=resolved,
        instruction="Stable system prompt",
    )
    await llm.generate([Prompt.user("Cache this turn")])
    body = json.loads(messages_requests[0].content)
    assert isinstance(body, dict)
    marker = {"type": "ephemeral", "ttl": override or resolved_ttl or AnthropicSettings().cache_ttl}
    assert body["system"][-1]["cache_control"] == marker
    assert body["messages"][-1]["content"][-1]["cache_control"] == marker
    assert "cache_control" not in body


@pytest.mark.asyncio
async def test_messages_overlay_inherits_catalog_cache_ttl(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    broker: FakeBroker,
    cache_context: Context,
    messages_requests: list[httpx2.Request],
) -> None:
    overlays = tmp_path / "model-overlays"
    overlays.mkdir()
    (overlays / "cached-opus.yaml").write_text(
        "name: cached-opus\nprovider: copilot\nmodel: claude-opus-5\n"
        "metadata:\n  model_specific: Custom overlay instruction\n",
        encoding="utf-8",
    )
    registry = load_model_overlay_registry(home=tmp_path)
    monkeypatch.setattr(
        "fast_agent.llm.model_factory.load_model_overlay_registry", lambda: registry
    )
    resolved = ModelFactory.resolve_model_spec("cached-opus")
    native = ModelFactory.resolve_model_spec("claude-opus-5")
    assert resolved.source == "overlay"
    assert resolved.model_params is not None and native.model_params is not None
    assert resolved.model_params.cache_ttl == native.model_params.cache_ttl
    assert resolved.model_params.cache_ttl is not None
    llm = CopilotMessagesLLM(
        context=cache_context,
        model=resolved.wire_model_name,
        resolved_model_spec=resolved,
        instruction="Stable system prompt",
    )
    await llm.generate([Prompt.user("Cache this turn")])
    body = json.loads(messages_requests[0].content)
    assert isinstance(body, dict)
    marker = {"type": "ephemeral", "ttl": resolved.model_params.cache_ttl}
    assert body["system"][-1]["cache_control"] == marker
    assert body["messages"][-1]["content"][-1]["cache_control"] == marker


def test_factory_opus48_high_reasoning(context: Context) -> None:
    from fast_agent.agents.agent_types import AgentConfig
    from fast_agent.agents.tool_agent import ToolAgent

    agent = ToolAgent(AgentConfig("opus48-high"), context=context)
    llm = ModelFactory.create_factory("copilot.claude-opus-4-8?reasoning=high")(agent)
    assert isinstance(llm, CopilotMessagesLLM)
    assert llm.provider is Provider.COPILOT
    assert llm.default_request_params.model == "claude-opus-4-8"
    arguments, enabled = llm._resolve_thinking_arguments("claude-opus-4-8", 4096, None)
    assert enabled
    assert arguments["thinking"] == {"type": "adaptive"}
    assert arguments["output_config"]["effort"] == "high"
