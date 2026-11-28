"""Gemini Copilot contracts through real Chat Completions SDK serialization."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Literal

import httpx2
import pytest
from mcp import Tool
from mcp.types import CallToolRequest, CallToolRequestParams, TextContent
from openai import AsyncOpenAI

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.tool_agent import ToolAgent
from fast_agent.config import GoogleSettings, OpenAISettings, Settings
from fast_agent.constants import (
    COPILOT_REASONING_OPAQUE,
    DEFAULT_HIGH_REASONING_STREAMING_TIMEOUT,
    FAST_AGENT_PROVIDER_ERROR,
    FAST_AGENT_RETRY,
    REASONING,
)
from fast_agent.context import Context
from fast_agent.core.exceptions import ModelConfigError
from fast_agent.llm.model_database import ModelDatabase
from fast_agent.llm.model_factory import ModelFactory
from fast_agent.llm.provider.copilot.broker import CopilotEndpoint
from fast_agent.llm.provider.copilot.chat_completions import CopilotChatCompletionsLLM
from fast_agent.llm.provider.copilot.policy import request_headers
from fast_agent.llm.provider_types import Provider
from fast_agent.mcp.prompt import Prompt
from fast_agent.types import PromptMessageExtended, RequestParams
from fast_agent.types.llm_stop_reason import LlmStopReason

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionMessageParam

    from fast_agent.llm.stream_types import StreamChunk

MODELS = ("gemini-4-neon", "gemini-4-argon", "gemini-4-carbon", "gemini-4-hydrogen")


@pytest.mark.parametrize("model", MODELS)
def test_factory_selects_chat_completions_and_isolates_native_metadata(model: str) -> None:
    public_id = f"copilot.{model}"
    resolved = ModelFactory.resolve_model_spec(public_id)
    assert resolved.provider is Provider.COPILOT
    assert resolved.wire_model_name == model
    assert resolved.model_config.transport == "sse"
    context = Context(config=Settings())
    llm = ModelFactory.create_factory(public_id)(
        ToolAgent(AgentConfig("test"), context=context), context=context
    )
    assert isinstance(llm, CopilotChatCompletionsLLM)
    assert llm.provider is Provider.COPILOT
    assert not llm.supports_files_api()
    for query in (
        "transport=websocket",
        "transport=auto",
        "service_tier=flex",
        "compaction=native",
    ):
        with pytest.raises(ModelConfigError):
            ModelFactory.resolve_model_spec(f"{public_id}?{query}")
    native = ModelDatabase.get_model_params(model)
    route = resolved.model_params
    assert native is not None and route is not None
    assert native.reasoning == "google_thinking"
    assert native.google_search_supported
    assert route.reasoning == "openai"
    assert route.reasoning_effort_spec is not None
    expected_default = "high" if model.endswith(("hydrogen", "neon")) else "medium"
    assert route.reasoning_effort_spec.default is not None
    assert route.reasoning_effort_spec.default.kind == "effort"
    assert route.reasoning_effort_spec.default.value == expected_default
    assert route.reasoning_effort_spec.allowed_efforts == (
        ["low", "medium", "high", "xhigh"]
        if model.endswith(("hydrogen", "neon"))
        else ["low", "medium", "high"]
    )
    assert not route.google_search_supported
    assert route.google_service_tiers == ()
    assert route.json_mode is None
    assert not ModelDatabase.supports_mime(public_id, "video/mp4")
    assert ModelDatabase.get_model_specific(public_id) == ""


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model", "effort"),
    [
        ("gemini-4-neon", "xhigh"),
        ("gemini-4-argon", "high"),
        ("gemini-4-carbon", "medium"),
        ("gemini-4-hydrogen", "xhigh"),
    ],
)
async def test_gateway_wire_auth_effort_tools_and_usage(
    model: str, effort: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    requests: list[httpx2.Request] = []
    broker_calls: list[str] = []

    class Broker:
        async def resolve(
            self, model_id: str, *, owner_id: str, transport: Literal["sse", "websocket"]
        ) -> CopilotEndpoint:
            broker_calls.append(model_id)
            assert owner_id
            assert transport == "sse"
            return CopilotEndpoint(
                model_id=model_id,
                wire_api="chat_completions",
                transport=transport,
                base_url="https://copilot.example",
                headers={
                    "authorization": "Bearer broker-token",
                    "copilot-integration-id": "copilot-sdk",
                },
            )

    monkeypatch.setattr(
        "fast_agent.llm.provider.copilot.broker.get_copilot_broker", lambda context: Broker()
    )
    for key in ("OPENAI_API_KEY", "OPENAI_ADMIN_KEY", "OPENAI_ORG_ID", "OPENAI_PROJECT_ID"):
        monkeypatch.setenv(key, "ambient-secret")
    monkeypatch.setenv("OPENAI_CUSTOM_HEADERS", "X-Ambient-Secret: ambient-secret")

    def respond(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        chunks = [
            {
                "id": "chat-1",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "OK"},
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "chat-1",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
            },
        ]
        content = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=content)

    def client_factory(**kwargs: Any) -> AsyncOpenAI:
        kwargs["http_client"] = httpx2.AsyncClient(transport=httpx2.MockTransport(respond))
        return AsyncOpenAI(**kwargs)

    monkeypatch.setattr("fast_agent.llm.provider.copilot.openai_client.AsyncOpenAI", client_factory)
    context = Context(
        config=Settings(
            openai=OpenAISettings(
                api_key="direct-secret", base_url="https://wrong.example", reasoning="low"
            ),
            google=GoogleSettings(api_key="google-secret"),
        )
    )
    llm = ModelFactory.create_factory(f"copilot.{model}?reasoning={effort}")(
        ToolAgent(AgentConfig("test"), context=context), context=context
    )
    assert isinstance(llm, CopilotChatCompletionsLLM)
    result = await llm._openai_completion(
        [{"role": "user", "content": "hello"}],
        RequestParams(max_tokens=37),
        tools=[Tool(name="local_tool", input_schema={"type": "object", "properties": {}})],
    )
    assert result.first_text() == "OK"
    assert broker_calls == [model]
    assert len(requests) == 1
    request = requests[0]
    assert str(request.url) == "https://copilot.example/chat/completions"
    assert request.headers["authorization"] == "Bearer broker-token"
    assert request.headers["copilot-integration-id"] == "copilot-sdk"
    assert request.headers["x-initiator"] == "user"
    assert "ambient-secret" not in str(request.headers)
    body = json.loads(request.content)
    assert body["model"] == model
    assert body["reasoning_effort"] == effort
    assert body["max_completion_tokens"] == 37
    assert body["tools"][0]["function"]["name"] == "local_tool"
    assert body["stream"] is True
    assert not {"thinking_config", "generation_config", "service_tier"} & body.keys()
    assert len(llm.usage_accumulator.turns) == 1
    assert llm.usage_accumulator.turns[0].provider is Provider.COPILOT


def test_unset_effort_does_not_borrow_direct_config_or_replay_native_reasoning() -> None:
    llm = CopilotChatCompletionsLLM(
        model="gemini-4-hydrogen",
        context=Context(config=Settings(openai=OpenAISettings(reasoning="low"))),
    )
    assert llm.reasoning_effort is None
    message = Prompt.assistant("visible")
    message.channels = {REASONING: [TextContent(type="text", text="native-private-reasoning")]}
    converted = llm._convert_extended_messages_to_provider([message])
    assert converted == [{"role": "assistant", "content": "visible"}]
    arguments = llm._prepare_api_request(converted, None, RequestParams())
    assert "reasoning_effort" not in arguments
    assert "native-private-reasoning" not in json.dumps(arguments)


def test_chat_headers_classify_tool_results_and_images() -> None:
    assert request_headers([{"role": "tool", "tool_call_id": "call-1", "content": "done"}]) == {
        "x-initiator": "agent"
    }
    assert request_headers(
        [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}
                ],
            }
        ]
    ) == {"x-initiator": "user", "copilot-vision-request": "true"}


@pytest.mark.parametrize(
    "metadata",
    [
        {"extra_headers": {"Authorization": "override"}},
        {"model": "gemini-4-carbon"},
        {"service_tier": "flex"},
        {"tools": [{"type": "web_search"}]},
        {"tool_choice": "required"},
        {
            "messages": [
                {"role": "user", "content": [{"type": "file", "file": {"file_id": "file-1"}}]}
            ]
        },
    ],
)
def test_chat_policy_rejects_unsupported_or_unsafe_requests(metadata: dict[str, Any]) -> None:
    llm = CopilotChatCompletionsLLM(model="gemini-4-argon", context=Context(config=Settings()))
    with pytest.raises(ValueError):
        llm._prepare_api_request(
            [{"role": "user", "content": "hello"}], None, RequestParams(metadata=metadata)
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("first_deltas", "retry_text", "calls_tool"),
    [
        (("<end_of_image>",), "Done.", False),
        (("<end_", "of_image><end_of_image>", "\n<end_of_image>"), "Done.", False),
        (("<eos><unused0>\n",), "Done.", False),
        (("<end_of_image>",), "<end_of_image>", False),
        (("The token is `<end_of_image>`.",), None, False),
        (("<end_of_image>",), None, True),
    ],
)
async def test_gemini_marker_only_reply_retries_completed_tool_boundary(
    first_deltas: tuple[str, ...],
    retry_text: str | None,
    calls_tool: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = "gemini-4-carbon"
    bodies: list[dict[str, Any]] = []
    sent_headers: list[httpx2.Headers] = []
    replies = [first_deltas] if retry_text is None else [first_deltas, (retry_text,)]

    class Broker:
        async def resolve(
            self, model_id: str, *, owner_id: str, transport: Literal["sse", "websocket"]
        ) -> CopilotEndpoint:
            return CopilotEndpoint(
                model_id=model_id,
                wire_api="chat_completions",
                transport=transport,
                base_url="https://copilot.example",
                headers={"authorization": "Bearer dummy-broker-token"},
            )

    def respond(request: httpx2.Request) -> httpx2.Response:
        index = len(bodies)
        bodies.append(json.loads(request.content))
        sent_headers.append(request.headers)
        common = {
            "id": f"chat-{index}",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": model,
        }
        chunks: list[dict[str, object]] = [
            {
                **common,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": text},
                        "finish_reason": None,
                    }
                ],
            }
            for text in replies[index]
        ]
        if calls_tool:
            chunks.append(
                {
                    **common,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call-next",
                                        "type": "function",
                                        "function": {"name": "run", "arguments": "{}"},
                                    }
                                ]
                            },
                            "finish_reason": None,
                        }
                    ],
                }
            )
        chunks.append(
            {
                **common,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "tool_calls" if calls_tool else "stop",
                    }
                ],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
            }
        )
        content = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
        return httpx2.Response(
            200, headers={"content-type": "text/event-stream"}, content=content + "data: [DONE]\n\n"
        )

    def client_factory(**kwargs: Any) -> AsyncOpenAI:
        kwargs["http_client"] = httpx2.AsyncClient(transport=httpx2.MockTransport(respond))
        return AsyncOpenAI(**kwargs)

    monkeypatch.setattr(
        "fast_agent.llm.provider.copilot.broker.get_copilot_broker", lambda context: Broker()
    )
    monkeypatch.setattr("fast_agent.llm.provider.copilot.openai_client.AsyncOpenAI", client_factory)
    llm = CopilotChatCompletionsLLM(model=model, context=Context(config=Settings()))
    events: list[StreamChunk] = []
    llm.add_stream_listener(events.append)
    messages: list[ChatCompletionMessageParam] = [
        {
            "role": "assistant",
            "tool_calls": [
                {"id": "call-1", "type": "function", "function": {"name": "run", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "already ran once"},
    ]
    result = await llm._openai_completion(messages)
    assert bodies[0]["messages"] == messages
    if retry_text is None:
        assert len(bodies) == 1
        assert result.first_text() == "".join(first_deltas)
        if calls_tool:
            assert result.stop_reason is LlmStopReason.TOOL_USE
            assert result.tool_calls is not None and "call-next" in result.tool_calls
        else:
            assert result.stop_reason is LlmStopReason.END_TURN
        assert not any(event.event == "rollback" for event in events)
        return

    assert len(bodies) == 2
    assert bodies[1]["messages"][:-1] == bodies[0]["messages"]
    assert bodies[1]["messages"][-1]["role"] == "user"
    assert "no usable output" in bodies[1]["messages"][-1]["content"]
    assert sent_headers[0]["x-initiator"] == "agent"
    assert sent_headers[1]["x-initiator"] == "agent"
    assert result.channels is not None
    assert FAST_AGENT_RETRY in result.channels
    saved_history = json.dumps(llm.history.get())
    assert "<end_of_image>" not in saved_history
    assert "no usable output" not in saved_history
    if retry_text == "<end_of_image>":
        assert result.stop_reason is LlmStopReason.ERROR
        assert "only <end_of_image> markers" in result.all_text()
        assert FAST_AGENT_PROVIDER_ERROR in result.channels
        assert sum(event.event == "rollback" for event in events) == 2
    else:
        assert result.first_text() == retry_text
        assert result.stop_reason is LlmStopReason.END_TURN
        assert sum(event.event == "rollback" for event in events) == 1


@pytest.mark.parametrize("model", MODELS)
def test_gemini_route_defaults_compaction_trigger_and_streaming_timeout(model: str) -> None:
    public_id = f"copilot.{model}"
    resolved = ModelFactory.resolve_model_spec(public_id)
    assert resolved.model_params is not None
    assert resolved.model_params.compaction_trigger_tokens == 300_000

    context = Context(config=Settings())
    llm = ModelFactory.create_factory(f"{public_id}?reasoning=medium")(
        ToolAgent(AgentConfig("test"), context=context), context=context
    )
    assert isinstance(llm, CopilotChatCompletionsLLM)
    params = llm.get_request_params()
    assert params.streaming_timeout == DEFAULT_HIGH_REASONING_STREAMING_TIMEOUT
    assert "streaming_timeout" not in params.model_fields_set

    explicit = ModelFactory.create_factory(f"{public_id}?reasoning=medium&streaming_timeout=45")(
        ToolAgent(AgentConfig("test"), context=context), context=context
    )
    assert explicit.get_request_params().streaming_timeout == 45.0


@pytest.mark.asyncio
async def test_gemini_streams_reasoning_text_replays_opaque_and_tracks_top_level_reasoning_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = "gemini-4-carbon"
    bodies: list[dict[str, Any]] = []

    class Broker:
        async def resolve(
            self, model_id: str, *, owner_id: str, transport: Literal["sse", "websocket"]
        ) -> CopilotEndpoint:
            return CopilotEndpoint(
                model_id=model_id,
                wire_api="chat_completions",
                transport=transport,
                base_url="https://copilot.example",
                headers={"authorization": "Bearer dummy-broker-token"},
            )

    def respond(request: httpx2.Request) -> httpx2.Response:
        turn = len(bodies)
        bodies.append(json.loads(request.content))
        common = {
            "id": f"chat-{turn}",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": model,
        }
        if turn == 0:
            chunks = [
                {
                    **common,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "role": "assistant",
                                "content": None,
                                "reasoning_text": "Inspecting workspace. ",
                            },
                            "finish_reason": None,
                        }
                    ],
                },
                {
                    **common,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call-1",
                                        "type": "function",
                                        "function": {"name": "run", "arguments": "{}"},
                                    }
                                ],
                                "reasoning_opaque": "opaque-sig-turn-1",
                            },
                            "finish_reason": None,
                        }
                    ],
                },
                {
                    **common,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 50,
                        "total_tokens": 150,
                        "reasoning_tokens": 32,
                    },
                },
            ]
        else:
            chunks = [
                {
                    **common,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "role": "assistant",
                                "content": "Finished.<end_of_image><end_of_image>",
                                "reasoning_opaque": "opaque-sig-turn-2",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 180, "completion_tokens": 12, "total_tokens": 192},
                }
            ]
        content = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=content)

    def client_factory(**kwargs: Any) -> AsyncOpenAI:
        kwargs["http_client"] = httpx2.AsyncClient(transport=httpx2.MockTransport(respond))
        return AsyncOpenAI(**kwargs)

    monkeypatch.setattr(
        "fast_agent.llm.provider.copilot.broker.get_copilot_broker", lambda context: Broker()
    )
    monkeypatch.setattr("fast_agent.llm.provider.copilot.openai_client.AsyncOpenAI", client_factory)

    llm = CopilotChatCompletionsLLM(
        model=model, reasoning_effort="medium", context=Context(config=Settings())
    )
    stream_events: list[StreamChunk] = []
    llm.add_stream_listener(stream_events.append)

    tools = [Tool(name="run", input_schema={"type": "object", "properties": {}})]
    user_msg = Prompt.user("Run the check.")
    turn1 = await llm.generate([user_msg], tools=tools)

    assert turn1.stop_reason is LlmStopReason.TOOL_USE
    assert any(
        ev.is_reasoning and "Inspecting workspace" in (ev.text or "") for ev in stream_events
    )
    assert turn1.channels is not None
    assert COPILOT_REASONING_OPAQUE in turn1.channels
    assert REASONING in turn1.channels
    assert llm.usage_accumulator.turns[0].completion.reasoning == 32

    from mcp.types import CallToolResult

    from fast_agent.mcp.prompt_message_extended import PromptMessageExtended

    tool_reply = PromptMessageExtended(
        role="user",
        tool_results={"call-1": CallToolResult(content=[TextContent(type="text", text="ok")])},
    )
    turn2 = await llm.generate([user_msg, turn1, tool_reply], tools=tools)

    assert len(bodies) == 2
    replayed_assistant = bodies[1]["messages"][1]
    assert replayed_assistant["role"] == "assistant"
    assert replayed_assistant["reasoning_opaque"] == "opaque-sig-turn-1"
    assert replayed_assistant["reasoning_text"] == "Inspecting workspace. "
    # Trailing <end_of_image> markers after real text are stripped cleanly.
    assert turn2.first_text() == "Finished."


@pytest.mark.asyncio
async def test_gemini_empty_stream_skips_non_streaming_fallback_and_recovers_with_agent_nudge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = "gemini-4-carbon"
    bodies: list[dict[str, Any]] = []
    sent_headers: list[httpx2.Headers] = []

    class Broker:
        async def resolve(
            self, model_id: str, *, owner_id: str, transport: Literal["sse", "websocket"]
        ) -> CopilotEndpoint:
            return CopilotEndpoint(
                model_id=model_id,
                wire_api="chat_completions",
                transport=transport,
                base_url="https://copilot.example",
                headers={"authorization": "Bearer dummy-broker-token"},
            )

    def respond(request: httpx2.Request) -> httpx2.Response:
        attempt = len(bodies)
        body = json.loads(request.content)
        bodies.append(body)
        sent_headers.append(request.headers)
        common = {
            "id": f"chat-{attempt}",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": model,
        }
        if attempt == 0:
            # Empty stop chunk (completion_tokens=0), reproducing production-planning turn 1.
            chunks = [
                {
                    **common,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": None},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 6853, "completion_tokens": 0, "total_tokens": 6853},
                }
            ]
        else:
            chunks = [
                {
                    **common,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": "Recovered."},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 6875, "completion_tokens": 4, "total_tokens": 6879},
                }
            ]
        content = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=content)

    def client_factory(**kwargs: Any) -> AsyncOpenAI:
        kwargs["http_client"] = httpx2.AsyncClient(transport=httpx2.MockTransport(respond))
        return AsyncOpenAI(**kwargs)

    monkeypatch.setattr(
        "fast_agent.llm.provider.copilot.broker.get_copilot_broker", lambda context: Broker()
    )
    monkeypatch.setattr("fast_agent.llm.provider.copilot.openai_client.AsyncOpenAI", client_factory)

    from mcp.types import CallToolRequest, CallToolRequestParams, CallToolResult

    from fast_agent.mcp.prompt_message_extended import PromptMessageExtended

    llm = CopilotChatCompletionsLLM(
        model=model, reasoning_effort="medium", context=Context(config=Settings())
    )
    llm.retry_count = 1
    llm.retry_backoff_seconds = 0.0

    history = [
        Prompt.user("Plan production."),
        PromptMessageExtended(
            role="assistant",
            tool_calls={
                "call-1": CallToolRequest(
                    method="tools/call",
                    params=CallToolRequestParams(
                        name="exec_sandbox", arguments={"code": "print(1)"}
                    ),
                )
            },
        ),
        PromptMessageExtended(
            role="user",
            tool_results={"call-1": CallToolResult(content=[TextContent(type="text", text="1")])},
        ),
    ]
    result = await llm.generate(history)

    assert result.first_text() == "Recovered."
    # Must NOT have issued a stream=False fallback request; only 2 SSE requests total.
    assert len(bodies) == 2
    assert all(body["stream"] is True for body in bodies)
    assert bodies[1]["messages"][-1]["role"] == "user"
    assert "no usable output" in bodies[1]["messages"][-1]["content"]
    assert sent_headers[0]["x-initiator"] == "agent"
    assert sent_headers[1]["x-initiator"] == "agent"


@pytest.mark.asyncio
async def test_gemini_stream_start_timeout_retries_with_lower_effort_and_persists_error_on_exhaustion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = "gemini-4-carbon"
    efforts: list[str | None] = []
    initiators: list[str] = []
    last_roles: list[str] = []

    class Broker:
        async def resolve(
            self, model_id: str, *, owner_id: str, transport: Literal["sse", "websocket"]
        ) -> CopilotEndpoint:
            return CopilotEndpoint(
                model_id=model_id,
                wire_api="chat_completions",
                transport=transport,
                base_url="https://copilot.example",
                headers={"authorization": "Bearer dummy-broker-token"},
            )

    monkeypatch.setattr(
        "fast_agent.llm.provider.copilot.broker.get_copilot_broker", lambda context: Broker()
    )

    from fast_agent.llm.provider.openai.llm_openai import OpenAILLM

    async def always_timeout(self: OpenAILLM, request: Any) -> Any:
        efforts.append(request.arguments.get("reasoning_effort"))
        initiators.append(request.arguments["extra_headers"]["x-initiator"])
        last_roles.append(request.arguments["messages"][-1]["role"])
        raise TimeoutError("OpenAI stream did not start within 300.0 seconds.")

    monkeypatch.setattr(OpenAILLM, "_run_openai_completion_request", always_timeout)

    from mcp.types import CallToolRequest, CallToolRequestParams, CallToolResult

    from fast_agent.mcp.prompt_message_extended import PromptMessageExtended

    llm = CopilotChatCompletionsLLM(
        model=model, reasoning_effort="medium", context=Context(config=Settings())
    )
    llm.retry_count = 1
    llm.retry_backoff_seconds = 0.0

    history = [
        Prompt.user("Compute SA-CCR."),
        PromptMessageExtended(
            role="assistant",
            tool_calls={
                "call-1": CallToolRequest(
                    method="tools/call",
                    params=CallToolRequestParams(
                        name="exec_sandbox", arguments={"code": "print(1)"}
                    ),
                )
            },
        ),
        PromptMessageExtended(
            role="user",
            tool_results={"call-1": CallToolResult(content=[TextContent(type="text", text="ok")])},
        ),
    ]
    result = await llm.generate(history)

    assert efforts == ["medium", "low"]
    assert initiators == ["agent", "agent"]
    assert last_roles == ["tool", "user"]
    assert result.stop_reason is LlmStopReason.ERROR
    assert result.channels is not None
    assert FAST_AGENT_PROVIDER_ERROR in result.channels


@pytest.mark.asyncio
async def test_gemini_image_cap_omits_oldest_images_above_ten_and_keeps_newest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import base64
    from io import BytesIO

    from mcp.types import CallToolResult, ImageContent
    from PIL import Image

    from fast_agent.llm.provider.request_images import IMAGE_OMITTED_PLACEHOLDER
    from fast_agent.mcp.prompt_message_extended import PromptMessageExtended

    model = "gemini-4-carbon"
    bodies: list[dict[str, Any]] = []

    class Broker:
        async def resolve(
            self, model_id: str, *, owner_id: str, transport: Literal["sse", "websocket"]
        ) -> CopilotEndpoint:
            return CopilotEndpoint(
                model_id=model_id,
                wire_api="chat_completions",
                transport=transport,
                base_url="https://copilot.example",
                headers={"authorization": "Bearer dummy-broker-token"},
            )

    def respond(request: httpx2.Request) -> httpx2.Response:
        bodies.append(json.loads(request.content))
        chunk = {
            "id": "chat-img",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "Checked."},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 100, "completion_tokens": 2, "total_tokens": 102},
        }
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n",
        )

    def client_factory(**kwargs: Any) -> AsyncOpenAI:
        kwargs["http_client"] = httpx2.AsyncClient(transport=httpx2.MockTransport(respond))
        return AsyncOpenAI(**kwargs)

    monkeypatch.setattr(
        "fast_agent.llm.provider.copilot.broker.get_copilot_broker", lambda context: Broker()
    )
    monkeypatch.setattr("fast_agent.llm.provider.copilot.openai_client.AsyncOpenAI", client_factory)

    def make_png(color: int) -> ImageContent:
        buf = BytesIO()
        Image.new("RGB", (16, 16), (color, 20, 30)).save(buf, format="PNG")
        return ImageContent(
            type="image",
            mime_type="image/png",
            data=base64.b64encode(buf.getvalue()).decode("ascii"),
        )

    eleven_images = [make_png(i * 20) for i in range(11)]
    history: list[PromptMessageExtended] = [Prompt.user("Inspect 11 CAD renders.")]
    for idx, img in enumerate(eleven_images):
        history.append(
            PromptMessageExtended(
                role="user",
                tool_results={f"call-{idx}": CallToolResult(content=[img])},
            )
        )

    llm = CopilotChatCompletionsLLM(model=model, context=Context(config=Settings()))
    result = await llm.generate(history)
    assert result.first_text() == "Checked."
    assert len(bodies) == 1

    sent_urls = [
        part["image_url"]["url"]
        for msg in bodies[0]["messages"]
        if isinstance(msg.get("content"), list)
        for part in msg["content"]
        if isinstance(part, dict) and part.get("type") == "image_url"
    ]
    assert len(sent_urls) == 10
    assert json.dumps(bodies[0]).count(IMAGE_OMITTED_PLACEHOLDER) == 1
    assert sent_urls[-1].endswith(eleven_images[-1].data)
    assert not any(url.endswith(eleven_images[0].data) for url in sent_urls)
    # Canonical history on the caller side is untouched (still has 11 ImageContent blocks).
    total_history_images = sum(
        1
        for msg in history
        for tr in (msg.tool_results or {}).values()
        for blk in tr.content
        if isinstance(blk, ImageContent)
    )
    assert total_history_images == 11


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model", "expected_retention"),
    [
        ("gemini-4-neon", 1800),
        ("gemini-4-hydrogen", 900),
        ("gemini-4-carbon", 900),
        ("gemini-4-argon", 240),
    ],
)
async def test_copilot_gemini_warm_cache_retention_and_agent_initiator(
    model: str,
    expected_retention: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[httpx2.Request] = []
    bodies: list[dict[str, Any]] = []

    class Broker:
        async def resolve(
            self, model_id: str, *, owner_id: str, transport: Literal["sse", "websocket"]
        ) -> CopilotEndpoint:
            return CopilotEndpoint(
                model_id=model_id,
                wire_api="chat_completions",
                transport=transport,
                base_url="https://copilot.example",
                headers={"authorization": "Bearer dummy-broker-token"},
            )

    def respond(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        bodies.append(json.loads(request.content))
        chunk = {
            "id": "chat-warm",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "OK"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 12000,
                "completion_tokens": 2,
                "total_tokens": 12002,
                "prompt_tokens_details": {"cached_tokens": 10000},
            },
        }
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n",
        )

    def client_factory(**kwargs: Any) -> AsyncOpenAI:
        kwargs["http_client"] = httpx2.AsyncClient(transport=httpx2.MockTransport(respond))
        return AsyncOpenAI(**kwargs)

    monkeypatch.setattr(
        "fast_agent.llm.provider.copilot.broker.get_copilot_broker", lambda context: Broker()
    )
    monkeypatch.setattr("fast_agent.llm.provider.copilot.openai_client.AsyncOpenAI", client_factory)

    llm = CopilotChatCompletionsLLM(model=model, context=Context(config=Settings()))
    assert llm.cache_retention_seconds == expected_retention
    assert llm.warm_cache_reports_cache_use is True
    assert llm.warms_cache_mid_turn is True

    # Mid-turn conversation with a pending assistant tool call is trimmed before warming.
    pending_tool_turn = PromptMessageExtended(
        role="assistant",
        tool_calls={
            "call-1": CallToolRequest(
                method="tools/call", params=CallToolRequestParams(name="slow_tool", arguments={})
            )
        },
    )
    usage = await llm.warm_cache([Prompt.user("Delegate task"), pending_tool_turn])
    assert usage.prompt.total == 12000
    assert usage.prompt.cache_read == 10000

    [request], [body] = requests, bodies
    assert request.headers["x-initiator"] == "agent"
    assert body["messages"][-1]["role"] == "user"
    assert not any(msg.get("tool_calls") for msg in body["messages"])
