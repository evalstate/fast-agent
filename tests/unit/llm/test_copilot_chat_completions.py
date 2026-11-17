"""Gemini Copilot contracts through real Chat Completions SDK serialization."""

from __future__ import annotations

import json
from typing import Any, Literal

import httpx2
import pytest
from mcp import Tool
from mcp.types import TextContent
from openai import AsyncOpenAI

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.tool_agent import ToolAgent
from fast_agent.config import GoogleSettings, OpenAISettings, Settings
from fast_agent.constants import REASONING
from fast_agent.context import Context
from fast_agent.core.exceptions import ModelConfigError
from fast_agent.llm.model_database import ModelDatabase
from fast_agent.llm.model_factory import ModelFactory
from fast_agent.llm.provider.copilot.broker import CopilotEndpoint
from fast_agent.llm.provider.copilot.chat_completions import CopilotChatCompletionsLLM
from fast_agent.llm.provider.copilot.policy import request_headers
from fast_agent.llm.provider_types import Provider
from fast_agent.mcp.prompt import Prompt
from fast_agent.types import RequestParams

MODELS = ("gemini-4-argon", "gemini-4-carbon", "gemini-4-hydrogen")


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
    assert route.reasoning_effort_spec.default is None
    assert route.reasoning_effort_spec.allowed_efforts == (
        ["low", "medium", "high", "xhigh"]
        if model.endswith("hydrogen")
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
    [("gemini-4-argon", "high"), ("gemini-4-carbon", "medium"), ("gemini-4-hydrogen", "xhigh")],
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

    monkeypatch.setattr(
        "fast_agent.llm.provider.copilot.chat_completions.AsyncOpenAI", client_factory
    )
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
