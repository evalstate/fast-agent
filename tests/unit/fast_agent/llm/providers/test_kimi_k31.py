"""Kimi K3.1 Pro routing, request, and preserved-thinking contracts."""

import json
from collections.abc import AsyncIterator

import pytest
from mcp.types import CallToolResult, TextContent
from openai.types.chat import ChatCompletionChunk

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.llm_agent import LlmAgent
from fast_agent.config import HuggingFaceSettings, Settings
from fast_agent.constants import REASONING
from fast_agent.context import Context
from fast_agent.llm.model_database import ModelDatabase
from fast_agent.llm.model_factory import ModelFactory
from fast_agent.llm.provider.openai.llm_huggingface import HuggingFaceLLM
from fast_agent.llm.provider.openai.llm_moonshot import MoonshotLLM
from fast_agent.llm.provider.openai.llm_openai import _OpenAICompletionResponse
from fast_agent.llm.provider_types import Provider
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended

_ROUTES = ("moonshot.kimi-k3.1-pro", "hf.moonshotai/Kimi-K3.1-Pro:together")


def _llm(model: str, *, backend: str | None = None) -> MoonshotLLM | HuggingFaceLLM:
    llm = ModelFactory.create_factory(model)(
        agent=LlmAgent(AgentConfig(name="kimi-k31-contract")),
        context=Context(config=Settings(hf=HuggingFaceSettings(default_provider=backend))),
    )
    assert isinstance(llm, MoonshotLLM | HuggingFaceLLM)
    return llm


@pytest.mark.parametrize(
    "model",
    [
        "kimi-k3.1-pro",
        "moonshot.kimi-k3.1-pro",
        "kimik31",
        "moonshotai/Kimi-K3.1-Pro",
        "hf.moonshotai/Kimi-K3.1-Pro:together",
        "Kimi K3.1 Pro (together)",
    ],
)
def test_model_resolution_and_capabilities(model: str) -> None:
    parsed = ModelFactory.parse_model_string(model)
    params = ModelDatabase.get_model_params(parsed.model_name, provider=parsed.provider)
    assert params is not None
    native = parsed.model_name == "kimi-k3.1-pro"
    assert parsed.provider is (Provider.MOONSHOT if native else Provider.HUGGINGFACE)
    assert params.context_window == 1_048_576
    assert params.max_output_tokens == 131_072
    assert params.reasoning == "reasoning_content"
    assert params.stream_mode == "manual"
    assert params.json_mode == "schema"
    assert params.structured_tool_policy == "no_tools"
    assert "image/png" in params.tokenizes
    assert ("video/mp4" in params.tokenizes) is native


@pytest.mark.parametrize("route", _ROUTES)
@pytest.mark.parametrize("effort", [None, "low", "high", "max", "off"])
def test_request_contract(route: str, effort: str | None) -> None:
    query = f"?reasoning={effort}" if effort else ""
    llm = _llm(route + query)
    request = llm._prepare_api_request(
        [{"role": "user", "content": "hello"}], None, llm.default_request_params
    )
    assert request["model"] == route.split(".", 1)[1]
    assert request["reasoning_effort"] == (effort if effort in {"low", "high"} else "max")
    assert "extra_body" not in request
    assert "thinking" not in request
    if isinstance(llm, MoonshotLLM):
        assert request["max_completion_tokens"] == 131_072
        assert "max_tokens" not in request
        assert "temperature" not in request
        assert "top_p" not in request
        assert "parallel_tool_calls" not in request
    else:
        assert request["max_tokens"] == 131_072
        assert "max_completion_tokens" not in request
        assert llm._resolve_usage_attribution("moonshotai/Kimi-K3.1-Pro", request) == (
            "moonshotai/Kimi-K3.1-Pro",
            "together",
        )


@pytest.mark.parametrize("route", (*_ROUTES, "kimi", "kimithink", "kimik31"))
def test_ptc_is_enabled_by_default_and_can_be_disabled(route: str) -> None:
    assert ModelFactory.resolve_model_spec(route).resolve_ptc_enabled()
    assert not ModelFactory.resolve_model_spec(f"{route}?ptc=off").resolve_ptc_enabled()


@pytest.mark.parametrize("provider", [MoonshotLLM, HuggingFaceLLM])
def test_provider_defaults_promote_kimi_k31(
    provider: type[MoonshotLLM] | type[HuggingFaceLLM],
) -> None:
    llm = provider(context=Context(config=Settings()), model="")
    request = llm._prepare_api_request(
        [{"role": "user", "content": "hello"}], None, llm.default_request_params
    )
    assert request["model"] == (
        "kimi-k3.1-pro" if provider is MoonshotLLM else "moonshotai/Kimi-K3.1-Pro:together"
    )
    assert request["reasoning_effort"] == "max"


def test_configured_hf_backend_uses_same_contract() -> None:
    llm = _llm("hf.moonshotai/Kimi-K3.1-Pro?reasoning=low", backend="together")
    request = llm._prepare_api_request(
        [{"role": "user", "content": "hello"}], None, llm.default_request_params
    )
    assert request["model"] == "moonshotai/Kimi-K3.1-Pro:together"
    assert request["reasoning_effort"] == "low"


async def _stream(parts: list[str], *, with_tools: bool) -> AsyncIterator[ChatCompletionChunk]:
    deltas: list[dict[str, object]] = [{"reasoning_content": part} for part in parts]
    if with_tools:
        deltas.append(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "lookup-1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": '{"key":'},
                    }
                ]
            }
        )
        deltas.append({"tool_calls": [{"index": 0, "function": {"arguments": '"answer"}'}}]})
    deltas.append({"content": "done"})
    for index, delta in enumerate(deltas):
        yield ChatCompletionChunk.model_validate(
            {
                "id": "simulated",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "kimi-k3.1-pro",
                "choices": [
                    {
                        "index": 0,
                        "delta": delta,
                        "finish_reason": (
                            ("tool_calls" if with_tools else "stop")
                            if index == len(deltas) - 1
                            else None
                        ),
                    }
                ],
            }
        )


@pytest.mark.parametrize("route", (*_ROUTES, "moonshot.kimi-k3"))
@pytest.mark.parametrize("with_tools", [False, True])
@pytest.mark.asyncio
async def test_stream_replays_complete_reasoning_and_tool_calls(
    route: str, with_tools: bool
) -> None:
    llm = _llm(route)
    reasoning = " \tfirst.\n\nSecond  sentence\t \n"
    for parts in ([reasoning], list(reasoning)):
        request = llm._openai_completion_request(
            [{"role": "user", "content": "Look up the answer."}], None, None
        )
        completion, segments = await llm._process_stream_manual(
            _stream(parts, with_tools=with_tools), request.model_name
        )
        response = await llm._finalize_openai_completion(
            request, _OpenAICompletionResponse(completion, segments)
        )
        assert response.channels is not None
        assert response.channels[REASONING] == [TextContent(type="text", text=reasoning)]
        history = [response]
        if with_tools:
            history.append(
                PromptMessageExtended(
                    role="user",
                    tool_results={
                        "lookup-1": CallToolResult(content=[TextContent(type="text", text="42")])
                    },
                )
            )
        messages = llm._convert_extended_messages_to_provider(history)
        assert messages[0].get("reasoning_content") == reasoning
        assert messages[0]["content"] == "done"
        if with_tools:
            assert messages[1] == {"role": "tool", "tool_call_id": "lookup-1", "content": "42"}
            assistant = messages[0]
            assert assistant["role"] == "assistant"
            calls = list(assistant.get("tool_calls") or [])
            assert len(calls) == 1
            call = calls[0]
            assert call["type"] == "function"
            assert call["id"] == "lookup-1"
            assert call["function"]["name"] == "lookup"
            assert json.loads(call["function"]["arguments"]) == {"key": "answer"}
