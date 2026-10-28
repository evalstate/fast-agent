"""HF GLM-5.3 contracts exercised through model resolution and request shaping."""

import json
import os
import subprocess
import sys
from collections.abc import AsyncIterator
from copy import deepcopy

import pytest
from mcp.types import CallToolResult, TextContent
from openai.types.chat import ChatCompletionChunk

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.llm_agent import LlmAgent
from fast_agent.config import HuggingFaceSettings, Settings
from fast_agent.constants import REASONING
from fast_agent.context import Context
from fast_agent.llm.model_factory import ModelFactory
from fast_agent.llm.provider.openai.llm_huggingface import HuggingFaceLLM
from fast_agent.llm.provider.openai.llm_openai import _OpenAICompletionResponse
from fast_agent.llm.provider_types import Provider
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended

_MODELS = ("GLM-5.3", "GLM-5.3-Flash")
_BACKENDS = ("together", "novita", "zai-org")


def _llm(model: str, *, configured_backend: str | None = None) -> HuggingFaceLLM:
    llm = ModelFactory.create_factory(model)(
        agent=LlmAgent(AgentConfig(name="glm53-contract")),
        context=Context(
            config=Settings(hf=HuggingFaceSettings(default_provider=configured_backend))
        ),
    )
    assert isinstance(llm, HuggingFaceLLM)
    return llm


def _request(llm: HuggingFaceLLM) -> dict[str, object]:
    return llm._prepare_api_request(
        [{"role": "user", "content": "hello"}], None, llm.default_request_params
    )


def _assert_reasoning_wire(request: dict[str, object], backend: str, effort: str) -> None:
    assert request["reasoning_effort"] == effort
    extra = request.get("extra_body", {})
    assert isinstance(extra, dict)
    assert "reasoning_effort" not in extra
    assert "separate_reasoning" not in extra
    assert "thinking" not in request
    if backend == "zai-org":
        assert extra["thinking"] == {"type": "enabled", "clear_thinking": False}
    else:
        assert "thinking" not in extra
    if backend == "together":
        template = extra["chat_template_kwargs"]
        assert isinstance(template, dict)
        assert template["clear_thinking"] is False
        assert "reasoning_effort" not in template


@pytest.mark.parametrize("model", _MODELS)
@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize(
    "setting, expected",
    (
        (None, "max"),
        ("low", "low"),
        ("high", "high"),
        ("max", "max"),
        ("medium", "high"),
        ("none", "max"),
        ("off", "max"),
        ("on", "max"),
        ("false", "max"),
        ("auto", "max"),
        ("minimal", "low"),
        ("xhigh", "max"),
    ),
)
def test_glm53_reasoning_contract(
    model: str, backend: str, setting: str | None, expected: str
) -> None:
    wire_model = f"zai-org/{model}:{backend}"
    query = f"?reasoning={setting}" if setting else ""
    request = _request(_llm(f"hf.{wire_model}{query}"))

    assert request["model"] == wire_model
    _assert_reasoning_wire(request, backend, expected)
    if backend == "novita":
        assert "extra_body" not in request


@pytest.mark.parametrize("model", _MODELS)
@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("explicit_suffix", (False, True))
def test_glm53_configured_backend_and_suffix_precedence(
    model: str, backend: str, explicit_suffix: bool
) -> None:
    suffix = f":{backend}" if explicit_suffix else ""
    llm = _llm(
        f"hf.zai-org/{model}{suffix}?reasoning=low",
        configured_backend="auto" if explicit_suffix else backend,
    )
    request = _request(llm)

    assert request["model"] == f"zai-org/{model}:{backend}"
    _assert_reasoning_wire(request, backend, "low")


@pytest.mark.parametrize("backend", _BACKENDS)
def test_glm53_environment_backend_resolution(backend: str) -> None:
    # A real child environment tests precedence without patching process globals.
    script = """
import json
from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.llm_agent import LlmAgent
from fast_agent.config import HuggingFaceSettings, Settings
from fast_agent.context import Context
from fast_agent.llm.model_factory import ModelFactory

requests = []
for model in ("GLM-5.3", "GLM-5.3-Flash"):
    for configured, suffix in ((None, ""), ("zai-org", ""), ("zai-org", ":together")):
        llm = ModelFactory.create_factory(f"hf.zai-org/{model}{suffix}?reasoning=high")(
            agent=LlmAgent(AgentConfig(name="env-contract")),
            context=Context(config=Settings(hf=HuggingFaceSettings(default_provider=configured))),
        )
        requests.append(llm._prepare_api_request(
            [{"role": "user", "content": "hello"}], None, llm.default_request_params
        ))
print(json.dumps(requests))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "HF_DEFAULT_PROVIDER": backend},
        capture_output=True,
        text=True,
        check=True,
    )
    requests = json.loads(result.stdout)
    for model, offset in zip(_MODELS, (0, 3), strict=True):
        for request, expected in zip(
            requests[offset : offset + 3], (backend, "zai-org", "together"), strict=True
        ):
            assert request["model"] == f"zai-org/{model}:{expected}"
            _assert_reasoning_wire(request, expected, "high")


@pytest.mark.parametrize("model", _MODELS)
@pytest.mark.parametrize("backend", _BACKENDS)
def test_glm53_preserves_unrelated_extra_fields(model: str, backend: str) -> None:
    llm = _llm(f"hf.zai-org/{model}:{backend}")
    params = llm.default_request_params.model_copy(
        update={
            "metadata": {
                "extra_body": {
                    "preserved": {"value": 1},
                    "thinking": {"type": "disabled"},
                    "chat_template_kwargs": {"custom": True, "clear_thinking": True},
                }
            }
        }
    )
    request = llm._prepare_api_request([{"role": "user", "content": "hello"}], None, params)

    _assert_reasoning_wire(request, backend, "max")
    extra = request["extra_body"]
    assert isinstance(extra, dict)
    assert extra["preserved"] == {"value": 1}
    assert extra["chat_template_kwargs"] == {
        "custom": True,
        "clear_thinking": backend != "together",
    }


@pytest.mark.parametrize("model", _MODELS)
@pytest.mark.parametrize("backend", _BACKENDS)
def test_glm53_request_shaping_does_not_mutate_metadata(model: str, backend: str) -> None:
    llm = _llm(f"hf.zai-org/{model}:{backend}")
    metadata = {
        "extra_body": {
            "thinking": {"type": "disabled"},
            "chat_template_kwargs": {"custom": True, "clear_thinking": True},
        }
    }
    original = deepcopy(metadata)
    params = llm.default_request_params.model_copy(update={"metadata": metadata})

    first = llm._prepare_api_request([{"role": "user", "content": "hello"}], None, params)
    second = llm._prepare_api_request([{"role": "user", "content": "hello"}], None, params)

    assert metadata == original
    assert first == second
    _assert_reasoning_wire(first, backend, "max")


@pytest.mark.parametrize("model", _MODELS)
@pytest.mark.parametrize("backend", _BACKENDS)
def test_glm53_hf_metadata(model: str, backend: str) -> None:
    llm = _llm(f"hf.zai-org/{model}:{backend}")
    params = llm.resolved_model.model_params
    assert params is not None
    assert params.default_provider == Provider.HUGGINGFACE
    expected_media = ["text/plain"]
    if model.endswith("Flash"):
        expected_media += ["image/jpeg", "image/png"]
    assert params.tokenizes == expected_media
    assert params.reasoning == "reasoning_content"
    assert params.stream_mode == "manual"
    assert params.json_mode == "object"
    assert params.reasoning_effort_spec is not None
    assert params.reasoning_effort_spec.allowed_efforts == ["low", "high", "max"]
    assert llm.usage_accumulator.context_window_size == 1_000_000
    assert _request(llm)["max_tokens"] == 131_072


async def _reasoning_stream(
    parts: list[str], *, with_tools: bool
) -> AsyncIterator[ChatCompletionChunk]:
    """Simulate independent reasoning, visible content, and fragmented tool deltas."""
    deltas: list[dict[str, object]] = [{"reasoning_content": part} for part in parts]
    final: dict[str, object] = {"content": "done"}
    if with_tools:
        deltas[0]["tool_calls"] = [
            {
                "index": 0,
                "id": "lookup-1",
                "type": "function",
                "function": {"name": "lookup", "arguments": '{"key":'},
            }
        ]
        final["tool_calls"] = [{"index": 0, "function": {"arguments": '"answer"}'}}]
    deltas.append(final)
    for index, delta in enumerate(deltas):
        yield ChatCompletionChunk.model_validate(
            {
                "id": "simulated",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "simulated",
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


async def _stream_response(
    llm: HuggingFaceLLM, parts: list[str], *, with_tools: bool
) -> PromptMessageExtended:
    request = llm._openai_completion_request(
        [{"role": "user", "content": "Look up the value."}], None, None
    )
    response, reasoning = await llm._process_stream_manual(
        _reasoning_stream(parts, with_tools=with_tools), request.model_name
    )
    # Exercise production assembly, not a synthetic history block per stream delta.
    return await llm._finalize_openai_completion(
        request, _OpenAICompletionResponse(response, reasoning)
    )


@pytest.mark.parametrize("model", _MODELS)
@pytest.mark.parametrize("backend", (*_BACKENDS, "auto"))
@pytest.mark.parametrize("explicit_suffix", (False, True))
@pytest.mark.parametrize("with_tools", (False, True))
@pytest.mark.parametrize(
    "reasoning", ("first.Second sentence", " \tfirst.\n\nSecond  sentence\t \n")
)
@pytest.mark.asyncio
async def test_glm53_reasoning_replay_is_chunk_boundary_invariant(
    model: str, backend: str, explicit_suffix: bool, with_tools: bool, reasoning: str
) -> None:
    suffix = f":{backend}" if explicit_suffix else ""
    llm = _llm(f"hf.zai-org/{model}{suffix}?reasoning=low", configured_backend=backend)
    # Whole, character-wise (including whitespace-only chunks), and every two-way split.
    partitions = [[reasoning], list(reasoning)]
    partitions.extend([reasoning[:cut], reasoning[cut:]] for cut in range(1, len(reasoning)))
    for parts in partitions:
        response = await _stream_response(llm, parts, with_tools=with_tools)
        assert response.channels is not None
        assert response.channels[REASONING] == [TextContent(type="text", text=reasoning)]
        assert response.content == [TextContent(type="text", text="done")]
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
        request = llm._prepare_api_request(
            llm._convert_extended_messages_to_provider(history), None, llm.default_request_params
        )
        messages = request["messages"]
        assert messages[0]["reasoning_content"] == reasoning
        assert messages[0]["content"] == "done"
        if with_tools:
            assert [message["role"] for message in messages] == ["assistant", "tool"]
            call = messages[0]["tool_calls"][0]
            assert call["id"] == messages[1]["tool_call_id"] == "lookup-1"
            assert call["function"]["name"] == "lookup"
            assert json.loads(call["function"]["arguments"]) == {"key": "answer"}
            assert messages[1]["content"] == "42"
        else:
            assert not response.tool_calls


@pytest.mark.parametrize("backend", ("zai-org", "together", "auto"))
@pytest.mark.asyncio
async def test_glm52_preserved_thinking_replays_raw_reasoning(backend: str) -> None:
    llm = _llm(f"hf.zai-org/GLM-5.2:{backend}")
    extra = _request(llm)["extra_body"]
    assert isinstance(extra, dict)
    assert extra["thinking"]["clear_thinking"] is False
    parts = [" \t", "first.", "Second", "\n", " sentence "]
    response = await _stream_response(llm, parts, with_tools=True)
    replay = llm._convert_extended_messages_to_provider([response])
    assert replay[0].get("reasoning_content") == "".join(parts)


@pytest.mark.parametrize(
    "model",
    (
        "zai-org/GLM-5.2:deepinfra",
        "zai-org/GLM-5.2:fireworks-ai",
        "zai-org/GLM-5",
        "moonshotai/Kimi-K2.5:novita",
    ),
)
@pytest.mark.asyncio
async def test_other_hf_routes_keep_reasoning_normalization(model: str) -> None:
    llm = _llm(f"hf.{model}")
    response = await _stream_response(llm, ["first.", "Second sentence"], with_tools=False)
    assert response.channels is not None
    assert response.channels[REASONING] == [TextContent(type="text", text="first. Second sentence")]
