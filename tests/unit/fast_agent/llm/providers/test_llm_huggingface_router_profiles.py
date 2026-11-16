import json
from collections.abc import AsyncIterator
from typing import Any

import httpx2
import pytest
from mcp import Tool
from mcp.types import CallToolResult, ImageContent, TextContent
from openai import AsyncOpenAI, BadRequestError
from openai.types.chat import ChatCompletionChunk

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.llm_agent import LlmAgent
from fast_agent.config import HuggingFaceSettings, Settings
from fast_agent.constants import REASONING
from fast_agent.context import Context
from fast_agent.llm.model_factory import ModelFactory
from fast_agent.llm.provider.openai.llm_huggingface import HuggingFaceLLM
from fast_agent.llm.provider.openai.llm_openai_compatible import OpenAICompatibleLLM
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended


def _factory_request(model: str) -> dict[str, object]:
    agent = LlmAgent(AgentConfig(name="router-profile-test"))
    llm = ModelFactory.create_factory(model)(agent=agent)
    assert isinstance(llm, HuggingFaceLLM)
    return llm._prepare_api_request(
        [{"role": "user", "content": "hello"}],
        None,
        llm.default_request_params,
    )


def _request_with_extra_body(model: str) -> dict[str, object]:
    llm = HuggingFaceLLM(
        context=Context(config=Settings()),
        model=model,
    )
    request_params = llm.default_request_params.model_copy(
        update={
            "metadata": {
                "extra_body": {
                    "preserved": True,
                    "thinking": {"legacy": True},
                    "disable_reasoning": True,
                    "chat_template_kwargs": {"legacy": True},
                }
            }
        }
    )
    return llm._prepare_api_request(
        [{"role": "user", "content": "hello"}],
        None,
        request_params,
    )


@pytest.mark.parametrize(
    ("backend", "max_tokens"),
    (("baseten", 384_000), ("deepinfra", 131_072)),
)
@pytest.mark.parametrize(("query", "effort"), (("", "max"), ("?reasoning=none", "none")))
def test_deepseek_0731_routes_by_model_id_apply_route_profiles(
    backend: str, max_tokens: int, query: str, effort: str
) -> None:
    request = _factory_request(f"hf.deepseek-ai/DeepSeek-V4-Flash-0731:{backend}{query}")

    assert request["model"] == f"deepseek-ai/DeepSeek-V4-Flash-0731:{backend}"
    assert request["max_tokens"] == max_tokens
    assert request["reasoning_effort"] == effort


@pytest.mark.parametrize(
    ("model", "reasoning_strength"),
    (
        ("glimmer", "high"),
        ("glimmer?reasoning=low", "low"),
        ("glimmer?reasoning=xhigh", "xhigh"),
        ("glimmer?reasoning=medium", "medium"),
        ("glimmer?reasoning=high", "high"),
    ),
)
def test_muse_glimmer_together_applies_top_level_effort(
    model: str,
    reasoning_strength: str,
) -> None:
    request = _factory_request(model)
    extra_body = request.get("extra_body")

    assert request["model"] == "meta-models/Muse-Glimmer-30B:together"
    assert request["temperature"] == 1.0
    assert request["top_p"] == 0.95
    assert "max_tokens" not in request
    assert request["reasoning_effort"] == reasoning_strength
    assert extra_body == {"top_k": 64}


def test_muse_glimmer_together_uses_effective_prompt_context_window() -> None:
    agent = LlmAgent(AgentConfig(name="router-profile-test"))
    llm = ModelFactory.create_factory("glimmer")(agent=agent)

    assert isinstance(llm, HuggingFaceLLM)
    assert llm.usage_accumulator.context_window_size == 98_304


def test_muse_glimmer_other_backend_uses_effective_prompt_context_window() -> None:
    agent = LlmAgent(AgentConfig(name="router-profile-test"))
    llm = ModelFactory.create_factory("hf.meta-models/Muse-Glimmer-30B:novita")(agent=agent)

    assert isinstance(llm, HuggingFaceLLM)
    assert llm.usage_accumulator.context_window_size == 98_304


def test_muse_glimmer_explicit_output_cap_adjusts_prompt_context_window() -> None:
    agent = LlmAgent(AgentConfig(name="router-profile-test"))
    llm = ModelFactory.create_factory("glimmer?max_tokens=4096")(agent=agent)

    assert isinstance(llm, HuggingFaceLLM)
    assert llm.usage_accumulator.context_window_size == 126_976


def test_muse_glimmer_other_backend_explicit_output_cap_adjusts_prompt_window() -> None:
    agent = LlmAgent(AgentConfig(name="router-profile-test"))
    llm = ModelFactory.create_factory("hf.meta-models/Muse-Glimmer-30B:novita?max_tokens=4096")(
        agent=agent
    )

    assert isinstance(llm, HuggingFaceLLM)
    assert llm.usage_accumulator.context_window_size == 126_976


@pytest.mark.parametrize(
    "model",
    (
        "glimmer?max_tokens=4096",
        "hf.meta-models/Muse-Glimmer-30B:novita?max_tokens=4096",
    ),
)
def test_muse_glimmer_preserves_explicit_max_tokens(model: str) -> None:
    request = _factory_request(model)

    assert request["max_tokens"] == 4096


def test_muse_glimmer_other_backend_omits_default_max_tokens() -> None:
    request = _factory_request("hf.meta-models/Muse-Glimmer-30B:novita")

    assert "max_tokens" not in request
    assert "extra_body" not in request


@pytest.mark.parametrize(
    ("model", "reasoning_effort"),
    (
        ("Inkling 2 Small (together)", "high"),
        ("Inkling 2 Small (together)?reasoning=low", "low"),
        ("Inkling 2 Small (together)?reasoning=minimal", "minimal"),
        ("hf.thinkingmachines/Inkling-2-Small:together?reasoning=max", "max"),
        ("hf.thinkingmachines/inkling-2-small:together", "high"),
    ),
)
def test_inkling_together_request_contract(model: str, reasoning_effort: str) -> None:
    request = _factory_request(model)

    assert request["model"] == "thinkingmachines/Inkling-2-Small:together"
    assert "max_tokens" not in request
    assert request["reasoning_effort"] == reasoning_effort
    assert "extra_body" not in request


_INKLING_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}


def test_inkling_together_preserves_explicit_max_tokens() -> None:
    request = _factory_request("Inkling 2 Small (together)?max_tokens=4096")

    assert request["max_tokens"] == 4096


def test_inkling_together_uses_legacy_schema_envelope() -> None:
    llm = HuggingFaceLLM(
        context=Context(config=Settings()),
        model="thinkingmachines/Inkling-2-Small:together",
    )
    params = llm.default_request_params.model_copy(
        update={"response_format": llm.schema_to_response_format(_INKLING_SCHEMA)}
    )

    request = llm._prepare_api_request([{"role": "user", "content": "hello"}], None, params)

    assert request["response_format"] == {"type": "json_object", "schema": _INKLING_SCHEMA}


@pytest.mark.parametrize(
    ("model", "wire_model", "reasoning_effort"),
    (
        ("inkling", "thinkingmachines/Inkling-2-Small:deepinfra", "high"),
        ("inkling?reasoning=minimal", "thinkingmachines/Inkling-2-Small:deepinfra", "minimal"),
        (
            "hf.thinkingmachines/Inkling-2-Small:deepinfra",
            "thinkingmachines/Inkling-2-Small:deepinfra",
            "high",
        ),
        (
            "hf.thinkingmachines/Inkling-2-Small:deepinfra?reasoning=low",
            "thinkingmachines/Inkling-2-Small:deepinfra",
            "low",
        ),
        (
            "hf.thinkingmachines/Inkling-2-Small:deepinfra?reasoning=minimal",
            "thinkingmachines/Inkling-2-Small:deepinfra",
            "minimal",
        ),
        (
            "hf.thinkingmachines/Inkling-2-Small:deepinfra?reasoning=max",
            "thinkingmachines/Inkling-2-Small:deepinfra",
            "max",
        ),
        ("hf.thinkingmachines/Inkling-2-Small", "thinkingmachines/Inkling-2-Small", "high"),
        (
            "hf.thinkingmachines/Inkling-2-Small?reasoning=minimal",
            "thinkingmachines/Inkling-2-Small",
            "minimal",
        ),
    ),
)
def test_inkling_deepinfra_and_unpinned_request_contract(
    model: str,
    wire_model: str,
    reasoning_effort: str,
) -> None:
    request = _factory_request(model)

    assert request["model"] == wire_model
    assert "max_tokens" not in request
    assert request["reasoning_effort"] == reasoning_effort
    assert "extra_body" not in request


@pytest.mark.parametrize(
    "model",
    (
        "inkling?max_tokens=4096",
        "hf.thinkingmachines/Inkling-2-Small:deepinfra?max_tokens=4096",
        "hf.thinkingmachines/Inkling-2-Small?max_tokens=4096",
    ),
)
def test_inkling_deepinfra_and_unpinned_preserve_explicit_max_tokens(model: str) -> None:
    request = _factory_request(model)

    assert request["max_tokens"] == 4096


def test_inkling_deepinfra_clamps_max_tokens_to_the_backend_output_cap() -> None:
    """DeepInfra stops at 65,536 output tokens whatever is requested; Together does not."""
    assert _factory_request("inkling?max_tokens=100000")["max_tokens"] == 65_536
    together = _factory_request("Inkling 2 Small (together)?max_tokens=100000")
    assert together["max_tokens"] == 100_000


def test_inkling_configured_backend_selects_route_profile() -> None:
    settings = Settings(hf=HuggingFaceSettings(default_provider="together"))
    llm = HuggingFaceLLM(
        context=Context(config=settings),
        model="thinkingmachines/Inkling-2-Small",
    )
    params = llm.default_request_params.model_copy(
        update={"response_format": llm.schema_to_response_format(_INKLING_SCHEMA)}
    )

    request = llm._prepare_api_request([{"role": "user", "content": "hello"}], None, params)

    assert request["model"] == "thinkingmachines/Inkling-2-Small:together"
    assert request["response_format"] == {"type": "json_object", "schema": _INKLING_SCHEMA}


@pytest.mark.parametrize(
    ("model", "settings", "has_inkling_profile"),
    (
        # Un-pinned router request: provisional DeepInfra fallback.
        ("thinkingmachines/Inkling-2-Small", Settings(), True),
        # Explicit untested backend: unchanged generic behaviour.
        ("thinkingmachines/Inkling-2-Small:fireworks-ai", Settings(), False),
        # Dedicated endpoint without a backend suffix: unchanged generic behaviour.
        (
            "thinkingmachines/Inkling-2-Small",
            Settings(hf=HuggingFaceSettings(base_url="https://dedicated.example.test/v1")),
            False,
        ),
    ),
)
def test_inkling_unpinned_fallback_is_scoped_to_router_choice(
    model: str,
    settings: Settings,
    has_inkling_profile: bool,
) -> None:
    llm = HuggingFaceLLM(context=Context(config=settings), model=model)

    request = llm._prepare_api_request(
        [{"role": "user", "content": "hello"}],
        None,
        llm.default_request_params,
    )

    assert request["model"] == model
    assert ("max_tokens" not in request) is has_inkling_profile
    assert ("reasoning_effort" in request) is has_inkling_profile
    assert llm._preserves_raw_reasoning_history(model) is has_inkling_profile
    assert llm._get_model_stream_mode(model) == ("manual" if has_inkling_profile else "openai")


@pytest.mark.parametrize(
    ("model", "stream_mode"),
    (
        ("thinkingmachines/Inkling-2-Small:together", "openai"),
        ("thinkingmachines/Inkling-2-Small:deepinfra", "manual"),
    ),
)
def test_inkling_stream_path_is_selected_per_route(model: str, stream_mode: str) -> None:
    llm = HuggingFaceLLM(context=Context(config=Settings()), model=model)

    assert llm._get_model_stream_mode(model) == stream_mode


@pytest.mark.parametrize(
    ("alias", "wire_model", "stream_mode"),
    (
        # Maintainer decision: DeepInfra is the default Inkling backend.
        ("inkling", "thinkingmachines/Inkling-2-Small:deepinfra", "manual"),
        ("Inkling 2 Small (together)", "thinkingmachines/Inkling-2-Small:together", "openai"),
    ),
)
def test_inkling_aliases_select_backend_and_stream_path(
    alias: str, wire_model: str, stream_mode: str
) -> None:
    agent = LlmAgent(AgentConfig(name="router-profile-test"))
    llm = ModelFactory.create_factory(alias)(agent=agent)
    assert isinstance(llm, HuggingFaceLLM)

    request = llm._prepare_api_request(
        [{"role": "user", "content": "hello"}], None, llm.default_request_params
    )

    assert request["model"] == wire_model
    assert llm._get_model_stream_mode(llm.default_request_params.model) == stream_mode


@pytest.mark.parametrize(
    "model",
    (
        "thinkingmachines/Inkling-2-Small:deepinfra",
        "thinkingmachines/Inkling-2-Small",
    ),
)
def test_inkling_deepinfra_and_unpinned_use_standard_schema_envelope(model: str) -> None:
    llm = HuggingFaceLLM(context=Context(config=Settings()), model=model)
    response_format = llm.schema_to_response_format(_INKLING_SCHEMA)
    params = llm.default_request_params.model_copy(update={"response_format": response_format})

    request = llm._prepare_api_request([{"role": "user", "content": "hello"}], None, params)

    assert request["response_format"] == response_format
    assert request["response_format"]["type"] == "json_schema"
    assert request["response_format"]["json_schema"]["schema"] == _INKLING_SCHEMA


@pytest.mark.parametrize(
    "model",
    (
        "thinkingmachines/Inkling-2-Small:together",
        "thinkingmachines/Inkling-2-Small:deepinfra",
        "thinkingmachines/Inkling-2-Small",
    ),
)
def test_inkling_replays_raw_reasoning_chunks(model: str) -> None:
    llm = HuggingFaceLLM(context=Context(config=Settings()), model=model)
    message = PromptMessageExtended(
        role="assistant",
        content=[TextContent(type="text", text="answer")],
        channels={
            REASONING: [
                TextContent(type="text", text="first "),
                TextContent(type="text", text="second"),
            ]
        },
    )

    converted = llm._convert_extended_messages_to_provider([message])

    assert converted == [
        {"role": "assistant", "content": "answer", "reasoning_content": "first second"}
    ]


@pytest.mark.parametrize(
    ("model", "reasoning_effort"),
    (
        ("qwen/qwen3.8-27b", "medium"),
        ("qwen/qwen3.8-27b?reasoning=low", "low"),
        ("qwen/qwen3.8-27b?reasoning=xhigh", "xhigh"),
    ),
)
def test_qwen38_applies_reasoning_effort_route_contract(
    model: str,
    reasoning_effort: str,
) -> None:
    request = _factory_request(model)

    assert request["model"] == "Qwen/Qwen3.8-27B"
    assert request["reasoning_effort"] == reasoning_effort


def test_qwen38_disables_thinking_through_chat_template_contract() -> None:
    request = _factory_request("qwen/qwen3.8-27b?reasoning=off")

    assert request["model"] == "Qwen/Qwen3.8-27B"
    assert "reasoning_effort" not in request
    assert request["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False},
    }


@pytest.mark.parametrize(
    ("suffix", "query", "enabled"),
    (
        ("", "", False),
        ("", "?reasoning=on", True),
        (":other-backend", "?reasoning=off", False),
        (":other-backend", "?reasoning=on", True),
    ),
)
def test_gemma4_generic_route_sends_chat_template_toggle(
    suffix: str, query: str, enabled: bool
) -> None:
    llm = ModelFactory.create_factory(f"hf.google/gemma-4-31B-it{suffix}{query}")(
        agent=LlmAgent(AgentConfig(name="router-profile-test"))
    )
    assert isinstance(llm, HuggingFaceLLM)
    # The model spec must only offer what the generic HF rule can put on the wire.
    assert llm.reasoning_effort_spec is not None
    assert llm.reasoning_effort_spec.kind == "toggle"
    request = llm._prepare_api_request(
        [{"role": "user", "content": "hello"}], None, llm.default_request_params
    )

    assert request["model"] == f"google/gemma-4-31B-it{suffix}"
    assert "reasoning_effort" not in request
    assert request["extra_body"] == {"chat_template_kwargs": {"enable_thinking": enabled}}


@pytest.mark.parametrize(
    "model",
    (
        "Qwen/Qwen3.8-27B",
        "deepseek-ai/DeepSeek-V4.1-Flash:novita",
        "deepseek-ai/DeepSeek-V4.1-Flash:fireworks-ai",
    ),
)
def test_hf_replays_reasoning_as_reasoning_content(model: str) -> None:
    llm = HuggingFaceLLM(
        context=Context(config=Settings()),
        model=model,
    )
    message = PromptMessageExtended(
        role="assistant",
        content=[TextContent(type="text", text="answer")],
        channels={REASONING: [TextContent(type="text", text="private reasoning")]},
    )

    converted = llm._convert_extended_messages_to_provider([message])

    assert converted == [
        {
            "role": "assistant",
            "content": "answer",
            "reasoning_content": "private reasoning",
        }
    ]


async def _stream_chunks(
    chunks: list[ChatCompletionChunk],
) -> AsyncIterator[ChatCompletionChunk]:
    for chunk in chunks:
        yield chunk


def _glimmer_chunk(
    *,
    delta: dict[str, object],
    finish_reason: str | None = None,
    usage: dict[str, int] | None = None,
) -> ChatCompletionChunk:
    return ChatCompletionChunk.model_validate(
        {
            "id": "glimmer-chunk",
            "created": 0,
            "model": "meta-models/Muse-Glimmer-30B",
            "object": "chat.completion.chunk",
            "choices": [
                {
                    "index": 0,
                    "delta": delta,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": usage,
        }
    )


@pytest.mark.asyncio
async def test_muse_glimmer_manual_stream_reassembles_together_tool_fragments() -> None:
    llm = HuggingFaceLLM(
        context=Context(config=Settings()),
        model="meta-models/Muse-Glimmer-30B:together",
    )
    chunks = [
        _glimmer_chunk(delta={"role": "assistant"}),
        _glimmer_chunk(delta={"reasoning": "Use the shell tool."}),
        _glimmer_chunk(
            delta={
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "chatcmpl-tool-glimmer",
                        "type": "function",
                        "function": {"name": "bash", "arguments": "{"},
                    }
                ]
            }
        ),
        _glimmer_chunk(
            delta={
                "tool_calls": [
                    {
                        "index": 0,
                        "id": None,
                        "type": None,
                        "function": {
                            "name": None,
                            "arguments": '"command":"printf glimmer-tool-ok"',
                        },
                    }
                ]
            }
        ),
        _glimmer_chunk(
            delta={
                "tool_calls": [
                    {
                        "index": 0,
                        "id": None,
                        "type": None,
                        "function": {"name": None, "arguments": "}"},
                    }
                ]
            },
            finish_reason="tool_calls",
            usage={
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "total_tokens": 120,
            },
        ),
    ]

    completion, reasoning = await llm._process_stream_manual(
        _stream_chunks(chunks),
        "meta-models/Muse-Glimmer-30B",
    )

    message = completion.choices[0].message
    assert reasoning == ["Use the shell tool."]
    assert completion.choices[0].finish_reason == "tool_calls"
    assert completion.usage.prompt_tokens == 100
    assert message.tool_calls is not None
    assert len(message.tool_calls) == 1
    assert message.tool_calls[0].id == "chatcmpl-tool-glimmer"
    assert message.tool_calls[0].function.name == "bash"
    assert message.tool_calls[0].function.arguments == ('{"command":"printf glimmer-tool-ok"}')


def _deepinfra_tool_fragment(
    index: int,
    *,
    call_id: str | None = None,
    name: str | None = None,
    arguments: str = "",
    finish_reason: str | None = None,
) -> ChatCompletionChunk:
    # Sanitized DeepInfra shape: role/content placeholders and explicit nulls on continuations.
    return ChatCompletionChunk.model_validate(
        {
            "id": "deepinfra-chunk",
            "created": 0,
            "model": "thinkingmachines/Inkling-2-Small",
            "object": "chat.completion.chunk",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": None,
                        "content": "",
                        "tool_calls": [
                            {
                                "index": index,
                                "id": call_id,
                                "type": "function" if call_id else None,
                                "function": {"name": name, "arguments": arguments},
                            }
                        ],
                    },
                    "logprobs": None,
                    "finish_reason": finish_reason,
                }
            ],
        }
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model",
    (
        "thinkingmachines/Inkling-2-Small:deepinfra",
        "thinkingmachines/Inkling-2-Small",
    ),
)
async def test_inkling_deepinfra_stream_reassembles_null_placeholder_tool_fragments(
    model: str,
) -> None:
    llm = HuggingFaceLLM(context=Context(config=Settings()), model=model)
    chunks = [
        ChatCompletionChunk.model_validate(
            {
                "id": "deepinfra-chunk",
                "created": 0,
                "model": "thinkingmachines/Inkling-2-Small",
                "object": "chat.completion.chunk",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "", "reasoning_content": "Two."},
                        "finish_reason": None,
                    }
                ],
            }
        ),
    ]
    for index, (call_id, city) in enumerate((("call-paris", "Paris"), ("call-tokyo", "Tokyo"))):
        chunks += [
            _deepinfra_tool_fragment(index, call_id=call_id, name="get_weather"),
            _deepinfra_tool_fragment(index),
            _deepinfra_tool_fragment(
                index,
                arguments=f'{{"city":"{city}"}}',
                finish_reason="tool_calls" if index == 1 else None,
            ),
        ]
    chunks.append(
        ChatCompletionChunk.model_validate(
            {
                "id": "deepinfra-chunk",
                "created": 0,
                "model": "thinkingmachines/Inkling-2-Small",
                "object": "chat.completion.chunk",
                "choices": [],
                "usage": {"prompt_tokens": 118, "completion_tokens": 96, "total_tokens": 214},
            }
        )
    )

    completion, reasoning = await llm._process_stream(
        _stream_chunks(chunks),
        "thinkingmachines/Inkling-2-Small",
    )

    message = completion.choices[0].message
    assert reasoning == ["Two."]
    assert completion.choices[0].finish_reason == "tool_calls"
    assert completion.usage.prompt_tokens == 118
    assert message.tool_calls is not None
    assert [
        (call.id, call.type, call.function.name, call.function.arguments)
        for call in message.tool_calls
    ] == [
        ("call-paris", "function", "get_weather", '{"city":"Paris"}'),
        ("call-tokyo", "function", "get_weather", '{"city":"Tokyo"}'),
    ]


@pytest.mark.parametrize(
    "model",
    (
        "hf.deepseek-ai/DeepSeek-V4-Flash-0731",
        "hf.deepseek-ai/DeepSeek-V4-Flash-0731:together",
    ),
)
def test_deepseek_routes_fall_back_to_model_profile(model: str) -> None:
    request = _factory_request(model)

    assert request["reasoning_effort"] == "max"


def test_deepseek_profile_uses_configured_hf_backend() -> None:
    settings = Settings(hf=HuggingFaceSettings(default_provider="deepinfra"))
    llm = HuggingFaceLLM(
        context=Context(config=settings),
        model="deepseek-ai/DeepSeek-V4-Flash-0731",
    )

    request = llm._prepare_api_request(
        [{"role": "user", "content": "hello"}],
        None,
        llm.default_request_params,
    )

    assert request["model"] == "deepseek-ai/DeepSeek-V4-Flash-0731:deepinfra"
    assert request["reasoning_effort"] == "max"


@pytest.mark.parametrize(
    ("reasoning_effort", "expected_effort"),
    (
        (None, "max"),
        ("none", "none"),
    ),
)
def test_deepseek_custom_endpoint_uses_reasoning_without_wire_suffix(
    reasoning_effort: str | None,
    expected_effort: str,
) -> None:
    settings = Settings(hf=HuggingFaceSettings(base_url="https://dedicated.example.test/v1"))
    kwargs: dict[str, object] = {
        "context": Context(config=settings),
        "model": "deepseek-ai/DeepSeek-V4-Flash-0731",
    }
    if reasoning_effort is not None:
        kwargs["reasoning_effort"] = reasoning_effort
    llm = HuggingFaceLLM(**kwargs)

    request = llm._prepare_api_request(
        [{"role": "user", "content": "hello"}],
        None,
        llm.default_request_params,
    )

    assert llm._provider_base_url() == "https://dedicated.example.test/v1"
    assert request["model"] == "deepseek-ai/DeepSeek-V4-Flash-0731"
    assert request["reasoning_effort"] == expected_effort


def test_deepseek_constructor_endpoint_uses_custom_route_profile() -> None:
    llm = HuggingFaceLLM(
        context=Context(config=Settings()),
        model="deepseek-ai/DeepSeek-V4-Flash-0731",
        base_url="https://dedicated.example.test/v1",
    )

    request = llm._prepare_api_request(
        [{"role": "user", "content": "hello"}],
        None,
        llm.default_request_params,
    )

    assert llm._base_url() == "https://dedicated.example.test/v1"
    assert request["model"] == "deepseek-ai/DeepSeek-V4-Flash-0731"
    assert request["reasoning_effort"] == "max"


@pytest.mark.parametrize(
    "base_url",
    (
        "https://router.huggingface.co/v1/",
        "HTTPS://ROUTER.HUGGINGFACE.CO/v1",
        "https://router.huggingface.co:443/v1",
    ),
)
def test_equivalent_huggingface_router_urls_use_model_profile(
    base_url: str,
) -> None:
    llm = HuggingFaceLLM(
        context=Context(config=Settings()),
        model="deepseek-ai/DeepSeek-V4-Flash-0731",
        base_url=base_url,
    )

    request = llm._prepare_api_request(
        [{"role": "user", "content": "hello"}],
        None,
        llm.default_request_params,
    )

    assert request["model"] == "deepseek-ai/DeepSeek-V4-Flash-0731"
    assert request["reasoning_effort"] == "max"


def test_deepseek_custom_endpoint_does_not_override_explicit_router_backend() -> None:
    settings = Settings(hf=HuggingFaceSettings(base_url="https://dedicated.example.test/v1"))
    llm = HuggingFaceLLM(
        context=Context(config=settings),
        model="deepseek-ai/DeepSeek-V4-Flash-0731:together",
    )

    request = llm._prepare_api_request(
        [{"role": "user", "content": "hello"}],
        None,
        llm.default_request_params,
    )

    assert request["model"] == "deepseek-ai/DeepSeek-V4-Flash-0731:together"
    assert request["reasoning_effort"] == "max"


def test_deepseek_custom_endpoint_uses_nested_hf_base_url_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF__BASE_URL", "https://dedicated.example.test/v1")
    settings = Settings()
    llm = HuggingFaceLLM(
        context=Context(config=settings),
        model="deepseek-ai/DeepSeek-V4-Flash-0731",
    )

    request = llm._prepare_api_request(
        [{"role": "user", "content": "hello"}],
        None,
        llm.default_request_params,
    )

    assert settings.hf is not None
    assert settings.hf.base_url == "https://dedicated.example.test/v1"
    assert request["model"] == "deepseek-ai/DeepSeek-V4-Flash-0731"
    assert request["reasoning_effort"] == "max"


@pytest.mark.parametrize(
    ("model", "expected_extra_body_keys"),
    (
        (
            "deepseek-ai/DeepSeek-V4-Flash-0731:deepinfra",
            {"preserved", "thinking", "chat_template_kwargs", "disable_reasoning"},
        ),
        (
            "moonshotai/Kimi-K3:together",
            {"preserved", "disable_reasoning"},
        ),
        (
            "zai-org/GLM-5.2:deepinfra",
            {"preserved", "chat_template_kwargs", "disable_reasoning"},
        ),
    ),
)
def test_hf_route_profiles_preserve_route_specific_cleanup(
    model: str,
    expected_extra_body_keys: set[str],
) -> None:
    request = _request_with_extra_body(model)
    extra_body = request.get("extra_body")

    assert isinstance(extra_body, dict)
    assert set(extra_body) == expected_extra_body_keys
    assert extra_body.get("preserved") is True


@pytest.mark.parametrize("model", ("GLM-5.3", "GLM-5.3-Flash"))
@pytest.mark.parametrize("backend", ("", ":auto", ":zai-org", ":deepinfra"))
@pytest.mark.parametrize(
    "query, effort", (("", "max"), ("?reasoning=low", "low"), ("?reasoning=high", "high"))
)
def test_glm_53_routes_fall_back_to_model_profile(
    model: str, backend: str, query: str, effort: str
) -> None:
    wire_model = f"zai-org/{model}{backend}"
    request = _factory_request(f"hf.{wire_model}{query}")

    assert request["model"] == wire_model
    assert request["reasoning_effort"] == effort
    assert "extra_body" not in request


@pytest.mark.parametrize("model", ("GLM-5.3", "GLM-5.3-Flash"))
def test_glm_53_hf_keeps_required_reasoning_enabled(model: str) -> None:
    request = _factory_request(f"hf.zai-org/{model}:together?reasoning=none")

    assert request["reasoning_effort"] == "low"


@pytest.mark.parametrize(
    "backend, limit",
    (("baseten", 384_000), ("scaleway", 32_768)),
)
@pytest.mark.parametrize("query", ("", "?max_tokens=393216"))
def test_deepseek_route_output_limit(backend: str, limit: int, query: str) -> None:
    request = _factory_request(f"hf.deepseek-ai/DeepSeek-V4-Flash-0731:{backend}{query}")

    assert request["max_tokens"] == limit
    assert request["reasoning_effort"] == "max"


@pytest.mark.parametrize("max_tokens, expected", ((128, 128), (393_216, 384_000)))
def test_baseten_output_limit_applies_to_per_request_overrides(
    max_tokens: int, expected: int
) -> None:
    llm = HuggingFaceLLM(
        context=Context(config=Settings()),
        model="deepseek-ai/DeepSeek-V4-Flash-0731:baseten",
    )
    assert llm.default_request_params.max_tokens == 384_000
    params = llm.default_request_params.model_copy(update={"max_tokens": max_tokens})
    request = llm._prepare_api_request([{"role": "user", "content": "hello"}], None, params)

    assert request["max_tokens"] == expected
    assert params.max_tokens == max_tokens


def test_configured_baseten_backend_uses_route_output_limit() -> None:
    llm = HuggingFaceLLM(
        context=Context(config=Settings(hf=HuggingFaceSettings(default_provider="baseten"))),
        model="deepseek-ai/DeepSeek-V4-Flash-0731",
    )
    request = llm._prepare_api_request(
        [{"role": "user", "content": "hello"}], None, llm.default_request_params
    )

    assert request["model"] == "deepseek-ai/DeepSeek-V4-Flash-0731:baseten"
    assert request["max_tokens"] == 384_000


# Backend and reasoning are independent; cover each value once rather than the product.
@pytest.mark.parametrize(
    "backend, query, effort",
    (
        ("", "", "max"),
        (":novita", "?reasoning=max", "max"),
        (":fireworks-ai", "?reasoning=off", "low"),
        (":other-backend", "?reasoning=none", "low"),
        ("", "?reasoning=low", "low"),
        (":novita", "?reasoning=high", "high"),
    ),
)
def test_deepseek_v41_hf_model_profile_fallback(backend: str, query: str, effort: str) -> None:
    wire_model = f"deepseek-ai/DeepSeek-V4.1-Flash{backend}"
    request = _factory_request(f"hf.{wire_model}{query}")

    assert request["model"] == wire_model
    if backend == ":novita":
        assert "max_tokens" not in request
    else:
        assert request["max_tokens"] == 393_216
    assert request["reasoning_effort"] == effort
    assert "extra_body" not in request


@pytest.mark.parametrize(
    "query, effort",
    (
        ("", "max"),
        ("?reasoning=low", "low"),
        ("?reasoning=high", "high"),
        ("?reasoning=max", "max"),
        ("?reasoning=none", "none"),
        ("?reasoning=off", "none"),
    ),
)
def test_deepseek_f1017_novita_request_contract(query: str, effort: str) -> None:
    # Lowercase input must reach the case-sensitive provider with its canonical ID.
    request = _factory_request(f"hf.deepseek-ai/deepseek-v4.1-flash-1017:novita{query}")

    assert request["model"] == "deepseek-ai/DeepSeek-V4.1-Flash-1017:novita"
    assert request["reasoning_effort"] == effort
    assert "max_tokens" not in request
    assert "max_completion_tokens" not in request
    assert "extra_body" not in request


def test_deepseek_f1017_novita_explicit_output_override() -> None:
    request = _factory_request("deepseek-hf?max_tokens=128&reasoning=off")

    assert request["model"] == "deepseek-ai/DeepSeek-V4.1-Flash-1017:novita"
    assert request["max_tokens"] == 128
    assert request["reasoning_effort"] == "none"


def _structured_phase_request(
    model: str, after_tool: bool
) -> tuple[OpenAICompatibleLLM, dict[str, Any]]:
    """Build the real wire request for a structured turn with one tool available.

    ``after_tool`` selects the turn after a tool result, i.e. the final phase under defer.
    """
    agent = LlmAgent(AgentConfig(name="router-profile-test"))
    llm = ModelFactory.create_factory(model)(agent=agent)
    assert isinstance(llm, OpenAICompatibleLLM)
    params = llm.default_request_params.model_copy(update={"structured_schema": _INKLING_SCHEMA})
    tools = [
        Tool(
            name="lookup",
            description="Lookup a value.",
            input_schema={"type": "object", "properties": {}, "additionalProperties": False},
        )
    ]
    messages = [
        PromptMessageExtended(role="user", content=[TextContent(type="text", text="lookup")])
    ]
    if after_tool:
        messages.append(
            PromptMessageExtended(
                role="user",
                content=[],
                tool_results={
                    "call-lookup": CallToolResult(content=[TextContent(type="text", text="42")])
                },
            )
        )

    # Exercise the same offline preparation steps as generate, then build the real wire request.
    prepared_messages, prepared_params = llm._prepare_structured_request(messages, params, tools)
    prepared_tools = (
        None
        if llm._should_suppress_tools_for_structured_final(
            prepared_messages, prepared_params, tools
        )
        else tools
    )
    request = llm._openai_completion_request(
        llm._convert_extended_messages_to_provider(prepared_messages),
        prepared_params,
        prepared_tools,
    ).arguments
    assert params.structured_schema == _INKLING_SCHEMA
    assert params.response_format is None
    return llm, request


@pytest.mark.parametrize(
    ("model", "deferred", "max_tokens"),
    (
        ("deepseek-hf", True, None),
        ("deepseek-hf?structured_tools=always", False, None),
        ("deepseek41-hf", True, None),
        ("deepseek-v41-hf", True, None),
        ("deepseek41-hf?structured_tools=always", False, None),
        # The unlisted 0731 checkpoint shares the defer default; route limits still apply.
        ("hf.deepseek-ai/DeepSeek-V4-Flash-0731:baseten", True, 384_000),
        ("hf.deepseek-ai/DeepSeek-V4-Flash-0731:deepinfra", True, 131_072),
        ("hf.deepseek-ai/DeepSeek-V4-Flash-0731:scaleway?structured_tools=always", False, 32_768),
    ),
)
@pytest.mark.parametrize("after_tool", (False, True))
def test_deepseek_flash_hf_structured_request_phases(
    model: str, deferred: bool, max_tokens: int | None, after_tool: bool
) -> None:
    llm, request = _structured_phase_request(model, after_tool)

    if deferred and not after_tool:
        assert "response_format" not in request
    else:
        assert request["response_format"] == llm.schema_to_response_format(_INKLING_SCHEMA)
        assert request["response_format"]["type"] == "json_schema"
        assert request["response_format"]["json_schema"]["strict"] is True
    if deferred and after_tool:
        assert "tools" not in request
    else:
        assert len(request["tools"]) == 1
        assert request["tools"][0]["function"]["name"] == "lookup"
    assert request.get("max_tokens") == max_tokens
    assert request["reasoning_effort"] == "max"


@pytest.mark.parametrize(
    ("model", "policy"),
    (
        # HF routes from the 2026-11-15 sweep: one per family.
        ("GLM 5.3 (novita)", "defer"),
        ("glm5", "defer"),
        ("kimi26", "defer"),
        ("minimax2.5", "defer"),
        # Native providers share the catalog constants and keep their policy.
        ("zai.glm-5.3", "always"),
        ("moonshot.moonshotai/kimi-k2.6", "no_tools"),
    ),
)
@pytest.mark.parametrize("after_tool", (False, True))
def test_structured_with_tool_request_phases_by_route(
    model: str, policy: str, after_tool: bool
) -> None:
    _, request = _structured_phase_request(model, after_tool)

    # defer: tools only until a tool result arrives, then a schema-only final turn.
    schema_expected = policy != "defer" or after_tool
    tools_expected = policy == "always" or (policy == "defer" and not after_tool)
    assert ("response_format" in request) is schema_expected
    assert bool(request.get("tools")) is tools_expected
    if tools_expected:
        assert [tool["function"]["name"] for tool in request["tools"]] == ["lookup"]


@pytest.mark.parametrize(
    ("model", "format_type"),
    (
        # Novita rejects (GLM 5, MiniMax M2.5) or ignores (Kimi K2.6) json_schema.
        ("glm5", "json_object"),
        ("minimax2.5", "json_object"),
        ("kimi26", "json_object"),
        ("hf.zai-org/GLM-5:together", "json_schema"),
        ("hf.moonshotai/Kimi-K2.6:together", "json_schema"),
        ("kimi25", "json_schema"),
    ),
)
def test_structured_final_response_format_by_backend(model: str, format_type: str) -> None:
    _, request = _structured_phase_request(model, after_tool=True)

    assert request["response_format"]["type"] == format_type


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model", "image_sent"),
    (
        ("hf.deepseek-ai/DeepSeek-V4-Flash-0731:baseten", False),
        ("hf.deepseek-ai/DeepSeek-V4-Flash-0731:deepinfra", False),
        ("deepseek-hf", True),
    ),
)
async def test_deepseek_flash_hf_image_input_reaches_wire_only_from_v41(
    model: str, image_sent: bool
) -> None:
    agent = LlmAgent(AgentConfig(name="router-profile-test"))
    llm = await agent.attach_llm(ModelFactory.create_factory(model))
    assert isinstance(llm, HuggingFaceLLM)
    message = PromptMessageExtended(
        role="user",
        content=[
            TextContent(type="text", text="describe"),
            ImageContent(type="image", data="iVBORw0KGgo=", mime_type="image/png"),
        ],
    )

    sanitized, removed = agent._sanitize_messages_for_llm([message])
    request = llm._openai_completion_request(
        llm._convert_extended_messages_to_provider(sanitized),
        llm.default_request_params,
        None,
    ).arguments
    [user_message] = [m for m in request["messages"] if m["role"] == "user"]
    content = user_message["content"]
    image_on_wire = isinstance(content, list) and any(
        part["type"] == "image_url" for part in content
    )

    assert image_on_wire is image_sent
    assert (removed is None) is image_sent
    assert request["reasoning_effort"] == "max"


@pytest.mark.asyncio
async def test_deepseek_f1017_novita_image_stream_cumulative_usage() -> None:
    agent = LlmAgent(AgentConfig(name="router-profile-test"))
    llm = ModelFactory.create_factory("deepseek-hf")(agent=agent)
    assert isinstance(llm, HuggingFaceLLM)
    # Synthetic values/text based only on the sanitised conformance report (D11),
    # not captured traces: role-only lead, reasoning then text, usage on every chunk,
    # stop followed by an empty-choices usage chunk; reasoning equals completion usage.
    deltas = [
        {},
        {"reasoning_content": "Inspect the image."},
        {"content": "A blue"},
        {"content": " square."},
        {},
        None,
    ]
    chunks = [
        ChatCompletionChunk.model_validate(
            {
                "id": "synthetic-novita-image",
                "created": 0,
                "model": "deepseek-v4.1-flash-1017",
                "object": "chat.completion.chunk",
                "choices": (
                    []
                    if delta is None
                    else [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", **delta},
                            "finish_reason": "stop" if index == 4 else None,
                        }
                    ]
                ),
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": index * 4,
                    "total_tokens": 100 + index * 4,
                    "completion_tokens_details": {"reasoning_tokens": index * 4},
                },
            }
        )
        for index, delta in enumerate(deltas)
    ]
    # Use the selected SDK stream path, not the manual accumulator.
    model = "deepseek-ai/DeepSeek-V4.1-Flash-1017:novita"
    completion, reasoning = await llm._process_stream(_stream_chunks(chunks), model)

    assert completion.choices[0].message.content == "A blue square."
    assert completion.choices[0].finish_reason == "stop"
    assert reasoning == ["Inspect the image."]
    assert completion.usage.prompt_tokens == 100
    assert completion.usage.completion_tokens == 20
    assert completion.usage.total_tokens == 120
    assert completion.usage.completion_tokens_details.reasoning_tokens == 20

    llm._track_openai_response_usage(completion, model, upstream_provider="novita")
    [turn] = llm.usage_accumulator.turns
    assert turn.prompt.total == 100
    assert turn.completion.total == turn.completion.reasoning == 20
    assert turn.total == 120
    assert turn.upstream_provider == "novita"
    assert turn.prompt.cache_read is None
    assert turn.prompt.cache_write is None
    assert turn.prompt.uncached is None
    assert turn.prompt.tool_use is None
    assert turn.cost_usd is None
    assert turn.timing is None


@pytest.mark.parametrize(
    "alias", ("deepseek41-hf", "deepseek-v41-hf", "DeepSeek V4.1 Flash (novita)")
)
def test_deepseek_v41_hf_alias_request(alias: str) -> None:
    request = _factory_request(f"{alias}?reasoning=low&max_tokens=128")

    assert request["model"] == "deepseek-ai/DeepSeek-V4.1-Flash:novita"
    assert request["reasoning_effort"] == "low"
    assert request["max_tokens"] == 128


@pytest.mark.parametrize("backend", ("novita", "fireworks-ai"))
def test_deepseek_v41_configured_backend_uses_model_profile(backend: str) -> None:
    llm = HuggingFaceLLM(
        context=Context(config=Settings(hf=HuggingFaceSettings(default_provider=backend))),
        model="deepseek-ai/DeepSeek-V4.1-Flash",
    )
    request = llm._prepare_api_request(
        [{"role": "user", "content": "hello"}], None, llm.default_request_params
    )

    assert request["model"] == f"deepseek-ai/DeepSeek-V4.1-Flash:{backend}"
    assert request["reasoning_effort"] == "max"
    assert llm._structured_json_mode(llm.default_request_params) == "schema"


@pytest.mark.parametrize(
    "model",
    (
        "deepseek-ai/DeepSeek-V4-Flash-0731:together",
        "deepseek-ai/DeepSeek-V4.1-Flash:novita",
    ),
)
@pytest.mark.parametrize("limit", (None, 4096, 393216))
def test_deepseek_routes_omit_only_default_output_limit(model: str, limit: int | None) -> None:
    query = f"?max_tokens={limit}" if limit is not None else ""
    request = _factory_request(f"hf.{model}{query}")
    if limit is None:
        assert "max_tokens" not in request
    else:
        assert request["max_tokens"] == limit


@pytest.mark.parametrize("limit", (None, 4096, 393216))
def test_deepinfra_deepseek_output_default_is_not_a_hard_cap(limit: int | None) -> None:
    query = f"?max_tokens={limit}" if limit is not None else ""
    request = _factory_request(f"hf.deepseek-ai/DeepSeek-V4-Flash-0731:deepinfra{query}")
    assert request["max_tokens"] == (131072 if limit is None else limit)


@pytest.mark.parametrize("explicit_backend", (True, False))
def test_scaleway_deepseek_uses_backend_context_for_compaction(explicit_backend: bool) -> None:
    from fast_agent.config import CompactionSettings
    from fast_agent.history.compaction import should_auto_compact

    model = "deepseek-ai/DeepSeek-V4-Flash-0731"
    llm = HuggingFaceLLM(
        context=Context(config=Settings(hf=HuggingFaceSettings(default_provider="scaleway"))),
        model=f"{model}:scaleway" if explicit_backend else model,
    )
    usage = llm.usage_accumulator
    assert usage.context_window_size == 262144
    assert not should_auto_compact(usage, CompactionSettings(), projected_context_tokens=222822)
    assert should_auto_compact(usage, CompactionSettings(), projected_context_tokens=222823)


@pytest.mark.parametrize(
    "model",
    (
        "deepseek-ai/DeepSeek-V4-Flash-0731:together",
        "deepseek-ai/DeepSeek-V4-Flash-0731:deepinfra",
        "deepseek-ai/DeepSeek-V4.1-Flash:novita",
    ),
)
def test_deepseek_route_defaults_preserve_per_call_output_override(model: str) -> None:
    from fast_agent.types import RequestParams

    llm = HuggingFaceLLM(context=Context(config=Settings()), model=model)
    params = llm.get_request_params(RequestParams(max_tokens=393216))
    request = llm._prepare_api_request([{"role": "user", "content": "hello"}], None, params)
    assert request["max_tokens"] == 393216


@pytest.mark.parametrize("model", ("kimi25", "kimi26", "glm5", "qwen35", "qwen36"))
@pytest.mark.parametrize("setting", (None, "on", "off"))
def test_routed_toggles_use_only_top_level_effort(model: str, setting: str | None) -> None:
    request = _factory_request(model + (f"?reasoning={setting}" if setting else ""))
    assert request.get("reasoning_effort") == ("none" if setting == "off" else None)
    extra = request.get("extra_body", {})
    assert isinstance(extra, dict)
    assert not {"thinking", "chat_template_kwargs", "disable_reasoning"} & (
        request.keys() | extra.keys()
    )


@pytest.mark.parametrize(
    "model",
    (
        "glm52",
        "glimmer",
        "inkling",
        "Inkling 2 Small (together)",
        "deepseek41-hf",
        "hf.zai-org/GLM-5.3:zai-org",
        "hf.zai-org/GLM-5.3-Flash:zai-org",
    ),
)
def test_routed_effort_ladders_reach_wire_without_backend_controls(model: str) -> None:
    llm = ModelFactory.create_factory(model)(agent=LlmAgent(AgentConfig(name="effort-contract")))
    spec = llm.reasoning_effort_spec
    assert spec is not None and spec.allowed_efforts and spec.default
    for setting in (None, *spec.allowed_efforts):
        request = _factory_request(model + (f"?reasoning={setting}" if setting else ""))
        expected = setting or spec.default.value
        if model == "glm52" and expected == "minimal":
            expected = "none"
        assert request["reasoning_effort"] == expected
        extra = request.get("extra_body", {})
        assert isinstance(extra, dict)
        assert not {"thinking", "chat_template_kwargs", "disable_reasoning"} & (
            request.keys() | extra.keys()
        )


@pytest.mark.parametrize("setting", ("none", "minimal", "off"))
def test_glm52_disabled_efforts_use_none(setting: str) -> None:
    request = _factory_request(f"glm52?reasoning={setting}")
    assert request["reasoning_effort"] == "none"
    assert "extra_body" not in request


@pytest.mark.parametrize(
    "model, lowest",
    (
        ("inkling", "minimal"),
        ("Inkling 2 Small (together)", "minimal"),
        ("hf.thinkingmachines/Inkling-2-Small", "minimal"),
        ("deepseek41-hf", "low"),
        ("glm53", "low"),
        ("glm53flash", "low"),
        ("kimi", "low"),
        ("glimmer", "low"),
    ),
)
@pytest.mark.parametrize("setting", ("none", "off"))
def test_mandatory_reasoning_uses_lowest_effort(model: str, lowest: str, setting: str) -> None:
    request = _factory_request(f"{model}?reasoning={setting}")
    assert request["reasoning_effort"] == lowest


def test_mandatory_reasoning_adjustment_is_logged_once(monkeypatch: pytest.MonkeyPatch) -> None:
    from fast_agent.llm.reasoning_effort import parse_reasoning_setting

    llm = ModelFactory.create_factory("inkling")(agent=LlmAgent(AgentConfig(name="mandatory")))
    assert isinstance(llm, HuggingFaceLLM)
    warnings: list[str] = []
    monkeypatch.setattr(llm.logger, "warning", warnings.append)
    for setting in ("none", "off"):
        llm.set_reasoning_effort(parse_reasoning_setting(setting))
        llm._prepare_api_request(
            [{"role": "user", "content": "hello"}], None, llm.default_request_params
        )
    assert len(warnings) == 1
    assert "cannot be disabled" in warnings[0]
    assert "minimal" in warnings[0]


def test_minimax_direct_route_has_no_reasoning_control() -> None:
    llm = ModelFactory.create_factory("minimax2.5")(agent=LlmAgent(AgentConfig(name="direct")))
    assert llm.reasoning_effort_spec is None
    for query in ("", "?reasoning=none", "?reasoning=off", "?reasoning=on"):
        request = _factory_request(f"minimax2.5{query}")
        assert "reasoning_effort" not in request
        extra = request.get("extra_body", {})
        assert isinstance(extra, dict)
        assert not {"thinking", "chat_template_kwargs", "disable_reasoning"} & extra.keys()


@pytest.mark.parametrize(
    "model",
    (
        "moonshotai/Kimi-K2.5:novita",
        "moonshotai/Kimi-K2.6:novita",
        "zai-org/GLM-5:novita",
        "zai-org/GLM-5.2:zai-org",
        "Qwen/Qwen3.5-397B-A17B:novita",
        "Qwen/Qwen3.6-35B-A3B:deepinfra",
        "meta-models/Muse-Glimmer-30B:together",
    ),
)
def test_routed_profiles_strip_legacy_controls_from_metadata(model: str) -> None:
    request = _request_with_extra_body(model)
    assert request["extra_body"] == {"preserved": True}


@pytest.mark.parametrize(
    "model, extra_body",
    (
        ("moonshotai/Kimi-K2.5", {"thinking": {"type": "disabled"}}),
        ("moonshotai/Kimi-K2.6", {"chat_template_kwargs": {"thinking": False}}),
        ("Qwen/Qwen3.5-397B-A17B", {"chat_template_kwargs": {"enable_thinking": False}}),
        ("Qwen/Qwen3.6-35B-A3B", {"chat_template_kwargs": {"enable_thinking": False}}),
        ("zai-org/GLM-5", {"disable_reasoning": True}),
        ("zai-org/GLM-5.2", {"thinking": {"type": "disabled"}}),
    ),
)
def test_dedicated_endpoints_keep_direct_reasoning_controls(
    model: str, extra_body: dict[str, object]
) -> None:
    llm = HuggingFaceLLM(
        context=Context(
            config=Settings(hf=HuggingFaceSettings(base_url="https://endpoint.test/v1"))
        ),
        model=model,
        reasoning_effort=False,
    )
    request = llm._prepare_api_request(
        [{"role": "user", "content": "hello"}], None, llm.default_request_params
    )
    assert request["extra_body"] == extra_body
    assert "reasoning_effort" not in request


@pytest.mark.parametrize("model", ("zai-org/GLM-5.3", "zai-org/GLM-5.3-Flash"))
@pytest.mark.parametrize("effort", ("low", "max"))
def test_dedicated_glm53_endpoints_keep_thinking_and_effort(model: str, effort: str) -> None:
    llm = HuggingFaceLLM(
        context=Context(
            config=Settings(hf=HuggingFaceSettings(base_url="https://endpoint.test/v1"))
        ),
        model=model,
        reasoning_effort=effort,
    )
    request = llm._prepare_api_request(
        [{"role": "user", "content": "hello"}], None, llm.default_request_params
    )
    assert request["reasoning_effort"] == effort
    assert request["extra_body"] == {"thinking": {"type": "enabled", "clear_thinking": False}}


@pytest.mark.parametrize(
    "model",
    ("hf.Qwen/Qwen3.8-27B", "hf.moonshotai/Kimi-K2.6:novita"),
)
@pytest.mark.parametrize("setting", ("none", "off"))
def test_disable_capable_hf_routes_keep_reasoning_disabled(
    model: str, setting: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fast_agent.llm.reasoning_effort import ReasoningEffortSetting, parse_reasoning_setting

    llm = ModelFactory.create_factory(model)(agent=LlmAgent(AgentConfig(name="toggle")))
    assert isinstance(llm, HuggingFaceLLM)
    warnings: list[str] = []
    monkeypatch.setattr(llm.logger, "warning", warnings.append)
    llm.set_reasoning_effort(parse_reasoning_setting(setting))
    assert llm.reasoning_effort == ReasoningEffortSetting(kind="toggle", value=False)
    assert warnings == []

    request = _factory_request(f"{model}?reasoning={setting}")
    if model.startswith("hf.Qwen/"):
        assert "reasoning_effort" not in request
        assert request["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    else:
        assert request["reasoning_effort"] == "none"
        assert "extra_body" not in request


@pytest.mark.parametrize(
    ("query", "effort"),
    (
        ("", "max"),
        ("?reasoning=minimal", "minimal"),
        ("?reasoning=low", "low"),
        ("?reasoning=medium", "medium"),
        ("?reasoning=high", "high"),
        ("?reasoning=xhigh", "xhigh"),
        ("?reasoning=max", "max"),
        ("?reasoning=none", "minimal"),
        ("?reasoning=off", "minimal"),
    ),
)
def test_beam_request_contract(query: str, effort: str) -> None:
    # Lowercase input must recover the case-sensitive HF wire identifier.
    request = _factory_request(f"hf.reflection/beam-1.0-501b-a23b:together{query}")

    assert request["model"] == "reflection/Beam-1.0-501B-A23B:together"
    assert request["reasoning_effort"] == effort
    assert "max_tokens" not in request
    assert "max_completion_tokens" not in request
    assert "extra_body" not in request


@pytest.mark.parametrize(("requested", "expected"), ((4096, 4096), (200000, 131072)))
def test_beam_explicit_output_budget_and_schema(requested: int, expected: int) -> None:
    agent = LlmAgent(AgentConfig(name="beam-contract-test"))
    llm = ModelFactory.create_factory(f"beam?max_tokens={requested}")(agent=agent)
    assert isinstance(llm, HuggingFaceLLM)
    schema = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
        "additionalProperties": False,
    }
    params = llm.default_request_params.model_copy(
        update={"response_format": llm.schema_to_response_format(schema)}
    )

    request = llm._prepare_api_request([{"role": "user", "content": "hello"}], None, params)

    assert request["max_tokens"] == expected
    assert "max_completion_tokens" not in request
    assert llm._structured_json_mode(params) == "schema"
    response_format = request["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["schema"] == schema


def test_beam_removes_obsolete_reasoning_controls() -> None:
    request = _request_with_extra_body("reflection/Beam-1.0-501B-A23B:together")

    assert request["reasoning_effort"] == "max"
    assert request["extra_body"] == {"preserved": True}


@pytest.mark.parametrize("backend", ("together", "other-backend"))
def test_beam_replays_reasoning_without_inventing_missing_history(backend: str) -> None:
    llm = HuggingFaceLLM(
        context=Context(config=Settings()), model=f"reflection/Beam-1.0-501B-A23B:{backend}"
    )
    messages = [
        PromptMessageExtended(
            role="assistant",
            content=[TextContent(type="text", text="answer")],
            channels={REASONING: [TextContent(type="text", text="Check the result.")]},
        ),
        PromptMessageExtended(
            role="assistant", content=[TextContent(type="text", text="imported")]
        ),
    ]

    converted = llm._convert_extended_messages_to_provider(messages)

    assert converted == [
        {"role": "assistant", "content": "answer", "reasoning_content": "Check the result."},
        {"role": "assistant", "content": "imported"},
    ]


# Sanitised HTTP body from fast-agent-model-lab:
# onboarding/models/beam-1/evidence/conformance-missing-reasoning-400.json
# (2026-11-16, HF router -> OpenRouter -> Nebius). Never retain the account ID.
_BEAM_MISSING_REASONING_BODY = {
    "error": {
        "message": "Provider returned error",
        "code": 400,
        "metadata": {
            "raw": '{"error":{"type":"BadRequest","message":"The assistant turn at messages[1] is missing its reasoning. Dropping reasoning from earlier turns degrades this model\'s output. Send it back in \'reasoning_content\', or set allow_missing_thinking=true to skip this check.","param":null,"code":"400"},"detail":"Invalid request. Please check the parameters and try again. Details: The assistant turn at messages[1] is missing its reasoning. Dropping reasoning from earlier turns degrades this model\'s output. Send it back in \'reasoning_content\', or set allow_missing_thinking=true to skip this check."}',
            "provider_name": "Nebius",
            "is_byok": False,
            "provider_error_code": "400",
        },
    },
    "user_id": "REDACTED",
}


class _ReplayedHuggingFaceError(HuggingFaceLLM):
    """Real SDK + completion path; only the HTTP transport is an offline replay."""

    def __init__(
        self,
        body: dict[str, object],
        *,
        model: str = "reflection/Beam-1.0-501B-A23B:together",
        inference_provider: str = "openrouter",
        base_url: str = "https://router.huggingface.co/v1",
    ) -> None:
        super().__init__(
            context=Context(config=Settings(hf=HuggingFaceSettings(base_url=base_url))),
            model=model,
        )
        self.body = body
        self.inference_provider = inference_provider
        self.requests: list[dict[str, object]] = []

    def _respond(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(json.loads(request.content))
        return httpx2.Response(
            400,
            headers={
                "content-type": "application/json",
                "x-inference-provider": self.inference_provider,
            },
            json=self.body,
        )

    def _openai_client(self) -> AsyncOpenAI:
        return AsyncOpenAI(
            api_key="unused-offline",
            base_url=self._base_url(),
            http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(self._respond)),
        )


async def _replay_hf_fatal_error(llm: _ReplayedHuggingFaceError) -> BadRequestError:
    with pytest.raises(BadRequestError) as caught:
        await llm._execute_with_retry(
            llm._apply_prompt_provider_specific,
            [
                PromptMessageExtended(role=role, content=[TextContent(type="text", text="hello")])
                for role in ("user", "assistant", "user")
            ],
        )
    assert len(llm.requests) == 1  # Neither SDK nor harness retries this fatal 400.
    assert llm.requests[0]["messages"] == [
        {"role": role, "content": "hello"} for role in ("user", "assistant", "user")
    ]  # D10: no placeholder reasoning was inserted.
    return caught.value


@pytest.mark.asyncio
@pytest.mark.parametrize("with_provider_name", (True, False))
@pytest.mark.parametrize("model", ("reflection/Beam-1.0-501B-A23B:together", "other/model:nebius"))
async def test_hf_openrouter_missing_reasoning_sdk_error_carries_upstream_detail(
    with_provider_name: bool,
    model: str,
) -> None:
    body = json.loads(json.dumps(_BEAM_MISSING_REASONING_BODY))
    if not with_provider_name:
        body["error"]["metadata"].pop("provider_name")
    llm = _ReplayedHuggingFaceError(body, model=model)
    error = await _replay_hf_fatal_error(llm)

    assert error.status_code == 400
    assert error.body == body["error"]  # SDK unwraps the HTTP error envelope.
    assert "Error code: 400 - Provider returned error" in str(error)
    assert "The assistant turn at messages[1] is missing its reasoning." in str(error)
    assert "Send it back in 'reasoning_content'" in str(error)
    assert ("Nebius" in str(error)) is with_provider_name
    assert "user_id" not in str(error)
    assert "REDACTED" not in str(error)
    assert str(error) == error.message


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("body", "inference_provider", "base_url", "expected"),
    (
        (
            {"error": {"message": "Invalid request", "type": "invalid_request_error"}},
            "nebius",
            "https://router.huggingface.co/v1",
            "Error code: 400 - Invalid request",
        ),
        (
            {
                "error": {
                    "message": "Provider returned error",
                    "metadata": {"provider_name": "Nebius"},
                }
            },
            "openrouter",
            "https://router.huggingface.co/v1",
            "Error code: 400 - Provider returned error",
        ),
        (
            _BEAM_MISSING_REASONING_BODY,
            "nebius",
            "https://router.huggingface.co/v1",
            "Error code: 400 - Provider returned error",
        ),
        (
            _BEAM_MISSING_REASONING_BODY,
            "openrouter",
            "https://dedicated-endpoint.test/v1",
            "Error code: 400 - Provider returned error",
        ),
    ),
)
async def test_hf_other_provider_errors_keep_exact_sdk_message(
    body: dict[str, object], inference_provider: str, base_url: str, expected: str
) -> None:
    llm = _ReplayedHuggingFaceError(body, inference_provider=inference_provider, base_url=base_url)
    error = await _replay_hf_fatal_error(llm)
    assert str(error) == error.message == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("raw", "detail"),
    (
        ("Upstream request rejected", "Upstream request rejected"),
        (" Upstream\nrequest\r\nrejected ", "Upstream request rejected"),
        (
            '{"message":"Upstream request rejected","user_id":"REDACTED"}',
            "Upstream request rejected",
        ),
        ('"Upstream request rejected"', "Upstream request rejected"),
        ("", None),
        ("   ", None),
        (None, None),
        (42, None),
        ('{"user_id":"REDACTED"}', None),
        ('{"error":{"message":null},"user_id":"REDACTED"}', None),
        ("[]", None),
    ),
)
async def test_hf_openrouter_raw_detail_is_additive_and_allowlisted(
    raw: object, detail: str | None
) -> None:
    llm = _ReplayedHuggingFaceError(
        {"error": {"message": "Provider returned error", "metadata": {"raw": raw}}}
    )
    error = await _replay_hf_fatal_error(llm)
    expected = "Error code: 400 - Provider returned error"
    if detail is not None:
        expected += f" — {detail}"
    assert str(error) == error.message == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raw",
    (
        "[" * 10000 + "0" + "]" * 10000,  # RecursionError from valid, nested JSON.
        "9" * 50000,  # ValueError from the JSON integer conversion limit.
    ),
    ids=("nested-json", "oversized-integer"),
)
async def test_hf_openrouter_raw_parse_failure_preserves_fatal_400(raw: str) -> None:
    body: dict[str, object] = {
        "error": {"message": "Provider returned error", "metadata": {"raw": raw}},
        "user_id": "REDACTED",
    }
    error = await _replay_hf_fatal_error(_ReplayedHuggingFaceError(body))

    assert error.status_code == 400
    assert error.body == body["error"]
    assert str(error) == error.message == "Error code: 400 - Provider returned error"


@pytest.mark.asyncio
@pytest.mark.parametrize("json_detail", (True, False), ids=("json", "plain-text"))
@pytest.mark.parametrize("length", (399, 400, 401, 50000))
@pytest.mark.parametrize("provider", (None, "Nebius\nupstream", "P" * 1000))
async def test_hf_openrouter_upstream_detail_is_single_line_and_bounded(
    json_detail: bool, length: int, provider: str | None
) -> None:
    text = "\n " + "x" * length + "\r\n"
    raw = json.dumps({"error": {"message": text}, "user_id": "REDACTED"}) if json_detail else text
    body: dict[str, object] = {
        "error": {
            "message": "Provider returned error",
            "metadata": {"raw": raw, "provider_name": provider},
        },
        "user_id": "REDACTED",
    }
    error = await _replay_hf_fatal_error(_ReplayedHuggingFaceError(body))

    prefix = "Error code: 400 - Provider returned error — "
    assert error.status_code == 400
    assert error.body == body["error"]
    assert str(error) == error.message
    assert str(error).startswith(prefix)
    detail = str(error).removeprefix(prefix)
    expected = (" ".join(provider.split()) + ": " if provider else "") + "x" * length
    assert detail == (expected if len(expected) <= 400 else expected[:399] + "…")
    assert len(detail) <= 400
    assert "\n" not in detail and "\r" not in detail
    assert "user_id" not in str(error) and "REDACTED" not in str(error)


@pytest.mark.asyncio
async def test_beam_streamed_reasoning_replays_on_tool_turn() -> None:
    model = "reflection/Beam-1.0-501B-A23B:together"
    llm = HuggingFaceLLM(context=Context(config=Settings()), model=model)
    # Synthetic fragments matching the contract's ordinary OpenAI stream shape.
    deltas = [
        {"role": "assistant"},
        {"reasoning": "Check the result."},
        {
            "tool_calls": [
                {
                    "index": 0,
                    "id": "beam-tool",
                    "type": "function",
                    "function": {"name": "check", "arguments": "{}"},
                }
            ]
        },
    ]
    chunks = [
        ChatCompletionChunk.model_validate(
            {
                "id": "beam-chunk",
                "created": 0,
                "model": model,
                "object": "chat.completion.chunk",
                "choices": [
                    {
                        "index": 0,
                        "delta": delta,
                        "finish_reason": "tool_calls" if index == 2 else None,
                    }
                ],
            }
        )
        for index, delta in enumerate(deltas)
    ]

    completion, reasoning = await llm._process_stream(_stream_chunks(chunks), model)

    assert reasoning == ["Check the result."]
    choice = completion.choices[0]
    stop = await llm._openai_stop_result(choice, choice.message)
    assistant = PromptMessageExtended(
        role="assistant",
        tool_calls=stop.requested_tool_calls,
        channels={REASONING: [TextContent(type="text", text=reasoning[0])]},
    )
    converted = llm._convert_extended_messages_to_provider([assistant])
    assert converted == [
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "Check the result.",
            "tool_calls": [
                {
                    "id": "beam-tool",
                    "type": "function",
                    "function": {"name": "check", "arguments": "{}"},
                }
            ],
        }
    ]
