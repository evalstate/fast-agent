from collections.abc import AsyncIterator

import pytest
from mcp.types import TextContent
from openai.types.chat import ChatCompletionChunk

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.llm_agent import LlmAgent
from fast_agent.config import HuggingFaceSettings, Settings
from fast_agent.constants import REASONING
from fast_agent.context import Context
from fast_agent.llm.model_factory import ModelFactory
from fast_agent.llm.provider.openai.llm_huggingface import HuggingFaceLLM
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
    ("alias", "wire_model", "max_tokens"),
    (
        (
            "DeepSeek V4 Flash 0731 (baseten)",
            "deepseek-ai/DeepSeek-V4-Flash-0731:baseten",
            384_000,
        ),
        (
            "DeepSeek V4 Flash 0731 (deepinfra)",
            "deepseek-ai/DeepSeek-V4-Flash-0731:deepinfra",
            131_072,
        ),
    ),
)
def test_deepseek_picker_aliases_apply_route_profiles(
    alias: str,
    wire_model: str,
    max_tokens: int,
) -> None:
    request = _factory_request(alias)

    assert request["model"] == wire_model
    assert request["max_tokens"] == max_tokens
    assert request["reasoning_effort"] == "max"


@pytest.mark.parametrize(
    "alias",
    (
        "DeepSeek V4 Flash 0731 (baseten)",
        "DeepSeek V4 Flash 0731 (deepinfra)",
    ),
)
def test_deepseek_picker_aliases_apply_reasoning_override(alias: str) -> None:
    request = _factory_request(f"{alias}?reasoning=none")

    assert request["reasoning_effort"] == "none"


@pytest.mark.parametrize(
    ("model", "reasoning_strength"),
    (
        ("glimmer", "high"),
        ("glimmer?reasoning=low", "low"),
        ("glimmer?reasoning=xhigh", "xhigh"),
    ),
)
def test_muse_glimmer_together_applies_chat_template_contract(
    model: str,
    reasoning_strength: str,
) -> None:
    request = _factory_request(model)
    extra_body = request.get("extra_body")

    assert request["model"] == "meta-models/Muse-Glimmer-30B:together"
    assert request["temperature"] == 1.0
    assert request["top_p"] == 0.95
    assert "max_tokens" not in request
    assert "reasoning_effort" not in request
    assert isinstance(extra_body, dict)
    assert extra_body == {
        "top_k": 64,
        "chat_template_kwargs": {
            "reasoning_strength": reasoning_strength,
        },
    }


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
            {"preserved", "thinking", "chat_template_kwargs"},
        ),
        (
            "moonshotai/Kimi-K3:together",
            {"preserved"},
        ),
        (
            "zai-org/GLM-5.2:deepinfra",
            {"preserved", "chat_template_kwargs"},
        ),
        (
            "google/gemma-4-31B-it:cerebras",
            {"preserved", "thinking"},
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
@pytest.mark.parametrize("backend", ("", ":together", ":deepinfra"))
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
    assert request["extra_body"] == {"thinking": {"type": "enabled", "clear_thinking": False}}


@pytest.mark.parametrize("model", ("GLM-5.3", "GLM-5.3-Flash"))
def test_glm_53_hf_keeps_required_reasoning_enabled(model: str) -> None:
    request = _factory_request(f"hf.zai-org/{model}:together?reasoning=none")

    assert request["reasoning_effort"] == "max"


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
        (":fireworks-ai", "?reasoning=off", "none"),
        (":other-backend", "?reasoning=none", "none"),
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
    "alias", ("deepseek41-hf", "deepseek-v41-hf", "DeepSeek V4.1 Flash (novita)")
)
def test_deepseek_v41_hf_alias_request(alias: str) -> None:
    request = _factory_request(f"{alias}?reasoning=off&max_tokens=128")

    assert request["model"] == "deepseek-ai/DeepSeek-V4.1-Flash:novita"
    assert request["reasoning_effort"] == "none"
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
