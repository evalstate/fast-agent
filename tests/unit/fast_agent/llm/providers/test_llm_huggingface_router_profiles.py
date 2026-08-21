from collections.abc import AsyncIterator

import pytest
from openai.types.chat import ChatCompletionChunk

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.llm_agent import LlmAgent
from fast_agent.config import HuggingFaceSettings, Settings
from fast_agent.context import Context
from fast_agent.llm.model_factory import ModelFactory
from fast_agent.llm.provider.openai.llm_huggingface import HuggingFaceLLM


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
            393_216,
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
    assert "reasoning_effort" not in request
    assert isinstance(extra_body, dict)
    assert extra_body == {
        "top_k": 64,
        "chat_template_kwargs": {
            "reasoning_strength": reasoning_strength,
        },
    }


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
def test_deepseek_unprofiled_route_does_not_invent_wire_contract(model: str) -> None:
    request = _factory_request(model)

    assert "reasoning_effort" not in request


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
def test_equivalent_huggingface_router_urls_do_not_use_custom_profile(
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
    assert "reasoning_effort" not in request


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
    assert "reasoning_effort" not in request


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
