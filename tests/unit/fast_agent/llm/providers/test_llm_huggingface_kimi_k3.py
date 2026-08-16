from types import SimpleNamespace

import pytest
from openai.types.completion_usage import CompletionUsage

from fast_agent.config import Settings
from fast_agent.context import Context
from fast_agent.llm.model_database import ModelDatabase
from fast_agent.llm.model_factory import ModelFactory
from fast_agent.llm.provider.openai.llm_huggingface import HuggingFaceLLM
from fast_agent.llm.provider_types import Provider
from fast_agent.llm.reasoning_effort import ReasoningEffortLevel, ReasoningEffortSetting


def _request(
    provider: str,
    effort: ReasoningEffortLevel | None = None,
) -> dict[str, object]:
    kwargs: dict[str, object] = {}
    if effort is not None:
        kwargs["reasoning_effort"] = ReasoningEffortSetting(kind="effort", value=effort)
    llm = HuggingFaceLLM(
        context=Context(config=Settings()),
        model=f"moonshotai/Kimi-K3:{provider}",
        **kwargs,
    )
    return llm._prepare_api_request(
        [{"role": "user", "content": "hello"}],
        None,
        llm.default_request_params,
    )


def _llm(provider: str) -> HuggingFaceLLM:
    return HuggingFaceLLM(
        context=Context(config=Settings()),
        model=f"moonshotai/Kimi-K3:{provider}",
    )


@pytest.mark.parametrize("provider", ["fireworks-ai", "together"])
def test_hf_kimi_k3_profile_is_image_only(provider: str) -> None:
    parsed = ModelFactory.parse_model_string(f"hf.moonshotai/Kimi-K3:{provider}")
    params = ModelDatabase.get_model_params(parsed.model_name, provider=parsed.provider)

    assert parsed.provider is Provider.HUGGINGFACE
    assert params is not None
    assert params.context_window == 1_048_576
    assert params.max_output_tokens == 131_072
    assert params.reasoning == "reasoning_content"
    assert params.stream_mode == "manual"
    assert params.default_provider is Provider.HUGGINGFACE
    assert params.shell_output_byte_limit == 16_000
    assert "image/png" in params.tokenizes
    assert "video/mp4" not in params.tokenizes


@pytest.mark.parametrize("provider", ["fireworks-ai", "together"])
@pytest.mark.parametrize("effort", ["low", "high", "max"])
def test_hf_kimi_k3_sends_top_level_reasoning_effort(
    provider: str,
    effort: ReasoningEffortLevel,
) -> None:
    request = _request(provider, effort)

    assert request["model"] == f"moonshotai/Kimi-K3:{provider}"
    assert request["reasoning_effort"] == effort
    assert request["max_tokens"] == 131_072
    assert "max_completion_tokens" not in request
    assert "extra_body" not in request


@pytest.mark.parametrize("provider", ["fireworks-ai", "together"])
def test_hf_kimi_k3_defaults_to_max_reasoning(provider: str) -> None:
    assert _request(provider)["reasoning_effort"] == "max"


@pytest.mark.parametrize("provider", ["fireworks-ai", "together"])
def test_hf_usage_attribution_preserves_upstream_provider(provider: str) -> None:
    llm = _llm(provider)
    arguments = llm._prepare_api_request(
        [{"role": "user", "content": "hello"}],
        None,
        llm.default_request_params,
    )

    model, upstream_provider = llm._resolve_usage_attribution(
        "moonshotai/Kimi-K3",
        arguments,
    )

    assert model == "moonshotai/Kimi-K3"
    assert upstream_provider == provider

    response = SimpleNamespace(
        usage=CompletionUsage(
            completion_tokens=20,
            prompt_tokens=100,
            total_tokens=120,
        )
    )
    llm._track_openai_response_usage(
        response,
        model,
        upstream_provider=upstream_provider,
    )
    turn = llm.usage_accumulator.turns[-1]

    assert turn.provider is Provider.HUGGINGFACE
    assert turn.model == "moonshotai/Kimi-K3"
    assert turn.upstream_provider == provider
