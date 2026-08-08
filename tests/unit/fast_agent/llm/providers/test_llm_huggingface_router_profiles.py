import pytest

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


def test_deepseek_unprofiled_route_does_not_invent_wire_contract() -> None:
    request = _factory_request("hf.deepseek-ai/DeepSeek-V4-Flash-0731:together")

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
