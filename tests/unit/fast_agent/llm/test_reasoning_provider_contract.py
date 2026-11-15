"""HF disable fallback must not change other providers' reasoning contracts."""

import pytest

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.llm_agent import LlmAgent
from fast_agent.commands.model_capabilities import resolve_reasoning_effort
from fast_agent.config import Settings
from fast_agent.context import Context
from fast_agent.llm.model_factory import ModelFactory
from fast_agent.llm.provider_types import Provider
from fast_agent.llm.reasoning_effort import format_reasoning_setting


@pytest.mark.parametrize(
    "model, none_expected, off_expected",
    [
        # Captured from 06bc7293e through ModelFactory, not reconstructed from specs.
        pytest.param("claude-sonnet-5-5", "effort=high", "disabled", id="anthropic-toggle"),
        pytest.param("claude-fable-5", "auto", "auto", id="anthropic-always-on"),
        pytest.param("gpt-6-astra", "effort=medium", "effort=medium", id="openai-responses"),
        pytest.param("gemini-2.5-pro", "auto", "disabled", id="google-budget-toggle"),
        pytest.param(
            "gemini-3-flash-preview", "effort=medium", "effort=medium", id="google-effort"
        ),
        pytest.param("grok-4.7", "effort=high", "effort=high", id="xai-responses"),
        pytest.param("zai.glm-5.3", "effort=max", "effort=max", id="zai-native-not-hf"),
        pytest.param(
            "moonshot.kimi-k3.1-pro", "effort=max", "effort=max", id="moonshot-native-not-hf"
        ),
    ],
)
@pytest.mark.parametrize("requested", ("none", "off"))
def test_non_hf_disable_resolution_keeps_parent_contract(
    model: str, none_expected: str, off_expected: str, requested: str
) -> None:
    llm = ModelFactory.create_factory(f"{model}?reasoning={requested}")(
        agent=LlmAgent(AgentConfig(name="reasoning-contract")),
        context=Context(config=Settings()),
    )

    assert llm.provider is not Provider.HUGGINGFACE
    expected = none_expected if requested == "none" else off_expected
    assert format_reasoning_setting(resolve_reasoning_effort(llm)) == expected
