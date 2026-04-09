"""
Regression tests for ACP maxTokens merging and model switching.

Testing notes:

- This module owns the ACP-specific regression around RequestParams default
  values becoming explicit during dump/recreate flows and accidentally
  overriding model-aware maxTokens.
- Prefer a few end-to-end seam tests (initialize LLM, attach through factory,
  merge ACP request params, switch models) over restating ModelDatabase rows.
- Generic ModelDatabase capability lookups belong in test_model_database.py;
  alias resolution semantics belong in test_model_factory.py.
"""

from typing import TypeGuard

import pytest

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.llm_agent import LlmAgent
from fast_agent.config import HuggingFaceSettings, Settings
from fast_agent.context import Context
from fast_agent.interfaces import FastAgentLLMProtocol
from fast_agent.llm.fastagent_llm import FastAgentLLM
from fast_agent.llm.model_factory import ModelFactory
from fast_agent.llm.provider.openai.llm_huggingface import HuggingFaceLLM
from fast_agent.types import RequestParams

EXPECTED_KIMI_MAX_TOKENS = 16_384


def _is_fastagent_llm(value: FastAgentLLMProtocol) -> TypeGuard[FastAgentLLM]:
    return isinstance(value, FastAgentLLM)


def _make_hf_llm(model: str) -> HuggingFaceLLM:
    settings = Settings(hf=HuggingFaceSettings())
    context = Context(config=settings)
    return HuggingFaceLLM(context=context, model=model, name="test-agent")


@pytest.mark.parametrize(
    ("params", "expected_dump"),
    [
        (RequestParams(systemPrompt="test prompt"), {"systemPrompt": "test prompt"}),
        (RequestParams(systemPrompt="test", maxTokens=8192), {"systemPrompt": "test", "maxTokens": 8192}),
    ],
)
def test_request_params_exclude_unset_tracks_only_explicit_max_tokens(
    params: RequestParams, expected_dump: dict[str, object]
) -> None:
    assert params.model_dump(exclude_unset=True) == expected_dump


def test_request_params_dump_recreate_turns_default_max_tokens_into_explicit_field() -> None:
    """
    Document the bug shape that originally caused ACP to reset model-aware limits.

    A RequestParams created with only systemPrompt has the class default
    maxTokens=2048, but that value is *unset*. Dumping and recreating the model
    turns maxTokens into an explicitly set field, so later merges may override
    the model-aware value unless maxTokens is excluded.
    """

    original = RequestParams(systemPrompt="test")
    recreated = RequestParams(**original.model_dump(exclude={"model"}))

    assert "maxTokens" in recreated.model_dump(exclude_unset=True)
    assert recreated.maxTokens == 2048


@pytest.mark.parametrize(
    "model",
    [
        "moonshotai/kimi-k2-instruct-0905",
        "moonshotai/Kimi-K2-Instruct-0905",
    ],
)
def test_huggingface_llm_initialization_uses_model_aware_max_tokens(model: str) -> None:
    llm = _make_hf_llm(model)

    assert llm.default_request_params.maxTokens == EXPECTED_KIMI_MAX_TOKENS


def test_acp_request_param_merge_preserves_model_aware_max_tokens() -> None:
    llm = _make_hf_llm("moonshotai/kimi-k2-instruct-0905")

    merged = llm.get_request_params(RequestParams(systemPrompt="Updated system prompt for ACP session"))

    assert merged.systemPrompt == "Updated system prompt for ACP session"
    assert merged.maxTokens == EXPECTED_KIMI_MAX_TOKENS


def test_acp_request_param_merge_allows_explicit_max_tokens_override() -> None:
    llm = _make_hf_llm("moonshotai/kimi-k2-instruct-0905")

    merged = llm.get_request_params(RequestParams(systemPrompt="test", maxTokens=4096))

    assert merged.maxTokens == 4096


@pytest.mark.parametrize(
    "model_spec",
    [
        "hf.moonshotai/kimi-k2-instruct-0905",
        "kimi",
        "kimithink",
    ],
)
@pytest.mark.asyncio
async def test_attach_llm_paths_preserve_model_aware_max_tokens(model_spec: str) -> None:
    agent = LlmAgent(AgentConfig(name="Test Agent"))

    llm = await agent.attach_llm(ModelFactory.create_factory(model_spec))
    assert _is_fastagent_llm(llm)

    assert llm.default_request_params.maxTokens == EXPECTED_KIMI_MAX_TOKENS


@pytest.mark.asyncio
async def test_apply_model_flow_recomputes_model_aware_max_tokens_after_dump_recreate() -> None:
    """
    Simulate the apply_model flow, which preserves user params but must drop
    model-specific fields so the newly selected model can restore its own limits.
    """

    original_params = RequestParams(systemPrompt="original prompt")
    recreated_params = RequestParams(
        **original_params.model_dump(exclude={"model", "maxTokens"})
    )
    agent = LlmAgent(AgentConfig(name="Test Agent"))

    llm = await agent.attach_llm(ModelFactory.create_factory("kimi"), request_params=recreated_params)
    assert _is_fastagent_llm(llm)

    assert llm.default_request_params.systemPrompt == "original prompt"
    assert llm.default_request_params.maxTokens == EXPECTED_KIMI_MAX_TOKENS


@pytest.mark.asyncio
async def test_attach_llm_then_acp_merge_preserves_model_aware_max_tokens() -> None:
    """Full seam smoke test for the reported user-visible ACP failure path."""

    agent = LlmAgent(AgentConfig(name="Test Agent"))
    llm = await agent.attach_llm(ModelFactory.create_factory("hf.moonshotai/kimi-k2-instruct-0905"))
    assert _is_fastagent_llm(llm)

    merged = llm.get_request_params(RequestParams(systemPrompt="Updated for ACP session"))

    assert merged.systemPrompt == "Updated for ACP session"
    assert merged.maxTokens == EXPECTED_KIMI_MAX_TOKENS
