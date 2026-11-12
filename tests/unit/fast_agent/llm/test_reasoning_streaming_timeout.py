"""Automatic timeout policy and explicit overrides, without provider calls."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from mcp_types import TextContent, Tool

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.llm_agent import LlmAgent
from fast_agent.config import GoogleSettings, Settings
from fast_agent.constants import OPENAI_COMPACTION_ITEMS
from fast_agent.context import Context
from fast_agent.llm.model_factory import ModelFactory
from fast_agent.llm.provider.copilot.responses import CopilotResponsesLLM
from fast_agent.llm.provider.google.llm_google_native import GoogleNativeLLM
from fast_agent.llm.provider.openai.compaction_replay import capture_compaction_replay
from fast_agent.llm.reasoning_effort import ReasoningEffortSetting
from fast_agent.llm.request_param_resolution import (
    apply_reasoning_streaming_timeout,
    merge_request_params,
)
from fast_agent.mcp.prompt import Prompt
from fast_agent.session import SessionHydrator, SessionManager
from fast_agent.types import PromptMessageExtended, RequestParams

if TYPE_CHECKING:
    from pathlib import Path


async def _agent(
    model: str,
    params: RequestParams | None = None,
    *,
    defaults: RequestParams | None = None,
) -> LlmAgent:
    agent = LlmAgent(
        AgentConfig("test", default_request_params=defaults), context=Context(config=Settings())
    )
    await agent.attach_llm(ModelFactory.create_factory(model), request_params=params)
    return agent


@pytest.mark.asyncio
@pytest.mark.parametrize("effort", ["medium", "high", "xhigh", "max"])
async def test_effective_named_reasoning_sets_automatic_timeout(effort: str) -> None:
    agent = await _agent(f"copilot.gpt-6.1-sol?reasoning={effort}")
    assert agent.llm is not None
    resolved = agent.llm.get_request_params(RequestParams(temperature=0.2))
    assert resolved.streaming_timeout == (150.0 if effort == "medium" else 300.0)
    assert "streaming_timeout" not in resolved.model_fields_set
    assert "streaming_timeout" not in agent.llm.default_request_params.model_fields_set


@pytest.mark.asyncio
async def test_reasoning_changes_recompute_timeout_and_keep_defaults_unchanged() -> None:
    agent = await _agent("responses.gpt-5.5?reasoning=high")
    assert agent.llm is not None
    llm = agent.llm
    defaults = llm.default_request_params.model_copy(deep=True)
    high = llm.get_request_params()
    assert high.streaming_timeout == 300.0
    assert llm.get_request_params(high).streaming_timeout == 300.0
    llm.set_reasoning_effort(ReasoningEffortSetting(kind="effort", value="low"))
    assert llm.get_request_params(high).streaming_timeout == 150.0
    llm.set_reasoning_effort(ReasoningEffortSetting(kind="effort", value="max"))
    assert llm.get_request_params().streaming_timeout == 300.0
    assert llm.default_request_params == defaults
    assert llm.default_request_params.model_fields_set == defaults.model_fields_set


@pytest.mark.asyncio
async def test_model_default_reasoning_is_used_after_reset() -> None:
    agent = await _agent("xai.grok-4.7")
    assert agent.llm is not None
    llm = agent.llm
    assert llm.get_request_params().streaming_timeout == 300.0
    llm.set_reasoning_effort(ReasoningEffortSetting(kind="effort", value="low"))
    assert llm.get_request_params().streaming_timeout == 150.0
    llm.set_reasoning_effort(None)
    assert llm.get_request_params().streaming_timeout == 300.0


@pytest.mark.parametrize(
    "reasoning",
    [
        None,
        ReasoningEffortSetting(kind="effort", value="auto"),
        ReasoningEffortSetting(kind="effort", value="none"),
        ReasoningEffortSetting(kind="effort", value="minimal"),
        ReasoningEffortSetting(kind="effort", value="low"),
        ReasoningEffortSetting(kind="budget", value=32768),
        ReasoningEffortSetting(kind="toggle", value=True),
        ReasoningEffortSetting(kind="toggle", value=False),
    ],
)
def test_other_reasoning_settings_keep_base_timeout(
    reasoning: ReasoningEffortSetting | None,
) -> None:
    assert apply_reasoning_streaming_timeout(RequestParams(), reasoning).streaming_timeout == 150.0


@pytest.mark.parametrize("timeout", [None, 45.0, 150.0, 600.0])
@pytest.mark.parametrize("source", ["default", "request"])
def test_timeout_merge_preserves_explicit_provenance(timeout: float | None, source: str) -> None:
    implicit = RequestParams(temperature=0.2)
    explicit = RequestParams(streaming_timeout=timeout)
    resolved = merge_request_params(
        explicit if source == "default" else implicit,
        implicit if source == "default" else explicit,
    )
    assert resolved.streaming_timeout == timeout
    assert "streaming_timeout" in resolved.model_fields_set
    assert (
        apply_reasoning_streaming_timeout(
            resolved, ReasoningEffortSetting(kind="effort", value="high")
        ).streaming_timeout
        == timeout
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout", [None, 45.0, 150.0, 600.0])
async def test_timeout_overrides_win_at_attachment_and_per_request(timeout: float | None) -> None:
    agent = await _agent(
        "responses.gpt-5.5?reasoning=high&streaming_timeout=75",
        RequestParams(streaming_timeout=timeout),
        defaults=RequestParams(streaming_timeout=90),
    )
    assert agent.llm is not None
    llm = agent.llm
    assert llm.get_request_params().streaming_timeout == timeout
    assert llm.get_request_params(RequestParams(temperature=0.2)).streaming_timeout == timeout
    assert llm.get_request_params(RequestParams(streaming_timeout=30)).streaming_timeout == 30
    assert llm.get_request_params().streaming_timeout == timeout


@pytest.mark.asyncio
@pytest.mark.parametrize(("query", "timeout"), [("none", None), ("150", 150.0), ("45", 45.0)])
async def test_model_timeout_override_wins_after_reasoning_changes(
    query: str, timeout: float | None
) -> None:
    agent = await _agent(f"copilot.gpt-6.1-sol?reasoning=max&streaming_timeout={query}")
    assert agent.llm is not None
    assert agent.llm.get_request_params().streaming_timeout == timeout
    agent.llm.set_reasoning_effort(ReasoningEffortSetting(kind="effort", value="medium"))
    assert agent.llm.get_request_params(RequestParams(temperature=0.2)).streaming_timeout == timeout


@pytest.mark.parametrize("timeout", [None, 45.0, 150.0])
def test_google_flex_preserves_explicit_timeout(timeout: float | None) -> None:
    llm = GoogleNativeLLM(
        context=Context(config=Settings(google=GoogleSettings(api_key="test-key"))),
        model="gemini-3.8-flash",
        reasoning_effort="high",
    )
    params = llm.get_request_params(RequestParams(service_tier="flex", streaming_timeout=timeout))
    assert params.streaming_timeout == timeout


def test_google_flex_default_does_not_pin_reused_request_params() -> None:
    llm = GoogleNativeLLM(
        context=Context(config=Settings(google=GoogleSettings(api_key="test-key"))),
        model="gemini-3.8-flash",
        reasoning_effort="high",
    )
    params = llm.get_request_params(RequestParams(service_tier="flex"))
    assert params.streaming_timeout == 900.0
    assert "streaming_timeout" not in params.model_fields_set
    standard = params.model_copy(update={"service_tier": None})
    assert llm.get_request_params(standard).streaming_timeout == 300.0
    llm.set_reasoning_effort(ReasoningEffortSetting(kind="effort", value="low"))
    assert llm.get_request_params(standard).streaming_timeout == 150.0


class _CheckpointSimulator(CopilotResponsesLLM):
    def __init__(self, params: RequestParams | None) -> None:
        super().__init__(
            context=Context(config=Settings()),
            model="gpt-6.1-sol",
            reasoning_effort="max",
            request_params=params,
        )
        self.timeouts: list[float | None] = []

    async def _apply_prompt_provider_specific(
        self,
        multipart_messages: list[PromptMessageExtended],
        request_params: RequestParams | None = None,
        tools: list[Tool] | None = None,
        is_template: bool = False,
    ) -> PromptMessageExtended:
        assert request_params is not None
        self.timeouts.append(request_params.streaming_timeout)
        envelope = capture_compaction_replay(
            {"type": "compaction", "id": "checkpoint", "encrypted_content": "opaque"},
            origin="copilot",
        )
        assert envelope is not None
        return PromptMessageExtended(
            role="assistant",
            content=[],
            channels={
                OPENAI_COMPACTION_ITEMS: [TextContent(type="text", text=json.dumps(envelope))]
            },
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "params", [None, RequestParams(streaming_timeout=45), RequestParams(streaming_timeout=None)]
)
async def test_native_checkpoint_inherits_effective_timeout(params: RequestParams | None) -> None:
    llm = _CheckpointSimulator(params)
    await llm.native_compaction([Prompt.user("context")])
    assert llm.timeouts == [300.0 if params is None else params.streaming_timeout]


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout", ["automatic", None, 45.0, 150.0])
@pytest.mark.parametrize("legacy", [False, True])
async def test_session_round_trip_preserves_timeout_policy(
    tmp_path: Path, timeout: str | float | None, legacy: bool
) -> None:
    params = None if timeout == "automatic" else RequestParams(streaming_timeout=timeout)
    agent = await _agent("responses.gpt-5.5?reasoning=high", params)
    agent.load_message_history([Prompt.user("saved context")])
    manager = SessionManager(
        cwd=tmp_path, home_override=tmp_path / ".fast-agent", respect_env_override=False
    )
    try:
        session = manager.create_session()
        manager.set_current_session(session)
        await session.save_history(agent, agent_registry={"test": agent})
        snapshot = session.load_snapshot().continuation.agents["test"].request_settings
        assert snapshot is not None
        assert snapshot.streaming_timeout_configured is (timeout != "automatic")
        if legacy:
            # Old snapshots cannot distinguish automatic values from explicit overrides.
            path = session.directory / "session.json"
            payload = json.loads(path.read_text())
            del payload["continuation"]["agents"]["test"]["request_settings"][
                "streaming_timeout_configured"
            ]
            path.write_text(json.dumps(payload))
        restored = await _agent("responses.gpt-5.5", RequestParams(streaming_timeout=75))
        await SessionHydrator().hydrate_session(
            session=session, agents={"test": restored}, fallback_agent_name="test"
        )
        assert restored.llm is not None
        assert restored.llm.get_request_params().streaming_timeout == (
            (150.0 if legacy else 300.0) if timeout == "automatic" else timeout
        )
        restored.llm.set_reasoning_effort(ReasoningEffortSetting(kind="effort", value="low"))
        assert restored.llm.get_request_params().streaming_timeout == (
            150.0 if timeout == "automatic" else timeout
        )
        await restored.set_model("responses.gpt-5.5?reasoning=high")
        assert restored.llm is not None
        assert restored.llm.get_request_params().streaming_timeout == (
            (150.0 if legacy else 300.0) if timeout == "automatic" else timeout
        )
    finally:
        manager.close()
