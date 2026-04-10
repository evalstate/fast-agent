from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, cast

from fast_agent.agents.workflow.parallel_agent import ParallelAgent
from fast_agent.ui.attachment_indicator import DraftAttachmentSummary
from fast_agent.ui.prompt.attachment_tokens import build_local_attachment_token
from fast_agent.ui.prompt.input_toolbar import (
    ToolbarAgentState,
    ToolbarRenderCache,
    _build_middle_segment,
    _resolve_attachment_summary,
    _resolve_toolbar_agent_state_cached,
    _should_resolve_attachment_summary,
)

if TYPE_CHECKING:
    from fast_agent.core.agent_app import AgentApp


@dataclass
class _StubMessage:
    role: str
    channels: dict = field(default_factory=dict)


@dataclass
class _StubConfig:
    model: str | None = None
    default_request_params: object | None = None


@dataclass
class _StubAgent:
    config: _StubConfig
    message_history: list[_StubMessage]
    usage_accumulator: object | None = None
    _llm: object | None = None


@dataclass
class _StubLlm:
    model_name: str | None = None
    resolved_model: object | None = None


class _StubAgentProvider:
    def __init__(self, agent: object) -> None:
        self._stub_agent = agent

    def _agent(self, agent_name: str | None) -> object:
        del agent_name
        return self._stub_agent


def test_build_middle_segment_prefixes_overlay_models() -> None:
    middle = _build_middle_segment(
        ToolbarAgentState(
            model_display="haikutiny",
            is_overlay_model=True,
            turn_count=3,
        ),
        shortcut_text="",
    )

    assert "▼haikutiny" in middle


def test_build_middle_segment_prefixes_codex_before_overlay() -> None:
    middle = _build_middle_segment(
        ToolbarAgentState(
            model_display="gpt-5-codex",
            is_codex_responses_model=True,
            is_overlay_model=True,
            turn_count=3,
        ),
        shortcut_text="",
    )

    assert "∞gpt-5-codex" in middle
    assert "▼gpt-5-codex" not in middle


def test_build_middle_segment_renders_attachment_indicator() -> None:
    middle = _build_middle_segment(
        ToolbarAgentState(
            model_display="gpt-4.1",
            model_name="gpt-4.1",
            model_gauges="RG",
            tdv_segment="TVD",
            service_tier_indicator="FAST",
            web_search_indicator="WEB",
            turn_count=3,
        ),
        shortcut_text="",
        attachment_summary=DraftAttachmentSummary(
            count=2,
            mime_types=("image/png",),
            any_questionable=False,
        ),
    )

    assert "▲2" in middle
    assert middle.index("TVD") < middle.index("▲2") < middle.index("RG") < middle.index("gpt-4.1")
    assert middle.index("gpt-4.1") < middle.index("FAST") < middle.index("WEB")


def test_should_resolve_attachment_summary_only_for_attachment_tokens() -> None:
    assert not _should_resolve_attachment_summary("hello world")
    assert not _should_resolve_attachment_summary("^server:resource")
    assert _should_resolve_attachment_summary("^file:/tmp/example.txt")
    assert _should_resolve_attachment_summary("look ^url:https://example.com")


def test_toolbar_agent_state_cache_hits_until_history_changes() -> None:
    agent = _StubAgent(
        config=_StubConfig(model="haiku"),
        message_history=[_StubMessage(role="user"), _StubMessage(role="assistant")],
    )
    provider = cast("AgentApp", _StubAgentProvider(agent))
    cache = ToolbarRenderCache()

    _, _, cache_hit = _resolve_toolbar_agent_state_cached("agent", provider, cache=cache)
    assert cache_hit is False

    _, _, cache_hit = _resolve_toolbar_agent_state_cached("agent", provider, cache=cache)
    assert cache_hit is True

    agent.message_history.append(_StubMessage(role="user"))

    _, _, cache_hit = _resolve_toolbar_agent_state_cached("agent", provider, cache=cache)
    assert cache_hit is False


def test_attachment_summary_cache_invalidates_when_file_appears(tmp_path: Path) -> None:
    attachment_path = tmp_path / "draft.txt"
    cache = ToolbarRenderCache()
    text = build_local_attachment_token(attachment_path)

    summary, cache_hit, skipped = _resolve_attachment_summary(
        current_input_text=text,
        model_name="gpt-4.1",
        provider=None,
        cwd=tmp_path,
        cache=cache,
    )

    assert skipped is False
    assert cache_hit is False
    assert summary is not None
    assert summary.any_questionable is True

    attachment_path.write_text("hello", encoding="utf-8")

    summary, cache_hit, skipped = _resolve_attachment_summary(
        current_input_text=text,
        model_name="gpt-4.1",
        provider=None,
        cwd=tmp_path,
        cache=cache,
    )

    assert skipped is False
    assert cache_hit is False
    assert summary is not None
    assert summary.any_questionable is False
    assert summary.mime_types == ("text/plain",)


def test_toolbar_agent_state_cache_invalidates_when_parallel_child_model_changes() -> None:
    child = _StubAgent(
        config=_StubConfig(model="anthropic.haiku"),
        message_history=[],
    )
    parallel_agent = cast("ParallelAgent", object.__new__(ParallelAgent))
    setattr(parallel_agent, "config", _StubConfig(model=None))
    setattr(parallel_agent, "_message_history", [])
    setattr(parallel_agent, "_llm", None)
    setattr(parallel_agent, "fan_out_agents", [child])

    provider = cast("AgentApp", _StubAgentProvider(parallel_agent))
    cache = ToolbarRenderCache()

    state, _, cache_hit = _resolve_toolbar_agent_state_cached("agent", provider, cache=cache)
    assert cache_hit is False
    assert state.model_display == "haiku"

    state, _, cache_hit = _resolve_toolbar_agent_state_cached("agent", provider, cache=cache)
    assert cache_hit is True
    assert state.model_display == "haiku"

    child.config.model = "anthropic.sonnet"

    state, _, cache_hit = _resolve_toolbar_agent_state_cached("agent", provider, cache=cache)
    assert cache_hit is False
    assert state.model_display == "sonnet"

    _, _, cache_hit = _resolve_toolbar_agent_state_cached("agent", provider, cache=cache)
    assert cache_hit is True
