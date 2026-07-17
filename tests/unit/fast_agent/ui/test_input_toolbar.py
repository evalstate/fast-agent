from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, cast

from fast_agent.agents.workflow.parallel_agent import ParallelAgent
from fast_agent.llm.provider_types import Provider
from fast_agent.ui import notification_tracker
from fast_agent.ui.attachment_indicator import DraftAttachmentSummary
from fast_agent.ui.prompt.attachment_tokens import build_local_attachment_token
from fast_agent.ui.prompt.input_toolbar import (
    AttachmentResourceSnapshot,
    ToolbarAgentState,
    ToolbarRenderCache,
    _build_copy_notice_segment,
    _build_middle_segment,
    _build_notification_segment,
    _format_toolbar_prefix,
    _resolve_attachment_summary,
    _resolve_toolbar_agent_state_cached,
    _should_resolve_attachment_summary,
    _snapshot_local_attachment_path,
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
    context: object | None = None
    shell_runtime: object | None = None

    @property
    def llm(self) -> object | None:
        return self._llm


@dataclass
class _StubLlm:
    model_name: str | None = None
    model_info: object | None = None
    resolved_model: object | None = None
    provider: object | None = None
    default_request_params: object | None = None
    reasoning_effort: object | None = None
    reasoning_effort_spec: object | None = None
    text_verbosity: object | None = None
    text_verbosity_spec: object | None = None
    service_tier: object | None = None
    service_tier_supported: bool = False
    available_service_tiers: tuple[str, ...] = ()
    web_search_supported: bool = False
    web_search_enabled: bool = False
    x_search_supported: bool = False
    x_search_enabled: bool = False
    web_fetch_supported: bool = False
    web_fetch_enabled: bool = False
    task_budget_supported: bool = False
    task_budget_tokens: int | None = None


class _MinimalToolbarLlm(_StubLlm):
    def __init__(self) -> None:
        super().__init__(
            model_name="unknown.custom",
            default_request_params=None,
            provider=Provider.GENERIC,
        )


class _StubAgentProvider:
    def __init__(self, agent: object) -> None:
        self._stub_agent = agent

    def _agent(self, agent_name: str | None) -> object:
        del agent_name
        return self._stub_agent


def test_build_notification_segment_pluralizes_multiple_event_types() -> None:
    notification_tracker.clear()
    notification_tracker.add_warning("one")
    notification_tracker.add_tool_update("server")

    try:
        assert _build_notification_segment() == " | ◀ 2 events (tool:1 warn:1)"
    finally:
        notification_tracker.clear()


def test_build_notification_segment_escapes_active_server_name() -> None:
    notification_tracker.clear()
    notification_tracker.start_sampling("server<one&two>")

    try:
        assert _build_notification_segment() == (
            " | <style fg='ansired' bg='ansiblack'>◀ SAMPLING (server&lt;one&amp;two&gt;)</style>"
        )
    finally:
        notification_tracker.clear()


def test_build_copy_notice_segment_escapes_text_and_style() -> None:
    segment = _build_copy_notice_segment(
        "copied <draft&1>",
        copy_notice_until=float("inf"),
        mode_style="bad'color",
    )

    assert segment.html == (
        " | <style fg='bad&#x27;color' bg='ansiblack'> copied &lt;draft&amp;1&gt; </style>"
    )
    assert segment.should_clear is False


def test_format_toolbar_prefix_escapes_mode_text_and_style() -> None:
    prefix = _format_toolbar_prefix(
        agent_identity_segment="agent",
        middle="",
        mode_style="bad'color",
        mode_text="mode<draft&1>",
    )

    assert prefix == (
        " agent Mode: <style fg='bad&#x27;color' bg='ansiblack'> "
        "mode&lt;draft&amp;1&gt; </style> | "
    )


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


def test_build_middle_segment_places_active_processes_before_attachments() -> None:
    middle = _build_middle_segment(
        ToolbarAgentState(
            model_display="gpt-4.1",
            model_name="gpt-4.1",
            model_gauges="RG",
            active_process_count=2,
            turn_count=3,
        ),
        shortcut_text="",
        attachment_summary=DraftAttachmentSummary(
            count=1,
            mime_types=("image/png",),
            any_questionable=False,
        ),
    )

    assert "▲1" in middle
    assert "↻" in middle
    assert "ansiyellow" in middle
    assert middle.index("↻") < middle.index("▲1") < middle.index("RG")


def test_build_middle_segment_renders_muted_process_indicator_when_idle() -> None:
    middle = _build_middle_segment(
        ToolbarAgentState(
            model_display="gpt-4.1",
            active_process_count=0,
            turn_count=3,
        ),
        shortcut_text="",
    )

    assert "↻" in middle
    assert "ansibrightblack" in middle


def test_build_middle_segment_warns_when_process_capacity_exceeds_seventy_five_percent() -> None:
    at_threshold = _build_middle_segment(
        ToolbarAgentState(
            model_display="gpt-4.1",
            active_process_count=24,
            turn_count=3,
        ),
        shortcut_text="",
    )
    over_threshold = _build_middle_segment(
        ToolbarAgentState(
            model_display="gpt-4.1",
            active_process_count=25,
            turn_count=3,
        ),
        shortcut_text="",
    )

    assert "ansiyellow" in at_threshold
    assert "ansired" not in at_threshold
    assert "ansired" in over_threshold
    assert "↻24" not in at_threshold
    assert "↻25" not in over_threshold


def test_should_resolve_attachment_summary_only_for_attachment_tokens() -> None:
    assert not _should_resolve_attachment_summary("hello world")
    assert not _should_resolve_attachment_summary("^server:resource")
    assert _should_resolve_attachment_summary("^file:/tmp/example.txt")
    assert _should_resolve_attachment_summary("look ^url:https://example.com")


def test_local_attachment_snapshot_uses_named_cache_key(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.txt"
    missing_snapshot = _snapshot_local_attachment_path(missing_path)

    assert missing_snapshot == AttachmentResourceSnapshot(
        kind="file",
        path=str(missing_path),
        exists=False,
        is_file=False,
        mtime_ns=None,
        size=None,
    )

    attachment_path = tmp_path / "draft.txt"
    attachment_path.write_text("hello", encoding="utf-8")
    existing_snapshot = _snapshot_local_attachment_path(attachment_path)

    assert existing_snapshot.kind == "file"
    assert existing_snapshot.path == str(attachment_path)
    assert existing_snapshot.exists is True
    assert existing_snapshot.is_file is True
    assert existing_snapshot.size == 5
    assert isinstance(existing_snapshot.mtime_ns, int)


def test_toolbar_agent_state_cache_hits_until_history_changes() -> None:
    agent = _StubAgent(
        config=_StubConfig(model="haiku"),
        message_history=[_StubMessage(role="user"), _StubMessage(role="assistant")],
    )
    provider = cast("AgentApp", _StubAgentProvider(agent))
    cache = ToolbarRenderCache()

    result = _resolve_toolbar_agent_state_cached("agent", provider, cache=cache)
    assert result.cache_hit is False

    result = _resolve_toolbar_agent_state_cached("agent", provider, cache=cache)
    assert result.cache_hit is True

    agent.message_history.append(_StubMessage(role="user"))

    result = _resolve_toolbar_agent_state_cached("agent", provider, cache=cache)
    assert result.cache_hit is False


def test_toolbar_agent_state_cache_refreshes_when_active_process_count_changes() -> None:
    @dataclass
    class _Runtime:
        active_process_count: int = 0

    runtime = _Runtime()
    agent = _StubAgent(
        config=_StubConfig(model="unknown.custom"),
        message_history=[],
        _llm=_MinimalToolbarLlm(),
        shell_runtime=runtime,
    )
    provider = cast("AgentApp", _StubAgentProvider(agent))
    cache = ToolbarRenderCache()

    idle = _resolve_toolbar_agent_state_cached("agent", provider, cache=cache)
    cached = _resolve_toolbar_agent_state_cached("agent", provider, cache=cache)
    runtime.active_process_count = 1
    active = _resolve_toolbar_agent_state_cached("agent", provider, cache=cache)

    assert idle.state.active_process_count == 0
    assert cached.cache_hit is True
    assert active.cache_hit is False
    assert active.state.active_process_count == 1


def test_toolbar_agent_state_uses_protocol_default_capabilities() -> None:
    agent = _StubAgent(
        config=_StubConfig(model=None),
        message_history=[_StubMessage(role="user")],
        _llm=_MinimalToolbarLlm(),
    )
    provider = cast("AgentApp", _StubAgentProvider(agent))
    cache = ToolbarRenderCache()

    result = _resolve_toolbar_agent_state_cached("agent", provider, cache=cache)

    assert result.cache_hit is False
    assert result.state.model_name == "unknown.custom"
    assert result.state.model_display == "unknown.custom"
    assert result.state.model_gauges == ""
    assert result.state.service_tier_indicator is None
    assert result.state.web_search_indicator is None
    assert result.state.web_fetch_indicator is None


def test_attachment_summary_cache_invalidates_when_file_appears(tmp_path: Path) -> None:
    attachment_path = tmp_path / "draft.txt"
    cache = ToolbarRenderCache()
    text = build_local_attachment_token(attachment_path)

    result = _resolve_attachment_summary(
        current_input_text=text,
        model_name="gpt-4.1",
        provider=None,
        cwd=tmp_path,
        cache=cache,
    )

    assert result.skipped is False
    assert result.cache_hit is False
    assert result.summary is not None
    assert result.summary.any_questionable is True

    attachment_path.write_text("hello", encoding="utf-8")

    result = _resolve_attachment_summary(
        current_input_text=text,
        model_name="gpt-4.1",
        provider=None,
        cwd=tmp_path,
        cache=cache,
    )

    assert result.skipped is False
    assert result.cache_hit is False
    assert result.summary is not None
    assert result.summary.any_questionable is False
    assert result.summary.mime_types == ("text/plain",)


def test_toolbar_agent_state_cache_invalidates_when_parallel_child_model_changes() -> None:
    child = _StubAgent(
        config=_StubConfig(model="anthropic.haiku"),
        message_history=[],
    )
    parallel_agent = cast("ParallelAgent", object.__new__(ParallelAgent))
    setattr(parallel_agent, "config", _StubConfig(model=None))
    setattr(parallel_agent, "_message_history", [])
    setattr(parallel_agent, "_llm", None)
    setattr(parallel_agent, "_context", None)
    setattr(parallel_agent, "fan_out_agents", [child])

    provider = cast("AgentApp", _StubAgentProvider(parallel_agent))
    cache = ToolbarRenderCache()

    result = _resolve_toolbar_agent_state_cached("agent", provider, cache=cache)
    assert result.cache_hit is False
    assert result.state.model_display == "haiku"

    result = _resolve_toolbar_agent_state_cached("agent", provider, cache=cache)
    assert result.cache_hit is True
    assert result.state.model_display == "haiku"

    child.config.model = "anthropic.sonnet"

    result = _resolve_toolbar_agent_state_cached("agent", provider, cache=cache)
    assert result.cache_hit is False
    assert result.state.model_display == "sonnet"


def test_toolbar_agent_state_deduplicates_parallel_child_models() -> None:
    children = [
        _StubAgent(config=_StubConfig(model="anthropic.haiku"), message_history=[]),
        _StubAgent(config=_StubConfig(model="anthropic.haiku"), message_history=[]),
        _StubAgent(config=_StubConfig(model="anthropic.sonnet"), message_history=[]),
    ]
    parallel_agent = cast("ParallelAgent", object.__new__(ParallelAgent))
    setattr(parallel_agent, "config", _StubConfig(model=None))
    setattr(parallel_agent, "_message_history", [])
    setattr(parallel_agent, "_llm", None)
    setattr(parallel_agent, "_context", None)
    setattr(parallel_agent, "fan_out_agents", children)

    provider = cast("AgentApp", _StubAgentProvider(parallel_agent))
    result = _resolve_toolbar_agent_state_cached(
        "agent",
        provider,
        cache=ToolbarRenderCache(),
    )

    assert result.state.model_display == "haiku,sonnet"
