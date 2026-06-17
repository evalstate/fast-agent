"""Toolbar rendering helpers for interactive prompt input."""

from __future__ import annotations

import time
from dataclasses import dataclass
from html import escape as escape_html
from pathlib import Path
from typing import TYPE_CHECKING, cast

from prompt_toolkit.formatted_text import HTML

from fast_agent.agents.workflow.parallel_agent import ParallelAgent
from fast_agent.commands.model_capabilities import (
    resolve_model_info,
    resolve_reasoning_effort,
    resolve_reasoning_effort_spec,
    resolve_resolved_model,
    resolve_service_tier,
    resolve_service_tier_supported,
    resolve_text_verbosity,
    resolve_text_verbosity_spec,
    resolve_web_fetch_enabled,
    resolve_web_fetch_supported,
    resolve_web_search_enabled,
    resolve_web_search_supported,
)
from fast_agent.llm.model_display_name import resolve_model_display_name
from fast_agent.llm.model_info import ModelInfo
from fast_agent.llm.provider_types import Provider
from fast_agent.ui import notification_tracker
from fast_agent.ui.attachment_indicator import (
    DraftAttachmentSummary,
    render_attachment_indicator,
    summarize_draft_attachments,
)
from fast_agent.ui.context_usage_display import (
    ContextUsageAccumulator,
    resolve_context_usage_percent,
)
from fast_agent.ui.model_chip_display import render_model_chip
from fast_agent.ui.prompt.alert_flags import _resolve_alert_flags_from_history
from fast_agent.ui.prompt.toolbar import (
    _can_fit_shell_path_and_version,
    _fit_shell_identity_for_toolbar,
    _fit_shell_path_for_toolbar,
    _format_context_usage_percent_for_toolbar,
    _format_toolbar_agent_identity,
    _render_model_gauges,
    _resolve_toolbar_width,
    _toolbar_markup_width,
)
from fast_agent.ui.service_tier_display import render_service_tier_indicator
from fast_agent.ui.web_fetch_display import render_web_fetch_indicator
from fast_agent.ui.web_search_display import render_web_search_indicator
from fast_agent.utils.collections import unique_preserve_order
from fast_agent.utils.count_display import format_count

if TYPE_CHECKING:
    from fast_agent.core.agent_app import AgentApp
    from fast_agent.interfaces import AgentProtocol, FastAgentLLMProtocol
    from fast_agent.llm.usage_tracking import UsageAccumulator


@dataclass(slots=True)
class ToolbarRenderCache:
    agent_state_key: tuple[object, ...] | None = None
    agent_state: ToolbarAgentState | None = None
    attachment_summary_key: tuple[object, ...] | None = None
    attachment_summary: DraftAttachmentSummary | None = None


@dataclass(frozen=True, slots=True)
class AttachmentResourceSnapshot:
    kind: str
    path: str
    exists: bool
    is_file: bool
    mtime_ns: int | None
    size: int | None


@dataclass(slots=True)
class ShellToolbarState:
    enabled: bool = False
    working_dir: Path | None = None
    started_at: float = 0.0
    show_path_segment: bool = False


@dataclass(slots=True)
class ToolbarRenderResult:
    html: HTML
    show_shell_path_segment: bool
    clear_copy_notice: bool = False
    agent_state_cache_hit: bool = False
    attachment_summary_cache_hit: bool = False
    attachment_summary_skipped: bool = False


@dataclass(slots=True)
class ToolbarAgentState:
    agent: object | None = None
    model_name: str | None = None
    model_display: str | None = None
    tdv_segment: str | None = None
    turn_count: int = 0
    context_pct: float | None = None
    is_codex_responses_model: bool = False
    is_overlay_model: bool = False
    model_gauges: str = ""
    service_tier_indicator: str | None = None
    web_search_indicator: str | None = None
    web_fetch_indicator: str | None = None


@dataclass(slots=True)
class ModelVisualState:
    is_codex_responses_model: bool = False
    is_overlay_model: bool = False
    model_gauges: str = ""
    service_tier_indicator: str | None = None
    web_search_indicator: str | None = None
    web_fetch_indicator: str | None = None


@dataclass(frozen=True, slots=True)
class ToolbarMode:
    style: str
    text: str


@dataclass(frozen=True, slots=True)
class AgentUsageContext:
    context_pct: float | None
    usage_accumulator: "UsageAccumulator | None"


@dataclass(frozen=True, slots=True)
class ResolvedAttachmentSummary:
    summary: DraftAttachmentSummary | None
    cache_hit: bool = False
    skipped: bool = False


@dataclass(frozen=True, slots=True)
class ResolvedToolbarAgentState:
    state: ToolbarAgentState
    llm: "FastAgentLLMProtocol | None"
    cache_hit: bool = False


@dataclass(frozen=True, slots=True)
class CopyNoticeSegment:
    html: str
    should_clear: bool = False


@dataclass(frozen=True, slots=True)
class ToolbarIdentitySegment:
    html: str
    show_shell_path_segment: bool


def resolve_active_llm(
    agent_provider: "AgentApp | None",
    agent_name: str,
) -> "FastAgentLLMProtocol | None":
    agent = _resolve_current_agent(agent_provider, agent_name)
    if agent is None:
        return None

    return _resolve_agent_llm(agent)


def render_input_toolbar(
    *,
    agent_name: str,
    toolbar_color: str,
    agent_provider: "AgentApp | None",
    multiline_mode: bool,
    shell_state: ShellToolbarState,
    app_version: str,
    copy_notice: str | None,
    copy_notice_until: float,
    shell_path_switch_delay_seconds: float,
    current_input_text: str = "",
    cache: ToolbarRenderCache | None = None,
) -> ToolbarRenderResult:
    mode = _resolve_toolbar_mode(multiline_mode)
    shortcut_text = ""
    resolved_agent_state = _resolve_toolbar_agent_state_cached(
        agent_name, agent_provider, cache=cache
    )
    agent_identity_segment = _format_toolbar_agent_identity(
        agent_name,
        toolbar_color,
        resolved_agent_state.state.agent,
    )
    attachment_summary = _resolve_attachment_summary(
        current_input_text=current_input_text,
        model_name=resolved_agent_state.state.model_name,
        provider=resolved_agent_state.llm.provider
        if resolved_agent_state.llm is not None
        else None,
        cwd=shell_state.working_dir,
        cache=cache,
    )
    middle = _build_middle_segment(
        resolved_agent_state.state,
        shortcut_text,
        attachment_summary=attachment_summary.summary,
    )
    notification_segment = _build_notification_segment()
    copy_notice_segment = _build_copy_notice_segment(
        copy_notice,
        copy_notice_until,
        mode.style,
    )
    toolbar_identity_segment = _resolve_toolbar_identity_segment(
        shell_state=shell_state,
        middle=middle,
        agent_identity_segment=agent_identity_segment,
        mode_style=mode.style,
        mode_text=mode.text,
        version_segment=f"fast-agent {app_version}",
        notification_segment=notification_segment,
        copy_notice_segment=copy_notice_segment.html,
        shell_path_switch_delay_seconds=shell_path_switch_delay_seconds,
    )
    html = _build_toolbar_html(
        agent_identity_segment=agent_identity_segment,
        middle=middle,
        mode_style=mode.style,
        mode_text=mode.text,
        toolbar_identity_segment=toolbar_identity_segment.html,
        notification_segment=notification_segment,
        copy_notice_segment=copy_notice_segment.html,
    )
    return ToolbarRenderResult(
        html=html,
        show_shell_path_segment=toolbar_identity_segment.show_shell_path_segment,
        clear_copy_notice=copy_notice_segment.should_clear,
        agent_state_cache_hit=resolved_agent_state.cache_hit,
        attachment_summary_cache_hit=attachment_summary.cache_hit,
        attachment_summary_skipped=attachment_summary.skipped,
    )


def _resolve_attachment_summary(
    *,
    current_input_text: str,
    model_name: str | None,
    provider: Provider | None,
    cwd: Path | None,
    cache: ToolbarRenderCache | None,
) -> ResolvedAttachmentSummary:
    if not _should_resolve_attachment_summary(current_input_text):
        return ResolvedAttachmentSummary(summary=None, skipped=True)

    cache_key = _build_attachment_summary_cache_key(
        current_input_text=current_input_text,
        model_name=model_name,
        provider=provider,
        cwd=cwd,
    )
    if cache is not None and cache.attachment_summary_key == cache_key:
        return ResolvedAttachmentSummary(summary=cache.attachment_summary, cache_hit=True)

    attachment_summary = summarize_draft_attachments(
        current_input_text,
        model_name=model_name,
        provider=provider,
        cwd=cwd,
    )
    if cache is not None:
        cache.attachment_summary_key = cache_key
        cache.attachment_summary = attachment_summary
    return ResolvedAttachmentSummary(summary=attachment_summary)


def _build_attachment_summary_cache_key(
    *,
    current_input_text: str,
    model_name: str | None,
    provider: Provider | None,
    cwd: Path | None,
) -> tuple[object, ...]:
    return (
        current_input_text,
        model_name,
        provider,
        cwd,
        _attachment_resource_cache_snapshot(current_input_text, cwd=cwd),
    )


def _attachment_resource_cache_snapshot(
    current_input_text: str,
    *,
    cwd: Path | None,
) -> tuple[object, ...]:
    from fast_agent.ui.prompt.attachment_tokens import FILE_MENTION_SERVER, URL_MENTION_SERVER
    from fast_agent.ui.prompt.resource_mentions import parse_mentions

    parsed = parse_mentions(current_input_text, cwd=cwd)
    snapshots: list[object] = []
    for mention in parsed.mentions:
        if mention.server_name == FILE_MENTION_SERVER:
            snapshots.append(_snapshot_local_attachment_path(Path(mention.resource_uri)))
        elif mention.server_name == URL_MENTION_SERVER:
            snapshots.append((URL_MENTION_SERVER, mention.resource_uri))
    return tuple(snapshots)


def _snapshot_local_attachment_path(path: Path) -> AttachmentResourceSnapshot:
    try:
        stat_result = path.stat()
    except OSError:
        return AttachmentResourceSnapshot(
            kind="file",
            path=str(path),
            exists=False,
            is_file=False,
            mtime_ns=None,
            size=None,
        )

    return AttachmentResourceSnapshot(
        kind="file",
        path=str(path),
        exists=True,
        is_file=path.is_file(),
        mtime_ns=stat_result.st_mtime_ns,
        size=stat_result.st_size,
    )


def _should_resolve_attachment_summary(current_input_text: str) -> bool:
    return "^file:" in current_input_text or "^url:" in current_input_text


def _resolve_toolbar_mode(multiline_mode: bool) -> ToolbarMode:
    if multiline_mode:
        return ToolbarMode(style="ansired", text="MLTI")
    return ToolbarMode(style="ansigreen", text="NRML")


def _resolve_toolbar_agent_state(
    agent_name: str,
    agent_provider: "AgentApp | None",
) -> ToolbarAgentState:
    agent = _resolve_current_agent(agent_provider, agent_name)
    llm = _resolve_agent_llm(agent) if agent is not None else None
    return _build_toolbar_agent_state(agent, llm=llm)


def _resolve_toolbar_agent_state_cached(
    agent_name: str,
    agent_provider: "AgentApp | None",
    *,
    cache: ToolbarRenderCache | None,
) -> ResolvedToolbarAgentState:
    agent = _resolve_current_agent(agent_provider, agent_name)
    llm = _resolve_agent_llm(agent) if agent is not None else None
    cache_key = _build_toolbar_agent_state_cache_key(agent, llm=llm)
    if cache is not None and cache.agent_state_key == cache_key and cache.agent_state is not None:
        return ResolvedToolbarAgentState(state=cache.agent_state, llm=llm, cache_hit=True)

    state = _build_toolbar_agent_state(agent, llm=llm)
    if cache is not None:
        cache.agent_state_key = cache_key
        cache.agent_state = state
    return ResolvedToolbarAgentState(state=state, llm=llm)


def _build_toolbar_agent_state(
    agent: AgentProtocol | None, *, llm: "FastAgentLLMProtocol | None"
) -> ToolbarAgentState:
    if agent is None:
        return ToolbarAgentState()

    turn_count = _turn_count_for_agent(agent)
    usage_context = _usage_context_for_agent(agent)
    model_name = _resolve_model_name(agent, llm)
    model_display = _resolve_model_display(agent, model_name, llm=llm)
    model_visuals = _resolve_model_visuals(model_name, llm)
    context_pct = _resolve_context_pct(
        usage_context.context_pct,
        usage_context.usage_accumulator,
        model_name,
        llm,
    )
    tdv_segment = _resolve_tdv_segment(agent, model_name, llm)
    return ToolbarAgentState(
        agent=agent,
        model_name=model_name,
        model_display=model_display,
        tdv_segment=tdv_segment,
        turn_count=turn_count,
        context_pct=context_pct,
        is_codex_responses_model=model_visuals.is_codex_responses_model,
        is_overlay_model=model_visuals.is_overlay_model,
        model_gauges=model_visuals.model_gauges,
        service_tier_indicator=model_visuals.service_tier_indicator,
        web_search_indicator=model_visuals.web_search_indicator,
        web_fetch_indicator=model_visuals.web_fetch_indicator,
    )


def _build_toolbar_agent_state_cache_key(
    agent: AgentProtocol | None,
    *,
    llm: "FastAgentLLMProtocol | None",
) -> tuple[object, ...] | None:
    if agent is None:
        return None

    model_name = _resolve_model_name(agent, llm)
    message_history = agent.message_history
    history_len = len(message_history)
    last_message_id = id(message_history[-1]) if message_history else None

    usage_accumulator = agent.usage_accumulator
    return (
        id(agent),
        id(llm) if llm is not None else None,
        model_name,
        history_len,
        last_message_id,
        _safe_cache_value(usage_accumulator.turn_count if usage_accumulator is not None else None),
        _safe_cache_value(
            usage_accumulator.current_context_tokens if usage_accumulator is not None else None
        ),
        _safe_cache_value(
            usage_accumulator.context_window_size if usage_accumulator is not None else None
        ),
        _safe_cache_value(resolve_reasoning_effort(llm)),
        _safe_cache_value(resolve_text_verbosity(llm)),
        _safe_cache_value(resolve_service_tier(llm)),
        _safe_cache_value(resolve_web_search_enabled(llm)),
        _safe_cache_value(resolve_web_fetch_enabled(llm)),
        _parallel_fan_out_model_cache_key(agent),
    )


def _safe_cache_value(value: object) -> object:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    return repr(value)


def _parallel_fan_out_model_cache_key(agent: AgentProtocol) -> tuple[object, ...] | None:
    if not isinstance(agent, ParallelAgent):
        return None

    return tuple(
        _fan_out_agent_model_cache_key(fan_out_agent) for fan_out_agent in agent.fan_out_agents
    )


def _fan_out_agent_model_cache_key(agent: AgentProtocol) -> tuple[object, ...]:
    llm = _resolve_agent_llm(agent)
    model_name = _resolve_model_name(agent, llm)
    return (
        id(agent),
        id(llm) if llm is not None else None,
        model_name,
        _safe_cache_value(resolve_model_display_name(model_name, llm=llm)),
    )


def _resolve_current_agent(
    agent_provider: "AgentApp | None",
    agent_name: str,
) -> AgentProtocol | None:
    if agent_provider is None:
        return None
    try:
        return cast("AgentProtocol", agent_provider._agent(agent_name))
    except Exception:
        return None


def _turn_count_for_agent(agent: AgentProtocol) -> int:
    return sum(1 for message in agent.message_history if message.role == "user")


def _usage_context_for_agent(agent: AgentProtocol) -> AgentUsageContext:
    usage_accumulator = agent.usage_accumulator
    if usage_accumulator is None:
        return AgentUsageContext(context_pct=None, usage_accumulator=None)
    return AgentUsageContext(
        context_pct=usage_accumulator.context_usage_percentage,
        usage_accumulator=usage_accumulator,
    )


def _resolve_agent_llm(agent: AgentProtocol) -> "FastAgentLLMProtocol | None":
    return agent.llm


def _resolve_model_name(agent: AgentProtocol, llm: "FastAgentLLMProtocol | None") -> str | None:
    if llm is not None:
        model_name = llm.model_name
        if model_name:
            return model_name
        default_request_params = llm.default_request_params
        fallback_name = default_request_params.model if default_request_params is not None else None
        if fallback_name:
            return fallback_name

    config = agent.config
    model_name = config.model
    if model_name:
        return model_name

    default_request_params = config.default_request_params
    fallback_name = default_request_params.model if default_request_params is not None else None
    if fallback_name:
        return fallback_name

    context = agent.context
    return (
        context.config.default_model if context is not None and context.config is not None else None
    )


def _resolve_model_display(
    agent: AgentProtocol,
    model_name: str | None,
    *,
    llm: "FastAgentLLMProtocol | None" = None,
) -> str | None:
    llm = llm or _resolve_agent_llm(agent)
    resolved_display = resolve_model_display_name(model_name, llm=llm)
    if resolved_display:
        return _truncate_model_display(resolved_display)
    if isinstance(agent, ParallelAgent):
        return _resolve_parallel_model_display(agent)
    return "unknown"


def _resolve_parallel_model_display(agent: ParallelAgent) -> str:
    parallel_models: list[str] = []
    for fan_out_agent in agent.fan_out_agents:
        child_llm = _resolve_agent_llm(fan_out_agent)
        child_model_name = _resolve_model_name(fan_out_agent, child_llm)
        child_display = resolve_model_display_name(child_model_name, llm=child_llm)
        if child_display:
            parallel_models.append(child_display)

    if not parallel_models:
        return "parallel"
    deduped_models = unique_preserve_order(parallel_models)
    return _truncate_model_display(",".join(deduped_models))


def _truncate_model_display(display_name: str) -> str:
    max_len = 25
    return display_name[: max_len - 1] + "…" if len(display_name) > max_len else display_name


def _resolve_model_visuals(
    model_name: str | None,
    llm: "FastAgentLLMProtocol | None",
) -> ModelVisualState:
    visuals = ModelVisualState()
    if model_name is None or llm is None:
        return visuals

    visuals.is_codex_responses_model = llm.provider == Provider.CODEX_RESPONSES
    resolved_model = resolve_resolved_model(llm)
    visuals.is_overlay_model = (
        resolved_model.overlay is not None if resolved_model is not None else False
    )
    visuals.model_gauges = _render_model_gauges(
        resolve_reasoning_effort(llm),
        resolve_reasoning_effort_spec(llm),
        resolve_text_verbosity(llm),
        resolve_text_verbosity_spec(llm),
    )
    visuals.service_tier_indicator = render_service_tier_indicator(
        supported=resolve_service_tier_supported(llm),
        service_tier=resolve_service_tier(llm),
    )
    visuals.web_search_indicator = render_web_search_indicator(
        supported=resolve_web_search_supported(llm),
        enabled=resolve_web_search_enabled(llm),
    )
    visuals.web_fetch_indicator = render_web_fetch_indicator(
        supported=resolve_web_fetch_supported(llm),
        enabled=resolve_web_fetch_enabled(llm),
    )
    return visuals


def _resolve_context_pct(
    context_pct: float | None,
    usage_accumulator: ContextUsageAccumulator | None,
    model_name: str | None,
    llm: "FastAgentLLMProtocol | None",
) -> float | None:
    if context_pct is not None or usage_accumulator is None:
        return context_pct

    info = _resolve_model_info(model_name, llm)
    fallback_window_size = info.context_window if info else None
    return resolve_context_usage_percent(
        context_pct=context_pct,
        usage_accumulator=usage_accumulator,
        fallback_window_size=fallback_window_size,
    )


def _resolve_tdv_segment(
    agent: AgentProtocol,
    model_name: str | None,
    llm: "FastAgentLLMProtocol | None",
) -> str | None:
    info = _resolve_model_info(model_name, llm)
    t, d, v = info.tdv_flags if info else (True, False, False)
    alert_flags = _resolve_alert_flags_from_history(agent.message_history)
    return "".join(
        _style_tdv_flag(letter, supported, alert_flags)
        for letter, supported in (("T", t), ("V", v), ("D", d))
    )


def _resolve_model_info(
    model_name: str | None,
    llm: "FastAgentLLMProtocol | None",
) -> ModelInfo | None:
    if llm is not None:
        info = resolve_model_info(llm)
        if info:
            return info
        resolved_model = resolve_resolved_model(llm)
        if resolved_model is not None:
            return ModelInfo.from_resolved_model(resolved_model)
    if model_name:
        return ModelInfo.from_name(model_name)
    return None


def _style_tdv_flag(letter: str, supported: bool, alert_flags: set[str]) -> str:
    if letter in alert_flags:
        return _toolbar_style_segment(letter, foreground="ansired")
    if supported:
        return _toolbar_style_segment(letter, foreground="ansigreen")
    return _toolbar_style_segment(letter, foreground="ansiblack", background="ansiwhite")


def _toolbar_style_segment(
    text: str,
    *,
    foreground: str,
    background: str = "ansiblack",
    padded: bool = False,
) -> str:
    content = f" {text} " if padded else text
    escaped_foreground = escape_html(foreground, quote=True)
    escaped_background = escape_html(background, quote=True)
    escaped_content = escape_html(content, quote=False)
    return f"<style fg='{escaped_foreground}' bg='{escaped_background}'>{escaped_content}</style>"


def _build_middle_segment(
    agent_state: ToolbarAgentState,
    shortcut_text: str,
    *,
    attachment_summary=None,
) -> str:
    middle_segments: list[str] = []
    if agent_state.model_display:
        model_prefix = ""
        if agent_state.is_codex_responses_model:
            model_prefix = "∞"
        elif agent_state.is_overlay_model:
            model_prefix = "▼"
        model_label = f"{model_prefix}{agent_state.model_display}"
        attachment_indicator = render_attachment_indicator(attachment_summary)
        model_chip = render_model_chip(
            model_label=model_label,
            web_search_indicator=agent_state.web_search_indicator,
            web_fetch_indicator=agent_state.web_fetch_indicator,
            service_tier_indicator=agent_state.service_tier_indicator,
        )
        prefix = ""
        if agent_state.tdv_segment:
            prefix += agent_state.tdv_segment
        if attachment_indicator:
            prefix += attachment_indicator
        if agent_state.model_gauges:
            prefix += agent_state.model_gauges
        middle_segments.append(f"{prefix} {model_chip}" if prefix else model_chip)

    context_chip = _format_context_usage_percent_for_toolbar(agent_state.context_pct)
    middle_segments.append(
        context_chip if context_chip is not None else f"{agent_state.turn_count:03d}"
    )
    if shortcut_text:
        middle_segments.append(shortcut_text)
    return " | ".join(middle_segments)


def _build_notification_segment() -> str:
    active_status = notification_tracker.get_active_status()
    if active_status:
        event_type = active_status["type"].upper()
        server = active_status["server"]
        return f" | {_toolbar_style_segment(f'◀ {event_type} ({server})', foreground='ansired')}"

    if notification_tracker.get_count() <= 0:
        return ""

    counts_by_type = notification_tracker.get_counts_by_type()
    total_events = sum(counts_by_type.values()) if counts_by_type else 0
    if len(counts_by_type) == 1:
        event_type, count = next(iter(counts_by_type.items()))
        label_text = notification_tracker.format_event_label(event_type, count)
        return f" | ◀ {label_text}"

    summary = notification_tracker.get_summary(compact=True)
    return f" | ◀ {format_count(total_events, 'event')} ({summary})"


def _build_copy_notice_segment(
    copy_notice: str | None,
    copy_notice_until: float,
    mode_style: str,
) -> CopyNoticeSegment:
    if not copy_notice:
        return CopyNoticeSegment(html="")
    if time.monotonic() >= copy_notice_until:
        return CopyNoticeSegment(html="", should_clear=True)
    return CopyNoticeSegment(
        html=f" | {_toolbar_style_segment(copy_notice, foreground=mode_style, padded=True)}"
    )


def _resolve_toolbar_identity_segment(
    *,
    shell_state: ShellToolbarState,
    middle: str,
    agent_identity_segment: str,
    mode_style: str,
    mode_text: str,
    version_segment: str,
    notification_segment: str,
    copy_notice_segment: str,
    shell_path_switch_delay_seconds: float,
) -> ToolbarIdentitySegment:
    if not shell_state.enabled:
        return ToolbarIdentitySegment(
            html=version_segment,
            show_shell_path_segment=shell_state.show_path_segment,
        )

    working_dir = shell_state.working_dir or Path.cwd()
    left_prefix = _format_toolbar_prefix(
        agent_identity_segment=agent_identity_segment,
        middle=middle,
        mode_style=mode_style,
        mode_text=mode_text,
    )
    right_suffix = f"{notification_segment}{copy_notice_segment}"
    available_width = (
        _resolve_toolbar_width()
        - _toolbar_markup_width(left_prefix)
        - _toolbar_markup_width(right_suffix)
    )
    if _can_fit_shell_path_and_version(working_dir, version_segment, available_width):
        return ToolbarIdentitySegment(
            html=_fit_shell_identity_for_toolbar(working_dir, version_segment, available_width),
            show_shell_path_segment=True,
        )

    show_path_segment = shell_state.show_path_segment
    if (
        not show_path_segment
        and (time.monotonic() - shell_state.started_at) >= shell_path_switch_delay_seconds
    ):
        show_path_segment = True
    if show_path_segment:
        return ToolbarIdentitySegment(
            html=_fit_shell_path_for_toolbar(working_dir, available_width),
            show_shell_path_segment=True,
        )
    return ToolbarIdentitySegment(html=version_segment, show_shell_path_segment=False)


def _format_toolbar_prefix(
    *,
    agent_identity_segment: str,
    middle: str,
    mode_style: str,
    mode_text: str,
) -> str:
    if middle:
        return (
            f" {agent_identity_segment} "
            f" {middle} | {_toolbar_style_segment(mode_text, foreground=mode_style, padded=True)} | "
        )
    return (
        f" {agent_identity_segment} "
        f"Mode: {_toolbar_style_segment(mode_text, foreground=mode_style, padded=True)} | "
    )


def _build_toolbar_html(
    *,
    agent_identity_segment: str,
    middle: str,
    mode_style: str,
    mode_text: str,
    toolbar_identity_segment: str,
    notification_segment: str,
    copy_notice_segment: str,
) -> HTML:
    return HTML(
        (
            _format_toolbar_prefix(
                agent_identity_segment=agent_identity_segment,
                middle=middle,
                mode_style=mode_style,
                mode_text=mode_text,
            )
            + f"{toolbar_identity_segment}{notification_segment}{copy_notice_segment}"
        )
    )
