"""Toolbar rendering helpers for interactive prompt input."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

from prompt_toolkit.formatted_text import HTML

from fast_agent.agents.workflow.parallel_agent import ParallelAgent
from fast_agent.llm.model_display_name import resolve_model_display_name
from fast_agent.llm.model_info import ModelInfo
from fast_agent.llm.provider_types import Provider
from fast_agent.ui import notification_tracker
from fast_agent.ui.attachment_indicator import (
    render_attachment_indicator,
    summarize_draft_attachments,
)
from fast_agent.ui.context_usage_display import resolve_context_usage_percent
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

if TYPE_CHECKING:
    from fast_agent.core.agent_app import AgentApp
    from fast_agent.interfaces import FastAgentLLMProtocol


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


def resolve_active_llm(
    agent_provider: "AgentApp | None",
    agent_name: str,
) -> "FastAgentLLMProtocol | None":
    agent = _resolve_current_agent(agent_provider, agent_name)
    if agent is None:
        return None

    llm = _resolve_agent_llm(agent)
    return llm


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
) -> ToolbarRenderResult:
    mode_style, mode_text = _resolve_toolbar_mode(multiline_mode)
    shortcut_text = ""
    agent_state = _resolve_toolbar_agent_state(agent_name, agent_provider)
    active_llm = resolve_active_llm(agent_provider, agent_name)
    agent_identity_segment = _format_toolbar_agent_identity(
        agent_name,
        toolbar_color,
        agent_state.agent,
    )
    attachment_summary = summarize_draft_attachments(
        current_input_text,
        model_name=agent_state.model_name,
        provider=getattr(active_llm, "provider", None),
        cwd=shell_state.working_dir,
    )
    middle = _build_middle_segment(agent_state, shortcut_text, attachment_summary=attachment_summary)
    notification_segment = _build_notification_segment()
    copy_notice_segment, clear_copy_notice = _build_copy_notice_segment(
        copy_notice,
        copy_notice_until,
        mode_style,
    )
    toolbar_identity_segment, show_shell_path_segment = _resolve_toolbar_identity_segment(
        shell_state=shell_state,
        middle=middle,
        agent_identity_segment=agent_identity_segment,
        mode_style=mode_style,
        mode_text=mode_text,
        version_segment=f"fast-agent {app_version}",
        notification_segment=notification_segment,
        copy_notice_segment=copy_notice_segment,
        shell_path_switch_delay_seconds=shell_path_switch_delay_seconds,
    )
    html = _build_toolbar_html(
        agent_identity_segment=agent_identity_segment,
        middle=middle,
        mode_style=mode_style,
        mode_text=mode_text,
        toolbar_identity_segment=toolbar_identity_segment,
        notification_segment=notification_segment,
        copy_notice_segment=copy_notice_segment,
    )
    return ToolbarRenderResult(
        html=html,
        show_shell_path_segment=show_shell_path_segment,
        clear_copy_notice=clear_copy_notice,
    )


def _resolve_toolbar_mode(multiline_mode: bool) -> tuple[str, str]:
    if multiline_mode:
        return "ansired", "MLTI"
    return "ansigreen", "NRML"


def _resolve_toolbar_agent_state(
    agent_name: str,
    agent_provider: "AgentApp | None",
) -> ToolbarAgentState:
    agent = _resolve_current_agent(agent_provider, agent_name)
    if agent is None:
        return ToolbarAgentState()

    turn_count = _turn_count_for_agent(agent)
    context_pct, usage_accumulator = _usage_context_for_agent(agent)
    llm = _resolve_agent_llm(agent)
    model_name = _resolve_model_name(agent, llm)
    model_display = _resolve_model_display(agent, model_name)
    model_visuals = _resolve_model_visuals(model_name, llm)
    context_pct = _resolve_context_pct(context_pct, usage_accumulator, model_name, llm)
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


def _resolve_current_agent(
    agent_provider: "AgentApp | None",
    agent_name: str,
) -> object | None:
    if agent_provider is None:
        return None
    try:
        return agent_provider._agent(agent_name)
    except Exception:
        return None


def _turn_count_for_agent(agent: object) -> int:
    message_history = getattr(agent, "message_history", [])
    return sum(1 for message in message_history if getattr(message, "role", None) == "user")


def _usage_context_for_agent(agent: object) -> tuple[float | None, object | None]:
    usage_accumulator = getattr(agent, "usage_accumulator", None)
    if usage_accumulator is None:
        return None, None
    try:
        return usage_accumulator.context_usage_percentage, usage_accumulator
    except Exception:
        return None, usage_accumulator


def _resolve_agent_llm(agent: object) -> "FastAgentLLMProtocol | None":
    try:
        llm = getattr(agent, "llm")
    except AssertionError:
        llm = getattr(agent, "_llm", None)
    except Exception:
        llm = getattr(agent, "_llm", None)
    return cast("FastAgentLLMProtocol | None", llm)


def _resolve_model_name(agent: object, llm: object | None) -> str | None:
    if llm is not None:
        model_name = getattr(llm, "model_name", None)
        if model_name:
            return model_name
        default_request_params = getattr(llm, "default_request_params", None)
        fallback_name = getattr(default_request_params, "model", None)
        if fallback_name:
            return fallback_name

    config = getattr(agent, "config", None)
    model_name = getattr(config, "model", None)
    if model_name:
        return model_name

    default_request_params = getattr(config, "default_request_params", None)
    fallback_name = getattr(default_request_params, "model", None)
    if fallback_name:
        return fallback_name

    try:
        context = getattr(agent, "context")
    except Exception:
        context = None
    config = getattr(context, "config", None)
    return getattr(config, "default_model", None)


def _resolve_model_display(agent: object, model_name: str | None) -> str | None:
    llm = _resolve_agent_llm(agent)
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
    deduped_models = list(dict.fromkeys(parallel_models))
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

    visuals.is_codex_responses_model = getattr(llm, "provider", None) == Provider.CODEX_RESPONSES
    try:
        visuals.is_overlay_model = getattr(getattr(llm, "resolved_model", None), "overlay", None) is not None
    except Exception:
        visuals.is_overlay_model = False
    try:
        visuals.model_gauges = _render_model_gauges(
            llm.reasoning_effort,
            llm.reasoning_effort_spec,
            llm.text_verbosity,
            llm.text_verbosity_spec,
        )
        visuals.service_tier_indicator = render_service_tier_indicator(
            supported=llm.service_tier_supported,
            service_tier=llm.service_tier,
        )
        visuals.web_search_indicator = render_web_search_indicator(
            supported=llm.web_search_supported,
            enabled=llm.web_search_enabled,
        )
        visuals.web_fetch_indicator = render_web_fetch_indicator(
            supported=llm.web_fetch_supported,
            enabled=llm.web_fetch_enabled,
        )
    except Exception:
        return visuals
    return visuals


def _resolve_context_pct(
    context_pct: float | None,
    usage_accumulator: object | None,
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
    agent: object,
    model_name: str | None,
    llm: "FastAgentLLMProtocol | None",
) -> str | None:
    info = _resolve_model_info(model_name, llm)
    t, d, v = info.tdv_flags if info else (True, False, False)
    alert_flags = _resolve_alert_flags_from_history(getattr(agent, "message_history", []))
    return "".join(
        _style_tdv_flag(letter, supported, alert_flags)
        for letter, supported in (("T", t), ("V", v), ("D", d))
    )


def _resolve_model_info(
    model_name: str | None,
    llm: "FastAgentLLMProtocol | None",
) -> ModelInfo | None:
    if llm is not None:
        info = ModelInfo.from_llm(llm)
        if info:
            return info
        return ModelInfo.from_resolved_model(llm.resolved_model)
    if model_name:
        return ModelInfo.from_name(model_name)
    return None


def _style_tdv_flag(letter: str, supported: bool, alert_flags: set[str]) -> str:
    if letter in alert_flags:
        return f"<style fg='ansired' bg='ansiblack'>{letter}</style>"
    if supported:
        return f"<style fg='ansigreen' bg='ansiblack'>{letter}</style>"
    return f"<style fg='ansiblack' bg='ansiwhite'>{letter}</style>"


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
        return f" | <style fg='ansired' bg='ansiblack'>◀ {event_type} ({server})</style>"

    if notification_tracker.get_count() <= 0:
        return ""

    counts_by_type = notification_tracker.get_counts_by_type()
    total_events = sum(counts_by_type.values()) if counts_by_type else 0
    if len(counts_by_type) == 1:
        event_type, count = next(iter(counts_by_type.items()))
        label_text = notification_tracker.format_event_label(event_type, count)
        return f" | ◀ {label_text}"

    summary = notification_tracker.get_summary(compact=True)
    heading = "event" if total_events == 1 else "events"
    return f" | ◀ {total_events} {heading} ({summary})"


def _build_copy_notice_segment(
    copy_notice: str | None,
    copy_notice_until: float,
    mode_style: str,
) -> tuple[str, bool]:
    if not copy_notice:
        return "", False
    if time.monotonic() >= copy_notice_until:
        return "", True
    return f" | <style fg='{mode_style}' bg='ansiblack'> {copy_notice} </style>", False


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
) -> tuple[str, bool]:
    if not shell_state.enabled:
        return version_segment, shell_state.show_path_segment

    working_dir = shell_state.working_dir or Path.cwd()
    left_prefix = _toolbar_left_prefix(
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
        return (
            _fit_shell_identity_for_toolbar(working_dir, version_segment, available_width),
            True,
        )

    show_path_segment = shell_state.show_path_segment
    if not show_path_segment and (time.monotonic() - shell_state.started_at) >= shell_path_switch_delay_seconds:
        show_path_segment = True
    if show_path_segment:
        return _fit_shell_path_for_toolbar(working_dir, available_width), True
    return version_segment, False


def _toolbar_left_prefix(
    *,
    agent_identity_segment: str,
    middle: str,
    mode_style: str,
    mode_text: str,
) -> str:
    if middle:
        return (
            f" {agent_identity_segment} "
            f" {middle} | <style fg='{mode_style}' bg='ansiblack'> {mode_text} </style> | "
        )
    return (
        f" {agent_identity_segment} "
        f"Mode: <style fg='{mode_style}' bg='ansiblack'> {mode_text} </style> | "
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
    if middle:
        return HTML(
            f" {agent_identity_segment} "
            f" {middle} | <style fg='{mode_style}' bg='ansiblack'> {mode_text} </style> | "
            f"{toolbar_identity_segment}{notification_segment}{copy_notice_segment}"
        )
    return HTML(
        f" {agent_identity_segment} "
        f"Mode: <style fg='{mode_style}' bg='ansiblack'> {mode_text} </style> | "
        f"{toolbar_identity_segment}{notification_segment}{copy_notice_segment}"
    )
