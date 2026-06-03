"""Status summary builders for ACP slash commands."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pydantic import ValidationError

from fast_agent.agents.agent_types import AgentType
from fast_agent.commands.model_capabilities import resolve_model_info, resolve_resolved_model
from fast_agent.commands.protocols import (
    HfDisplayInfoProvider,
    ParallelAgentProtocol,
    WarningAwareAgent,
)
from fast_agent.commands.summary_utils import JsonObject, json_object, optional_string
from fast_agent.constants import FAST_AGENT_ERROR_CHANNEL
from fast_agent.llm.model_display_name import resolve_llm_display_name
from fast_agent.llm.model_info import ModelInfo
from fast_agent.mcp.helpers.content_helpers import get_text
from fast_agent.types.conversation_summary import ConversationSummary
from fast_agent.utils.collections import unique_preserve_order
from fast_agent.utils.count_display import format_count
from fast_agent.utils.numeric import (
    finite_number_or_none,
    nonnegative_int_or_none,
    positive_int_or_none,
)
from fast_agent.utils.text import strip_to_none

MODEL_CAPABILITY_LABELS = ("Text", "Document", "Vision")
DEFAULT_ERROR_SUMMARY_LIMIT = 3
DEFAULT_WARNING_SUMMARY_LIMIT = 5
ERROR_BLOCK_PREVIEW_LENGTH = 60

if TYPE_CHECKING:
    from fast_agent.core.fastagent import AgentInstance
    from fast_agent.interfaces import AgentProtocol


@dataclass(slots=True)
class ClientInfoSummary:
    name: str | None = None
    version: str | None = None
    title: str | None = None
    protocol_version: str | None = None
    filesystem_caps: JsonObject = field(default_factory=dict)
    terminal: str | None = None
    meta_caps: JsonObject = field(default_factory=dict)


@dataclass(slots=True)
class AgentModelSummary:
    agent_name: str
    provider: str
    provider_display: str
    model_name: str
    wire_model_name: str | None
    context_window: int | None
    capabilities: list[str]
    hf_provider: str | None = None


@dataclass(slots=True)
class ParallelModelSummary:
    fan_out_agents: list[AgentModelSummary]
    fan_in_agent: AgentModelSummary | None


@dataclass(slots=True)
class ToolUsageSummary:
    name: str
    count: int


@dataclass(slots=True)
class TokenEstimate:
    tokens: int
    characters: int


@dataclass(slots=True)
class ConversationStatsSummary:
    agent_name: str
    turns: int
    message_count: int
    user_message_count: int
    assistant_message_count: int
    tool_calls: int
    tool_successes: int
    tool_errors: int
    context_usage_line: str
    total_llm_time_seconds: float | None = None
    conversation_runtime_seconds: float | None = None
    tool_breakdown: list[ToolUsageSummary] = field(default_factory=list)


@dataclass(slots=True)
class ErrorHandlingSummary:
    channel_label: str
    recent_entries: list[str]


@dataclass(slots=True)
class StatusSummary:
    fast_agent_version: str
    client_info: ClientInfoSummary | None
    model_summary: AgentModelSummary | None
    parallel_summary: ParallelModelSummary | None
    model_source: str | None
    conversation_stats: ConversationStatsSummary
    uptime_seconds: float
    error_report: ErrorHandlingSummary
    warnings: list[str]


@dataclass(slots=True)
class SystemPromptSummary:
    agent_name: str
    system_prompt: str | None
    server_count: int = 0


@dataclass(slots=True)
class PermissionsSummary:
    heading: str
    message: str
    path: str


def _collect_client_info(
    *,
    client_info: Mapping[str, object] | None,
    client_capabilities: Mapping[str, object] | None,
    protocol_version: str | None,
) -> ClientInfoSummary | None:
    if not client_info and not client_capabilities and not protocol_version:
        return None

    summary = ClientInfoSummary(protocol_version=protocol_version)
    if client_info:
        summary.name = optional_string(client_info.get("name"))
        summary.version = optional_string(client_info.get("version"))
        summary.title = optional_string(client_info.get("title"))

    if client_capabilities:
        summary.filesystem_caps = json_object(client_capabilities.get("fs"))
        terminal = client_capabilities.get("terminal")
        if terminal is not None:
            summary.terminal = optional_string(str(terminal))
        summary.meta_caps = json_object(client_capabilities.get("_meta"))

    return summary


def _resolve_model_info_from_llm(llm: object | None) -> ModelInfo | None:
    model_info = resolve_model_info(llm)
    if model_info is not None or llm is None:
        return model_info

    resolved_model = resolve_resolved_model(llm)
    if resolved_model is None:
        return None
    return ModelInfo.from_resolved_model(resolved_model)


def _model_capability_labels(model_info: ModelInfo) -> list[str]:
    return [
        label
        for supported, label in zip(
            model_info.tdv_flags,
            MODEL_CAPABILITY_LABELS,
            strict=True,
        )
        if supported
    ]


def _hf_provider_display(llm: object | None) -> str | None:
    if not isinstance(llm, HfDisplayInfoProvider):
        return None
    provider = llm.get_hf_display_info().get("provider")
    if isinstance(provider, str) and provider:
        return provider
    return "auto-routing"


def _build_agent_model_summary(agent: "AgentProtocol") -> AgentModelSummary:
    model_name = "unknown"
    wire_model_name: str | None = None
    provider = "unknown"
    provider_display = "unknown"
    context_window = None
    capabilities: list[str] = []

    resolved_model = resolve_resolved_model(agent.llm)
    model_info = _resolve_model_info_from_llm(agent.llm)
    if model_info:
        model_name = model_info.name
        provider = str(model_info.provider.value)
        provider_display = model_info.provider.display_name
        context_window = model_info.context_window
        capabilities = _model_capability_labels(model_info)
    if resolved_model:
        model_name = resolve_llm_display_name(agent.llm) or resolved_model.wire_model_name
        if model_name != resolved_model.wire_model_name:
            wire_model_name = resolved_model.wire_model_name

    hf_provider = _hf_provider_display(agent.llm)

    return AgentModelSummary(
        agent_name=agent.name,
        provider=provider,
        provider_display=provider_display,
        model_name=model_name,
        wire_model_name=wire_model_name,
        context_window=context_window,
        capabilities=capabilities,
        hf_provider=hf_provider,
    )


def _build_parallel_model_summary(agent: ParallelAgentProtocol) -> ParallelModelSummary:
    fan_out_agents = [
        _build_agent_model_summary(fan_out_agent)
        for fan_out_agent in agent.fan_out_agents or []
    ]

    fan_in_agent = None
    if agent.fan_in_agent:
        fan_in_agent = _build_agent_model_summary(agent.fan_in_agent)

    return ParallelModelSummary(
        fan_out_agents=fan_out_agents,
        fan_in_agent=fan_in_agent,
    )


def _context_usage_line(summary: ConversationSummary, agent: "AgentProtocol") -> str:
    usage = agent.usage_accumulator
    if usage:
        usage_line = _usage_accumulator_context_line(
            window=usage.context_window_size,
            tokens=usage.current_context_tokens,
            percentage=usage.context_usage_percentage,
        )
        if usage_line is not None:
            return usage_line

    estimate = _estimate_tokens(summary, agent)

    model_info = _resolve_model_info_from_llm(agent.llm)
    if model_info and model_info.context_window:
        percentage = (
            (estimate.tokens / model_info.context_window) * 100
            if model_info.context_window
            else 0.0
        )
        percentage = min(percentage, 100.0)
        return (
            "Context Used: "
            f"{percentage:.1f}% (~{estimate.tokens:,} tokens of {model_info.context_window:,})"
        )

    token_text = f"~{estimate.tokens:,} tokens" if estimate.tokens else "~0 tokens"
    return f"Context Used: {estimate.characters:,} chars ({token_text} est.)"


def _usage_accumulator_context_line(
    *,
    window: object,
    tokens: object,
    percentage: object,
) -> str | None:
    parsed_window = positive_int_or_none(window)
    parsed_tokens = nonnegative_int_or_none(tokens)
    parsed_percentage = finite_number_or_none(percentage)
    if parsed_window is not None and parsed_tokens is not None and parsed_percentage is not None:
        safe_percentage = min(max(parsed_percentage, 0.0), 100.0)
        return (
            "Context Used: "
            f"{safe_percentage:.1f}% (~{parsed_tokens:,} tokens of {parsed_window:,})"
        )
    if parsed_tokens is not None and parsed_tokens > 0:
        return f"Context Used: ~{parsed_tokens:,} tokens (window unknown)"
    return None


def _estimate_tokens(
    summary: ConversationSummary, agent: "AgentProtocol"
) -> TokenEstimate:
    text_parts: list[str] = []
    for message in summary.messages:
        for content in message.content:
            text = get_text(content)
            if text:
                text_parts.append(text)

    combined = "\n".join(text_parts)
    char_count = len(combined)
    if not combined:
        return TokenEstimate(tokens=0, characters=0)

    model_name = None
    llm = agent.llm
    if llm:
        model_name = llm.model_name

    token_count = _count_tokens_with_tiktoken(combined, model_name)
    return TokenEstimate(tokens=token_count, characters=char_count)


def _count_tokens_with_tiktoken(text: str, model_name: str | None) -> int:
    try:
        import tiktoken

        if model_name:
            encoding = tiktoken.encoding_for_model(model_name)
        else:
            encoding = tiktoken.get_encoding("cl100k_base")

        return len(encoding.encode(text))
    except (ImportError, KeyError, ValueError):
        return max(1, (len(text) + 3) // 4)


def _empty_conversation_stats(
    *,
    agent_name: str,
    context_usage_line: str,
) -> ConversationStatsSummary:
    return ConversationStatsSummary(
        agent_name=agent_name,
        turns=0,
        message_count=0,
        user_message_count=0,
        assistant_message_count=0,
        tool_calls=0,
        tool_successes=0,
        tool_errors=0,
        context_usage_line=context_usage_line,
    )


def _tool_usage_breakdown(summary: ConversationSummary) -> list[ToolUsageSummary]:
    return [
        ToolUsageSummary(name=tool_name, count=count)
        for tool_name, count in sorted(
            summary.tool_call_map.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    ]


def _positive_milliseconds_to_seconds(milliseconds: int | float) -> float | None:
    positive_milliseconds = finite_number_or_none(milliseconds)
    if positive_milliseconds is None or positive_milliseconds <= 0:
        return None
    return positive_milliseconds / 1000


def build_conversation_stats_summary(
    agent: "AgentProtocol | None",
    *,
    fallback_agent_name: str,
) -> ConversationStatsSummary:
    if not agent:
        return _empty_conversation_stats(
            agent_name=fallback_agent_name,
            context_usage_line="Context Used: 0%",
        )

    try:
        summary = ConversationSummary(messages=agent.message_history)
    except (AttributeError, TypeError, ValueError, ValidationError) as exc:
        return _empty_conversation_stats(
            agent_name=agent.name,
            context_usage_line=f"Context Used: error ({exc})",
        )

    turns = min(summary.user_message_count, summary.assistant_message_count)
    context_usage = _context_usage_line(summary, agent)

    return ConversationStatsSummary(
        agent_name=agent.name,
        turns=turns,
        message_count=summary.message_count,
        user_message_count=summary.user_message_count,
        assistant_message_count=summary.assistant_message_count,
        tool_calls=summary.tool_calls,
        tool_successes=summary.tool_successes,
        tool_errors=summary.tool_errors,
        context_usage_line=context_usage,
        total_llm_time_seconds=_positive_milliseconds_to_seconds(
            summary.total_elapsed_time_ms
        ),
        conversation_runtime_seconds=_positive_milliseconds_to_seconds(
            summary.conversation_span_ms
        ),
        tool_breakdown=_tool_usage_breakdown(summary),
    )


def _error_block_summary(block: object) -> str | None:
    text = get_text(block)
    if text:
        return strip_to_none(text.replace("\n", " "))

    block_str = str(block)
    if len(block_str) > ERROR_BLOCK_PREVIEW_LENGTH:
        return (
            f"{block_str[:ERROR_BLOCK_PREVIEW_LENGTH]}... "
            f"({format_count(len(block_str), 'character')})"
        )
    return block_str


def build_error_handling_summary(
    agent: "AgentProtocol | None",
    *,
    max_entries: int = DEFAULT_ERROR_SUMMARY_LIMIT,
) -> ErrorHandlingSummary:
    channel_label = f"Error Channel: {FAST_AGENT_ERROR_CHANNEL}"
    if not agent:
        return ErrorHandlingSummary(channel_label=channel_label, recent_entries=[])

    recent_entries: list[str] = []
    history = agent.message_history

    for message in reversed(history):
        channels = message.channels or {}
        channel_blocks = channels.get(FAST_AGENT_ERROR_CHANNEL)
        if not channel_blocks:
            continue

        for block in channel_blocks:
            summary = _error_block_summary(block)
            if summary:
                recent_entries.append(summary)
            if len(recent_entries) >= max_entries:
                break
        if len(recent_entries) >= max_entries:
            break

    return ErrorHandlingSummary(channel_label=channel_label, recent_entries=recent_entries)


def build_warning_summary(
    agent: "AgentProtocol | None",
    *,
    instance: "AgentInstance | None",
    max_entries: int = DEFAULT_WARNING_SUMMARY_LIMIT,
) -> list[str]:
    warnings: list[str] = []

    if instance is not None:
        warnings.extend(_instance_card_collision_warnings(instance))

    if isinstance(agent, WarningAwareAgent):
        warnings.extend(agent.warnings)
        if agent.skill_registry:
            warnings.extend(agent.skill_registry.warnings)

    return _normalize_warning_summary(warnings, max_entries=max_entries)


def _instance_card_collision_warnings(instance: "AgentInstance") -> list[str]:
    warnings_attr = instance.app.card_collision_warnings
    if isinstance(warnings_attr, Iterable) and not isinstance(warnings_attr, (str, bytes)):
        return [str(item) for item in warnings_attr]
    if warnings_attr:
        return [str(warnings_attr)]
    return []


def _normalize_warning_summary(warnings: Iterable[object], *, max_entries: int) -> list[str]:
    cleaned = unique_preserve_order(
        message for warning in warnings if (message := strip_to_none(str(warning)))
    )

    if not cleaned:
        return []

    trimmed = cleaned[:max_entries]
    if len(cleaned) > max_entries:
        trimmed.append(f"... ({format_count(len(cleaned) - max_entries, 'more warning')})")
    return trimmed


def _resolve_model_source(agent: "AgentProtocol | None") -> str | None:
    if agent is None or agent.context is None or agent.context.config is None:
        return None

    source = agent.context.config.model_source
    return optional_string(source)


def build_status_summary(
    *,
    fast_agent_version: str,
    agent: "AgentProtocol | None",
    client_info: Mapping[str, object] | None,
    client_capabilities: Mapping[str, object] | None,
    protocol_version: str | None,
    uptime_seconds: float,
    instance: "AgentInstance | None",
) -> StatusSummary:
    client_summary = _collect_client_info(
        client_info=client_info,
        client_capabilities=client_capabilities,
        protocol_version=protocol_version,
    )

    model_summary = None
    parallel_summary = None
    if agent:
        if agent.agent_type == AgentType.PARALLEL and isinstance(agent, ParallelAgentProtocol):
            parallel_summary = _build_parallel_model_summary(agent)
        else:
            model_summary = _build_agent_model_summary(agent)

    conversation_stats = build_conversation_stats_summary(
        agent,
        fallback_agent_name=agent.name if agent else "Unknown",
    )
    error_report = build_error_handling_summary(agent)
    warnings = build_warning_summary(agent, instance=instance)

    return StatusSummary(
        fast_agent_version=fast_agent_version,
        client_info=client_summary,
        model_summary=model_summary,
        parallel_summary=parallel_summary,
        model_source=_resolve_model_source(agent),
        conversation_stats=conversation_stats,
        uptime_seconds=uptime_seconds,
        error_report=error_report,
        warnings=warnings,
    )


def build_system_prompt_summary(
    *,
    agent: "AgentProtocol | None",
    session_instructions: dict[str, str],
    current_agent_name: str,
) -> SystemPromptSummary:
    agent_name = current_agent_name
    system_prompt = None

    if agent:
        agent_name = agent.name

    if agent_name in session_instructions:
        system_prompt = session_instructions[agent_name]
    elif agent:
        system_prompt = agent.instruction

    return SystemPromptSummary(agent_name=agent_name, system_prompt=system_prompt or None)


def build_permissions_summary(
    *,
    heading: str,
    message: str,
    path: str,
) -> PermissionsSummary:
    return PermissionsSummary(heading=heading, message=message, path=path)
