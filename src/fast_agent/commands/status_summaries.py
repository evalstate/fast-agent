"""Status summary builders for ACP slash commands."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from fast_agent.agents.agent_types import AgentType
from fast_agent.commands.protocols import (
    HfDisplayInfoProvider,
    InstructionAwareAgent,
    ParallelAgentProtocol,
    WarningAwareAgent,
)
from fast_agent.constants import FAST_AGENT_ERROR_CHANNEL
from fast_agent.llm.model_info import ModelInfo
from fast_agent.mcp.helpers.content_helpers import get_text
from fast_agent.types.conversation_summary import ConversationSummary

if TYPE_CHECKING:
    from fast_agent.interfaces import AgentProtocol


@dataclass(slots=True)
class ClientInfoSummary:
    name: str | None = None
    version: str | None = None
    title: str | None = None
    protocol_version: str | None = None
    filesystem_caps: dict[str, Any] = field(default_factory=dict)
    terminal: str | None = None
    meta_caps: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AgentModelSummary:
    agent_name: str
    provider: str
    provider_display: str
    model_name: str
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
    client_info: dict | None,
    client_capabilities: dict | None,
    protocol_version: str | None,
) -> ClientInfoSummary | None:
    if not client_info and not client_capabilities and not protocol_version:
        return None

    summary = ClientInfoSummary(protocol_version=protocol_version)
    if client_info:
        summary.name = client_info.get("name")
        summary.version = client_info.get("version")
        summary.title = client_info.get("title")

    if client_capabilities:
        fs_caps = client_capabilities.get("fs")
        if isinstance(fs_caps, dict):
            summary.filesystem_caps = dict(fs_caps)
        terminal = client_capabilities.get("terminal")
        if terminal is not None:
            summary.terminal = str(terminal)
        meta_caps = client_capabilities.get("_meta")
        if isinstance(meta_caps, dict):
            summary.meta_caps = dict(meta_caps)

    return summary


def _build_agent_model_summary(agent: "AgentProtocol") -> AgentModelSummary:
    model_name = "unknown"
    provider = "unknown"
    provider_display = "unknown"
    context_window = None
    capabilities: list[str] = []

    model_info = ModelInfo.from_llm(agent.llm) if agent.llm else None
    if model_info:
        model_name = model_info.name
        provider = str(model_info.provider.value)
        provider_display = model_info.provider.display_name
        context_window = model_info.context_window
        if model_info.supports_text:
            capabilities.append("Text")
        if model_info.supports_document:
            capabilities.append("Document")
        if model_info.supports_vision:
            capabilities.append("Vision")

    hf_provider = None
    if agent.llm and isinstance(agent.llm, HfDisplayInfoProvider):
        hf_info = agent.llm.get_hf_display_info()
        if hf_info:
            hf_provider = hf_info.get("provider", "auto-routing")

    return AgentModelSummary(
        agent_name=getattr(agent, "name", "unknown"),
        provider=provider,
        provider_display=provider_display,
        model_name=model_name,
        context_window=context_window,
        capabilities=capabilities,
        hf_provider=hf_provider,
    )


def _build_parallel_model_summary(agent: ParallelAgentProtocol) -> ParallelModelSummary:
    fan_out_agents = []
    for fan_out_agent in agent.fan_out_agents or []:
        fan_out_agents.append(_build_agent_model_summary(fan_out_agent))

    fan_in_agent = None
    if agent.fan_in_agent:
        fan_in_agent = _build_agent_model_summary(agent.fan_in_agent)

    return ParallelModelSummary(
        fan_out_agents=fan_out_agents,
        fan_in_agent=fan_in_agent,
    )


def _context_usage_line(summary: ConversationSummary, agent: "AgentProtocol") -> str:
    usage = getattr(agent, "usage_accumulator", None)
    if usage:
        window = usage.context_window_size
        tokens = usage.current_context_tokens
        pct = usage.context_usage_percentage
        if window and pct is not None:
            return (
                "Context Used: "
                f"{min(pct, 100.0):.1f}% (~{tokens:,} tokens of {window:,})"
            )
        if tokens:
            return f"Context Used: ~{tokens:,} tokens (window unknown)"

    token_count, char_count = _estimate_tokens(summary, agent)

    model_info = ModelInfo.from_llm(agent.llm) if agent.llm else None
    if model_info and model_info.context_window:
        percentage = (
            (token_count / model_info.context_window) * 100
            if model_info.context_window
            else 0.0
        )
        percentage = min(percentage, 100.0)
        return (
            "Context Used: "
            f"{percentage:.1f}% (~{token_count:,} tokens of {model_info.context_window:,})"
        )

    token_text = f"~{token_count:,} tokens" if token_count else "~0 tokens"
    return f"Context Used: {char_count:,} chars ({token_text} est.)"


def _estimate_tokens(
    summary: ConversationSummary, agent: "AgentProtocol"
) -> tuple[int, int]:
    text_parts: list[str] = []
    for message in summary.messages:
        for content in message.content:
            text = get_text(content)
            if text:
                text_parts.append(text)

    combined = "\n".join(text_parts)
    char_count = len(combined)
    if not combined:
        return 0, 0

    model_name = None
    llm = getattr(agent, "llm", None)
    if llm:
        model_name = llm.model_name

    token_count = _count_tokens_with_tiktoken(combined, model_name)
    return token_count, char_count


def _count_tokens_with_tiktoken(text: str, model_name: str | None) -> int:
    try:
        import tiktoken

        if model_name:
            encoding = tiktoken.encoding_for_model(model_name)
        else:
            encoding = tiktoken.get_encoding("cl100k_base")

        return len(encoding.encode(text))
    except Exception:
        return max(1, (len(text) + 3) // 4)


def build_conversation_stats_summary(
    agent: "AgentProtocol | None",
    *,
    fallback_agent_name: str,
) -> ConversationStatsSummary:
    if not agent:
        return ConversationStatsSummary(
            agent_name=fallback_agent_name,
            turns=0,
            message_count=0,
            user_message_count=0,
            assistant_message_count=0,
            tool_calls=0,
            tool_successes=0,
            tool_errors=0,
            context_usage_line="Context Used: 0%",
        )

    try:
        summary = ConversationSummary(messages=agent.message_history)
        turns = min(summary.user_message_count, summary.assistant_message_count)
        context_usage = _context_usage_line(summary, agent)
        tool_breakdown = [
            ToolUsageSummary(name=tool_name, count=count)
            for tool_name, count in sorted(
                summary.tool_call_map.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        ]

        total_time = None
        if summary.total_elapsed_time_ms > 0:
            total_time = summary.total_elapsed_time_ms / 1000

        runtime = None
        if summary.conversation_span_ms > 0:
            runtime = summary.conversation_span_ms / 1000

        return ConversationStatsSummary(
            agent_name=getattr(agent, "name", fallback_agent_name),
            turns=turns,
            message_count=summary.message_count,
            user_message_count=summary.user_message_count,
            assistant_message_count=summary.assistant_message_count,
            tool_calls=summary.tool_calls,
            tool_successes=summary.tool_successes,
            tool_errors=summary.tool_errors,
            context_usage_line=context_usage,
            total_llm_time_seconds=total_time,
            conversation_runtime_seconds=runtime,
            tool_breakdown=tool_breakdown,
        )
    except Exception as exc:  # noqa: BLE001
        return ConversationStatsSummary(
            agent_name=fallback_agent_name,
            turns=0,
            message_count=0,
            user_message_count=0,
            assistant_message_count=0,
            tool_calls=0,
            tool_successes=0,
            tool_errors=0,
            context_usage_line=f"Context Used: error ({exc})",
        )


def build_error_handling_summary(
    agent: "AgentProtocol | None",
    *,
    max_entries: int = 3,
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
            text = get_text(block)
            if text:
                cleaned = text.replace("\n", " ").strip()
                if cleaned:
                    recent_entries.append(cleaned)
            else:
                block_str = str(block)
                if len(block_str) > 60:
                    recent_entries.append(f"{block_str[:60]}... ({len(block_str)} characters)")
                else:
                    recent_entries.append(block_str)
            if len(recent_entries) >= max_entries:
                break
        if len(recent_entries) >= max_entries:
            break

    return ErrorHandlingSummary(channel_label=channel_label, recent_entries=recent_entries)


def build_warning_summary(
    agent: "AgentProtocol | None",
    *,
    instance: object | None,
    max_entries: int = 5,
) -> list[str]:
    warnings: list[str] = []

    if instance and hasattr(instance, "app") and hasattr(instance.app, "card_collision_warnings"):
        warnings_attr = instance.app.card_collision_warnings
        if isinstance(warnings_attr, list):
            warnings.extend(str(item) for item in warnings_attr)
        elif isinstance(warnings_attr, tuple):
            warnings.extend(str(item) for item in warnings_attr)
        elif warnings_attr:
            from collections.abc import Iterable

            if isinstance(warnings_attr, Iterable) and not isinstance(
                warnings_attr, (str, bytes)
            ):
                warnings.extend(str(item) for item in warnings_attr)
            else:
                warnings.append(str(warnings_attr))

    if isinstance(agent, WarningAwareAgent):
        warnings.extend(agent.warnings)
        if agent.skill_registry:
            warnings.extend(agent.skill_registry.warnings)

    cleaned: list[str] = []
    seen: set[str] = set()
    for warning in warnings:
        message = str(warning).strip()
        if message and message not in seen:
            cleaned.append(message)
            seen.add(message)

    if not cleaned:
        return []

    trimmed = cleaned[:max_entries]
    if len(cleaned) > max_entries:
        trimmed.append(f"... ({len(cleaned) - max_entries} more)")
    return trimmed


def build_status_summary(
    *,
    fast_agent_version: str,
    agent: "AgentProtocol | None",
    client_info: dict | None,
    client_capabilities: dict | None,
    protocol_version: str | None,
    uptime_seconds: float,
    instance: object | None,
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
        fallback_agent_name=getattr(agent, "name", "Unknown") if agent else "Unknown",
    )
    error_report = build_error_handling_summary(agent)
    warnings = build_warning_summary(agent, instance=instance)

    return StatusSummary(
        fast_agent_version=fast_agent_version,
        client_info=client_summary,
        model_summary=model_summary,
        parallel_summary=parallel_summary,
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

    if agent and isinstance(agent, InstructionAwareAgent):
        agent_name = agent.name

    if agent_name in session_instructions:
        system_prompt = session_instructions[agent_name]
    elif agent and isinstance(agent, InstructionAwareAgent):
        system_prompt = agent.instruction

    return SystemPromptSummary(agent_name=agent_name, system_prompt=system_prompt or None)


def build_permissions_summary(
    *,
    heading: str,
    message: str,
    path: str,
) -> PermissionsSummary:
    return PermissionsSummary(heading=heading, message=message, path=path)
