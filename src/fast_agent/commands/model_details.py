"""Presentation helpers for model command handlers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from rich.text import Text

from fast_agent.commands.model_capabilities import (
    describe_service_tier_state,
    resolve_resolved_model,
    resolve_service_tier_supported,
    resolve_task_budget_supported,
    resolve_task_budget_tokens,
    resolve_text_verbosity,
    resolve_text_verbosity_spec,
    resolve_web_fetch_enabled,
    resolve_web_fetch_supported,
    resolve_web_search_enabled,
    resolve_web_search_supported,
    resolve_x_search_enabled,
    resolve_x_search_supported,
)
from fast_agent.constants import TERMINAL_BYTES_PER_TOKEN
from fast_agent.llm.model_display_name import (
    resolve_llm_display_name,
    resolve_resolved_model_display_name,
)
from fast_agent.llm.task_budget import format_task_budget_tokens
from fast_agent.llm.terminal_output_limits import (
    calculate_terminal_output_limit_for_max_tokens,
    calculate_terminal_output_limit_for_model,
)
from fast_agent.llm.text_verbosity import format_text_verbosity
from fast_agent.utils.action_normalization import (
    enabled_disabled_label as enabled_label,
)
from fast_agent.utils.numeric import finite_number_or_none, positive_int_or_none
from fast_agent.utils.text import strip_str_to_none

_STRUCTURED_OUTPUT_LABELS: dict[str | None, str] = {
    "schema": "schema",
    "object": "object",
}

_STRUCTURED_TOOL_POLICY_LABELS: dict[str | None, str] = {
    "defer": "two-phase (tools first, schema-only final)",
    "no_tools": "structured-only (regular tools suppressed)",
    "always": "same-request compatible",
}

if TYPE_CHECKING:
    from fast_agent.commands.context import CommandContext
    from fast_agent.commands.results import CommandOutcome
    from fast_agent.interfaces import FastAgentLLMProtocol
    from fast_agent.llm.resolved_model import ResolvedModelSpec
    from fast_agent.mcp.types import McpAgentProtocol


@runtime_checkable
class ResponseTransportAware(Protocol):
    @property
    def configured_transport(self) -> str | None: ...

    @property
    def active_transport(self) -> str | None: ...


@runtime_checkable
class ShellRuntimeAware(Protocol):
    @property
    def shell_runtime(self): ...


def _transport_label(value: object) -> str | None:
    return strip_str_to_none(value)


def format_shell_budget(byte_limit: int, source: str) -> str:
    estimated_tokens = max(int(byte_limit / TERMINAL_BYTES_PER_TOKEN), 1)
    return f"{byte_limit} bytes (~{_format_token_estimate(estimated_tokens)} tokens, {source})"


def _format_token_estimate(estimated_tokens: int) -> str:
    if estimated_tokens >= 1000:
        compact = estimated_tokens / 1000
        formatted = f"{compact:.1f}".rstrip("0").rstrip(".")
        return f"{formatted}k"
    return str(estimated_tokens)


def styled_model_line(
    label: str,
    value: str,
    *,
    suffix: str = ".",
    emphasize_value: bool = False,
) -> Text:
    line = Text()
    line.append(f"{label}: ", style="dim")
    value_style = "bold cyan" if emphasize_value else "cyan"
    line.append(value, style=value_style)
    if suffix:
        line.append(suffix, style="dim")
    return line


def styled_selected_with_allowed(label: str, selected: str, allowed: str) -> Text:
    line = Text()
    line.append(f"{label}: ", style="dim")
    line.append(selected, style="bold cyan")
    line.append(". Allowed values: ", style="dim")
    line.append(allowed, style="cyan")
    line.append(".", style="dim")
    return line


def styled_set_line(label: str, selected: str) -> Text:
    line = Text()
    line.append(f"{label}: ", style="dim")
    line.append("set to ", style="dim")
    line.append(selected, style="bold cyan")
    line.append(".", style="dim")
    return line


def styled_switch_line(previous: str, current: str) -> Text:
    line = Text()
    line.append("Model: ", style="dim")
    line.append("switched from ", style="dim")
    line.append(previous, style="cyan")
    line.append(" to ", style="dim")
    line.append(current, style="bold cyan")
    line.append(".", style="dim")
    return line


def _emit_model_line(
    outcome: CommandOutcome,
    label: str,
    value: str,
    *,
    emphasize_value: bool = False,
) -> None:
    outcome.add_message(
        styled_model_line(label, value, emphasize_value=emphasize_value),
        channel="system",
        right_info="model",
    )


def _render_sampling_overrides(llm: FastAgentLLMProtocol) -> str | None:
    request_params = llm.default_request_params
    if request_params is None:
        return None

    parts: list[str] = []
    for label, value in (
        ("temperature", request_params.temperature),
        ("top_p", request_params.top_p),
        ("top_k", request_params.top_k),
        ("min_p", request_params.min_p),
        ("presence_penalty", request_params.presence_penalty),
        ("frequency_penalty", request_params.frequency_penalty),
        ("repetition_penalty", request_params.repetition_penalty),
    ):
        sampling_value = finite_number_or_none(value)
        if sampling_value is None:
            continue
        parts.append(f"{label}={_format_sampling_value(sampling_value)}")

    if not parts:
        return None
    return ", ".join(parts)


def _format_sampling_value(value: object) -> str:
    if isinstance(value, float):
        rounded = round(value, 6)
        if rounded.is_integer():
            return f"{rounded:.1f}"
        return f"{rounded:.6f}".rstrip("0").rstrip(".")
    return str(value)


def _provider_value(llm: FastAgentLLMProtocol) -> str:
    return llm.provider.config_name


def _iter_model_identity_lines(
    llm: "FastAgentLLMProtocol",
) -> list[tuple[str, str, bool]]:
    resolved_model = resolve_resolved_model(llm)
    selected_model = (
        resolved_model.selected_model_name
        if resolved_model is not None
        else llm.model_name
    )
    wire_model = (
        resolved_model.wire_model_name
        if resolved_model is not None
        else llm.model_name
    )
    lines = [
        ("Provider", _provider_value(llm), True),
        ("Selected model", selected_model, True),
        ("Display model", resolve_llm_display_name(llm) or wire_model, False),
        ("Wire model", wire_model, False),
    ]

    context_window = (
        positive_int_or_none(resolved_model.context_window)
        if resolved_model is not None
        else None
    )
    if context_window is not None:
        lines.append(("Context window", str(context_window), False))

    if resolved_model is not None:
        lines.extend(_iter_structured_output_lines(resolved_model))

    sampling_overrides = _render_sampling_overrides(llm)
    if sampling_overrides:
        lines.append(("Sampling overrides", sampling_overrides, False))

    return [(label, value, emphasize) for label, value, emphasize in lines if value]


def _iter_structured_output_lines(
    resolved_model: "ResolvedModelSpec",
) -> list[tuple[str, str, bool]]:
    json_mode = resolved_model.json_mode
    structured_output = _STRUCTURED_OUTPUT_LABELS.get(json_mode)
    if structured_output is None and json_mode is None and resolved_model.model_params is not None:
        structured_output = "prompt-only + local validation"
    elif structured_output is None:
        structured_output = "unknown (defaulting to provider behavior)"

    policy = resolved_model.structured_tool_policy
    structured_tools = _STRUCTURED_TOOL_POLICY_LABELS.get(policy, "default policy")

    return [
        ("Structured output", structured_output, False),
        ("Structured + tools", structured_tools, False),
    ]


def _format_active_transport_value(
    configured_transport: str | None,
    active_transport: str,
) -> str:
    if configured_transport in {"websocket", "auto"} and active_transport == "sse":
        return f"{active_transport} (websocket fallback was used for this turn)"
    return active_transport


def _emit_transport_details(
    outcome: CommandOutcome,
    *,
    llm: "FastAgentLLMProtocol",
) -> None:
    resolved_model = resolve_resolved_model(llm)
    if resolved_model is None:
        return

    response_transports = resolved_model.response_transports
    if not response_transports:
        return

    _emit_model_line(outcome, "Model transports", ", ".join(response_transports))

    configured_transport = (
        _transport_label(llm.configured_transport)
        if isinstance(llm, ResponseTransportAware)
        else None
    )
    if configured_transport is not None:
        _emit_model_line(
            outcome,
            "Configured transport",
            configured_transport,
            emphasize_value=True,
        )

    active_transport = (
        _transport_label(llm.active_transport)
        if isinstance(llm, ResponseTransportAware)
        else None
    )
    if active_transport is not None:
        _emit_model_line(
            outcome,
            "Active transport",
            _format_active_transport_value(configured_transport, active_transport),
            emphasize_value=True,
        )


def _add_model_runtime_settings(
    outcome: CommandOutcome,
    *,
    llm: "FastAgentLLMProtocol",
) -> None:
    text_verbosity_spec = resolve_text_verbosity_spec(llm)
    if text_verbosity_spec is not None:
        _emit_model_line(
            outcome,
            "Text verbosity",
            format_text_verbosity(
                resolve_text_verbosity(llm) or text_verbosity_spec.default
            ),
        )

    if resolve_service_tier_supported(llm):
        _emit_model_line(outcome, "Service tier", describe_service_tier_state(llm))

    if resolve_task_budget_supported(llm):
        _emit_model_line(
            outcome,
            "Task budget",
            format_task_budget_tokens(resolve_task_budget_tokens(llm)),
        )

    if resolve_web_search_supported(llm):
        _emit_model_line(outcome, "Web search", enabled_label(resolve_web_search_enabled(llm)))

    if resolve_x_search_supported(llm):
        _emit_model_line(outcome, "X Search", enabled_label(resolve_x_search_enabled(llm)))

    if resolve_web_fetch_supported(llm):
        _emit_model_line(outcome, "Web fetch", enabled_label(resolve_web_fetch_enabled(llm)))


def _resolve_shell_budget_line(
    *,
    ctx: "CommandContext",
    agent: McpAgentProtocol | object,
    max_output_tokens: int | None,
    wire_model_name: str,
) -> str | None:
    if isinstance(agent, ShellRuntimeAware):
        shell_runtime = agent.shell_runtime
        if shell_runtime is not None:
            runtime_limit = shell_runtime.output_byte_limit
            if runtime_limit > 0:
                return format_shell_budget(runtime_limit, "active runtime")

    settings = ctx.resolve_settings()
    shell_config = settings.shell_execution
    config_limit = positive_int_or_none(shell_config.output_byte_limit)
    if config_limit is not None:
        return format_shell_budget(config_limit, "config override")

    max_output_tokens = positive_int_or_none(max_output_tokens)
    if max_output_tokens is not None:
        return format_shell_budget(
            calculate_terminal_output_limit_for_max_tokens(max_output_tokens),
            "auto from model",
        )

    if wire_model_name:
        return format_shell_budget(
            calculate_terminal_output_limit_for_model(wire_model_name),
            "auto from model",
        )

    return None


def _emit_shell_budget_details(
    outcome: CommandOutcome,
    *,
    ctx: "CommandContext",
    agent: object,
    llm: "FastAgentLLMProtocol",
) -> None:
    resolved_model = resolve_resolved_model(llm)
    if resolved_model is None:
        return

    max_output_tokens = positive_int_or_none(resolved_model.max_output_tokens)
    if max_output_tokens is not None:
        _emit_model_line(outcome, "Model max output tokens", str(max_output_tokens))

    shell_budget = _resolve_shell_budget_line(
        ctx=ctx,
        agent=agent,
        max_output_tokens=max_output_tokens,
        wire_model_name=resolved_model.wire_model_name,
    )
    if shell_budget is not None:
        _emit_model_line(outcome, "Shell output budget", shell_budget)


def add_model_details(
    outcome: CommandOutcome,
    *,
    ctx: "CommandContext",
    agent: object,
    llm: "FastAgentLLMProtocol",
    include_shell_budget: bool,
    include_runtime_settings: bool = False,
) -> None:
    for label, value, emphasize in _iter_model_identity_lines(llm):
        _emit_model_line(outcome, label, value, emphasize_value=emphasize)

    resolved_model = resolve_resolved_model(llm)
    wire_model_name = (
        resolved_model.wire_model_name
        if resolved_model is not None
        else llm.model_name or ""
    )
    if wire_model_name:
        _emit_transport_details(outcome, llm=llm)

    if include_runtime_settings:
        _add_model_runtime_settings(outcome, llm=llm)

    if include_shell_budget:
        _emit_shell_budget_details(outcome, ctx=ctx, agent=agent, llm=llm)


def format_model_switch_value(resolved_model: "ResolvedModelSpec | None") -> str:
    if resolved_model is None:
        return "<unknown>"

    display_name = (
        resolve_resolved_model_display_name(resolved_model) or resolved_model.wire_model_name
    )
    if display_name not in (
        resolved_model.selected_model_name,
        resolved_model.wire_model_name,
    ):
        return (
            f"{resolved_model.selected_model_name} "
            f"(display: {display_name}) → {resolved_model.wire_model_name}"
        )

    if resolved_model.selected_model_name != resolved_model.wire_model_name:
        return f"{resolved_model.selected_model_name} → {resolved_model.wire_model_name}"

    return resolved_model.selected_model_name or "<unknown>"
