"""Shared handler for model commands."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from rich.text import Text

from fast_agent.commands.results import CommandOutcome
from fast_agent.constants import TERMINAL_BYTES_PER_TOKEN
from fast_agent.llm.model_database import ModelDatabase
from fast_agent.llm.reasoning_effort import (
    ReasoningEffortLevel,
    ReasoningEffortSetting,
    ReasoningEffortSpec,
    available_reasoning_values,
    format_reasoning_setting,
    parse_reasoning_setting,
    validate_reasoning_setting,
)
from fast_agent.llm.terminal_output_limits import calculate_terminal_output_limit_for_model
from fast_agent.llm.text_verbosity import (
    available_text_verbosity_values,
    format_text_verbosity,
    parse_text_verbosity,
)

if TYPE_CHECKING:
    from fast_agent.commands.context import CommandContext


def _format_shell_budget(byte_limit: int, source: str) -> str:
    estimated_tokens = max(int(byte_limit / TERMINAL_BYTES_PER_TOKEN), 1)
    return f"{byte_limit} bytes (~{estimated_tokens} tokens, {source})"


def _styled_model_line(
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


def _styled_selected_with_allowed(label: str, selected: str, allowed: str) -> Text:
    line = Text()
    line.append(f"{label}: ", style="dim")
    line.append(selected, style="bold cyan")
    line.append(". Allowed values: ", style="dim")
    line.append(allowed, style="cyan")
    line.append(".", style="dim")
    return line


def _styled_set_line(label: str, selected: str) -> Text:
    line = Text()
    line.append(f"{label}: ", style="dim")
    line.append("set to ", style="dim")
    line.append(selected, style="bold cyan")
    line.append(".", style="dim")
    return line


def _add_model_details(
    outcome: CommandOutcome,
    *,
    ctx: "CommandContext",
    agent: object,
    llm: object,
    include_shell_budget: bool,
) -> None:
    provider = getattr(llm, "provider", None)
    provider_label: str | None = None
    if isinstance(provider, str) and provider.strip():
        provider_label = provider.strip()
    else:
        provider_value = getattr(provider, "value", None)
        if isinstance(provider_value, str) and provider_value.strip():
            provider_label = provider_value.strip()

    if provider_label:
        outcome.add_message(
            _styled_model_line("Provider", provider_label, emphasize_value=True),
            channel="system",
            right_info="model",
        )

    model_name = getattr(llm, "model_name", None)
    if model_name:
        outcome.add_message(
            _styled_model_line("Resolved model", model_name),
            channel="system",
            right_info="model",
        )

    response_transports = ModelDatabase.get_response_transports(model_name) if model_name else None
    if response_transports:
        allowed_transport_values = ", ".join(response_transports)
        outcome.add_message(
            _styled_model_line("Model transports", allowed_transport_values),
            channel="system",
            right_info="model",
        )

        configured_transport = getattr(llm, "configured_transport", None) or getattr(
            llm, "_transport", None
        )
        if isinstance(configured_transport, str) and configured_transport.strip():
            outcome.add_message(
                _styled_model_line(
                    "Configured transport",
                    configured_transport,
                    emphasize_value=True,
                ),
                channel="system",
                right_info="model",
            )

        active_transport = getattr(llm, "active_transport", None)
        if isinstance(active_transport, str) and active_transport.strip():
            transport_value = active_transport
            if (
                isinstance(configured_transport, str)
                and configured_transport in {"websocket", "auto"}
                and active_transport == "sse"
            ):
                transport_value = (
                    f"{active_transport} (websocket fallback was used for this turn)"
                )
            outcome.add_message(
                _styled_model_line(
                    "Active transport",
                    transport_value,
                    emphasize_value=True,
                ),
                channel="system",
                right_info="model",
            )

    if not include_shell_budget:
        return

    max_output_tokens = ModelDatabase.get_max_output_tokens(model_name) if model_name else None
    if max_output_tokens is not None:
        outcome.add_message(
            _styled_model_line("Model max output tokens", str(max_output_tokens)),
            channel="system",
            right_info="model",
        )

    shell_runtime = getattr(agent, "shell_runtime", None)
    runtime_limit = getattr(shell_runtime, "output_byte_limit", None)
    if isinstance(runtime_limit, int) and runtime_limit > 0:
        outcome.add_message(
            _styled_model_line(
                "Shell output budget",
                _format_shell_budget(runtime_limit, "active runtime"),
            ),
            channel="system",
            right_info="model",
        )
        return

    settings = ctx.resolve_settings()
    shell_config = getattr(settings, "shell_execution", None)
    config_limit = getattr(shell_config, "output_byte_limit", None)
    if isinstance(config_limit, int) and config_limit > 0:
        outcome.add_message(
            _styled_model_line(
                "Shell output budget",
                _format_shell_budget(config_limit, "config override"),
            ),
            channel="system",
            right_info="model",
        )
        return

    if model_name:
        outcome.add_message(
            _styled_model_line(
                "Shell output budget",
                _format_shell_budget(
                    calculate_terminal_output_limit_for_model(model_name),
                    "auto from model",
                ),
            ),
            channel="system",
            right_info="model",
        )


def _resolve_toggle_to_default(
    spec: ReasoningEffortSpec,
    value: bool,
) -> ReasoningEffortSetting:
    if not value:
        return ReasoningEffortSetting(kind="toggle", value=False)
    if spec.default:
        return spec.default
    if spec.kind == "effort":
        fallback: ReasoningEffortLevel = "medium"
        allowed = spec.allowed_efforts or [fallback]
        return ReasoningEffortSetting(
            kind="effort",
            value=cast("ReasoningEffortLevel", allowed[0]),
        )
    if spec.kind == "budget":
        budget = spec.min_budget_tokens or 1024
        return ReasoningEffortSetting(kind="budget", value=budget)
    return ReasoningEffortSetting(kind="toggle", value=True)


async def handle_model_reasoning(
    ctx: CommandContext,
    *,
    agent_name: str,
    value: str | None,
) -> CommandOutcome:
    outcome = CommandOutcome()
    agent = ctx.agent_provider._agent(agent_name)
    llm = getattr(agent, "llm", None) or getattr(agent, "_llm", None)
    if llm is None:
        outcome.add_message("No LLM attached to agent.", channel="warning", right_info="model")
        return outcome

    _add_model_details(
        outcome,
        ctx=ctx,
        agent=agent,
        llm=llm,
        include_shell_budget=value is None,
    )

    spec = llm.reasoning_effort_spec
    if spec is None:
        outcome.add_message(
            "Current model does not support reasoning effort configuration.",
            channel="warning",
            right_info="model",
        )
        return outcome

    if value is None:
        current = format_reasoning_setting(llm.reasoning_effort or spec.default)
        allowed = ", ".join(available_reasoning_values(spec))
        if spec.kind == "budget" and spec.budget_presets:
            allowed = f"{allowed} (presets; any value between {spec.min_budget_tokens} and {spec.max_budget_tokens} is allowed)"
        outcome.add_message(
            _styled_selected_with_allowed("Reasoning effort", current, allowed),
            channel="system",
            right_info="model",
        )
        return outcome

    parsed = parse_reasoning_setting(value)
    if parsed is None:
        allowed = ", ".join(available_reasoning_values(spec))
        outcome.add_message(
            f"Invalid reasoning value '{value}'. Allowed values: {allowed}.",
            channel="error",
            right_info="model",
        )
        return outcome

    if parsed.kind == "toggle":
        if (
            spec.kind == "effort"
            and parsed.value is False
            and "none" not in (spec.allowed_efforts or [])
            and not spec.allow_toggle_disable
        ):
            allowed = ", ".join(available_reasoning_values(spec))
            outcome.add_message(
                f"Reasoning disable is not supported for this model. Allowed values: {allowed}.",
                channel="error",
                right_info="model",
            )
            return outcome
        parsed = _resolve_toggle_to_default(spec, bool(parsed.value))

    if parsed.kind == "effort" and spec.kind == "budget":
        try:
            llm.set_reasoning_effort(parsed)
        except ValueError as exc:
            allowed = ", ".join(available_reasoning_values(spec))
            outcome.add_message(
                f"{exc} Allowed values: {allowed}.",
                channel="error",
                right_info="model",
            )
            return outcome

        outcome.add_message(
            _styled_set_line("Reasoning effort", format_reasoning_setting(llm.reasoning_effort)),
            channel="system",
            right_info="model",
        )
        return outcome

    try:
        parsed = validate_reasoning_setting(parsed, spec)
    except ValueError as exc:
        allowed = ", ".join(available_reasoning_values(spec))
        outcome.add_message(
            f"{exc} Allowed values: {allowed}.",
            channel="error",
            right_info="model",
        )
        return outcome

    llm.set_reasoning_effort(parsed)
    outcome.add_message(
        _styled_set_line("Reasoning effort", format_reasoning_setting(llm.reasoning_effort)),
        channel="system",
        right_info="model",
    )
    return outcome


async def handle_model_verbosity(
    ctx: CommandContext,
    *,
    agent_name: str,
    value: str | None,
) -> CommandOutcome:
    outcome = CommandOutcome()
    agent = ctx.agent_provider._agent(agent_name)
    llm = getattr(agent, "llm", None) or getattr(agent, "_llm", None)
    if llm is None:
        outcome.add_message("No LLM attached to agent.", channel="warning", right_info="model")
        return outcome

    _add_model_details(
        outcome,
        ctx=ctx,
        agent=agent,
        llm=llm,
        include_shell_budget=value is None,
    )

    spec = llm.text_verbosity_spec
    if spec is None:
        outcome.add_message(
            "Current model does not support text verbosity configuration.",
            channel="warning",
            right_info="model",
        )
        return outcome

    if value is None:
        current = format_text_verbosity(llm.text_verbosity or spec.default)
        allowed = ", ".join(available_text_verbosity_values(spec))
        outcome.add_message(
            _styled_selected_with_allowed("Text verbosity", current, allowed),
            channel="system",
            right_info="model",
        )
        return outcome

    parsed = parse_text_verbosity(value)
    if parsed is None:
        allowed = ", ".join(available_text_verbosity_values(spec))
        outcome.add_message(
            f"Invalid verbosity value '{value}'. Allowed values: {allowed}.",
            channel="error",
            right_info="model",
        )
        return outcome

    try:
        llm.set_text_verbosity(parsed)
    except ValueError as exc:
        allowed = ", ".join(available_text_verbosity_values(spec))
        outcome.add_message(
            f"{exc} Allowed values: {allowed}.",
            channel="error",
            right_info="model",
        )
        return outcome

    outcome.add_message(
        _styled_set_line("Text verbosity", format_text_verbosity(llm.text_verbosity)),
        channel="system",
        right_info="model",
    )
    return outcome
