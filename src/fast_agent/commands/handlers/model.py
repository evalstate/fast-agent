"""Shared handler for model commands."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from fast_agent.commands.results import CommandOutcome
from fast_agent.llm.reasoning_effort import (
    ReasoningEffortLevel,
    ReasoningEffortSetting,
    ReasoningEffortSpec,
    available_reasoning_values,
    format_reasoning_setting,
    parse_reasoning_setting,
    validate_reasoning_setting,
)
from fast_agent.llm.text_verbosity import (
    available_text_verbosity_values,
    format_text_verbosity,
    parse_text_verbosity,
)

if TYPE_CHECKING:
    from fast_agent.commands.context import CommandContext


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

    spec = llm.reasoning_effort_spec
    if spec is None:
        outcome.add_message(
            "Current model does not support reasoning effort configuration.",
            channel="warning",
            right_info="model",
        )
        return outcome

    model_name = getattr(llm, "_model_name", None)
    if model_name:
        outcome.add_message(
            f"Resolved model: {model_name}.",
            channel="info",
            right_info="model",
        )

    if value is None:
        current = format_reasoning_setting(llm.reasoning_effort or spec.default)
        allowed = ", ".join(available_reasoning_values(spec))
        if spec.kind == "budget" and spec.budget_presets:
            allowed = f"{allowed} (presets; any value between {spec.min_budget_tokens} and {spec.max_budget_tokens} is allowed)"
        outcome.add_message(
            f"Reasoning effort: {current}. Allowed values: {allowed}.",
            channel="info",
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
            f"Reasoning effort set to {format_reasoning_setting(llm.reasoning_effort)}.",
            channel="info",
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
        f"Reasoning effort set to {format_reasoning_setting(llm.reasoning_effort)}.",
        channel="info",
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

    spec = llm.text_verbosity_spec
    if spec is None:
        outcome.add_message(
            "Current model does not support text verbosity configuration.",
            channel="warning",
            right_info="model",
        )
        return outcome

    model_name = getattr(llm, "_model_name", None)
    if model_name:
        outcome.add_message(
            f"Resolved model: {model_name}.",
            channel="info",
            right_info="model",
        )

    if value is None:
        current = format_text_verbosity(llm.text_verbosity or spec.default)
        allowed = ", ".join(available_text_verbosity_values(spec))
        outcome.add_message(
            f"Text verbosity: {current}. Allowed values: {allowed}.",
            channel="info",
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
        f"Text verbosity set to {format_text_verbosity(llm.text_verbosity)}.",
        channel="info",
        right_info="model",
    )
    return outcome
