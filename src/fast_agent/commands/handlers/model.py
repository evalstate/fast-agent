"""Shared handler for model commands."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, cast

from fast_agent.commands import model_capabilities as _model_capabilities
from fast_agent.commands.model_capabilities import (
    available_service_tier_values,
    describe_service_tier_state,
    resolve_service_tier,
    resolve_service_tier_supported,
    resolve_web_fetch_enabled,
    resolve_web_fetch_supported,
    resolve_web_search_enabled,
    resolve_web_search_supported,
    service_tier_command_values,
    set_service_tier,
    set_web_fetch_enabled,
    set_web_search_enabled,
)
from fast_agent.commands.model_details import (
    add_model_details,
    format_model_switch_value,
    styled_model_line,
    styled_selected_with_allowed,
    styled_set_line,
    styled_switch_line,
)
from fast_agent.commands.results import CommandOutcome
from fast_agent.constants import REASONING_LABEL
from fast_agent.core.exceptions import ModelConfigError, format_fast_agent_error
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
from fast_agent.ui.model_picker_common import infer_initial_picker_provider

if TYPE_CHECKING:
    from fast_agent.commands.context import CommandContext
    from fast_agent.interfaces import FastAgentLLMProtocol, LlmAgentProtocol


model_supports_web_search = _model_capabilities.model_supports_web_search
model_supports_web_fetch = _model_capabilities.model_supports_web_fetch
model_supports_service_tier = _model_capabilities.model_supports_service_tier
model_supports_text_verbosity = _model_capabilities.model_supports_text_verbosity


def _enabled_label(value: bool) -> str:
    return "enabled" if value else "disabled"


def _parse_web_tool_setting(value: str) -> bool | None:
    normalized = value.strip().lower()
    if normalized in {"default", "auto", "unset"}:
        return None

    parsed = parse_reasoning_setting(normalized)
    if parsed is not None and parsed.kind == "toggle":
        return bool(parsed.value)

    raise ValueError("Allowed values: on, off, default.")


def _resolve_agent_llm(
    ctx: CommandContext,
    *,
    agent_name: str,
    outcome: CommandOutcome,
) -> tuple["LlmAgentProtocol", FastAgentLLMProtocol] | None:
    agent = cast("LlmAgentProtocol", ctx.agent_provider._agent(agent_name))
    llm_obj = getattr(agent, "llm", None) or getattr(agent, "_llm", None)
    if llm_obj is None:
        outcome.add_message("No LLM attached to agent.", channel="warning", right_info="model")
        return None
    llm = cast("FastAgentLLMProtocol", llm_obj)
    return agent, llm


async def _handle_model_web_tool(
    ctx: CommandContext,
    *,
    agent_name: str,
    value: str | None,
    label: str,
    setting_name: str,
    supported_resolver: Callable[[object], bool],
    enabled_resolver: Callable[[object], bool],
    setter: Callable[[object, bool | None], None],
) -> CommandOutcome:
    outcome = CommandOutcome()
    resolved = _resolve_agent_llm(ctx, agent_name=agent_name, outcome=outcome)
    if resolved is None:
        return outcome
    agent, llm = resolved

    add_model_details(
        outcome,
        ctx=ctx,
        agent=agent,
        llm=llm,
        include_shell_budget=value is None,
    )

    if not supported_resolver(llm):
        outcome.add_message(
            f"Current model does not support {setting_name} configuration.",
            channel="warning",
            right_info="model",
        )
        return outcome

    if value is None:
        outcome.add_message(
            styled_selected_with_allowed(label, _enabled_label(enabled_resolver(llm)), "on, off, default"),
            channel="system",
            right_info="model",
        )
        return outcome

    try:
        parsed = _parse_web_tool_setting(value)
    except ValueError as exc:
        outcome.add_message(
            f"Invalid {setting_name} value '{value}'. {exc}",
            channel="error",
            right_info="model",
        )
        return outcome

    try:
        setter(llm, parsed)
    except ValueError as exc:
        outcome.add_message(
            str(exc),
            channel="error",
            right_info="model",
        )
        return outcome

    current = _enabled_label(enabled_resolver(llm))
    selected = current if parsed is not None else f"default ({current})"
    outcome.add_message(
        styled_set_line(label, selected),
        channel="system",
        right_info="model",
    )
    return outcome


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
            value=allowed[0],
        )
    if spec.kind == "budget":
        budget = spec.min_budget_tokens or 1024
        return ReasoningEffortSetting(kind="budget", value=budget)
    return ReasoningEffortSetting(kind="toggle", value=True)


def _resolve_model_switch_initial_provider(llm: "FastAgentLLMProtocol") -> str | None:
    if llm.resolved_model.overlay is not None:
        return "overlays"
    return infer_initial_picker_provider(llm.resolved_model.selected_model_name)


async def handle_model_switch(
    ctx: CommandContext,
    *,
    agent_name: str,
    value: str | None,
) -> CommandOutcome:
    outcome = CommandOutcome()
    resolved = _resolve_agent_llm(ctx, agent_name=agent_name, outcome=outcome)
    if resolved is None:
        return outcome
    agent, llm = resolved
    previous_resolved_model = llm.resolved_model

    selected_model = value.strip() if value else ""
    if not selected_model:
        selected = await ctx.io.prompt_model_selection(
            initial_provider=_resolve_model_switch_initial_provider(llm),
            default_model=llm.resolved_model.selected_model_name,
        )
        if selected is None:
            outcome.add_message(
                "Model switch cancelled.",
                channel="warning",
                right_info="model",
            )
            return outcome
        selected_model = selected.strip()

    if not selected_model:
        outcome.add_message(
            "Model switch requires a non-empty model name.",
            channel="error",
            right_info="model",
        )
        return outcome

    try:
        await agent.set_model(selected_model)
    except ModelConfigError as exc:
        outcome.add_message(
            format_fast_agent_error(exc),
            channel="error",
            right_info="model",
        )
        return outcome
    except ValueError as exc:
        outcome.add_message(
            str(exc),
            channel="error",
            right_info="model",
        )
        return outcome

    updated_llm = agent.llm
    current_resolved_model = updated_llm.resolved_model if updated_llm is not None else None
    if (
        current_resolved_model is not None
        and current_resolved_model.selected_model_name == previous_resolved_model.selected_model_name
    ):
        outcome.add_message(
            styled_model_line(
                "Model",
                f"{format_model_switch_value(current_resolved_model)} (already active)",
                suffix="",
            ),
            channel="warning",
            right_info="model",
        )
        return outcome

    outcome.add_message(
        styled_switch_line(
            format_model_switch_value(previous_resolved_model),
            format_model_switch_value(current_resolved_model),
        ),
        channel="system",
        right_info="model",
    )
    outcome.reset_session = True
    return outcome


async def handle_model_reasoning(
    ctx: CommandContext,
    *,
    agent_name: str,
    value: str | None,
) -> CommandOutcome:
    outcome = CommandOutcome()
    resolved = _resolve_agent_llm(ctx, agent_name=agent_name, outcome=outcome)
    if resolved is None:
        return outcome
    agent, llm = resolved

    add_model_details(
        outcome,
        ctx=ctx,
        agent=agent,
        llm=llm,
        include_shell_budget=value is None,
        include_runtime_settings=value is None,
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
            styled_selected_with_allowed(REASONING_LABEL, current, allowed),
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
                f"{REASONING_LABEL} disable is not supported for this model. Allowed values: {allowed}.",
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
            styled_set_line(REASONING_LABEL, format_reasoning_setting(llm.reasoning_effort)),
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
        styled_set_line(REASONING_LABEL, format_reasoning_setting(llm.reasoning_effort)),
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
    resolved = _resolve_agent_llm(ctx, agent_name=agent_name, outcome=outcome)
    if resolved is None:
        return outcome
    agent, llm = resolved

    add_model_details(
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
            styled_selected_with_allowed("Text verbosity", current, allowed),
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
        styled_set_line("Text verbosity", format_text_verbosity(llm.text_verbosity)),
        channel="system",
        right_info="model",
    )
    return outcome


async def handle_model_web_search(
    ctx: CommandContext,
    *,
    agent_name: str,
    value: str | None,
) -> CommandOutcome:
    return await _handle_model_web_tool(
        ctx,
        agent_name=agent_name,
        value=value,
        label="Web search",
        setting_name="web_search",
        supported_resolver=resolve_web_search_supported,
        enabled_resolver=resolve_web_search_enabled,
        setter=set_web_search_enabled,
    )


async def handle_model_web_fetch(
    ctx: CommandContext,
    *,
    agent_name: str,
    value: str | None,
) -> CommandOutcome:
    return await _handle_model_web_tool(
        ctx,
        agent_name=agent_name,
        value=value,
        label="Web fetch",
        setting_name="web_fetch",
        supported_resolver=resolve_web_fetch_supported,
        enabled_resolver=resolve_web_fetch_enabled,
        setter=set_web_fetch_enabled,
    )


async def handle_model_fast(
    ctx: CommandContext,
    *,
    agent_name: str,
    value: str | None,
) -> CommandOutcome:
    outcome = CommandOutcome()
    resolved = _resolve_agent_llm(ctx, agent_name=agent_name, outcome=outcome)
    if resolved is None:
        return outcome
    agent, llm = resolved

    normalized_value = value.strip().lower() if value else None

    add_model_details(
        outcome,
        ctx=ctx,
        agent=agent,
        llm=llm,
        include_shell_budget=normalized_value == "status",
    )

    if not resolve_service_tier_supported(llm):
        outcome.add_message(
            "Current model does not support service tier configuration.",
            channel="warning",
            right_info="model",
        )
        return outcome

    allowed_values = service_tier_command_values(llm)
    allowed_values_text = ", ".join(allowed_values)

    if normalized_value == "status":
        outcome.add_message(
            styled_selected_with_allowed(
                "Service tier",
                describe_service_tier_state(llm),
                allowed_values_text,
            ),
            channel="system",
            right_info="model",
        )
        return outcome

    if normalized_value in {None, "toggle"}:
        current_value = resolve_service_tier(llm)
        if current_value == "fast":
            new_value = None
        elif current_value == "flex":
            new_value = None
        else:
            new_value = "fast"
    elif normalized_value == "on":
        new_value = "fast"
    elif normalized_value == "off":
        new_value = None
    elif normalized_value == "flex" and "flex" in available_service_tier_values(llm):
        new_value = "flex"
    else:
        outcome.add_message(
            f"Invalid service tier value '{value}'. Allowed values: {allowed_values_text}.",
            channel="error",
            right_info="model",
        )
        return outcome

    try:
        set_service_tier(llm, new_value)
    except ValueError as exc:
        outcome.add_message(
            str(exc),
            channel="error",
            right_info="model",
        )
        return outcome

    outcome.add_message(
        styled_set_line("Service tier", describe_service_tier_state(llm)),
        channel="system",
        right_info="model",
    )
    return outcome
