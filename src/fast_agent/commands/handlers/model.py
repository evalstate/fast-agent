"""Shared handler for model commands."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

from fast_agent.commands import model_capabilities as _model_capabilities
from fast_agent.commands.handlers import sessions as sessions_handlers
from fast_agent.commands.handlers.shared import clear_agent_histories
from fast_agent.commands.model_capabilities import (
    SERVICE_TIER_VALUES,
    ServiceTierValue,
    available_service_tier_values,
    describe_service_tier_state,
    resolve_reasoning_effort,
    resolve_reasoning_effort_spec,
    resolve_resolved_model,
    resolve_service_tier,
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
    service_tier_command_values,
    set_reasoning_effort,
    set_service_tier,
    set_task_budget_tokens,
    set_text_verbosity,
    set_web_fetch_enabled,
    set_web_search_enabled,
    set_x_search_enabled,
)
from fast_agent.commands.model_details import (
    add_model_details,
    enabled_label,
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
from fast_agent.llm.task_budget import (
    TASK_BUDGET_MIN_TOKENS,
    format_task_budget_tokens,
    parse_task_budget_tokens,
    validate_task_budget_tokens,
)
from fast_agent.llm.text_verbosity import (
    available_text_verbosity_values,
    format_text_verbosity,
    parse_text_verbosity,
)
from fast_agent.ui.model_picker_common import infer_initial_picker_provider
from fast_agent.utils.action_normalization import normalize_action_token, parse_boolean_alias
from fast_agent.utils.text import strip_to_none

if TYPE_CHECKING:
    from fast_agent.commands.context import CommandContext
    from fast_agent.core.logging.logger import Logger
    from fast_agent.interfaces import FastAgentLLMProtocol, LlmAgentProtocol


model_supports_web_search = _model_capabilities.model_supports_web_search
model_supports_x_search = _model_capabilities.model_supports_x_search
model_supports_web_fetch = _model_capabilities.model_supports_web_fetch
model_supports_service_tier = _model_capabilities.model_supports_service_tier
model_supports_task_budget = _model_capabilities.model_supports_task_budget
model_supports_text_verbosity = _model_capabilities.model_supports_text_verbosity

ServiceTierCommandValue = Literal["status", "toggle", "on", "off", "flex"]
ModelActionHandler = Callable[..., Awaitable[CommandOutcome]]
_DIRECT_SERVICE_TIER_COMMAND_VALUES: tuple[ServiceTierValue, ...] = tuple(
    tier for tier in SERVICE_TIER_VALUES if tier != "fast"
)
_SERVICE_TIER_COMMAND_VALUES: frozenset[str] = frozenset(
    ("status", "toggle", "on", "off", *_DIRECT_SERVICE_TIER_COMMAND_VALUES)
)
_FIXED_SERVICE_TIER_SELECTIONS: dict[ServiceTierCommandValue, ServiceTierValue | None] = {
    "on": "fast",
    "off": None,
}


@dataclass(frozen=True, slots=True)
class ResolvedAgentLlm:
    agent: "LlmAgentProtocol"
    llm: "FastAgentLLMProtocol"


@dataclass(frozen=True, slots=True)
class WebToolSetting:
    label: str
    setting_name: str
    supported: Callable[[object], bool]
    enabled: Callable[[object], bool]
    set_enabled: Callable[[object, bool | None], None]


WEB_SEARCH_SETTING = WebToolSetting(
    label="Web search",
    setting_name="web_search",
    supported=resolve_web_search_supported,
    enabled=resolve_web_search_enabled,
    set_enabled=set_web_search_enabled,
)
X_SEARCH_SETTING = WebToolSetting(
    label="X Search",
    setting_name="x_search",
    supported=resolve_x_search_supported,
    enabled=resolve_x_search_enabled,
    set_enabled=set_x_search_enabled,
)
WEB_FETCH_SETTING = WebToolSetting(
    label="Web fetch",
    setting_name="web_fetch",
    supported=resolve_web_fetch_supported,
    enabled=resolve_web_fetch_enabled,
    set_enabled=set_web_fetch_enabled,
)


def _parse_web_tool_setting(value: str) -> bool | None:
    normalized = normalize_action_token(value)
    if normalized in {"default", "auto", "unset"}:
        return None

    parsed_boolean = parse_boolean_alias(normalized)
    if parsed_boolean is not None:
        return parsed_boolean

    raise ValueError("Allowed values: on, off, default.")


def _normalize_service_tier_command_value(value: str | None) -> ServiceTierCommandValue | None:
    if value is None:
        return None
    normalized = normalize_action_token(value)
    if normalized in _SERVICE_TIER_COMMAND_VALUES:
        return cast("ServiceTierCommandValue", normalized)
    return None


def _resolve_service_tier_command_selection(
    *,
    command_value: ServiceTierCommandValue | None,
    current_value: str | None,
    flex_available: bool,
) -> ServiceTierValue | None:
    if command_value is None or command_value == "toggle":
        return None if current_value in SERVICE_TIER_VALUES else "fast"
    if command_value in _FIXED_SERVICE_TIER_SELECTIONS:
        return _FIXED_SERVICE_TIER_SELECTIONS[command_value]
    if command_value == "flex" and flex_available:
        return "flex"
    raise ValueError


def _add_invalid_service_tier_message(
    outcome: CommandOutcome,
    *,
    value: str | None,
    allowed_values_text: str,
) -> None:
    outcome.add_message(
        f"Invalid service tier value '{value}'. Allowed values: {allowed_values_text}.",
        channel="error",
        right_info="model",
    )


def _apply_service_tier_command(
    outcome: CommandOutcome,
    llm: "FastAgentLLMProtocol",
    *,
    value: str | None,
    command_value: ServiceTierCommandValue | None,
    allowed_values_text: str,
) -> None:
    if command_value == "status":
        outcome.add_message(
            styled_selected_with_allowed(
                "Service tier",
                describe_service_tier_state(llm),
                allowed_values_text,
            ),
            channel="system",
            right_info="model",
        )
        return

    if value is not None and command_value is None:
        _add_invalid_service_tier_message(
            outcome,
            value=value,
            allowed_values_text=allowed_values_text,
        )
        return

    try:
        new_value = _resolve_service_tier_command_selection(
            command_value=command_value,
            current_value=resolve_service_tier(llm),
            flex_available="flex" in available_service_tier_values(llm),
        )
    except ValueError:
        _add_invalid_service_tier_message(
            outcome,
            value=value,
            allowed_values_text=allowed_values_text,
        )
        return

    try:
        set_service_tier(llm, new_value)
    except ValueError as exc:
        outcome.add_message(
            str(exc),
            channel="error",
            right_info="model",
        )
        return

    outcome.add_message(
        styled_set_line("Service tier", describe_service_tier_state(llm)),
        channel="system",
        right_info="model",
    )


def _resolve_agent_llm(
    ctx: CommandContext,
    *,
    agent_name: str,
    outcome: CommandOutcome,
) -> ResolvedAgentLlm | None:
    agent = cast("LlmAgentProtocol", ctx.agent_provider._agent(agent_name))
    llm_obj = agent.llm
    if llm_obj is None:
        outcome.add_message("No LLM attached to agent.", channel="warning", right_info="model")
        return None
    return ResolvedAgentLlm(agent=agent, llm=llm_obj)


async def _handle_model_web_tool(
    ctx: CommandContext,
    *,
    agent_name: str,
    value: str | None,
    setting: WebToolSetting,
) -> CommandOutcome:
    outcome = CommandOutcome()
    resolved = _resolve_agent_llm(ctx, agent_name=agent_name, outcome=outcome)
    if resolved is None:
        return outcome
    agent = resolved.agent
    llm = resolved.llm

    add_model_details(
        outcome,
        ctx=ctx,
        agent=agent,
        llm=llm,
        include_shell_budget=value is None,
    )

    if not setting.supported(llm):
        outcome.add_message(
            f"Current model does not support {setting.setting_name} configuration.",
            channel="warning",
            right_info="model",
        )
        return outcome

    if value is None:
        outcome.add_message(
            styled_selected_with_allowed(
                setting.label,
                enabled_label(setting.enabled(llm)),
                "on, off, default",
            ),
            channel="system",
            right_info="model",
        )
        return outcome

    try:
        parsed = _parse_web_tool_setting(value)
    except ValueError as exc:
        outcome.add_message(
            f"Invalid {setting.setting_name} value '{value}'. {exc}",
            channel="error",
            right_info="model",
        )
        return outcome

    try:
        setting.set_enabled(llm, parsed)
    except ValueError as exc:
        outcome.add_message(
            str(exc),
            channel="error",
            right_info="model",
        )
        return outcome

    current = enabled_label(setting.enabled(llm))
    selected = current if parsed is not None else f"default ({current})"
    outcome.add_message(
        styled_set_line(setting.label, selected),
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


def _reasoning_allowed_values_text(spec: ReasoningEffortSpec) -> str:
    allowed = ", ".join(available_reasoning_values(spec))
    if spec.kind != "budget" or not spec.budget_presets:
        return allowed
    return (
        f"{allowed} (presets; any value between {spec.min_budget_tokens} "
        f"and {spec.max_budget_tokens} is allowed)"
    )


def _add_reasoning_status(
    outcome: CommandOutcome,
    llm: "FastAgentLLMProtocol",
    spec: ReasoningEffortSpec,
) -> None:
    current = format_reasoning_setting(resolve_reasoning_effort(llm) or spec.default)
    outcome.add_message(
        styled_selected_with_allowed(
            REASONING_LABEL,
            current,
            _reasoning_allowed_values_text(spec),
        ),
        channel="system",
        right_info="model",
    )


def _add_reasoning_validation_error(
    outcome: CommandOutcome,
    message: str,
    spec: ReasoningEffortSpec,
) -> None:
    outcome.add_message(
        f"{message} Allowed values: {_reasoning_allowed_values_text(spec)}.",
        channel="error",
        right_info="model",
    )


def _add_reasoning_set_message(
    outcome: CommandOutcome,
    llm: "FastAgentLLMProtocol",
) -> None:
    outcome.add_message(
        styled_set_line(
            REASONING_LABEL,
            format_reasoning_setting(resolve_reasoning_effort(llm)),
        ),
        channel="system",
        right_info="model",
    )


def _add_invalid_reasoning_message(
    outcome: CommandOutcome,
    *,
    value: str,
    spec: ReasoningEffortSpec,
) -> None:
    outcome.add_message(
        f"Invalid reasoning value '{value}'. "
        f"Allowed values: {_reasoning_allowed_values_text(spec)}.",
        channel="error",
        right_info="model",
    )


def _resolve_reasoning_command_setting(
    outcome: CommandOutcome,
    *,
    value: str,
    spec: ReasoningEffortSpec,
) -> ReasoningEffortSetting | None:
    parsed = parse_reasoning_setting(value)
    if parsed is None:
        _add_invalid_reasoning_message(outcome, value=value, spec=spec)
        return None

    if parsed.kind != "toggle":
        return parsed

    if (
        spec.kind == "effort"
        and parsed.value is False
        and "none" not in (spec.allowed_efforts or [])
        and not spec.allow_toggle_disable
    ):
        outcome.add_message(
            f"{REASONING_LABEL} disable is not supported for this model. "
            f"Allowed values: {_reasoning_allowed_values_text(spec)}.",
            channel="error",
            right_info="model",
        )
        return None
    return _resolve_toggle_to_default(spec, bool(parsed.value))


def _set_reasoning_command_setting(
    outcome: CommandOutcome,
    llm: "FastAgentLLMProtocol",
    *,
    setting: ReasoningEffortSetting,
    spec: ReasoningEffortSpec,
) -> None:
    if setting.kind == "effort" and spec.kind == "budget":
        try:
            set_reasoning_effort(llm, setting)
        except ValueError as exc:
            _add_reasoning_validation_error(outcome, str(exc), spec)
            return

        _add_reasoning_set_message(outcome, llm)
        return

    try:
        validated = validate_reasoning_setting(setting, spec)
    except ValueError as exc:
        _add_reasoning_validation_error(outcome, str(exc), spec)
        return

    set_reasoning_effort(llm, validated)
    _add_reasoning_set_message(outcome, llm)


def _resolve_model_switch_initial_provider(llm: "FastAgentLLMProtocol") -> str | None:
    resolved_model = resolve_resolved_model(llm)
    if resolved_model is None:
        return None
    if resolved_model.overlay is not None:
        return "overlays"
    return infer_initial_picker_provider(resolved_model.selected_model_name)


async def _set_agent_model_for_switch(
    outcome: CommandOutcome,
    agent: "LlmAgentProtocol",
    selected_model: str,
) -> bool:
    try:
        await agent.set_model(selected_model)
    except ModelConfigError as exc:
        outcome.add_message(
            format_fast_agent_error(exc),
            channel="error",
            right_info="model",
        )
        return False
    except ValueError as exc:
        outcome.add_message(
            str(exc),
            channel="error",
            right_info="model",
        )
        return False
    return True


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
    agent = resolved.agent
    llm = resolved.llm
    previous_resolved_model = resolve_resolved_model(llm)

    selected_model = strip_to_none(value) or ""
    if not selected_model:
        selected = await ctx.io.prompt_model_selection(
            initial_provider=_resolve_model_switch_initial_provider(llm),
            default_model=(
                previous_resolved_model.selected_model_name
                if previous_resolved_model is not None
                else llm.model_name
            ),
        )
        if selected is None:
            outcome.add_message(
                "Model switch cancelled.",
                channel="warning",
                right_info="model",
            )
            return outcome
        selected_model = strip_to_none(selected) or ""

    if not selected_model:
        outcome.add_message(
            "Model switch requires a non-empty model name.",
            channel="error",
            right_info="model",
        )
        return outcome

    if not await _set_agent_model_for_switch(outcome, agent, selected_model):
        return outcome

    updated_llm = agent.llm
    current_resolved_model = resolve_resolved_model(updated_llm)
    if (
        previous_resolved_model is not None
        and
        current_resolved_model is not None
        and current_resolved_model.selected_model_name
        == previous_resolved_model.selected_model_name
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


async def apply_model_switch_session_reset(
    ctx: CommandContext,
    outcome: CommandOutcome,
    *,
    logger: "Logger | None" = None,
) -> None:
    """Apply the shared session/history reset requested by a model switch."""
    if not outcome.reset_session:
        return

    if not ctx.noenv:
        outcome.add_message(
            "Model switch starts a new session to avoid mixing histories.",
            channel="info",
        )
        session_outcome = await sessions_handlers.handle_create_session(
            ctx,
            session_name=None,
            session_id=ctx.acp_session_id,
            replace_existing=ctx.acp_session_id is not None,
        )
        outcome.messages.extend(session_outcome.messages)
    else:
        outcome.add_message(
            "Model switch cleared in-memory history (--noenv disables session persistence).",
            channel="info",
        )

    cleared = clear_agent_histories(ctx.agent_provider.registered_agents(), logger)
    if cleared:
        outcome.add_message(
            f"Cleared agent history: {', '.join(sorted(cleared))}",
            channel="info",
        )


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
    agent = resolved.agent
    llm = resolved.llm

    add_model_details(
        outcome,
        ctx=ctx,
        agent=agent,
        llm=llm,
        include_shell_budget=value is None,
        include_runtime_settings=value is None,
    )

    spec = resolve_reasoning_effort_spec(llm)
    if spec is None:
        outcome.add_message(
            "Current model does not support reasoning effort configuration.",
            channel="warning",
            right_info="model",
        )
        return outcome

    if value is None:
        _add_reasoning_status(outcome, llm, spec)
        return outcome

    setting = _resolve_reasoning_command_setting(outcome, value=value, spec=spec)
    if setting is None:
        return outcome

    _set_reasoning_command_setting(outcome, llm, setting=setting, spec=spec)
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
    agent = resolved.agent
    llm = resolved.llm

    add_model_details(
        outcome,
        ctx=ctx,
        agent=agent,
        llm=llm,
        include_shell_budget=value is None,
    )

    spec = resolve_text_verbosity_spec(llm)
    if spec is None:
        outcome.add_message(
            "Current model does not support text verbosity configuration.",
            channel="warning",
            right_info="model",
        )
        return outcome

    if value is None:
        current = format_text_verbosity(resolve_text_verbosity(llm) or spec.default)
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
        set_text_verbosity(llm, parsed)
    except ValueError as exc:
        allowed = ", ".join(available_text_verbosity_values(spec))
        outcome.add_message(
            f"{exc} Allowed values: {allowed}.",
            channel="error",
            right_info="model",
        )
        return outcome

    outcome.add_message(
        styled_set_line("Text verbosity", format_text_verbosity(resolve_text_verbosity(llm))),
        channel="system",
        right_info="model",
    )
    return outcome


async def handle_model_task_budget(
    ctx: CommandContext,
    *,
    agent_name: str,
    value: str | None,
) -> CommandOutcome:
    outcome = CommandOutcome()
    resolved = _resolve_agent_llm(ctx, agent_name=agent_name, outcome=outcome)
    if resolved is None:
        return outcome
    agent = resolved.agent
    llm = resolved.llm

    add_model_details(
        outcome,
        ctx=ctx,
        agent=agent,
        llm=llm,
        include_shell_budget=value is None,
        include_runtime_settings=value is None,
    )

    if not resolve_task_budget_supported(llm):
        outcome.add_message(
            "Current model does not support task budget configuration.",
            channel="warning",
            right_info="model",
        )
        return outcome

    allowed = (
        f"off, 20k+, or shorthand values like 64k/128k/256k "
        f"(minimum {TASK_BUDGET_MIN_TOKENS:,} tokens)"
    )
    if value is None:
        current = format_task_budget_tokens(resolve_task_budget_tokens(llm))
        outcome.add_message(
            styled_selected_with_allowed("Task budget", current, allowed),
            channel="system",
            right_info="model",
        )
        return outcome

    try:
        parsed = validate_task_budget_tokens(parse_task_budget_tokens(value))
    except ValueError as exc:
        outcome.add_message(
            f"Invalid task budget value '{value}'. {exc}",
            channel="error",
            right_info="model",
        )
        return outcome

    try:
        set_task_budget_tokens(llm, parsed)
    except ValueError as exc:
        outcome.add_message(
            str(exc),
            channel="error",
            right_info="model",
        )
        return outcome

    outcome.add_message(
        styled_set_line("Task budget", format_task_budget_tokens(resolve_task_budget_tokens(llm))),
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
        setting=WEB_SEARCH_SETTING,
    )


async def handle_model_x_search(
    ctx: CommandContext,
    *,
    agent_name: str,
    value: str | None,
) -> CommandOutcome:
    return await _handle_model_web_tool(
        ctx,
        agent_name=agent_name,
        value=value,
        setting=X_SEARCH_SETTING,
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
        setting=WEB_FETCH_SETTING,
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
    agent = resolved.agent
    llm = resolved.llm

    command_value = _normalize_service_tier_command_value(value)

    add_model_details(
        outcome,
        ctx=ctx,
        agent=agent,
        llm=llm,
        include_shell_budget=command_value == "status",
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

    _apply_service_tier_command(
        outcome,
        llm,
        value=value,
        command_value=command_value,
        allowed_values_text=allowed_values_text,
    )
    return outcome


MODEL_ACTION_HANDLERS: dict[str, ModelActionHandler] = {
    "verbosity": handle_model_verbosity,
    "task_budget": handle_model_task_budget,
    "fast": handle_model_fast,
    "web_search": handle_model_web_search,
    "x_search": handle_model_x_search,
    "web_fetch": handle_model_web_fetch,
    "switch": handle_model_switch,
    "reasoning": handle_model_reasoning,
}


def get_model_action_handler(action: str) -> ModelActionHandler | None:
    """Return the shared handler for a direct /model action."""
    return MODEL_ACTION_HANDLERS.get(normalize_action_token(action))
