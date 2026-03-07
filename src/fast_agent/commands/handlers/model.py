"""Shared handler for model commands."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, cast

from rich.text import Text

from fast_agent.commands.results import CommandOutcome
from fast_agent.constants import REASONING_LABEL, TERMINAL_BYTES_PER_TOKEN
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
    from fast_agent.interfaces import FastAgentLLMProtocol, LlmAgentProtocol


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


def _styled_switch_line(previous: str, current: str) -> Text:
    line = Text()
    line.append("Model: ", style="dim")
    line.append("switched from ", style="dim")
    line.append(previous, style="cyan")
    line.append(" to ", style="dim")
    line.append(current, style="bold cyan")
    line.append(".", style="dim")
    return line


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


def _resolve_web_search_enabled(llm: object) -> bool:
    web_tools_enabled = getattr(llm, "web_tools_enabled", None)
    if isinstance(web_tools_enabled, tuple) and len(web_tools_enabled) >= 1:
        return bool(web_tools_enabled[0])

    enabled = getattr(llm, "web_search_enabled", None)
    if isinstance(enabled, bool):
        return enabled
    return False


def _resolve_web_fetch_enabled(llm: object) -> bool:
    web_tools_enabled = getattr(llm, "web_tools_enabled", None)
    if isinstance(web_tools_enabled, tuple) and len(web_tools_enabled) >= 2:
        return bool(web_tools_enabled[1])

    enabled = getattr(llm, "web_fetch_enabled", None)
    if isinstance(enabled, bool):
        return enabled
    return False


def _resolve_web_search_supported(llm: object) -> bool:
    supported = getattr(llm, "web_search_supported", None)
    return bool(supported) if isinstance(supported, bool) else False


def _resolve_web_fetch_supported(llm: object) -> bool:
    supported = getattr(llm, "web_fetch_supported", None)
    return bool(supported) if isinstance(supported, bool) else False


def _set_web_search_enabled(llm: object, value: bool | None) -> None:
    setter = getattr(llm, "set_web_search_enabled", None)
    if callable(setter):
        setter(value)
        return
    raise ValueError("Current model does not support web search configuration.")


def _set_web_fetch_enabled(llm: object, value: bool | None) -> None:
    setter = getattr(llm, "set_web_fetch_enabled", None)
    if callable(setter):
        setter(value)
        return
    raise ValueError("Current model does not support web fetch configuration.")


def _resolve_service_tier_supported(llm: object) -> bool:
    supported = getattr(llm, "service_tier_supported", None)
    return bool(supported) if isinstance(supported, bool) else False


def available_service_tier_values(llm: object) -> tuple[str, ...]:
    raw_values = getattr(llm, "available_service_tiers", None)
    if isinstance(raw_values, (tuple, list)):
        values = tuple(value for value in raw_values if value in {"fast", "flex"})
        if values:
            return values
    if _resolve_service_tier_supported(llm):
        return ("fast", "flex")
    return ()


def service_tier_command_values(llm: object) -> tuple[str, ...]:
    values = ["on", "off"]
    if "flex" in available_service_tier_values(llm):
        values.append("flex")
    values.append("status")
    return tuple(values)


def _resolve_service_tier(llm: object) -> str | None:
    value = getattr(llm, "service_tier", None)
    return value if value in {"fast", "flex"} else None


def _set_service_tier(llm: object, value: str | None) -> None:
    setter = getattr(llm, "set_service_tier", None)
    if callable(setter):
        setter(value)
        return
    raise ValueError("Current model does not support service tier configuration.")


def _describe_service_tier_state(llm: object) -> str:
    current_tier = _resolve_service_tier(llm)
    if current_tier == "fast":
        return "fast"
    if current_tier == "flex":
        return "flex"
    return "default"


def model_supports_web_search(llm: object) -> bool:
    """Return True when model/provider supports web_search runtime configuration."""
    return _resolve_web_search_supported(llm)


def model_supports_web_fetch(llm: object) -> bool:
    """Return True when model/provider supports web_fetch runtime configuration."""
    return _resolve_web_fetch_supported(llm)


def model_supports_service_tier(llm: object) -> bool:
    """Return True when model/provider supports service tier runtime configuration."""
    return _resolve_service_tier_supported(llm)


def model_supports_text_verbosity(llm: object) -> bool:
    """Return True when model exposes text verbosity controls."""
    return getattr(llm, "text_verbosity_spec", None) is not None


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

    _add_model_details(
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
            _styled_selected_with_allowed(
                label, _enabled_label(enabled_resolver(llm)), "on, off, default"
            ),
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
        _styled_set_line(label, selected),
        channel="system",
        right_info="model",
    )
    return outcome


def _add_model_runtime_settings(
    outcome: CommandOutcome,
    *,
    llm: "FastAgentLLMProtocol",
) -> None:
    text_verbosity_spec = llm.text_verbosity_spec
    if text_verbosity_spec is not None:
        current_text_verbosity = format_text_verbosity(
            llm.text_verbosity or text_verbosity_spec.default
        )
        outcome.add_message(
            _styled_model_line("Text verbosity", current_text_verbosity),
            channel="system",
            right_info="model",
        )

    if llm.service_tier_supported:
        outcome.add_message(
            _styled_model_line("Service tier", _describe_service_tier_state(llm)),
            channel="system",
            right_info="model",
        )

    if llm.web_search_supported:
        outcome.add_message(
            _styled_model_line("Web search", _enabled_label(llm.web_search_enabled)),
            channel="system",
            right_info="model",
        )

    if llm.web_fetch_supported:
        outcome.add_message(
            _styled_model_line("Web fetch", _enabled_label(llm.web_fetch_enabled)),
            channel="system",
            right_info="model",
        )



def _add_model_details(
    outcome: CommandOutcome,
    *,
    ctx: "CommandContext",
    agent: object,
    llm: "FastAgentLLMProtocol",
    include_shell_budget: bool,
    include_runtime_settings: bool = False,
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

    sampling_overrides = _render_sampling_overrides(llm)
    if sampling_overrides:
        outcome.add_message(
            _styled_model_line("Sampling overrides", sampling_overrides),
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
                transport_value = f"{active_transport} (websocket fallback was used for this turn)"
            outcome.add_message(
                _styled_model_line(
                    "Active transport",
                    transport_value,
                    emphasize_value=True,
                ),
                channel="system",
                right_info="model",
            )

    if include_runtime_settings:
        _add_model_runtime_settings(outcome, llm=llm)

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


def _render_sampling_overrides(llm: object) -> str | None:
    request_params = getattr(llm, "default_request_params", None)
    if request_params is None:
        return None

    sampling_fields = (
        ("temperature", "temperature"),
        ("top_p", "top_p"),
        ("top_k", "top_k"),
        ("min_p", "min_p"),
        ("presence_penalty", "presence_penalty"),
        ("frequency_penalty", "frequency_penalty"),
        ("repetition_penalty", "repetition_penalty"),
    )

    parts: list[str] = []
    for attribute, label in sampling_fields:
        value = getattr(request_params, attribute, None)
        if value is None:
            continue
        parts.append(f"{label}={value}")

    if not parts:
        return None
    return ", ".join(parts)


def _resolve_model_switch_initial_provider(llm: "FastAgentLLMProtocol") -> str | None:
    provider = getattr(llm, "provider", None)
    config_name = getattr(provider, "config_name", None)
    if isinstance(config_name, str) and config_name.strip():
        return config_name.strip()
    if isinstance(provider, str) and provider.strip():
        return provider.strip()
    provider_value = getattr(provider, "value", None)
    if isinstance(provider_value, str) and provider_value.strip():
        return provider_value.strip()
    return None


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
    previous_model = getattr(llm, "model_name", None)

    selected_model = value.strip() if value else ""
    if not selected_model:
        selected = await ctx.io.prompt_model_selection(
            initial_provider=_resolve_model_switch_initial_provider(llm),
            default_model=getattr(llm, "model_name", None),
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
    except ValueError as exc:
        outcome.add_message(
            str(exc),
            channel="error",
            right_info="model",
        )
        return outcome

    updated_llm = agent.llm
    resolved_model = (
        updated_llm.model_name if updated_llm is not None and updated_llm.model_name else selected_model
    )
    if previous_model and resolved_model == previous_model:
        outcome.add_message(
            _styled_model_line("Model", f"{resolved_model} (already active)", suffix=""),
            channel="warning",
            right_info="model",
        )
        return outcome

    outcome.add_message(
        _styled_switch_line(previous_model or "<unknown>", resolved_model),
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

    _add_model_details(
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
            _styled_selected_with_allowed(REASONING_LABEL, current, allowed),
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
            _styled_set_line(REASONING_LABEL, format_reasoning_setting(llm.reasoning_effort)),
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
        _styled_set_line(REASONING_LABEL, format_reasoning_setting(llm.reasoning_effort)),
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
        supported_resolver=_resolve_web_search_supported,
        enabled_resolver=_resolve_web_search_enabled,
        setter=_set_web_search_enabled,
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
        supported_resolver=_resolve_web_fetch_supported,
        enabled_resolver=_resolve_web_fetch_enabled,
        setter=_set_web_fetch_enabled,
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

    _add_model_details(
        outcome,
        ctx=ctx,
        agent=agent,
        llm=llm,
        include_shell_budget=normalized_value == "status",
    )

    if not _resolve_service_tier_supported(llm):
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
            _styled_selected_with_allowed(
                "Service tier",
                _describe_service_tier_state(llm),
                allowed_values_text,
            ),
            channel="system",
            right_info="model",
        )
        return outcome

    if normalized_value in {None, "toggle"}:
        current_value = _resolve_service_tier(llm)
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
        _set_service_tier(llm, new_value)
    except ValueError as exc:
        outcome.add_message(
            str(exc),
            channel="error",
            right_info="model",
        )
        return outcome

    outcome.add_message(
        _styled_set_line("Service tier", _describe_service_tier_state(llm)),
        channel="system",
        right_info="model",
    )
    return outcome
