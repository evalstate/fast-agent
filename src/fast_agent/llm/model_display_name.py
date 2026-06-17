from __future__ import annotations

from typing import TYPE_CHECKING

from fast_agent.llm.provider_types import Provider

if TYPE_CHECKING:
    from fast_agent.interfaces import FastAgentLLMProtocol
    from fast_agent.llm.resolved_model import ResolvedModelSpec

_ANTHROPIC_VERTEX_PREFIXES = (
    f"{Provider.ANTHROPIC_VERTEX.config_name}.",
    f"{Provider.ANTHROPIC_VERTEX.config_name}/",
)


def _model_without_query(model: str) -> str:
    return model.partition("?")[0].strip().rstrip("/")


def format_model_display_name(model: str | None, *, max_len: int | None = None) -> str | None:
    if not model:
        return model

    trimmed = _model_without_query(model)
    if not trimmed:
        return None
    for provider in Provider:
        dotted_prefix = f"{provider.config_name}."
        slash_prefix = f"{provider.config_name}/"
        if trimmed.startswith(dotted_prefix):
            trimmed = trimmed[len(dotted_prefix) :]
            break
        if trimmed.startswith(slash_prefix):
            trimmed = trimmed[len(slash_prefix) :]
            break
    display = (trimmed.split("/")[-1] or trimmed) if "/" in trimmed else trimmed

    if ":" in display:
        display = display.rsplit(":", 1)[0] or display

    return _truncate_display_name(display, max_len=max_len)


def resolve_resolved_model_display_name(
    resolved_model: ResolvedModelSpec | None,
    *,
    max_len: int | None = None,
) -> str | None:
    if resolved_model is None:
        return None

    if resolved_model.overlay_name is not None:
        display = resolved_model.overlay_name
    else:
        display = (
            format_model_display_name(resolved_model.wire_model_name)
            or resolved_model.wire_model_name
        )

    if resolved_model.provider == Provider.ANTHROPIC_VERTEX:
        display = f"{display} · Vertex"

    return _truncate_display_name(display, max_len=max_len)


def resolve_llm_display_name(
    llm: "FastAgentLLMProtocol | None",
    *,
    max_len: int | None = None,
) -> str | None:
    resolved_model = None if llm is None else llm.resolved_model
    return resolve_resolved_model_display_name(
        resolved_model,
        max_len=max_len,
    )


def resolve_model_display_name(
    model: str | None = None,
    *,
    llm: "FastAgentLLMProtocol | None" = None,
    max_len: int | None = None,
) -> str | None:
    resolved_display = resolve_llm_display_name(llm, max_len=max_len)
    if resolved_display is not None:
        return resolved_display
    display = format_model_display_name(model)
    if display is None:
        return None
    if model:
        trimmed = _model_without_query(model)
        if trimmed.startswith(_ANTHROPIC_VERTEX_PREFIXES):
            display = f"{display} · Vertex"
    return _truncate_display_name(display, max_len=max_len)


def _truncate_display_name(display: str, *, max_len: int | None) -> str:
    if max_len is None or len(display) <= max_len:
        return display
    if max_len <= 0:
        return ""
    if max_len == 1:
        return "…"
    return display[: max_len - 1] + "…"
