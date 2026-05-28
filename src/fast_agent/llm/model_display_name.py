from __future__ import annotations

from typing import TYPE_CHECKING

from fast_agent.llm.provider_types import Provider

if TYPE_CHECKING:
    from fast_agent.llm.resolved_model import ResolvedModelSpec


def format_model_display_name(model: str | None, *, max_len: int | None = None) -> str | None:
    if not model:
        return model

    trimmed = model.rstrip("/").partition("?")[0]
    for provider in Provider:
        dotted_prefix = f"{provider.config_name}."
        slash_prefix = f"{provider.config_name}/"
        if trimmed.startswith(dotted_prefix):
            trimmed = trimmed[len(dotted_prefix) :]
            break
        if trimmed.startswith(slash_prefix):
            trimmed = trimmed[len(slash_prefix) :]
            break
    if "/" in trimmed:
        display = trimmed.split("/")[-1] or trimmed
    else:
        display = trimmed

    if ":" in display:
        display = display.rsplit(":", 1)[0] or display

    if max_len is not None and len(display) > max_len:
        return display[: max_len - 1] + "…"
    return display


def resolve_resolved_model_display_name(
    resolved_model: "ResolvedModelSpec | None",
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

    if max_len is not None and len(display) > max_len:
        return display[: max_len - 1] + "…"
    return display


def resolve_llm_display_name(
    llm: object | None,
    *,
    max_len: int | None = None,
) -> str | None:
    if llm is None:
        return None
    return resolve_resolved_model_display_name(
        getattr(llm, "resolved_model", None),
        max_len=max_len,
    )


def resolve_model_display_name(
    model: str | None = None,
    *,
    llm: object | None = None,
    max_len: int | None = None,
) -> str | None:
    resolved_display = resolve_llm_display_name(llm, max_len=max_len)
    if resolved_display is not None:
        return resolved_display
    display = format_model_display_name(model)
    if display is None:
        return None
    if model:
        trimmed = model.partition("?")[0].strip()
        if trimmed.startswith(f"{Provider.ANTHROPIC_VERTEX.config_name}.") or trimmed.startswith(
            f"{Provider.ANTHROPIC_VERTEX.config_name}/"
        ):
            display = f"{display} · Vertex"
    if max_len is not None and len(display) > max_len:
        return display[: max_len - 1] + "…"
    return display
