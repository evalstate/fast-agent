from __future__ import annotations


def format_model_display(model: str | None, *, max_len: int | None = None) -> str | None:
    if not model:
        return model
    trimmed = model.rstrip("/")
    if "/" in trimmed:
        display = trimmed.split("/")[-1] or trimmed
    else:
        display = trimmed
    if max_len is not None and len(display) > max_len:
        return display[: max_len - 1] + "…"
    return display
