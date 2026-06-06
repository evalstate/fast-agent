"""Tiny row template renderer for batch runs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Final

from fast_agent.batch.input import RowError
from fast_agent.core.template_render import render_template_text
from fast_agent.utils.count_display import plural_label

DEFAULT_ROW_TEMPLATE: Final[str] = "Input record:\n\n{{row_json}}\n"


@dataclass(frozen=True, slots=True)
class RenderedRowTemplate:
    text: str | None = None
    error: RowError | None = None


def render_row_template(template: str, row: dict[str, Any]) -> RenderedRowTemplate:
    """Render supported placeholders against a top-level row dictionary."""

    values: dict[str, str] = {"row_json": json.dumps(row, ensure_ascii=False, indent=2)}
    for field_name, value in row.items():
        if isinstance(value, str):
            values[field_name] = value
        else:
            values[field_name] = json.dumps(value, ensure_ascii=False)

    result = render_template_text(template, values)
    if result.missing:
        names = ", ".join(result.missing)
        field_label = plural_label(len(result.missing), "field")
        return RenderedRowTemplate(
            error=RowError("MissingTemplateField", f"Missing template {field_label}: {names}")
        )
    return RenderedRowTemplate(text=result.text)
