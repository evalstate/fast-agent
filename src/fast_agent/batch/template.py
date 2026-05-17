"""Tiny row template renderer for batch runs."""

from __future__ import annotations

import json
from typing import Any, Final

from fast_agent.batch.input import RowError
from fast_agent.core.template_render import render_template_text

DEFAULT_ROW_TEMPLATE: Final[str] = "Input record:\n\n{{row_json}}\n"


def render_row_template(template: str, row: dict[str, Any]) -> tuple[str | None, RowError | None]:
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
        return None, RowError("MissingTemplateField", f"Missing template field(s): {names}")
    return result.text, None
