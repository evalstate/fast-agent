"""Tiny row template renderer for batch runs."""

from __future__ import annotations

import json
import re
from typing import Any, Final

from fast_agent.batch.input import RowError

DEFAULT_ROW_TEMPLATE: Final[str] = "Input record:\n\n{{row_json}}\n"
_PLACEHOLDER_RE: Final[re.Pattern[str]] = re.compile(r"{{\s*([A-Za-z_][A-Za-z0-9_]*|row_json)\s*}}")


def render_row_template(template: str, row: dict[str, Any]) -> tuple[str | None, RowError | None]:
    """Render supported placeholders against a top-level row dictionary."""

    def replace(match: re.Match[str]) -> str:
        field_name = match.group(1)
        if field_name == "row_json":
            return json.dumps(row, ensure_ascii=False, indent=2)
        if field_name not in row:
            missing_fields.append(field_name)
            return ""
        value = row[field_name]
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False)

    missing_fields: list[str] = []
    rendered = _PLACEHOLDER_RE.sub(replace, template)
    if missing_fields:
        names = ", ".join(dict.fromkeys(missing_fields))
        return None, RowError("MissingTemplateField", f"Missing template field(s): {names}")
    return rendered, None

