"""Tiny row template renderer for batch runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

from fast_agent.batch.input import RowError
from fast_agent.core.template_render import render_mapping_template
from fast_agent.utils.count_display import plural_label

if TYPE_CHECKING:
    from collections.abc import Mapping

DEFAULT_ROW_TEMPLATE: Final[str] = "Input record:\n\n{{row_json}}\n"


@dataclass(frozen=True, slots=True)
class RenderedRowTemplate:
    text: str | None = None
    error: RowError | None = None


def render_row_template(template: str, row: Mapping[str, Any]) -> RenderedRowTemplate:
    """Render supported placeholders against a top-level row dictionary."""

    result = render_mapping_template(template, row, json_placeholder="row_json")
    if result.missing:
        names = ", ".join(result.missing)
        field_label = plural_label(len(result.missing), "field")
        return RenderedRowTemplate(
            error=RowError("MissingTemplateField", f"Missing template {field_label}: {names}")
        )
    return RenderedRowTemplate(text=result.text)
