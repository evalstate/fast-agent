"""Small value-only ``{{placeholder}}`` renderer for prompt text."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

from fast_agent.utils.text import strip_to_none

if TYPE_CHECKING:
    from collections.abc import Mapping

_PLACEHOLDER_RE: Final[re.Pattern[str]] = re.compile(r"{{\s*([^}]+?)\s*}}")


@dataclass(frozen=True)
class TemplateRenderResult:
    text: str
    missing: tuple[str, ...] = ()


def _placeholder_name(match: re.Match[str]) -> str | None:
    return strip_to_none(match.group(1))


def extract_template_variables(text: str) -> set[str]:
    """Return placeholder names from ``text`` without braces."""
    return {
        name
        for match in _PLACEHOLDER_RE.finditer(text)
        if (name := _placeholder_name(match)) is not None
    }


def render_template_text(text: str, values: Mapping[str, Any]) -> TemplateRenderResult:
    """Replace placeholders with supplied values, preserving unresolved placeholders."""
    missing: list[str] = []

    def replace(match: re.Match[str]) -> str:
        name = _placeholder_name(match)
        if name is None:
            return match.group(0)
        if name not in values:
            missing.append(name)
            return match.group(0)
        return str(values[name])

    rendered = _PLACEHOLDER_RE.sub(replace, text)
    return TemplateRenderResult(text=rendered, missing=tuple(dict.fromkeys(missing)))


def render_mapping_template(
    template: str,
    values: Mapping[str, Any],
    *,
    json_placeholder: str,
) -> TemplateRenderResult:
    """Render a prompt template from mapping fields plus a full JSON placeholder."""
    rendered_values: dict[str, str] = {
        json_placeholder: json.dumps(dict(values), ensure_ascii=False, indent=2)
    }
    for field_name, value in values.items():
        rendered_values[field_name] = (
            value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        )
    return render_template_text(template, rendered_values)
