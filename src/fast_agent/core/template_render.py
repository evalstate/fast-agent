"""Small value-only ``{{placeholder}}`` renderer for prompt text."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Final, Mapping

_PLACEHOLDER_RE: Final[re.Pattern[str]] = re.compile(r"{{\s*([^}]+?)\s*}}")


@dataclass(frozen=True)
class TemplateRenderResult:
    text: str
    missing: tuple[str, ...] = ()


def extract_template_variables(text: str) -> set[str]:
    """Return placeholder names from ``text`` without braces."""
    return {
        name
        for match in _PLACEHOLDER_RE.finditer(text)
        if (name := match.group(1).strip())
    }


def render_template_text(text: str, values: Mapping[str, Any]) -> TemplateRenderResult:
    """Replace placeholders with supplied values, preserving unresolved placeholders."""
    missing: list[str] = []

    def replace(match: re.Match[str]) -> str:
        name = match.group(1).strip()
        if not name:
            return match.group(0)
        if name not in values:
            missing.append(name)
            return match.group(0)
        return str(values[name])

    rendered = _PLACEHOLDER_RE.sub(replace, text)
    return TemplateRenderResult(text=rendered, missing=tuple(dict.fromkeys(missing)))
