"""Small Markdown block helpers shared by command renderers."""

from __future__ import annotations

import textwrap

from fast_agent.utils.markdown import escape_markdown_text
from fast_agent.utils.text import strip_to_none


def normalize_markdown_heading(heading: str) -> str:
    return " ".join(heading.lstrip("# ").split())


def markdown_heading(heading: str, *, level: int = 1) -> str:
    normalized = normalize_markdown_heading(heading)
    if not normalized:
        return ""
    return f"{'#' * max(1, level)} {escape_markdown_text(normalized)}"


def wrapped_quote_lines(
    text: str | None,
    *,
    prefix: str = "> ",
    width: int = 88,
    max_lines: int = 4,
) -> list[str]:
    normalized = strip_to_none(text)
    if normalized is None or max_lines <= 0:
        return []

    wrapped = textwrap.wrap(normalized, width=max(1, width))
    lines = [f"{prefix}{line}" for line in wrapped[:max_lines]]
    if len(wrapped) > max_lines:
        lines.append(f"{prefix}…")
    return lines
