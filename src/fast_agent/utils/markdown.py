"""Small Markdown text helpers."""

from __future__ import annotations

import re

from fast_agent.utils.text import strip_to_none

_MARKDOWN_TEXT_ESCAPES = str.maketrans(
    {char: f"\\{char}" for char in r"\[]*_`"}
)
_MARKDOWN_TABLE_CELL_ESCAPES = str.maketrans(
    {char: f"\\{char}" for char in r"\[]*`|"}
)
_BACKTICK_RUN_PATTERN = re.compile(r"`+")


def escape_markdown_text(value: str) -> str:
    return value.translate(_MARKDOWN_TEXT_ESCAPES)


def escape_markdown_table_cell(value: str) -> str:
    single_line = " ".join(value.splitlines())
    return single_line.translate(_MARKDOWN_TABLE_CELL_ESCAPES)


def _longest_backtick_run(value: str) -> int:
    return max(
        (len(match.group(0)) for match in _BACKTICK_RUN_PATTERN.finditer(value)),
        default=0,
    )


def _code_span_fence(value: str) -> str:
    return "`" * max(1, _longest_backtick_run(value) + 1)


def _code_span_needs_padding(value: str, fence: str) -> bool:
    return len(fence) > 1 or (
        value.startswith(" ") and value.endswith(" ") and strip_to_none(value) is not None
    )


def markdown_code_span(value: str) -> str:
    if not value:
        return "` `"

    fence = _code_span_fence(value)
    if not _code_span_needs_padding(value, fence):
        return f"{fence}{value}{fence}"
    return f"{fence} {value} {fence}"
