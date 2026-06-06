from __future__ import annotations

import re
from functools import lru_cache
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

HTML_ESCAPE_CHARS: dict[str, str] = {
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
}
_BLOCKQUOTE_PREFIX_RE = re.compile(r"^(?P<prefix>[ ]{0,3}(?:>[ \t]?)+)")
_FENCE_PATTERN = re.compile(r"^```", re.MULTILINE)


def _flatten_tokens(tokens: Iterable[Any]) -> Iterator[Any]:
    """Recursively flatten markdown-it token trees."""
    for token in tokens:
        yield token
        if token.children:
            yield from _flatten_tokens(token.children)


def _is_protected_position(position: int, protected_ranges: list[tuple[int, int]]) -> bool:
    return any(start <= position < end for start, end in protected_ranges)


@lru_cache(maxsize=1)
def _get_markdown_parser() -> Any:
    from markdown_it import MarkdownIt

    return MarkdownIt()


@lru_cache(maxsize=32)
def _prepare_markdown_content_cached(content: str) -> str:
    parser = _get_markdown_parser()
    try:
        tokens = parser.parse(content)
    except Exception:
        return _escape_markdown_text(content)

    protected_ranges = _protected_markdown_ranges(content, tokens)
    return _escape_unprotected_markdown(content, protected_ranges)


def _protected_markdown_ranges(
    content: str,
    tokens: Iterable[Any],
) -> list[tuple[int, int]]:
    protected_ranges: list[tuple[int, int]] = []
    lines = content.split("\n")

    for token in _flatten_tokens(tokens):
        if token.map is not None and token.type in ("fence", "code_block"):
            protected_ranges.append(_line_map_to_range(lines, token.map))

        if token.type == "code_inline":
            protected_ranges.extend(
                _inline_code_ranges(content, token.content, protected_ranges)
            )

    incomplete_fence_range = _incomplete_fence_range(content, protected_ranges)
    if incomplete_fence_range is not None:
        protected_ranges.append(incomplete_fence_range)

    return _merge_ranges(protected_ranges)


def _line_map_to_range(lines: list[str], line_map: list[int]) -> tuple[int, int]:
    start_line, end_line = line_map
    start_pos = sum(len(line) + 1 for line in lines[:start_line])
    end_pos = sum(len(line) + 1 for line in lines[:end_line])
    return start_pos, end_pos


def _inline_code_ranges(
    content: str,
    code_content: str,
    protected_ranges: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    if not code_content:
        return []

    ranges: list[tuple[int, int]] = []
    pattern = f"`{code_content}`"
    start = 0
    while True:
        pos = content.find(pattern, start)
        if pos == -1:
            return ranges
        if not _is_protected_position(pos, protected_ranges):
            ranges.append((pos, pos + len(pattern)))
        start = pos + len(pattern)


def _incomplete_fence_range(
    content: str,
    protected_ranges: list[tuple[int, int]],
) -> tuple[int, int] | None:
    fences = list(_FENCE_PATTERN.finditer(content))
    if len(fences) % 2 == 0:
        return None

    last_fence_pos = fences[-1].start()
    if _is_protected_position(last_fence_pos, protected_ranges):
        return None
    return last_fence_pos, len(content)


def _merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    ranges.sort(key=lambda item: item[0])
    merged_ranges: list[tuple[int, int]] = []
    for start, end in ranges:
        if merged_ranges and start <= merged_ranges[-1][1]:
            merged_ranges[-1] = (merged_ranges[-1][0], max(end, merged_ranges[-1][1]))
        else:
            merged_ranges.append((start, end))
    return merged_ranges


def _escape_unprotected_markdown(
    content: str,
    protected_ranges: list[tuple[int, int]],
) -> str:
    result_segments: list[str] = []
    last_end = 0

    for start, end in protected_ranges:
        result_segments.append(_escape_markdown_text(content[last_end:start]))
        result_segments.append(content[start:end])
        last_end = end

    result_segments.append(_escape_markdown_text(content[last_end:]))

    return "".join(result_segments)


def _escape_markdown_text(text: str) -> str:
    if not text:
        return text

    escaped_lines: list[str] = []
    for raw_line in text.splitlines(keepends=True):
        line = raw_line.rstrip("\r\n")
        newline = raw_line[len(line) :]
        prefix = ""
        body = line

        match = _BLOCKQUOTE_PREFIX_RE.match(line)
        if match is not None:
            prefix = match.group("prefix")
            body = line[len(prefix) :]

        for char, replacement in HTML_ESCAPE_CHARS.items():
            body = body.replace(char, replacement)

        escaped_lines.append(f"{prefix}{body}{newline}")

    return "".join(escaped_lines)


def prepare_markdown_content(content: str, escape_xml: bool = True) -> str:
    """Prepare content for markdown rendering, escaping HTML/XML outside code blocks."""
    if not escape_xml or not isinstance(content, str):
        return content
    return _prepare_markdown_content_cached(content)


__all__ = ["HTML_ESCAPE_CHARS", "prepare_markdown_content"]
