from __future__ import annotations

from typing import Any, Iterable, Iterator

HTML_ESCAPE_CHARS: dict[str, str] = {
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
}


def _flatten_tokens(tokens: Iterable[Any]) -> Iterator[Any]:
    """Recursively flatten markdown-it token trees."""
    for token in tokens:
        yield token
        if token.children:
            yield from _flatten_tokens(token.children)


def prepare_markdown_content(content: str, escape_xml: bool = True) -> str:
    """Prepare content for markdown rendering, escaping HTML/XML outside code blocks."""
    if not escape_xml or not isinstance(content, str):
        return content

    from markdown_it import MarkdownIt

    parser = MarkdownIt()
    try:
        tokens = parser.parse(content)
    except Exception:
        result = content
        for char, replacement in HTML_ESCAPE_CHARS.items():
            result = result.replace(char, replacement)
        return result

    protected_ranges: list[tuple[int, int]] = []
    lines = content.split("\n")

    for token in _flatten_tokens(tokens):
        if token.map is not None:
            if token.type in ("fence", "code_block"):
                start_line = token.map[0]
                end_line = token.map[1]
                start_pos = sum(len(line) + 1 for line in lines[:start_line])
                end_pos = sum(len(line) + 1 for line in lines[:end_line])
                protected_ranges.append((start_pos, end_pos))

        if token.type == "code_inline":
            code_content = token.content
            if code_content:
                pattern = f"`{code_content}`"
                start = 0
                while True:
                    pos = content.find(pattern, start)
                    if pos == -1:
                        break
                    in_protected = any(s <= pos < e for s, e in protected_ranges)
                    if not in_protected:
                        protected_ranges.append((pos, pos + len(pattern)))
                    start = pos + len(pattern)

    import re

    fence_pattern = r"^```"
    fences = list(re.finditer(fence_pattern, content, re.MULTILINE))

    if len(fences) % 2 == 1:
        last_fence_pos = fences[-1].start()
        in_protected = any(s <= last_fence_pos < e for s, e in protected_ranges)
        if not in_protected:
            protected_ranges.append((last_fence_pos, len(content)))

    protected_ranges.sort(key=lambda x: x[0])

    merged_ranges: list[tuple[int, int]] = []
    for start, end in protected_ranges:
        if merged_ranges and start <= merged_ranges[-1][1]:
            merged_ranges[-1] = (merged_ranges[-1][0], max(end, merged_ranges[-1][1]))
        else:
            merged_ranges.append((start, end))

    result_segments: list[str] = []
    last_end = 0

    for start, end in merged_ranges:
        unprotected_text = content[last_end:start]
        for char, replacement in HTML_ESCAPE_CHARS.items():
            unprotected_text = unprotected_text.replace(char, replacement)
        result_segments.append(unprotected_text)

        result_segments.append(content[start:end])
        last_end = end

    remainder_text = content[last_end:]
    for char, replacement in HTML_ESCAPE_CHARS.items():
        remainder_text = remainder_text.replace(char, replacement)
    result_segments.append(remainder_text)

    return "".join(result_segments)


__all__ = ["HTML_ESCAPE_CHARS", "prepare_markdown_content"]
