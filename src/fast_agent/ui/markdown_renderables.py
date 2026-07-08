from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

from pygments.lexers import get_lexer_by_name
from pygments.util import ClassNotFound
from rich.console import Group
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.text import Text

from fast_agent.ui.apply_patch_preview import style_apply_patch_preview_text
from fast_agent.ui.markdown_helpers import prepare_markdown_content
from fast_agent.utils.text import strip_casefold

_FENCE_OPEN_LINE_RE = re.compile(r"^\s{0,3}(?P<delim>`{3,}|~{3,})(?P<info>.*)$")
_FENCE_INFO_LINE_RE = re.compile(
    r"^(?P<indent>\s{0,3})(?P<delim>`{3,}|~{3,})(?P<spacing>[ \t]*)(?P<lang>\S+)(?P<rest>.*)$",
    re.MULTILINE,
)
_FENCE_LANGUAGE_ALIASES = {
    "apply_patch": "diff",
    "patch": "diff",
    "cmd": "batch",
    "shellscript": "bash",
    "terminal": "console",
}
_APPLY_PATCH_LANGUAGES = frozenset({"apply_patch", "patch"})
_TABLE_SEPARATOR_LINE_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$")


@dataclass(frozen=True)
class FencedCodeBlock:
    language: str
    code: str
    complete: bool


@dataclass(frozen=True)
class _FenceStart:
    delimiter: str
    info: str

    @property
    def char(self) -> str:
        return self.delimiter[0]

    @property
    def length(self) -> int:
        return len(self.delimiter)


@dataclass(frozen=True)
class _FenceState:
    char: str = "`"
    length: int = 3

    @classmethod
    def from_start(cls, start: _FenceStart) -> _FenceState:
        return cls(char=start.char, length=start.length)

    @property
    def closing_fence(self) -> str:
        return self.char * self.length


@dataclass(frozen=True)
class _ReferenceDefinition:
    label: str
    url: str
    title: str


@dataclass(frozen=True)
class _ParsedMarkdownSpan:
    kind: Literal["code", "definition"]
    start: int
    end: int
    raw_text: str = ""
    code: str = ""
    language: str = "text"


@dataclass(frozen=True)
class _ParsedMarkdownDocument:
    spans: tuple[_ParsedMarkdownSpan, ...]
    reference_definitions: tuple[_ReferenceDefinition, ...]


@dataclass(frozen=True)
class _RenderableChunk:
    kind: Literal["markdown", "code"]
    text: str
    language: str = "text"
    reference_definitions: str = ""


@lru_cache(maxsize=64)
def _has_lexer(language: str) -> bool:
    try:
        get_lexer_by_name(language)
    except ClassNotFound:
        return False
    return True


def _normalize_code_language(language: str) -> str:
    normalized = strip_casefold(language)
    if not normalized:
        return "text"

    alias = _FENCE_LANGUAGE_ALIASES.get(normalized)
    if alias is not None and _has_lexer(alias):
        return alias
    if _has_lexer(normalized):
        return normalized
    return normalized


def _is_apply_patch_language(language: str) -> bool:
    return strip_casefold(language) in _APPLY_PATCH_LANGUAGES


def _line_indent_width(line: str) -> tuple[str, int]:
    stripped = line.lstrip(" ")
    return stripped, len(line) - len(stripped)


def _top_level_fence_start(line: str) -> _FenceStart | None:
    stripped, indent = _line_indent_width(line)
    if indent > 3:
        return None

    opening = _FENCE_OPEN_LINE_RE.match(line)
    if opening is None:
        return None

    delimiter = opening.group("delim")
    info = opening.group("info")
    if delimiter[0] == "`" and "`" in info:
        return None
    return _FenceStart(delimiter=delimiter, info=info)


def _is_closing_fence_line(line: str, fence: _FenceState) -> bool:
    stripped, indent = _line_indent_width(line)
    if indent > 3 or not stripped or stripped[0] != fence.char:
        return False

    marker_len = 0
    while marker_len < len(stripped) and stripped[marker_len] == fence.char:
        marker_len += 1
    return marker_len >= fence.length and stripped[marker_len:].strip() == ""


def _split_line_ending(raw_line: str) -> tuple[str, str]:
    line = raw_line.rstrip("\r\n")
    return line, raw_line[len(line) :]


def _trim_blank_line_bounds(lines: list[str]) -> tuple[int, int] | None:
    start_index = 0
    while start_index < len(lines) and not lines[start_index].strip():
        start_index += 1
    if start_index >= len(lines):
        return None

    end_index = len(lines) - 1
    while end_index >= start_index and not lines[end_index].strip():
        end_index -= 1
    return start_index, end_index


def _find_closing_fence_index(
    lines: list[str],
    *,
    start_index: int,
    end_index: int,
    fence: _FenceState,
) -> int | None:
    for index in range(start_index + 1, end_index + 1):
        if _is_closing_fence_line(lines[index], fence):
            return index
    return None


def _rewrite_fence_languages(text: str) -> str:
    if "```" not in text and "~~~" not in text:
        return text

    rewritten_lines: list[str] = []
    in_fence = False
    fence = _FenceState()

    for raw_line in text.splitlines(keepends=True):
        line, newline = _split_line_ending(raw_line)

        if not in_fence:
            start = _top_level_fence_start(line)
            if start is None:
                rewritten_lines.append(raw_line)
                continue

            output_line = raw_line
            info_match = _FENCE_INFO_LINE_RE.match(line)
            if info_match is not None:
                language = info_match.group("lang")
                rewritten = _normalize_code_language(language)
                if rewritten != language:
                    output_line = (
                        f"{info_match.group('indent')}{info_match.group('delim')}"
                        f"{info_match.group('spacing')}{rewritten}"
                        f"{info_match.group('rest')}{newline}"
                    )

            rewritten_lines.append(output_line)
            in_fence = True
            fence = _FenceState.from_start(start)
            continue

        rewritten_lines.append(raw_line)
        if _is_closing_fence_line(line, fence):
            in_fence = False

    return "".join(rewritten_lines)


def extract_single_fenced_code_block(text: str) -> FencedCodeBlock | None:
    if not text:
        return None

    lines = text.splitlines()
    bounds = _trim_blank_line_bounds(lines)
    if bounds is None:
        return None
    start_index, end_index = bounds

    start = _top_level_fence_start(lines[start_index])
    if start is None:
        return None

    fence = _FenceState.from_start(start)
    language = start.info.strip().split(" ", 1)[0] or "text"

    closing_index = _find_closing_fence_index(
        lines,
        start_index=start_index,
        end_index=end_index,
        fence=fence,
    )
    if closing_index is None:
        return FencedCodeBlock(
            language=language,
            code="\n".join(lines[start_index + 1 :]),
            complete=False,
        )

    if closing_index != end_index:
        return None

    return FencedCodeBlock(
        language=language,
        code="\n".join(lines[start_index + 1 : closing_index]),
        complete=True,
    )


def close_incomplete_code_blocks(text: str) -> str:
    if "```" not in text and "~~~" not in text:
        return text

    in_fence = False
    fence = _FenceState()

    for line in text.splitlines():
        if not in_fence:
            start = _top_level_fence_start(line)
            if start is None:
                continue

            in_fence = True
            fence = _FenceState.from_start(start)
            continue

        if _is_closing_fence_line(line, fence):
            in_fence = False

    if not in_fence:
        return text

    if text.endswith("\n"):
        return f"{text}{fence.closing_fence}\n"
    return f"{text}\n{fence.closing_fence}\n"


@lru_cache(maxsize=1)
def _get_markdown_parser():
    from markdown_it import MarkdownIt

    return MarkdownIt(options_update={"inline_definitions": True}).enable("table")


def _line_start_offsets(text: str) -> list[int]:
    offsets = [0]
    running = 0
    for line in text.split("\n"):
        running += len(line) + 1
        offsets.append(running)
    return offsets


def _format_reference_definition_block(
    definitions: tuple[_ReferenceDefinition, ...],
) -> str:
    if not definitions:
        return ""

    lines: list[str] = []
    for definition in definitions:
        title = definition.title.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")
        if title:
            lines.append(f'[{definition.label}]: {definition.url} "{title}"')
        else:
            lines.append(f"[{definition.label}]: {definition.url}")
    return "\n".join(lines)


def _parse_markdown_document(text: str) -> _ParsedMarkdownDocument | None:
    parser = _get_markdown_parser()
    env: dict[str, object] = {}
    try:
        tokens = parser.parse(text, env)
    except Exception:
        return None

    offsets = _line_start_offsets(text)
    spans: list[_ParsedMarkdownSpan] = []
    reference_definitions: list[_ReferenceDefinition] = []
    seen_reference_ids: set[str] = set()

    for token in tokens:
        if token.type not in ("fence", "code_block", "definition") or token.map is None:
            continue
        if getattr(token, "level", 0) != 0:
            continue

        start_line, end_line = token.map
        start = offsets[start_line]
        end = min(offsets[end_line], len(text))

        if token.type == "definition":
            meta = getattr(token, "meta", {}) or {}
            identifier = str(meta.get("id", "") or "")
            if identifier and identifier not in seen_reference_ids:
                seen_reference_ids.add(identifier)
                reference_definitions.append(
                    _ReferenceDefinition(
                        label=str(meta.get("label", "") or identifier),
                        url=str(meta.get("url", "") or ""),
                        title=str(meta.get("title", "") or ""),
                    )
                )
            spans.append(_ParsedMarkdownSpan(kind="definition", start=start, end=end))
            continue

        raw_text = text[start:end]
        info = (getattr(token, "info", "") or "").strip()
        language = info.split(" ", 1)[0] or "text"
        code = getattr(token, "content", "") or ""

        if token.type == "fence":
            fenced = extract_single_fenced_code_block(raw_text)
            if fenced is not None:
                code = fenced.code
                language = fenced.language or language

        spans.append(
            _ParsedMarkdownSpan(
                kind="code",
                start=start,
                end=end,
                raw_text=raw_text,
                code=code,
                language=language,
            )
        )

    return _ParsedMarkdownDocument(
        spans=tuple(spans),
        reference_definitions=tuple(reference_definitions),
    )


def _build_renderable_chunks(text: str) -> tuple[_RenderableChunk, ...] | None:
    document = _parse_markdown_document(text)
    if document is None:
        return None
    if not document.spans:
        return ()

    chunks: list[_RenderableChunk] = []
    reference_definitions = _format_reference_definition_block(document.reference_definitions)
    cursor = 0
    previous_span_kind: Literal["code", "definition"] | None = None
    for span in document.spans:
        if span.start > cursor:
            gap_text = text[cursor : span.start]
            if not (
                not gap_text.strip()
                and (span.kind == "definition" or previous_span_kind == "definition")
            ):
                chunks.append(
                    _RenderableChunk(
                        kind="markdown",
                        text=gap_text,
                        reference_definitions=reference_definitions,
                    )
                )
        if span.kind == "code":
            chunks.append(
                _RenderableChunk(
                    kind="code",
                    text=span.code,
                    language=span.language,
                )
            )
        cursor = span.end
        previous_span_kind = span.kind

    if cursor < len(text):
        tail_text = text[cursor:]
        if not (not tail_text.strip() and previous_span_kind == "definition"):
            chunks.append(
                _RenderableChunk(
                    kind="markdown",
                    text=tail_text,
                    reference_definitions=reference_definitions,
                )
            )

    return tuple(chunk for chunk in chunks if chunk.kind == "code" or chunk.text)


def _render_markdown_chunk(
    text: str,
    *,
    code_theme: str,
    escape_xml: bool,
    reference_definitions: str = "",
):
    if not text:
        return Text("")
    if not text.strip():
        return Text(text)

    prepared = prepare_markdown_content(text, escape_xml)
    prepared = _rewrite_fence_languages(prepared)
    if reference_definitions:
        prepared = f"{prepared}\n\n{reference_definitions}"
    if not prepared:
        return Text("")
    return Markdown(prepared, code_theme=code_theme)


def _ensure_final_table_row_newline(text: str) -> str:
    if not text.endswith("|"):
        return text

    lines = text.split("\n")
    if len(lines) < 3:
        return text

    final_line = lines[-1]
    separator_line = lines[-2]
    if "|" in final_line and _TABLE_SEPARATOR_LINE_RE.match(separator_line):
        return f"{text}\n"
    return text


def _render_code_chunk(
    text: str,
    *,
    language: str,
    code_theme: str,
    code_word_wrap: bool,
    pad_code_blocks: bool = True,
):
    render_text = f"\n{text}\n" if pad_code_blocks else text
    if _is_apply_patch_language(language):
        return style_apply_patch_preview_text(render_text, default_style="dim")
    return Syntax(
        render_text,
        _normalize_code_language(language),
        theme=code_theme,
        line_numbers=False,
        word_wrap=code_word_wrap,
    )


def _append_cursor_to_last_chunk(
    chunks: tuple[_RenderableChunk, ...],
    cursor_suffix: str,
) -> list[_RenderableChunk]:
    render_chunks = list(chunks)
    if not cursor_suffix:
        return render_chunks

    last = render_chunks[-1]
    render_chunks[-1] = _RenderableChunk(
        kind=last.kind,
        text=last.text + cursor_suffix,
        language=last.language,
        reference_definitions=last.reference_definitions,
    )
    return render_chunks


def _render_chunks(
    chunks: tuple[_RenderableChunk, ...],
    *,
    cursor_suffix: str,
    code_theme: str,
    escape_xml: bool,
    code_word_wrap: bool,
    pad_code_blocks: bool,
):
    renderables = [
        _render_markdown_chunk(
            chunk.text,
            code_theme=code_theme,
            escape_xml=escape_xml,
            reference_definitions=chunk.reference_definitions,
        )
        if chunk.kind == "markdown"
        else _render_code_chunk(
            chunk.text,
            language=chunk.language,
            code_theme=code_theme,
            code_word_wrap=code_word_wrap,
            pad_code_blocks=pad_code_blocks,
        )
        for chunk in _append_cursor_to_last_chunk(chunks, cursor_suffix)
    ]
    if len(renderables) == 1:
        return renderables[0]
    return Group(*renderables)


def build_markdown_renderable(
    text: str,
    *,
    code_theme: str,
    escape_xml: bool,
    cursor_suffix: str = "",
    close_incomplete_fences: bool = False,
    render_fences_with_syntax: bool = True,
    code_word_wrap: bool = True,
    pad_code_blocks: bool = True,
):
    if not text and not cursor_suffix:
        return Text("")

    if close_incomplete_fences:
        text = close_incomplete_code_blocks(text)

    if not cursor_suffix:
        text = _ensure_final_table_row_newline(text)

    if render_fences_with_syntax:
        code_block = extract_single_fenced_code_block(text)
        if code_block is not None:
            code = code_block.code
            if cursor_suffix:
                code += cursor_suffix
            return _render_code_chunk(
                code,
                language=code_block.language,
                code_theme=code_theme,
                code_word_wrap=code_word_wrap,
                pad_code_blocks=pad_code_blocks,
            )

        chunks = _build_renderable_chunks(text)
        if chunks:
            return _render_chunks(
                chunks,
                cursor_suffix=cursor_suffix,
                code_theme=code_theme,
                escape_xml=escape_xml,
                code_word_wrap=code_word_wrap,
                pad_code_blocks=pad_code_blocks,
            )

    prepared = prepare_markdown_content(text, escape_xml)
    prepared = _rewrite_fence_languages(prepared)
    if cursor_suffix:
        prepared += cursor_suffix
    if not prepared:
        return Text("")
    return Markdown(prepared, code_theme=code_theme)


__all__ = [
    "FencedCodeBlock",
    "build_markdown_renderable",
    "close_incomplete_code_blocks",
    "extract_single_fenced_code_block",
]
