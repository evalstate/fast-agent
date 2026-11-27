from __future__ import annotations

import re
import textwrap
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, ClassVar, Literal

from pygments.lexers import get_lexer_by_name
from pygments.util import ClassNotFound
from rich.console import Console, ConsoleOptions, Group, RenderableType, RenderResult
from rich.errors import StyleSyntaxError
from rich.markdown import (
    BlockQuote,
    Heading,
    HorizontalRule,
    ListItem,
    Markdown,
    MarkdownElement,
    Paragraph,
)
from rich.markdown import CodeBlock as RichCodeBlock
from rich.segment import Segment
from rich.style import Style, StyleType
from rich.syntax import Syntax
from rich.text import Text

from fast_agent.ui.apply_patch_preview import style_apply_patch_preview_text
from fast_agent.ui.markdown.content import prepare_markdown_content
from fast_agent.ui.message_styles import SHORT_RULE_WIDTH, build_code_rule
from fast_agent.utils.text import strip_casefold

if TYPE_CHECKING:
    from markdown_it.token import Token

    from fast_agent.command_actions.models import MarkdownTextStyle

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
    complete: bool = True


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
    complete: bool = True
    cursor_suffix: str = ""


@dataclass(frozen=True)
class _StyleMatch:
    start: int
    end: int
    style: Style


@dataclass(frozen=True)
class _LiteralStyledRenderable:
    renderable: RenderableType
    styles: tuple[MarkdownTextStyle, ...]

    def __rich_console__(
        self,
        console: Console,
        options: ConsoleOptions,
    ) -> RenderResult:
        segments = tuple(console.render(self.renderable, options))
        yield from _apply_literal_styles(segments, self.styles)


def _literal_style_matches(
    text: str,
    styles: tuple[MarkdownTextStyle, ...],
) -> tuple[_StyleMatch, ...]:
    matches: list[_StyleMatch] = []
    for text_style in styles:
        if not text_style.text:
            continue
        try:
            style = Style.parse(text_style.style)
        except StyleSyntaxError:
            continue
        start = 0
        while (index := text.find(text_style.text, start)) >= 0:
            end = index + len(text_style.text)
            starts_inside_token = (
                index > 0
                and (text_style.text[0].isalnum() or text_style.text[0] == "_")
                and (text[index - 1].isalnum() or text[index - 1] == "_")
            )
            ends_inside_token = (
                end < len(text)
                and (text_style.text[-1].isalnum() or text_style.text[-1] == "_")
                and (text[end].isalnum() or text[end] == "_")
            )
            if not starts_inside_token and not ends_inside_token:
                matches.append(_StyleMatch(start=index, end=end, style=style))
            start = end
    return tuple(matches)


def _apply_literal_styles(
    segments: tuple[Segment, ...],
    styles: tuple[MarkdownTextStyle, ...],
) -> tuple[Segment, ...]:
    visible = "".join(segment.text for segment in segments if not segment.control)
    matches = _literal_style_matches(visible, styles)
    if not matches:
        return segments

    output: list[Segment] = []
    offset = 0
    for segment in segments:
        if segment.control or not segment.text:
            output.append(segment)
            continue

        start = offset
        end = start + len(segment.text)
        offset = end
        boundaries = {start, end}
        for match in matches:
            if match.start < end and match.end > start:
                boundaries.add(max(start, match.start))
                boundaries.add(min(end, match.end))

        ordered = sorted(boundaries)
        for chunk_start, chunk_end in zip(ordered, ordered[1:], strict=False):
            style = segment.style
            for match in matches:
                if match.start <= chunk_start and chunk_end <= match.end:
                    style = (style or Style()) + match.style
            output.append(
                Segment(
                    segment.text[chunk_start - start : chunk_end - start],
                    style,
                    segment.control,
                )
            )
    return tuple(output)


def style_markdown_renderable(
    renderable: RenderableType,
    styles: tuple[MarkdownTextStyle, ...],
) -> RenderableType:
    """Apply Rich styles to literal visible text without changing Markdown source."""

    return _LiteralStyledRenderable(renderable=renderable, styles=styles)


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
        raw_lines = text.splitlines(keepends=True)
        return FencedCodeBlock(
            language=language,
            code="".join(raw_lines[start_index + 1 :]),
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
        if token.level != 0:
            continue

        start_line, end_line = token.map
        start = offsets[start_line]
        end = min(offsets[end_line], len(text))

        if token.type == "definition":
            meta = token.meta
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
        info = (token.info or "").strip()
        language = info.split(" ", 1)[0] or "text"
        code = token.content or ""
        complete = True

        if token.type == "fence":
            fenced = extract_single_fenced_code_block(raw_text)
            if fenced is not None:
                code = fenced.code
                language = fenced.language or language
                complete = fenced.complete

        spans.append(
            _ParsedMarkdownSpan(
                kind="code",
                start=start,
                end=end,
                raw_text=raw_text,
                code=code,
                language=language,
                complete=complete,
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
                    complete=span.complete,
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


@lru_cache(maxsize=64)
def syntax_theme_has_background(code_theme: str) -> bool:
    """Return True when the Rich/Pygments syntax theme paints its own background."""
    theme = Syntax.get_theme(code_theme)
    return theme.get_background_style().bgcolor is not None


class _ReflowSafeSyntax(Syntax):
    """Syntax renderable that emits unbroken logical lines when options.no_wrap is set."""

    def _prepare_code(self) -> str:
        processed = self.code if self.code.endswith("\n") else f"{self.code}\n"
        if self.dedent:
            processed = textwrap.dedent(processed)
        return processed.expandtabs(self.tab_size)

    def __rich_console__(
        self,
        console: Console,
        options: ConsoleOptions,
    ) -> RenderResult:
        if not options.no_wrap:
            yield from super().__rich_console__(console, options)
            return
        highlighted = self.highlight(self._prepare_code())
        if highlighted.plain.endswith("\n"):
            highlighted.remove_suffix("\n")
        for line in highlighted.split("\n", allow_blank=True):
            yield from line.render(console, end="\n")


class CursorSyntax(_ReflowSafeSyntax):
    """Append the stream cursor after lexing so it cannot become an error token."""

    def __init__(
        self,
        code: str,
        lexer: str,
        *,
        theme: str,
        word_wrap: bool,
        cursor_suffix: str,
    ) -> None:
        self._cursor_suffix = cursor_suffix
        super().__init__(
            code,
            lexer,
            theme=theme,
            line_numbers=False,
            word_wrap=word_wrap,
        )

    def highlight(
        self,
        code: str,
        line_range: tuple[int | None, int | None] | None = None,
    ) -> Text:
        highlighted = super().highlight(code, line_range)
        if self._cursor_suffix and code.endswith("\n") and not self.code.endswith("\n"):
            highlighted.remove_suffix("\n")
            highlighted.append(self._cursor_suffix)
            highlighted.append("\n")
        else:
            highlighted.append(self._cursor_suffix)
        return highlighted


def _token_is_open_fence(token: Token) -> bool:
    if token.type != "fence" or token.map is None:
        return False
    start_line, end_line = token.map
    body_line_count = len((token.content or "").splitlines())
    return (end_line - start_line) < (2 + body_line_count)


class _CodeRule:
    def __init__(self, label: str = "") -> None:
        self.label = label

    def __rich_console__(
        self,
        console: Console,
        options: ConsoleOptions,
    ) -> RenderResult:
        del console
        yield build_code_rule(self.label, max_width=options.max_width)


class _ReflowSafeCodeBlock(RichCodeBlock):
    """Rich Markdown CodeBlock with live incomplete-fence and post-lex cursor support."""

    def __init__(
        self,
        lexer_name: str,
        theme: str,
        *,
        is_open: bool = False,
        cursor_suffix: str = "",
    ) -> None:
        super().__init__(lexer_name, theme)
        self._is_open = is_open
        self._cursor_suffix = cursor_suffix

    @classmethod
    def create(cls, markdown: Markdown, token: Token) -> RichCodeBlock:
        node_info = token.info or ""
        lexer_name = node_info.partition(" ")[0] or "text"
        is_open = False
        cursor_suffix = ""
        if isinstance(markdown, _ReflowSafeMarkdown):
            is_open = token is markdown._open_fence_token
            if is_open:
                cursor_suffix = markdown._cursor_suffix
        return cls(
            lexer_name,
            markdown.code_theme,
            is_open=is_open,
            cursor_suffix=cursor_suffix,
        )

    def __rich_console__(
        self,
        console: Console,
        options: ConsoleOptions,
    ) -> RenderResult:
        del console, options
        raw_code = str(self.text)
        code = raw_code if self._is_open else raw_code.rstrip()
        normalized_language = _normalize_code_language(self.lexer_name)
        if self._cursor_suffix:
            yield CursorSyntax(
                code,
                normalized_language,
                theme=self.theme,
                word_wrap=True,
                cursor_suffix=self._cursor_suffix,
            )
            return
        yield _ReflowSafeSyntax(
            code,
            normalized_language,
            theme=self.theme,
            word_wrap=True,
            padding=1,
        )


class _FramedMarkdownCodeBlock(_ReflowSafeCodeBlock):
    """Rich Markdown CodeBlock framed by short rules for backgroundless themes."""

    def __rich_console__(
        self,
        console: Console,
        options: ConsoleOptions,
    ) -> RenderResult:
        del console
        raw_code = str(self.text)
        code = raw_code if self._is_open else raw_code.rstrip("\n")
        normalized_language = _normalize_code_language(self.lexer_name)
        label = "" if normalized_language == "text" else normalized_language
        syntax: Syntax = (
            CursorSyntax(
                code,
                normalized_language,
                theme=self.theme,
                word_wrap=True,
                cursor_suffix=self._cursor_suffix,
            )
            if self._cursor_suffix
            else _ReflowSafeSyntax(
                code,
                normalized_language,
                theme=self.theme,
                word_wrap=True,
            )
        )
        yield build_code_rule(label, max_width=options.max_width)
        yield syntax
        if not self._is_open:
            yield build_code_rule(max_width=options.max_width)


class _ReflowSafeParagraph(Paragraph):
    """Paragraph without right-edge space padding so terminal reflow stays clean."""

    @classmethod
    def create(cls, markdown: Markdown, token: Token) -> Paragraph:
        del token
        return cls(justify=markdown.justify or "default")


class _ReflowSafeHeading(Heading):
    """Heading rendered left-aligned without right-edge space padding."""

    def __rich_console__(
        self,
        console: Console,
        options: ConsoleOptions,
    ) -> RenderResult:
        del console, options
        text = self.text.copy()
        text.justify = "default"
        yield text


def _render_container_lines(
    console: Console,
    renderable: RenderableType,
    render_options: ConsoleOptions,
    *,
    style: StyleType,
) -> list[list[Segment]]:
    resolved_style = console.get_style(style) if style else None
    if render_options.no_wrap:
        segments = console.render(renderable, render_options)
        if resolved_style:
            segments = Segment.apply_style(segments, resolved_style)
        return list(Segment.split_lines(segments))
    return console.render_lines(renderable, render_options, style=resolved_style, pad=False)


class _ReflowSafeListItem(ListItem):
    """List item whose inner lines omit right-edge space padding."""

    def render_bullet(self, console: Console, options: ConsoleOptions) -> RenderResult:
        render_options = options.update(width=max(1, options.max_width - 3))
        lines = _render_container_lines(console, self.elements, render_options, style=self.style)
        bullet_style = console.get_style("markdown.item.bullet", default="none")

        bullet = Segment(" • ", bullet_style)
        padding = Segment(" " * 3, bullet_style)
        new_line = Segment("\n")
        for index, line in enumerate(lines):
            if index > 0 and not line:
                yield new_line
                continue
            yield bullet if index == 0 else padding
            yield from line
            yield new_line

    def render_number(
        self,
        console: Console,
        options: ConsoleOptions,
        number: int,
        last_number: int,
    ) -> RenderResult:
        number_width = len(str(last_number)) + 2
        render_options = options.update(width=max(1, options.max_width - number_width))
        lines = _render_container_lines(console, self.elements, render_options, style=self.style)
        number_style = console.get_style("markdown.item.number", default="none")

        new_line = Segment("\n")
        padding = Segment(" " * number_width, number_style)
        numeral = Segment(f"{number}".rjust(number_width - 1) + " ", number_style)
        for index, line in enumerate(lines):
            if index > 0 and not line:
                yield new_line
                continue
            yield numeral if index == 0 else padding
            yield from line
            yield new_line


class _ReflowSafeBlockQuote(BlockQuote):
    """Block quote whose inner lines omit right-edge space padding."""

    def __rich_console__(
        self,
        console: Console,
        options: ConsoleOptions,
    ) -> RenderResult:
        render_options = options.update(width=max(1, options.max_width - 4))
        lines = _render_container_lines(console, self.elements, render_options, style=self.style)
        style = self.style
        new_line = Segment("\n")
        padding = Segment("▌ ", style)
        empty_padding = Segment("▌", style)
        for line in lines:
            if line:
                yield padding
                yield from line
            else:
                yield empty_padding
            yield new_line


class _ReflowSafeHorizontalRule(HorizontalRule):
    """Horizontal rule capped to short-rule width so narrowing panes do not wrap dashes."""

    def __rich_console__(
        self,
        console: Console,
        options: ConsoleOptions,
    ) -> RenderResult:
        style = console.get_style("markdown.hr", default="dim")
        rule_width = max(0, min(options.max_width, SHORT_RULE_WIDTH))
        yield Text("─" * rule_width, style=style)
        yield Text()


class _ReflowSafeMarkdown(Markdown):
    elements: ClassVar[dict[str, type[MarkdownElement]]] = {
        **Markdown.elements,
        "paragraph_open": _ReflowSafeParagraph,
        "heading_open": _ReflowSafeHeading,
        "list_item_open": _ReflowSafeListItem,
        "blockquote_open": _ReflowSafeBlockQuote,
        "hr": _ReflowSafeHorizontalRule,
        "fence": _ReflowSafeCodeBlock,
        "code_block": _ReflowSafeCodeBlock,
    }

    def __init__(
        self,
        markup: str,
        *,
        code_theme: str = "monokai",
        style: str = "",
        live_incomplete_fences: bool = False,
        cursor_suffix: str = "",
        reference_definitions: str = "",
    ) -> None:
        self._live_incomplete_fences = live_incomplete_fences
        self._cursor_suffix = cursor_suffix
        self._open_fence_token: Token | None = None
        self._trailing_cursor = ""
        full_markup = f"{markup}\n\n{reference_definitions}" if reference_definitions else markup
        super().__init__(full_markup, code_theme=code_theme, style=style)

        last_leaf = next(
            (token for token in reversed(self.parsed) if token.nesting == 0),
            None,
        )
        if live_incomplete_fences and last_leaf is not None and _token_is_open_fence(last_leaf):
            body_tokens = self.parsed
            if reference_definitions:
                markup_only = Markdown(markup, code_theme=code_theme, style=style)
                body_tokens = markup_only.parsed
            max_mapped_end = max(
                (token.map[1] for token in body_tokens if token.map is not None),
                default=0,
            )
            if max_mapped_end >= len(markup.splitlines()):
                self._open_fence_token = last_leaf

        if not cursor_suffix:
            return

        if last_leaf is None or last_leaf.type in {"fence", "code_block", "hr"}:
            if self._open_fence_token is None:
                self._trailing_cursor = cursor_suffix
            return

        with_cursor = f"{markup}{cursor_suffix}"
        full_with_cursor = (
            f"{with_cursor}\n\n{reference_definitions}" if reference_definitions else with_cursor
        )
        super().__init__(full_with_cursor, code_theme=code_theme, style=style)

    def __rich_console__(
        self,
        console: Console,
        options: ConsoleOptions,
    ) -> RenderResult:
        yield from super().__rich_console__(console, options)
        if self._trailing_cursor:
            style = console.get_style(self.style, default="none")
            yield Text(self._trailing_cursor, style=style)


class _FramedMarkdown(_ReflowSafeMarkdown):
    elements: ClassVar[dict[str, type[MarkdownElement]]] = {
        **_ReflowSafeMarkdown.elements,
        "fence": _FramedMarkdownCodeBlock,
        "code_block": _FramedMarkdownCodeBlock,
    }


def build_rich_markdown(
    text: str,
    *,
    code_theme: str,
    style: str = "",
    live_incomplete_fences: bool = False,
    cursor_suffix: str = "",
    reference_definitions: str = "",
) -> Markdown:
    """Build a reflow-safe Rich Markdown renderable with short-rule framing for palette themes."""
    markdown_cls = (
        _ReflowSafeMarkdown if syntax_theme_has_background(code_theme) else _FramedMarkdown
    )
    return markdown_cls(
        text,
        code_theme=code_theme,
        style=style,
        live_incomplete_fences=live_incomplete_fences,
        cursor_suffix=cursor_suffix,
        reference_definitions=reference_definitions,
    )


def _render_markdown_chunk(
    text: str,
    *,
    code_theme: str,
    escape_xml: bool,
    reference_definitions: str = "",
    live_incomplete_fences: bool = False,
    cursor_suffix: str = "",
) -> RenderableType:
    if not text:
        return Text(cursor_suffix)
    if not text.strip():
        return Text(f"{text}{cursor_suffix}")

    prepared = prepare_markdown_content(text, escape_xml)
    prepared = _rewrite_fence_languages(prepared)
    if not prepared:
        return Text(cursor_suffix)
    return build_rich_markdown(
        prepared,
        code_theme=code_theme,
        live_incomplete_fences=live_incomplete_fences,
        cursor_suffix=cursor_suffix,
        reference_definitions=reference_definitions,
    )


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
    is_open: bool = False,
    cursor_suffix: str = "",
) -> RenderableType:
    if _is_apply_patch_language(language):
        body_text = f"{text}{cursor_suffix}"
        render_text = f"\n{body_text}\n" if pad_code_blocks and not is_open else body_text
        return style_apply_patch_preview_text(render_text, default_style="dim")

    normalized_language = _normalize_code_language(language)
    syntax: Syntax = (
        CursorSyntax(
            text,
            normalized_language,
            theme=code_theme,
            word_wrap=code_word_wrap,
            cursor_suffix=cursor_suffix,
        )
        if cursor_suffix
        else _ReflowSafeSyntax(
            text,
            normalized_language,
            theme=code_theme,
            line_numbers=False,
            word_wrap=code_word_wrap,
        )
    )
    if not syntax_theme_has_background(code_theme):
        label = "" if normalized_language == "text" else normalized_language
        if is_open:
            return Group(_CodeRule(label), syntax)
        return Group(_CodeRule(label), syntax, _CodeRule())

    if pad_code_blocks and not cursor_suffix:
        render_text = f"\n{text}\n"
        return _ReflowSafeSyntax(
            render_text,
            normalized_language,
            theme=code_theme,
            line_numbers=False,
            word_wrap=code_word_wrap,
        )
    if pad_code_blocks and cursor_suffix:
        top_pad = Text("")
        if is_open:
            return Group(top_pad, syntax)
        return Group(top_pad, syntax, Text(""))
    return syntax


def _append_cursor_to_last_chunk(
    chunks: tuple[_RenderableChunk, ...],
    cursor_suffix: str,
    *,
    live_incomplete_fences: bool = False,
) -> list[_RenderableChunk]:
    render_chunks = list(chunks)
    if not cursor_suffix:
        return render_chunks

    last = render_chunks[-1]
    if last.kind == "code":
        is_open = live_incomplete_fences and not last.complete
        if is_open:
            render_chunks[-1] = _RenderableChunk(
                kind="code",
                text=last.text,
                language=last.language,
                reference_definitions=last.reference_definitions,
                complete=last.complete,
                cursor_suffix=cursor_suffix,
            )
        else:
            render_chunks.append(
                _RenderableChunk(
                    kind="markdown",
                    text="",
                    cursor_suffix=cursor_suffix,
                )
            )
        return render_chunks

    render_chunks[-1] = _RenderableChunk(
        kind="markdown",
        text=last.text,
        language=last.language,
        reference_definitions=last.reference_definitions,
        complete=last.complete,
        cursor_suffix=cursor_suffix,
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
    live_incomplete_fences: bool = False,
) -> RenderableType:
    renderables = [
        _render_markdown_chunk(
            chunk.text,
            code_theme=code_theme,
            escape_xml=escape_xml,
            reference_definitions=chunk.reference_definitions,
            live_incomplete_fences=live_incomplete_fences,
            cursor_suffix=chunk.cursor_suffix,
        )
        if chunk.kind == "markdown"
        else _render_code_chunk(
            chunk.text,
            language=chunk.language,
            code_theme=code_theme,
            code_word_wrap=code_word_wrap,
            pad_code_blocks=pad_code_blocks,
            is_open=live_incomplete_fences and not chunk.complete,
            cursor_suffix=chunk.cursor_suffix,
        )
        for chunk in _append_cursor_to_last_chunk(
            chunks,
            cursor_suffix,
            live_incomplete_fences=live_incomplete_fences,
        )
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
) -> RenderableType:
    if not text and not cursor_suffix:
        return Text("")

    live_incomplete_fences = close_incomplete_fences
    if not cursor_suffix:
        text = _ensure_final_table_row_newline(text)

    if render_fences_with_syntax:
        code_block = extract_single_fenced_code_block(text)
        if code_block is not None:
            is_open = live_incomplete_fences and not code_block.complete
            rendered_code = _render_code_chunk(
                code_block.code,
                language=code_block.language,
                code_theme=code_theme,
                code_word_wrap=code_word_wrap,
                pad_code_blocks=pad_code_blocks,
                is_open=is_open,
                cursor_suffix=cursor_suffix if is_open else "",
            )
            if cursor_suffix and not is_open:
                return Group(rendered_code, Text(cursor_suffix))
            return rendered_code

        chunks = _build_renderable_chunks(text)
        if chunks:
            return _render_chunks(
                chunks,
                cursor_suffix=cursor_suffix,
                code_theme=code_theme,
                escape_xml=escape_xml,
                code_word_wrap=code_word_wrap,
                pad_code_blocks=pad_code_blocks,
                live_incomplete_fences=live_incomplete_fences,
            )

    prepared = prepare_markdown_content(text, escape_xml)
    prepared = _rewrite_fence_languages(prepared)
    if not prepared:
        return Text(cursor_suffix)
    return build_rich_markdown(
        prepared,
        code_theme=code_theme,
        live_incomplete_fences=live_incomplete_fences,
        cursor_suffix=cursor_suffix,
    )


__all__ = [
    "CursorSyntax",
    "FencedCodeBlock",
    "build_markdown_renderable",
    "build_rich_markdown",
    "close_incomplete_code_blocks",
    "extract_single_fenced_code_block",
    "style_markdown_renderable",
]
