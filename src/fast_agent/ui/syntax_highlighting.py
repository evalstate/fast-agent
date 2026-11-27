import io
import re
import tokenize
from dataclasses import dataclass
from pathlib import Path

from fast_agent.tools.shell_command import shell_heredoc_bodies, shell_inline_code_spans

_LANGUAGE_BY_EXTENSION: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".jsx": "jsx",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".md": "markdown",
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "bash",
    ".xml": "xml",
    ".html": "html",
    ".css": "css",
    ".sql": "sql",
}

_INTERPRETER_LANGUAGE: dict[str, str] = {
    "lua": "lua",
    "node": "javascript",
    "nodejs": "javascript",
    "osascript": "applescript",
    "perl": "perl",
    "php": "php",
    "ruby": "ruby",
    "tsx": "typescript",
}


@dataclass(frozen=True, slots=True)
class SyntaxBlock:
    code: str
    language: str


@dataclass(frozen=True, slots=True)
class _LanguageSpan:
    start: int
    end: int
    language: str


def syntax_language_for_path(path: str) -> str | None:
    return _LANGUAGE_BY_EXTENSION.get(Path(path).suffix.casefold())


def _syntax_language_for_interpreter(interpreter: str | None) -> str | None:
    if interpreter is None:
        return None
    if re.fullmatch(r"(?:python|pypy)(?:\d+(?:\.\d+)*)?", interpreter):
        return "python"
    return _INTERPRETER_LANGUAGE.get(interpreter)


def _language_spans(
    command: str,
    *,
    include_incomplete: bool,
) -> list[_LanguageSpan]:
    spans: list[_LanguageSpan] = []
    for body in shell_heredoc_bodies(command, include_incomplete=include_incomplete):
        language = _syntax_language_for_interpreter(body.stdin_interpreter) or (
            syntax_language_for_path(body.target_path) if body.target_path else None
        )
        if language is None or not command[body.start : body.end].strip():
            continue
        spans.append(_LanguageSpan(body.start, body.end, language))
    for inline in shell_inline_code_spans(command):
        language = _syntax_language_for_interpreter(inline.interpreter)
        if language is None or not command[inline.start : inline.end].strip():
            continue
        spans.append(_LanguageSpan(inline.start, inline.end, language))
    spans.sort(key=lambda span: span.start)
    merged: list[_LanguageSpan] = []
    for span in spans:
        if merged and span.start < merged[-1].end:
            # Prefer earlier (usually heredoc) span on overlap.
            continue
        merged.append(span)
    return merged


def shell_syntax_blocks(
    command: str,
    *,
    shell_language: str,
    include_incomplete: bool = False,
) -> list[SyntaxBlock]:
    language_spans = _language_spans(command, include_incomplete=include_incomplete)
    if not language_spans:
        return [SyntaxBlock(command.rstrip(), shell_language)]

    blocks: list[SyntaxBlock] = []
    cursor = 0
    for span in language_spans:
        if shell_text := command[cursor : span.start].rstrip():
            blocks.append(SyntaxBlock(shell_text, shell_language))
        blocks.append(
            SyntaxBlock(
                command[span.start : span.end].rstrip("\r\n"),
                span.language,
            )
        )
        cursor = span.end
    if shell_text := command[cursor:].rstrip():
        blocks.append(SyntaxBlock(shell_text, shell_language))
    return blocks


def _triple_quoted_inner_span(token_text: str) -> tuple[int, int] | None:
    """Return (start, end) offsets inside a non-f multiline triple-quoted Python string."""
    for quote in ('"""', "'''"):
        quote_index = token_text.find(quote)
        if quote_index == -1 or not token_text.endswith(quote):
            continue
        prefix = token_text[:quote_index].lower()
        if "f" in prefix:
            return None
        inner_start = quote_index + 3
        inner_end = len(token_text) - 3
        if inner_end <= inner_start or "\n" not in token_text[inner_start:inner_end]:
            return None
        return inner_start, inner_end
    return None


_TRIVIAL_TOKEN_TYPES = frozenset(
    {tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.COMMENT}
)


def _prev_significant_token(
    tokens: list[tokenize.TokenInfo],
    index: int,
) -> tuple[int, tokenize.TokenInfo] | None:
    i = index - 1
    while i >= 0:
        if tokens[i].type not in _TRIVIAL_TOKEN_TYPES:
            return i, tokens[i]
        i -= 1
    return None


def _is_shell_command_string(tokens: list[tokenize.TokenInfo], index: int) -> bool:
    """Return True when tokens[index] is the command argument of a shell(...) call."""
    prev = _prev_significant_token(tokens, index)
    if prev is None:
        return False
    prev_idx, prev_tok = prev

    # Positional: shell("""...""")
    if prev_tok.type == tokenize.OP and prev_tok.string == "(":
        callee = _prev_significant_token(tokens, prev_idx)
        return (
            callee is not None and callee[1].type == tokenize.NAME and callee[1].string == "shell"
        )

    # Keyword: shell(..., command="""...""")
    if not (prev_tok.type == tokenize.OP and prev_tok.string == "="):
        return False
    kw = _prev_significant_token(tokens, prev_idx)
    if kw is None or kw[1].type != tokenize.NAME or kw[1].string != "command":
        return False

    # Walk back to the enclosing '(' to verify the callee is `shell`.
    depth = 0
    i = kw[0] - 1
    while i >= 0:
        tok = tokens[i]
        if tok.type == tokenize.OP:
            if tok.string in {")", "]", "}"}:
                depth += 1
            elif tok.string in {"(", "[", "{"}:
                if depth == 0:
                    if tok.string != "(":
                        return False
                    callee = _prev_significant_token(tokens, i)
                    return (
                        callee is not None
                        and callee[1].type == tokenize.NAME
                        and callee[1].string == "shell"
                    )
                depth -= 1
        i -= 1
    return False


def _fill_bash_gaps(
    code: str,
    *,
    start: int,
    end: int,
    heredoc_spans: list[_LanguageSpan],
) -> list[_LanguageSpan]:
    """Fill whole-line interior ranges not claimed by heredoc spans with `bash`."""
    combined: list[_LanguageSpan] = []
    cursor = start
    for span in sorted(heredoc_spans, key=lambda item: item.start):
        if span.start > cursor and code[cursor : span.start].strip():
            combined.append(_LanguageSpan(cursor, span.start, "bash"))
        combined.append(span)
        cursor = max(cursor, span.end)
    if end > cursor and code[cursor:end].strip():
        combined.append(_LanguageSpan(cursor, end, "bash"))
    return combined


def _python_embedded_spans(code: str) -> list[_LanguageSpan]:
    lines = code.splitlines(keepends=True)
    if not lines:
        return []

    line_starts: list[int] = [0]
    for line in lines:
        line_starts.append(line_starts[-1] + len(line))

    spans: list[_LanguageSpan] = []
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(code).readline))
    except (tokenize.TokenError, IndentationError):
        return []

    for tok_index, tok in enumerate(tokens):
        if tok.type != tokenize.STRING:
            continue
        inner_bounds = _triple_quoted_inner_span(tok.string)
        if inner_bounds is None:
            continue
        tok_start = line_starts[tok.start[0] - 1] + tok.start[1]
        inner_offset, inner_end_offset = inner_bounds
        inner_start = tok_start + inner_offset
        inner_text = tok.string[inner_offset:inner_end_offset]

        heredoc_spans: list[_LanguageSpan] = []
        for body in shell_heredoc_bodies(inner_text, include_incomplete=False):
            language = _syntax_language_for_interpreter(body.stdin_interpreter) or (
                syntax_language_for_path(body.target_path) if body.target_path else None
            )
            if language is None or not inner_text[body.start : body.end].strip():
                continue
            abs_start = inner_start + body.start
            abs_end = inner_start + body.end
            # Only split on whole-line boundaries so outer/inner blocks align cleanly.
            if abs_start > 0 and code[abs_start - 1] != "\n":
                continue
            if abs_end < len(code) and code[abs_end - 1] != "\n":
                continue
            heredoc_spans.append(_LanguageSpan(abs_start, abs_end, language))

        end_line_index = tok.end[0] - 1
        closing_line_start = line_starts[end_line_index]
        closing_line_end = line_starts[end_line_index + 1]

        string_spans = heredoc_spans
        if _is_shell_command_string(tokens, tok_index):
            interior_start = line_starts[tok.start[0]]
            if closing_line_start > interior_start:
                string_spans = _fill_bash_gaps(
                    code,
                    start=interior_start,
                    end=closing_line_start,
                    heredoc_spans=heredoc_spans,
                )

        if not string_spans:
            continue

        spans.extend(string_spans)
        # Isolate the closing triple-quote line as plain text so its closing quotes
        # do not re-open a string in Pygments for subsequent Python lines.
        if closing_line_start >= string_spans[-1].end and closing_line_end > closing_line_start:
            spans.append(_LanguageSpan(closing_line_start, closing_line_end, "text"))

    spans.sort(key=lambda span: span.start)
    merged: list[_LanguageSpan] = []
    for span in spans:
        if merged and span.start < merged[-1].end:
            continue
        merged.append(span)
    return merged


def python_syntax_blocks(
    code: str,
    *,
    language: str = "python",
) -> list[SyntaxBlock]:
    """Split Python source when triple-quoted strings contain whole-line shell heredocs."""
    embedded_spans = _python_embedded_spans(code)
    if not embedded_spans:
        return [SyntaxBlock(code.rstrip(), language)]

    blocks: list[SyntaxBlock] = []
    cursor = 0
    for span in embedded_spans:
        if outer_text := code[cursor : span.start].rstrip("\r\n"):
            blocks.append(SyntaxBlock(outer_text, language))
        if inner_text := code[span.start : span.end].rstrip("\r\n"):
            blocks.append(SyntaxBlock(inner_text, span.language))
        cursor = span.end
    if tail_text := code[cursor:].rstrip():
        blocks.append(SyntaxBlock(tail_text, language))
    return blocks
