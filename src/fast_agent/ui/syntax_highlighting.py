import re
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
