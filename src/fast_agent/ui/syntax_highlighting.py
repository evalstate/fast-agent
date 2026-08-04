import re
from dataclasses import dataclass
from pathlib import Path

from fast_agent.tools.shell_command import shell_heredoc_bodies, shell_inline_code_bodies

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
    "perl": "perl",
    "ruby": "ruby",
    "tsx": "typescript",
}


@dataclass(frozen=True, slots=True)
class SyntaxBlock:
    code: str
    language: str


@dataclass(frozen=True, slots=True)
class _SyntaxSpan:
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


def shell_syntax_blocks(
    command: str,
    *,
    shell_language: str,
    include_incomplete: bool = False,
) -> list[SyntaxBlock]:
    spans = [
        _SyntaxSpan(start=body.start, end=body.end, language=language)
        for body in shell_heredoc_bodies(command, include_incomplete=include_incomplete)
        if (
            language := _syntax_language_for_interpreter(body.stdin_interpreter)
            or (syntax_language_for_path(body.target_path) if body.target_path else None)
        )
        and command[body.start : body.end].strip()
    ]
    spans.extend(
        _SyntaxSpan(start=body.start, end=body.end, language=language)
        for body in shell_inline_code_bodies(command, include_incomplete=include_incomplete)
        if (language := _syntax_language_for_interpreter(body.interpreter))
        and command[body.start : body.end].strip()
    )
    if not spans:
        return [SyntaxBlock(command.rstrip(), shell_language)]

    blocks: list[SyntaxBlock] = []
    cursor = 0
    for span in sorted(spans, key=lambda span: (span.start, span.end)):
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
