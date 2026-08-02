from dataclasses import dataclass
from pathlib import Path

from fast_agent.tools.shell_command import shell_heredoc_bodies

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


@dataclass(frozen=True, slots=True)
class SyntaxBlock:
    code: str
    language: str


def syntax_language_for_path(path: str) -> str | None:
    return _LANGUAGE_BY_EXTENSION.get(Path(path).suffix.casefold())


def shell_syntax_blocks(command: str, *, shell_language: str) -> list[SyntaxBlock]:
    heredoc_bodies = [
        (body, language)
        for body in shell_heredoc_bodies(command)
        if body.target_path
        and (language := syntax_language_for_path(body.target_path))
        and command[body.start : body.end].strip()
    ]
    if not heredoc_bodies:
        return [SyntaxBlock(command.rstrip(), shell_language)]

    blocks: list[SyntaxBlock] = []
    cursor = 0
    for body, language in heredoc_bodies:
        if shell_text := command[cursor : body.start].rstrip():
            blocks.append(SyntaxBlock(shell_text, shell_language))
        blocks.append(
            SyntaxBlock(
                command[body.start : body.end].rstrip("\r\n"),
                language,
            )
        )
        cursor = body.end
    if shell_text := command[cursor:].rstrip():
        blocks.append(SyntaxBlock(shell_text, shell_language))
    return blocks
