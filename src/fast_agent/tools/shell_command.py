from __future__ import annotations

import posixpath
import re
from collections import deque
from typing import Literal

type ShellDetachmentKind = Literal["none", "ambiguous", "service_detach"]

_HEREDOC_PATTERN = re.compile(
    r"<<-?\s*(?:'([^']+)'|\"([^\"]+)\"|([A-Za-z_][A-Za-z0-9_]*))"
)


def _heredoc_declarations(
    line: str,
    quote: str | None,
) -> tuple[list[tuple[str, bool]], str | None]:
    declarations: list[tuple[str, bool]] = []
    escaped = False
    index = 0
    while index < len(line):
        char = line[index]
        if escaped:
            escaped = False
            index += 1
            continue
        if quote is not None:
            if char == "\\" and quote == '"':
                escaped = True
            elif char == quote:
                quote = None
            index += 1
            continue
        if char in {"'", '"'}:
            quote = char
            index += 1
            continue
        if char == "\\":
            escaped = True
            index += 1
            continue
        if char == "<" and (index == 0 or line[index - 1] != "<"):
            match = _HEREDOC_PATTERN.match(line, index)
            if match is not None:
                delimiter = next(
                    group for group in match.groups() if group is not None
                )
                declarations.append(
                    (delimiter, match.group(0).startswith("<<-"))
                )
                index = match.end()
                continue
        index += 1
    return declarations, quote


def _without_heredoc_bodies(command: str) -> str:
    lines = command.splitlines(keepends=True)
    kept: list[str] = []
    delimiters: deque[tuple[str, bool]] = deque()
    quote: str | None = None
    for line in lines:
        if delimiters:
            delimiter, strip_tabs = delimiters[0]
            candidate = line.rstrip("\r\n")
            if strip_tabs:
                candidate = candidate.lstrip("\t")
            if candidate == delimiter:
                delimiters.popleft()
                kept.append("\n")
            continue
        kept.append(line)
        declarations, quote = _heredoc_declarations(line, quote)
        delimiters.extend(declarations)
    return "".join(kept)


def _command_chunks(words: list[tuple[str, bool]]) -> list[list[str]]:
    chunks: list[list[str]] = []
    for word, at_command_position in words:
        if at_command_position:
            chunks.append([])
        if chunks:
            chunks[-1].append(word)
    return chunks


def _skip_env_prefix(words: list[str], index: int) -> int:
    options_with_values = {
        "-C",
        "-S",
        "-u",
        "--argv0",
        "--block-signal",
        "--chdir",
        "--default-signal",
        "--ignore-signal",
        "--split-string",
        "--unset",
    }
    while index < len(words):
        word = words[index]
        if word == "--":
            index += 1
            break
        if not word.startswith("-") or word == "-":
            break
        index += 1
        if word in options_with_values and index < len(words):
            index += 1
    while index < len(words) and re.fullmatch(
        r"[A-Za-z_][A-Za-z0-9_]*=.*",
        words[index],
    ):
        index += 1
    return index


def _invoked_command_basename(words: list[str]) -> str | None:
    index = 0
    while index < len(words):
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", words[index]):
            index += 1
            continue
        command = posixpath.basename(words[index])
        index += 1
        if command == "command":
            if index < len(words) and words[index] in {"-v", "-V"}:
                return None
            while index < len(words) and words[index] in {"-p", "--"}:
                index += 1
            continue
        if command == "exec":
            while index < len(words):
                option = words[index]
                if option == "--":
                    index += 1
                    break
                if option == "-a" and index + 1 < len(words):
                    index += 2
                    continue
                if option in {"-c", "-l"}:
                    index += 1
                    continue
                break
            continue
        if command == "env":
            index = _skip_env_prefix(words, index)
            continue
        return command
    return None


def classify_shell_detachment(
    command: str,
    *,
    run_in_background: bool,
) -> ShellDetachmentKind:
    """Conservatively identify shell-level service detachment."""
    source = _without_heredoc_bodies(command)
    words: list[tuple[str, bool]] = []
    top_level_background = False
    token: list[str] = []
    command_position = True
    depth = 0
    quote: str | None = None
    escaped = False
    index = 0

    def finish_word() -> None:
        nonlocal command_position
        if not token:
            return
        words.append(("".join(token), command_position))
        token.clear()
        command_position = False

    while index < len(source):
        char = source[index]
        if escaped:
            token.append(char)
            escaped = False
            index += 1
            continue
        if quote is not None:
            if char == "\\" and quote == '"':
                escaped = True
            elif char == quote:
                quote = None
            else:
                token.append(char)
            index += 1
            continue
        if char in {"'", '"'}:
            quote = char
            index += 1
            continue
        if char == "\\":
            escaped = True
            index += 1
            continue
        if char == "#" and not token and (index == 0 or source[index - 1].isspace()):
            newline = source.find("\n", index)
            index = len(source) if newline < 0 else newline
            continue
        if char in {"(", ")"}:
            finish_word()
            depth = max(depth + (1 if char == "(" else -1), 0)
            command_position = char == "("
            index += 1
            continue
        if char == "&":
            previous = source[index - 1] if index else ""
            following = source[index + 1] if index + 1 < len(source) else ""
            if previous in {">", "<"}:
                token.append(char)
                index += 1
                continue
            if following == ">":
                finish_word()
                index += 1
                continue
            finish_word()
            if following == "&":
                command_position = True
                index += 2
                continue
            if depth == 0:
                top_level_background = True
            command_position = True
            index += 1
            continue
        if char == "|":
            finish_word()
            command_position = True
            index += 2 if index + 1 < len(source) and source[index + 1] in {"|", "&"} else 1
            continue
        if char in {";", "\n"}:
            finish_word()
            command_position = True
            index += 1
            continue
        if char.isspace() or char in {"<", ">"}:
            finish_word()
            index += 1
            continue
        token.append(char)
        index += 1
    finish_word()

    command_words = {
        invoked
        for chunk in _command_chunks(words)
        if (invoked := _invoked_command_basename(chunk)) is not None
    }
    if top_level_background and (
        run_in_background or "nohup" in command_words or "disown" in command_words
    ):
        return "service_detach"
    if top_level_background:
        return "ambiguous"
    return "none"
