from __future__ import annotations

import posixpath
import re
import shlex
from collections import deque
from dataclasses import dataclass
from typing import Literal

type ShellDetachmentKind = Literal["none", "ambiguous", "service_detach"]

_HEREDOC_PATTERN = re.compile(
    r"<<-?\s*(?:'([^']+)'|\"([^\"]+)\"|\\([A-Za-z_][A-Za-z0-9_]*)|([A-Za-z_][A-Za-z0-9_]*))"
)
_UV_RUN_FLAG_OPTIONS = frozenset(
    {
        "--active",
        "--all-extras",
        "--compile-bytecode",
        "--exact",
        "--frozen",
        "--isolated",
        "--locked",
        "--managed-python",
        "--no-dev",
        "--no-editable",
        "--no-managed-python",
        "--no-project",
        "--no-python-downloads",
        "--no-sources",
        "--no-sync",
        "--offline",
        "--quiet",
        "--verbose",
        "-q",
        "-v",
    }
)
_PNPM_EXEC_FLAG_OPTIONS = frozenset({"--recursive", "--silent", "--workspace-root", "-r", "-w"})
_PNPM_EXEC_VALUE_OPTIONS = frozenset({"--dir", "-C"})


@dataclass(frozen=True, slots=True)
class ShellHeredocBody:
    start: int
    end: int
    target_path: str | None
    stdin_interpreter: str | None = None


@dataclass(frozen=True, slots=True)
class ShellInlineCodeBody:
    start: int
    end: int
    interpreter: str


@dataclass(frozen=True, slots=True)
class _HeredocDeclaration:
    delimiter: str
    strip_tabs: bool
    start: int


@dataclass(slots=True)
class _PendingHeredoc:
    delimiter: str
    strip_tabs: bool
    body_start: int
    target_path: str | None
    stdin_interpreter: str | None


def _heredoc_declarations(
    line: str,
    quote: str | None,
) -> tuple[list[_HeredocDeclaration], str | None]:
    declarations: list[_HeredocDeclaration] = []
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
        if char == "#" and (index == 0 or line[index - 1].isspace()):
            break
        if char == "<" and (index == 0 or line[index - 1] != "<"):
            match = _HEREDOC_PATTERN.match(line, index)
            if match is not None:
                delimiter = next(group for group in match.groups() if group is not None)
                if line.rfind("((", 0, index) <= line.rfind("))", 0, index):
                    declarations.append(
                        _HeredocDeclaration(
                            delimiter=delimiter,
                            strip_tabs=match.group(0).startswith("<<-"),
                            start=index,
                        )
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
        delimiters.extend(
            (declaration.delimiter, declaration.strip_tabs) for declaration in declarations
        )
    return "".join(kept)


def _shell_command_span(line: str, position: int) -> tuple[int, int]:
    start = 0
    end = len(line)
    quote: str | None = None
    escaped = False
    index = 0
    while index < len(line):
        char = line[index]
        if escaped:
            escaped = False
        elif quote is not None:
            if char == "\\" and quote == '"':
                escaped = True
            elif char == quote:
                quote = None
        elif char in {"'", '"'}:
            quote = char
        elif char in {";", "|", "&"}:
            separator_end = index + (1 if index + 1 >= len(line) or line[index + 1] != char else 2)
            if index < position:
                start = separator_end
            else:
                end = index
                break
            index = separator_end - 1
        index += 1
    return start, end


def _shell_redirect_target(command: str) -> str | None:
    quote: str | None = None
    escaped = False
    target_path: str | None = None
    index = 0
    while index < len(command):
        char = command[index]
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
        if char != ">":
            index += 1
            continue

        fd_start = index
        while fd_start > 0 and command[fd_start - 1].isdigit():
            fd_start -= 1
        file_descriptor = command[fd_start:index]
        if file_descriptor and file_descriptor != "1":
            index += 1
            continue

        index += 2 if command.startswith(">>", index) else 1
        while index < len(command) and command[index].isspace():
            index += 1
        if index >= len(command):
            return None

        target_quote = command[index] if command[index] in {"'", '"'} else None
        if target_quote is not None:
            target_start = index + 1
            target_end = command.find(target_quote, target_start)
            if target_end < 0:
                return None
            target = command[target_start:target_end]
            if target_quote == '"' and any(character in target for character in "$`"):
                return None
            index = target_end + 1
        else:
            target_start = index
            while index < len(command) and not command[index].isspace():
                if command[index] in ";&|<>":
                    break
                index += 1
            target = command[target_start:index]
            if any(character in target for character in "$`*?[]{}()"):
                return None
        if not target:
            return None
        target_path = target
    return target_path


def _heredoc_redirect_target(line: str, declaration: _HeredocDeclaration) -> str | None:
    start, end = _shell_command_span(line, declaration.start)
    return _shell_redirect_target(line[start:end])


def _heredoc_stdin_interpreter(
    line: str,
    declaration: _HeredocDeclaration,
) -> str | None:
    start, end = _shell_command_span(line, declaration.start)
    try:
        tokens = shlex.split(line[start:end], posix=True)
    except ValueError:
        return None
    declaration_index = next(
        (index for index, token in enumerate(tokens) if token.startswith("<<")),
        None,
    )
    if declaration_index is None or declaration_index != len(tokens) - 1:
        return None
    command = tokens[:declaration_index]
    if len(command) == 2 and command[1] == "-":
        return posixpath.basename(command[0]).casefold()
    if (
        len(command) >= 4
        and posixpath.basename(command[0]).casefold() == "uv"
        and command[1] == "run"
        and command[-1] == "-"
        and all(option in _UV_RUN_FLAG_OPTIONS for option in command[2:-2])
    ):
        return posixpath.basename(command[-2]).casefold()
    if command and posixpath.basename(command[0]).casefold() == "pnpm" and command[-1] == "-":
        index = 1
        while index < len(command):
            option = command[index]
            if option in _PNPM_EXEC_FLAG_OPTIONS:
                index += 1
                continue
            if option in _PNPM_EXEC_VALUE_OPTIONS and index + 1 < len(command):
                index += 2
                continue
            if any(option.startswith(f"{name}=") for name in _PNPM_EXEC_VALUE_OPTIONS):
                index += 1
                continue
            break
        if command[index:] == ["exec", command[-2], "-"]:
            return posixpath.basename(command[-2]).casefold()
    return None


def shell_heredoc_bodies(
    command: str,
    *,
    include_incomplete: bool = False,
) -> list[ShellHeredocBody]:
    """Return heredoc bodies with static output or stdin-interpreter hints."""
    bodies: list[ShellHeredocBody] = []
    pending: deque[_PendingHeredoc] = deque()
    quote: str | None = None
    offset = 0

    for line in command.splitlines(keepends=True):
        line_end = offset + len(line)
        if pending:
            current = pending[0]
            candidate = line.rstrip("\r\n")
            if current.strip_tabs:
                candidate = candidate.lstrip("\t")
            if candidate == current.delimiter:
                if current.body_start >= 0:
                    bodies.append(
                        ShellHeredocBody(
                            start=current.body_start,
                            end=offset,
                            target_path=current.target_path,
                            stdin_interpreter=current.stdin_interpreter,
                        )
                    )
                pending.popleft()
                if pending:
                    pending[0].body_start = line_end
            offset = line_end
            continue

        declarations, quote = _heredoc_declarations(line, quote)
        target_path = (
            _heredoc_redirect_target(line, declarations[0]) if len(declarations) == 1 else None
        )
        stdin_interpreter = (
            _heredoc_stdin_interpreter(line, declarations[0]) if len(declarations) == 1 else None
        )
        for index, declaration in enumerate(declarations):
            pending.append(
                _PendingHeredoc(
                    delimiter=declaration.delimiter,
                    strip_tabs=declaration.strip_tabs,
                    body_start=line_end if index == 0 else -1,
                    target_path=target_path,
                    stdin_interpreter=stdin_interpreter,
                )
            )
        offset = line_end

    if include_incomplete and pending:
        current = pending[0]
        if current.body_start >= 0:
            bodies.append(
                ShellHeredocBody(
                    start=current.body_start,
                    end=len(command),
                    target_path=current.target_path,
                    stdin_interpreter=current.stdin_interpreter,
                )
            )

    return bodies


def _quoted_argument_end(command: str, start: int, quote: str) -> tuple[int | None, bool]:
    dynamic = False
    index = start
    while index < len(command):
        char = command[index]
        if quote == '"' and char == "\\":
            index += 2
            continue
        if quote == '"' and char in {"$", "`"}:
            dynamic = True
        if char == quote:
            return index, dynamic
        index += 1
    return None, dynamic


def _direct_python_inline_code_interpreter(prefix: str) -> str | None:
    prefix = prefix.replace("\\\r\n", "").replace("\\\n", "")
    try:
        words = shlex.split(prefix, posix=True)
    except ValueError:
        return None
    if len(words) != 2 or words[1] != "-c":
        return None
    interpreter = posixpath.basename(words[0]).casefold()
    if re.fullmatch(r"(?:python|pypy)(?:\d+(?:\.\d+)*)?", interpreter):
        return interpreter
    return None


def shell_inline_code_bodies(
    command: str,
    *,
    include_incomplete: bool = False,
) -> list[ShellInlineCodeBody]:
    """Return static multiline Python/PyPy ``-c`` bodies."""
    bodies: list[ShellInlineCodeBody] = []
    heredocs = iter(shell_heredoc_bodies(command, include_incomplete=True))
    heredoc = next(heredocs, None)
    command_start = 0
    index = 0

    while index < len(command):
        if heredoc is not None and index >= heredoc.start:
            index = heredoc.end
            heredoc = next(heredocs, None)
            continue

        char = command[index]
        if char == "\\":
            index += 2
            continue
        if char == "#" and (index == 0 or command[index - 1].isspace()):
            newline = command.find("\n", index)
            if newline < 0:
                break
            index = newline + 1
            command_start = index
            continue
        if char in {";", "&", "|", "(", ")", "\n"}:
            index += 1
            command_start = index
            continue
        if char not in {"'", '"'}:
            index += 1
            continue

        interpreter = _direct_python_inline_code_interpreter(command[command_start:index])
        quote_end, dynamic = _quoted_argument_end(command, index + 1, char)
        if heredoc is not None and heredoc.start < (quote_end or len(command)):
            index = heredoc.end
            heredoc = next(heredocs, None)
            continue
        if quote_end is None:
            if interpreter is not None and include_incomplete and not dynamic:
                body_start = index + 1
                if command.startswith("\r\n", body_start):
                    body_start += 2
                elif command.startswith("\n", body_start):
                    body_start += 1
                if "\n" in command[index + 1 :] or "\r" in command[index + 1 :]:
                    bodies.append(
                        ShellInlineCodeBody(
                            start=body_start,
                            end=len(command),
                            interpreter=interpreter,
                        )
                    )
            break

        quote_follow = quote_end + 1
        followed_by_boundary = quote_follow == len(command) or (
            command[quote_follow].isspace()
            or command[quote_follow] in {";", "&", "|", "(", ")", "<", ">"}
        )
        raw_body = command[index + 1 : quote_end]
        if (
            interpreter is not None
            and not dynamic
            and followed_by_boundary
            and ("\n" in raw_body or "\r" in raw_body)
        ):
            body_start = index + 1
            if command.startswith("\r\n", body_start):
                body_start += 2
            elif command.startswith("\n", body_start):
                body_start += 1
            bodies.append(
                ShellInlineCodeBody(
                    start=body_start,
                    end=quote_end,
                    interpreter=interpreter,
                )
            )
        index = quote_end + 1

    return bodies


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
    has_background_job = False
    token: list[str] = []
    command_position = True
    contexts: list[
        Literal[
            "arithmetic",
            "arithmetic_group",
            "quoted_arithmetic",
            "quoted_shell",
            "shell",
        ]
    ] = []
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
            elif quote == '"' and source.startswith("$((", index):
                token.append("$((")
                contexts.append("quoted_arithmetic")
                quote = None
                index += 3
                continue
            elif quote == '"' and source.startswith("$(", index):
                finish_word()
                contexts.append("quoted_shell")
                quote = None
                command_position = True
                index += 2
                continue
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
        context = contexts[-1] if contexts else None
        if context in {"arithmetic", "arithmetic_group", "quoted_arithmetic"}:
            if source.startswith("$((", index):
                token.append("$((")
                contexts.append("arithmetic")
                index += 3
                continue
            if source.startswith("$(", index):
                finish_word()
                contexts.append("shell")
                command_position = True
                index += 2
                continue
            if char == "(":
                token.append(char)
                contexts.append("arithmetic_group")
                index += 1
                continue
            if char == ")":
                if context in {"arithmetic", "quoted_arithmetic"} and source.startswith(
                    "))", index
                ):
                    token.append("))")
                    contexts.pop()
                    if context == "quoted_arithmetic":
                        quote = '"'
                    index += 2
                else:
                    token.append(char)
                    if context == "arithmetic_group":
                        contexts.pop()
                    index += 1
                continue
            token.append(char)
            index += 1
            continue
        if source.startswith("$((", index):
            token.append("$((")
            contexts.append("arithmetic")
            index += 3
            continue
        if source.startswith("((", index):
            token.append("((")
            contexts.append("arithmetic")
            index += 2
            continue
        if source.startswith("$(", index):
            finish_word()
            contexts.append("shell")
            command_position = True
            index += 2
            continue
        if char in {"(", ")"}:
            finish_word()
            if char == "(":
                contexts.append("shell")
            elif contexts and contexts[-1] in {"shell", "quoted_shell"}:
                closed_context = contexts.pop()
                if closed_context == "quoted_shell":
                    quote = '"'
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
            has_background_job = True
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
    if has_background_job and (
        run_in_background or "nohup" in command_words or "disown" in command_words
    ):
        return "service_detach"
    if has_background_job:
        return "ambiguous"
    return "none"
