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
        "--all-groups",
        "--all-packages",
        "--compile-bytecode",
        "--exact",
        "--frozen",
        "--isolated",
        "--locked",
        "--managed-python",
        "--no-binary",
        "--no-build",
        "--no-build-isolation",
        "--no-cache",
        "--no-config",
        "--no-default-groups",
        "--no-dev",
        "--no-editable",
        "--no-env-file",
        "--no-index",
        "--no-managed-python",
        "--no-progress",
        "--no-project",
        "--no-python-downloads",
        "--no-sources",
        "--no-sync",
        "--offline",
        "--only-dev",
        "--quiet",
        "--refresh",
        "--reinstall",
        "--system-certs",
        "--upgrade",
        "--verbose",
        "-U",
        "-n",
        "-q",
        "-v",
    }
)
_UV_RUN_VALUE_OPTIONS = frozenset(
    {
        "--allow-insecure-host",
        "--cache-dir",
        "--color",
        "--config-file",
        "--config-setting",
        "--config-settings-package",
        "--default-index",
        "--directory",
        "--env-file",
        "--exclude-newer",
        "--exclude-newer-package",
        "--extra",
        "--extra-index-url",
        "--find-links",
        "--fork-strategy",
        "--group",
        "--index",
        "--index-strategy",
        "--index-url",
        "--keyring-provider",
        "--link-mode",
        "--no-binary-package",
        "--no-build-isolation-package",
        "--no-build-package",
        "--no-editable-package",
        "--no-extra",
        "--no-group",
        "--no-sources-package",
        "--only-group",
        "--package",
        "--prerelease",
        "--project",
        "--python",
        "--python-platform",
        "--refresh-package",
        "--reinstall-package",
        "--resolution",
        "--upgrade-group",
        "--upgrade-package",
        "--with",
        "--with-editable",
        "--with-requirements",
        "-C",
        "-P",
        "-f",
        "-i",
        "-p",
        "-w",
    }
)
_UV_RUN_SHORT_FLAG_OPTIONS = frozenset(
    option[1:] for option in _UV_RUN_FLAG_OPTIONS if len(option) == 2
)
_UV_RUN_SHORT_VALUE_OPTIONS = frozenset(
    option[1:] for option in _UV_RUN_VALUE_OPTIONS if len(option) == 2
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
class ShellInlineCodeSpan:
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


@dataclass(frozen=True, slots=True)
class _ShellToken:
    value: str
    start: int
    end: int


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


def _uv_run_compact_option_width(option: str) -> Literal[1, 2] | None:
    if len(option) <= 2 or not option.startswith("-") or option.startswith("--"):
        return None
    for offset, short_option in enumerate(option[1:], start=2):
        if short_option in _UV_RUN_SHORT_FLAG_OPTIONS:
            continue
        if short_option in _UV_RUN_SHORT_VALUE_OPTIONS:
            return 1 if offset < len(option) else 2
        return None
    return 1


def _skip_uv_run_options(command: list[str], index: int) -> int | None:
    while index < len(command):
        option = command[index]
        if option == "--":
            return index + 1
        if option in _UV_RUN_FLAG_OPTIONS:
            index += 1
            continue
        if option in _UV_RUN_VALUE_OPTIONS:
            if index + 1 >= len(command):
                return None
            index += 2
            continue
        name, separator, value = option.partition("=")
        if separator and name in _UV_RUN_VALUE_OPTIONS:
            if not value:
                return None
            index += 1
            continue
        compact_width = _uv_run_compact_option_width(option)
        if compact_width is not None:
            if compact_width == 2 and index + 1 >= len(command):
                return None
            index += compact_width
            continue
        break
    return index


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
    if len(command) >= 3 and posixpath.basename(command[0]).casefold() == "uv":
        if command[1] != "run":
            return None
        command_index = _skip_uv_run_options(command, 2)
        if command_index is None:
            return None
        invocation = command[command_index:]
        if invocation == ["-"]:
            return "python"
        if len(invocation) == 2 and invocation[1] == "-":
            return posixpath.basename(invocation[0]).casefold()
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


def _skip_shell_redirect(text: str, index: int) -> int | None:
    length = len(text)
    while index < length and text[index].isdigit():
        index += 1
    if index >= length or text[index] not in {"<", ">"}:
        return None
    if text.startswith(">>", index) or text.startswith("<<", index):
        index += 2
    else:
        index += 1
    if index < length and text[index] == "&":
        index += 1
        target_start = index
        while index < length and text[index].isdigit():
            index += 1
        if index > target_start:
            return index
    while index < length and text[index].isspace():
        index += 1
    if index >= length:
        return index
    if text[index] in {"'", '"'}:
        quote = text[index]
        index += 1
        escaped = False
        while index < length:
            char = text[index]
            if escaped:
                escaped = False
            elif char == "\\" and quote == '"':
                escaped = True
            elif char == quote:
                return index + 1
            index += 1
        return None
    target_start = index
    while index < length and not text[index].isspace():
        if text[index] in ";&|<>()":
            break
        if text[index] == "\\" and index + 1 < length:
            index += 2
            continue
        index += 1
    return index if index > target_start else None


def _tokenize_shell_span(text: str, absolute_start: int) -> list[_ShellToken] | None:
    tokens: list[_ShellToken] = []
    length = len(text)
    index = 0
    while index < length:
        while index < length and text[index].isspace():
            index += 1
        if index >= length:
            break
        if text[index] in ";&|()":
            return None
        redirect_end = _skip_shell_redirect(text, index)
        if redirect_end is not None and redirect_end > index:
            index = redirect_end
            continue
        start = index
        quote = text[index] if text[index] in {"'", '"'} else None
        if quote is not None:
            index += 1
            escaped = False
            while index < length:
                char = text[index]
                if escaped:
                    escaped = False
                elif char == "\\" and quote == '"':
                    escaped = True
                elif char == quote:
                    index += 1
                    break
                index += 1
            else:
                return None
            try:
                value = shlex.split(text[start:index], posix=True)[0]
            except (ValueError, IndexError):
                return None
        else:
            while index < length and not text[index].isspace():
                if text[index] in ";&|<>()":
                    break
                if text[index] == "\\" and index + 1 < length:
                    index += 2
                    continue
                index += 1
            if start == index:
                return None
            try:
                value = shlex.split(text[start:index], posix=True)[0]
            except (ValueError, IndexError):
                return None
        tokens.append(
            _ShellToken(
                value=value,
                start=absolute_start + start,
                end=absolute_start + index,
            )
        )
    return tokens


def _inline_code_flag(interpreter: str) -> str | None:
    if re.fullmatch(r"(?:python|pypy)(?:\d+(?:\.\d+)*)?", interpreter):
        return "-c"
    basename = posixpath.basename(interpreter).casefold()
    if basename == "php":
        return "-r"
    if basename in {
        "lua",
        "node",
        "nodejs",
        "osascript",
        "perl",
        "ruby",
    }:
        return "-e"
    return None


def _match_inline_code_span(tokens: list[_ShellToken]) -> ShellInlineCodeSpan | None:
    if len(tokens) < 3:
        return None

    index = 0
    while index < len(tokens):
        value = tokens[index].value
        basename = posixpath.basename(value).casefold()
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", value):
            index += 1
            continue
        if basename == "command":
            index += 1
            if index < len(tokens) and tokens[index].value in {"-v", "-V"}:
                return None
            while index < len(tokens) and tokens[index].value in {"-p", "--"}:
                index += 1
            continue
        if basename == "exec":
            index += 1
            while index < len(tokens):
                option = tokens[index].value
                if option == "--":
                    index += 1
                    break
                if option == "-a" and index + 1 < len(tokens):
                    index += 2
                    continue
                if option in {"-c", "-l"}:
                    index += 1
                    continue
                break
            continue
        if basename == "env":
            index += 1
            while index < len(tokens):
                option = tokens[index].value
                if option == "--":
                    index += 1
                    break
                if not option.startswith("-") or option == "-":
                    break
                index += 1
                if option in {
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
                } and index < len(tokens):
                    index += 1
            while index < len(tokens) and re.fullmatch(
                r"[A-Za-z_][A-Za-z0-9_]*=.*",
                tokens[index].value,
            ):
                index += 1
            continue
        if basename == "uv":
            if index + 1 >= len(tokens) or tokens[index + 1].value != "run":
                return None
            command_index = _skip_uv_run_options(
                [token.value for token in tokens],
                index + 2,
            )
            if command_index is None:
                return None
            index = command_index
            continue
        break
    else:
        return None

    interpreter = posixpath.basename(tokens[index].value).casefold()
    flag = _inline_code_flag(interpreter)
    if flag is None:
        return None

    flag_index = index + 1
    if flag_index >= len(tokens) or tokens[flag_index].value != flag:
        return None
    code_index = flag_index + 1
    if code_index >= len(tokens):
        return None
    code_token = tokens[code_index]
    if not code_token.value:
        return None
    return ShellInlineCodeSpan(
        start=code_token.start,
        end=code_token.end,
        interpreter=interpreter,
    )


def _strip_shell_quotes(command: str, start: int, end: int) -> tuple[int, int]:
    if end - start < 2:
        return start, end
    opening = command[start]
    if opening not in {"'", '"'} or command[end - 1] != opening:
        return start, end
    return start + 1, end - 1


def shell_inline_code_spans(command: str) -> list[ShellInlineCodeSpan]:
    """Return inline interpreter payloads (`python -c`, `node -e`, ...)."""
    heredoc_bodies = shell_heredoc_bodies(command, include_incomplete=True)
    blocked = [(body.start, body.end) for body in heredoc_bodies]
    spans: list[ShellInlineCodeSpan] = []
    quote: str | None = None
    escaped = False
    segment_start = 0
    index = 0
    length = len(command)

    def in_heredoc_body(position: int) -> bool:
        return any(start <= position < end for start, end in blocked)

    def consume_segment(start: int, end: int) -> None:
        if start >= end or in_heredoc_body(start):
            return
        text = command[start:end]
        if not text.strip() or "<<" in text:
            # Heredoc declaration lines are handled by shell_heredoc_bodies.
            return
        tokens = _tokenize_shell_span(text, start)
        if tokens is None:
            return
        matched = _match_inline_code_span(tokens)
        if matched is not None:
            payload_start, payload_end = _strip_shell_quotes(
                command,
                matched.start,
                matched.end,
            )
            if command.startswith("\r\n", payload_start):
                payload_start += 2
            elif command.startswith("\n", payload_start):
                payload_start += 1
            if payload_start < payload_end:
                spans.append(
                    ShellInlineCodeSpan(
                        start=payload_start,
                        end=payload_end,
                        interpreter=matched.interpreter,
                    )
                )

    while index < length:
        if in_heredoc_body(index):
            if segment_start < index:
                consume_segment(segment_start, index)
            body_end = next(end for start, end in blocked if start <= index < end)
            # Skip the delimiter line that terminates a completed heredoc.
            index = body_end
            while index < length and command[index] not in {"\n", "\r"}:
                index += 1
            while index < length and command[index] in {"\n", "\r"}:
                index += 1
            segment_start = index
            quote = None
            escaped = False
            continue
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
        if char == "\\":
            escaped = True
            index += 1
            continue
        if char in {";", "|", "&"}:
            separator_end = index + (1 if index + 1 >= length or command[index + 1] != char else 2)
            consume_segment(segment_start, index)
            segment_start = separator_end
            index = separator_end
            continue
        index += 1
    consume_segment(segment_start, length)
    spans.sort(key=lambda span: span.start)
    return spans


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
