"""Cross-platform argv split/join helpers."""

from __future__ import annotations

import os
import shlex
import subprocess
from typing import TYPE_CHECKING, Literal, cast

import mslex

from fast_agent.utils.action_normalization import normalize_action_token

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

CommandLineSyntax = Literal["auto", "posix", "windows"]
ResolvedCommandLineSyntax = Literal["posix", "windows"]

_SUPPORTED_SYNTAXES: frozenset[ResolvedCommandLineSyntax] = frozenset(("posix", "windows"))
_AUTO_SYNTAX_BY_OS_NAME: dict[str, ResolvedCommandLineSyntax] = {"nt": "windows"}
_DEFAULT_AUTO_SYNTAX: ResolvedCommandLineSyntax = "posix"
_SPLITTERS: dict[ResolvedCommandLineSyntax, Callable[[str], list[str]]] = {
    "posix": lambda text: shlex.split(text, posix=True),
    "windows": mslex.split,
}
_JOINERS: dict[ResolvedCommandLineSyntax, Callable[[Sequence[str]], str]] = {
    "posix": shlex.join,
    "windows": subprocess.list2cmdline,
}


def resolve_commandline_syntax(
    syntax: CommandLineSyntax = "auto",
) -> ResolvedCommandLineSyntax:
    if not isinstance(syntax, str):
        raise ValueError(f"Unsupported command-line syntax: {syntax}")
    normalized_syntax = normalize_action_token(syntax)
    if normalized_syntax == "auto":
        return _AUTO_SYNTAX_BY_OS_NAME.get(os.name, _DEFAULT_AUTO_SYNTAX)
    if normalized_syntax in _SUPPORTED_SYNTAXES:
        return cast("ResolvedCommandLineSyntax", normalized_syntax)
    raise ValueError(f"Unsupported command-line syntax: {syntax}")


def split_commandline(
    text: str,
    *,
    syntax: CommandLineSyntax = "auto",
) -> list[str]:
    return _run_syntax_operation(syntax, _SPLITTERS, text)


def split_posix_like_preserving_backslashes(text: str) -> list[str]:
    """Split POSIX-style options while preserving Windows paths on non-Windows hosts."""
    tokens: list[str] = []
    current: list[str] = []
    quote: str | None = None
    token_started = False
    index = 0

    while index < len(text):
        char = text[index]
        if quote is None:
            if char.isspace():
                if token_started:
                    tokens.append("".join(current))
                    current.clear()
                    token_started = False
                index += 1
                continue
            if char in {'"', "'"}:
                quote = char
                token_started = True
                index += 1
                continue
            if char == "\\" and index + 1 < len(text):
                next_char = text[index + 1]
                if next_char.isspace() or next_char in {'"', "'"}:
                    current.append(next_char)
                    token_started = True
                    index += 2
                    continue
            current.append(char)
            token_started = True
            index += 1
            continue

        if char == quote:
            quote = None
            index += 1
            continue
        if quote == '"' and char == "\\" and index + 1 < len(text):
            next_char = text[index + 1]
            if next_char == '"':
                current.append(next_char)
                token_started = True
                index += 2
                continue
        current.append(char)
        token_started = True
        index += 1

    if quote is not None:
        raise ValueError("No closing quotation")
    if token_started:
        tokens.append("".join(current))
    return tokens


def join_commandline(
    argv: Sequence[str],
    *,
    syntax: CommandLineSyntax = "auto",
) -> str:
    normalized_argv = [str(token) for token in argv]
    return _run_syntax_operation(syntax, _JOINERS, normalized_argv)


def quote_commandline_token(
    token: str,
    *,
    syntax: CommandLineSyntax = "auto",
) -> str:
    return join_commandline([token], syntax=syntax)


def _run_syntax_operation[InputT, OutputT](
    syntax: CommandLineSyntax,
    operations: Mapping[ResolvedCommandLineSyntax, Callable[[InputT], OutputT]],
    value: InputT,
) -> OutputT:
    resolved = resolve_commandline_syntax(syntax)
    try:
        return operations[resolved](value)
    except ValueError as exc:
        raise ValueError(str(exc)) from exc
