"""Cross-platform argv split/join helpers."""

from __future__ import annotations

import os
import shlex
import subprocess
from typing import Literal, Sequence

import mslex

CommandLineSyntax = Literal["auto", "posix", "windows"]
ResolvedCommandLineSyntax = Literal["posix", "windows"]


def resolve_commandline_syntax(
    syntax: CommandLineSyntax = "auto",
) -> ResolvedCommandLineSyntax:
    if syntax == "auto":
        return "windows" if os.name == "nt" else "posix"
    if syntax in {"posix", "windows"}:
        return syntax
    raise ValueError(f"Unsupported command-line syntax: {syntax}")


def split_commandline(
    text: str,
    *,
    syntax: CommandLineSyntax = "auto",
) -> list[str]:
    resolved = resolve_commandline_syntax(syntax)
    try:
        if resolved == "windows":
            return mslex.split(text)
        return shlex.split(text, posix=True)
    except Exception as exc:  # noqa: BLE001 - normalize parsing failures
        raise ValueError(str(exc)) from exc


def join_commandline(
    argv: Sequence[str],
    *,
    syntax: CommandLineSyntax = "auto",
) -> str:
    resolved = resolve_commandline_syntax(syntax)
    normalized_argv = [str(token) for token in argv]
    try:
        if resolved == "windows":
            return subprocess.list2cmdline(normalized_argv)
        return shlex.join(normalized_argv)
    except Exception as exc:  # noqa: BLE001 - normalize join failures
        raise ValueError(str(exc)) from exc
