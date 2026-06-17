"""Shared shell detection helpers."""

from __future__ import annotations

import os
import platform
import shutil
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath

from fast_agent.utils.text import strip_to_none

WINDOWS_SHELL_CANDIDATES = ("pwsh", "powershell", "cmd")
POSIX_SHELL_CANDIDATES = ("bash", "zsh", "sh")


def _strip_outer_quotes(value: str) -> str:
    stripped = strip_to_none(value)
    if stripped is None:
        return ""
    if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in {"'", '"'}:
        return strip_to_none(stripped[1:-1]) or ""
    return stripped


@dataclass(frozen=True, slots=True)
class _ShellRuntimeInfo:
    name: str
    path: str | None

    @property
    def command(self) -> str:
        return self.path or self.name

    def as_dict(self) -> dict[str, str | None]:
        return {"name": self.name, "path": self.path}


def _first_available_shell(candidates: tuple[str, ...]) -> _ShellRuntimeInfo | None:
    return next(
        (
            _ShellRuntimeInfo(shell_name, shell_path)
            for shell_name in candidates
            if (shell_path := shutil.which(shell_name))
        ),
        None,
    )


def _shell_env_path() -> str | None:
    shell_env = _strip_outer_quotes(os.environ.get("SHELL") or "")
    if shell_env and Path(shell_env).exists():
        return shell_env
    return None


def _detect_windows_shell_runtime() -> _ShellRuntimeInfo:
    available_shell = _first_available_shell(WINDOWS_SHELL_CANDIDATES)
    if available_shell is not None:
        return available_shell

    comspec = _strip_outer_quotes(os.environ.get("COMSPEC") or "") or "cmd.exe"
    return _ShellRuntimeInfo(PureWindowsPath(comspec).name or "cmd.exe", comspec)


def _detect_posix_shell_runtime() -> _ShellRuntimeInfo:
    shell_env = _shell_env_path()
    if shell_env:
        return _ShellRuntimeInfo(Path(shell_env).name, shell_env)

    available_shell = _first_available_shell(POSIX_SHELL_CANDIDATES)
    if available_shell is not None:
        return available_shell

    return _ShellRuntimeInfo("sh", None)


def _detect_shell_runtime() -> _ShellRuntimeInfo:
    if platform.system() == "Windows":
        return _detect_windows_shell_runtime()
    return _detect_posix_shell_runtime()


def default_shell_command() -> str:
    """Return the best command to launch an interactive shell."""
    return _detect_shell_runtime().command


def shell_runtime_info() -> dict[str, str | None]:
    """Return display metadata for the local shell runtime."""
    return _detect_shell_runtime().as_dict()
