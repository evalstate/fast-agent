"""Helpers for formatting shell notice text."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from fast_agent.constants import SHELL_NOTICE_PREFIX

if TYPE_CHECKING:
    from fast_agent.tools.shell_runtime import ShellRuntime


def format_shell_notice(
    shell_access_modes: Sequence[str],
    shell_runtime: "ShellRuntime | None",
) -> str:
    modes_display = ", ".join(shell_access_modes or ("direct",))
    shell_name = None
    if shell_runtime is not None:
        runtime_info = shell_runtime.runtime_info()
        shell_name = runtime_info.get("name")
    shell_display = f"{modes_display}, {shell_name}" if shell_name else modes_display

    if shell_runtime is not None:
        working_dir = shell_runtime.working_directory()
        try:
            working_dir_display = str(working_dir.relative_to(Path.cwd()))
            if working_dir_display == ".":
                parts = Path.cwd().parts
                if len(parts) >= 2:
                    working_dir_display = "/".join(parts[-2:])
                elif len(parts) == 1:
                    working_dir_display = parts[0]
                else:
                    working_dir_display = str(Path.cwd())
        except ValueError:
            working_dir_display = str(working_dir)
        shell_display = f"{shell_display} | cwd: {working_dir_display}"

    return f"{SHELL_NOTICE_PREFIX}[dim] ({shell_display})[/dim]"
