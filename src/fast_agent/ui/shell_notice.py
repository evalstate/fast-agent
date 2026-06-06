"""Helpers for formatting shell notice text."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.text import Text

from fast_agent.constants import SHELL_NOTICE_PREFIX
from fast_agent.utils.path_display import format_working_directory

if TYPE_CHECKING:
    from collections.abc import Sequence

    from fast_agent.tools.shell_runtime import ShellRuntime


def format_shell_notice(
    shell_access_modes: Sequence[str],
    shell_runtime: "ShellRuntime | None",
) -> Text:
    modes_display = ", ".join(shell_access_modes or ("direct",))
    shell_name = None
    if shell_runtime is not None:
        runtime_info = shell_runtime.runtime_info()
        shell_name = runtime_info.get("name")
    shell_display = f"{modes_display}, {shell_name}" if shell_name else modes_display

    if shell_runtime is not None:
        working_dir = shell_runtime.working_directory()
        working_dir_display = format_working_directory(working_dir)
        shell_display = f"{shell_display} | cwd: {working_dir_display}"

    notice = Text.from_markup(SHELL_NOTICE_PREFIX)
    notice.append(Text(f" ({shell_display})", style="dim"))
    return notice
