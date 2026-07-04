"""Helpers for formatting shell notice text."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.text import Text

from fast_agent.constants import SHELL_NOTICE_PREFIX
from fast_agent.utils.path_display import format_working_directory

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path
    from typing import Protocol

    from fast_agent.tools.execution_environment import ShellRuntimeInfo

    class ShellNoticeRuntime(Protocol):
        def runtime_info(self) -> ShellRuntimeInfo: ...

        def working_directory(self) -> Path: ...


def format_shell_notice(
    shell_access_modes: Sequence[str],
    shell_runtime: "ShellNoticeRuntime | None",
) -> Text:
    modes_display = ", ".join(shell_access_modes or ("direct",))
    shell_name = None
    environment_display = None
    if shell_runtime is not None:
        runtime_info = shell_runtime.runtime_info()
        shell_name = runtime_info.name
        environment_parts: list[str] = []
        if runtime_info.environment_name:
            environment_parts.append(runtime_info.environment_name)
        environment_parts.append(runtime_info.kind)
        if runtime_info.provider and runtime_info.provider != runtime_info.kind:
            environment_parts.append(runtime_info.provider)
        environment_display = " ".join(environment_parts)
    shell_display = f"{modes_display}, {shell_name}" if shell_name else modes_display

    if shell_runtime is not None:
        working_dir = shell_runtime.working_directory()
        working_dir_display = format_working_directory(working_dir)
        if environment_display:
            shell_display = f"{shell_display} | env: {environment_display}"
        shell_display = f"{shell_display} | cwd: {working_dir_display}"

    notice = Text.from_markup(SHELL_NOTICE_PREFIX)
    notice.append(Text(f" ({shell_display})", style="dim"))
    return notice
