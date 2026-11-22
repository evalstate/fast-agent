from pathlib import Path

from fast_agent.tools.execution_environment import ShellRuntimeInfo
from fast_agent.ui.shell_notice import format_shell_notice


class _Runtime:
    def __init__(self, info: ShellRuntimeInfo, working_directory: str) -> None:
        self._info = info
        self._working_directory = working_directory

    def runtime_info(self) -> ShellRuntimeInfo:
        return self._info

    def working_directory(self) -> Path:
        return Path(self._working_directory)


def test_format_shell_notice_includes_environment_metadata() -> None:
    runtime = _Runtime(
        ShellRuntimeInfo(
            name="bash",
            path="/bin/bash",
            kind="docker",
            provider="docker",
            environment_name="ubuntu",
        ),
        "/workspace",
    )

    notice = format_shell_notice(("direct",), runtime)

    assert notice.plain == "Agents have shell (direct, bash | env: ubuntu docker | cwd: /workspace)"


def test_format_shell_notice_without_runtime_uses_access_modes() -> None:
    notice = format_shell_notice(("approval",), None)

    assert notice.plain == "Agents have shell (approval)"


def test_format_shell_notice_does_not_repeat_environment_kind() -> None:
    runtime = _Runtime(
        ShellRuntimeInfo(name="bash", kind="local", environment_name="local"),
        "/workspace",
    )

    notice = format_shell_notice(("resumed",), runtime)

    assert notice.plain == "Agents have shell (resumed, bash | env: local | cwd: /workspace)"
