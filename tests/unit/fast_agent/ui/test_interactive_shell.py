from __future__ import annotations

import shlex
import sys

from fast_agent.config import Settings, ShellSettings
from fast_agent.ui.interactive_shell import (
    _interactive_shell_prefers_pty,
    _is_windows_interactive_shell_command,
    _PtyCleanupState,
    _update_alt_screen_state,
    _windows_shell_token,
    run_interactive_shell_command,
)


def _python_shell_command(script: str) -> str:
    return f"{shlex.quote(sys.executable)} -c {shlex.quote(script)}"


def test_interactive_shell_prefers_pty_reads_typed_shell_setting(monkeypatch) -> None:
    monkeypatch.setattr("fast_agent.ui.interactive_shell.os.name", "posix")
    monkeypatch.setattr(
        "fast_agent.ui.interactive_shell.get_settings",
        lambda: Settings(shell_execution=ShellSettings(interactive_use_pty=False)),
    )

    assert not _interactive_shell_prefers_pty()


def test_run_interactive_shell_command_captures_output() -> None:
    command = _python_shell_command("print('hello from shell')")

    result = run_interactive_shell_command(command, show_output=False)

    assert result.exit_code == 0
    assert "hello from shell" in result.stdout


def test_run_interactive_shell_command_truncates_captured_output() -> None:
    command = _python_shell_command("print('x' * 32)")

    result = run_interactive_shell_command(
        command,
        max_output_chars=8,
        show_output=False,
    )

    assert result.exit_code == 0
    assert len(result.stdout) == 8
    assert result.stdout in {"xxxxxxx\n", "xxxxxx\r\n"}


def test_windows_shell_token_handles_quoted_paths() -> None:
    token, end = _windows_shell_token(r'"C:\Program Files\PowerShell\7\pwsh.exe"')

    assert token == r"C:\Program Files\PowerShell\7\pwsh.exe"
    assert end == len(r'"C:\Program Files\PowerShell\7\pwsh.exe"')


def test_windows_interactive_shell_detection(monkeypatch) -> None:
    monkeypatch.setattr("fast_agent.ui.interactive_shell.os.name", "nt")
    monkeypatch.setattr(
        "fast_agent.utils.shell_detection.default_shell_command",
        lambda: r"C:\Program Files\PowerShell\7\pwsh.exe",
    )

    assert _is_windows_interactive_shell_command(r"C:\Program Files\PowerShell\7\pwsh.exe")
    assert _is_windows_interactive_shell_command(r'"C:\Program Files\PowerShell\7\pwsh.exe"')
    assert _is_windows_interactive_shell_command("pwsh")
    assert not _is_windows_interactive_shell_command("pwsh -NoProfile -Command Get-Date")


def test_update_alt_screen_state_tracks_enter_and_exit_sequences() -> None:
    cleanup_state = _PtyCleanupState()

    _update_alt_screen_state(cleanup_state, b"\x1b[?1049hhello")
    assert "1049" in cleanup_state.alt_screen_modes
    assert cleanup_state.needs_scroll_reset is True

    _update_alt_screen_state(cleanup_state, b"\x1b[?1049l")
    assert "1049" not in cleanup_state.alt_screen_modes
