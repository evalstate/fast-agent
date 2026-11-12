from __future__ import annotations

import os
import shlex
import signal
import sys
import time

import pytest

from fast_agent.config import Settings, ShellSettings
from fast_agent.ui.interactive_shell import (
    _CapturedShellOutput,
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


def test_pipe_output_that_is_not_utf8_decodes_with_the_locale_encoding(monkeypatch) -> None:
    # A child writing its locale code page (e.g. Python 3.12-3.14 on cp1252 Windows)
    # must stay readable, alongside UTF-8 output from the same command.
    monkeypatch.setattr("fast_agent.utils.text.locale.getencoding", lambda: "cp1252")
    command = _python_shell_command(
        "import sys; sys.stdout.buffer.write(b'caf\\xe9\\n' + 'café\\n'.encode())"
    )

    result = run_interactive_shell_command(command, show_output=False)

    assert result.stdout.splitlines() == ["café", "café"]


def _use_pipe(monkeypatch) -> None:
    monkeypatch.setattr(
        "fast_agent.ui.interactive_shell._interactive_shell_prefers_pty", lambda: False
    )


def test_pipe_output_left_after_the_shell_exits_is_not_dropped(monkeypatch) -> None:
    # A reader that falls behind must still collect everything the shell wrote.
    _use_pipe(monkeypatch)
    append = _CapturedShellOutput.append

    def slow_append(self: _CapturedShellOutput, chunk: str) -> None:
        append(self, chunk)
        time.sleep(0.1)

    monkeypatch.setattr(_CapturedShellOutput, "append", slow_append)
    command = _python_shell_command("print('one'); print('two'); print('three')")

    result = run_interactive_shell_command(command, show_output=False)

    assert result.stdout.splitlines() == ["one", "two", "three"]


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX background job syntax")
def test_pipe_returns_when_a_background_job_keeps_the_pipe_open(monkeypatch) -> None:
    _use_pipe(monkeypatch)
    started = time.monotonic()

    result = run_interactive_shell_command(
        "echo start; sleep 30 & echo $!; echo end", show_output=False
    )
    elapsed = time.monotonic() - started
    start, pid, end = result.stdout.splitlines()
    os.kill(int(pid), signal.SIGTERM)  # closes the pipe and ends the reader

    assert elapsed < 10
    assert result.exit_code == 0
    assert (start, end) == ("start", "end")


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


@pytest.mark.skipif(sys.platform == "win32", reason="PTYs are unavailable on Windows")
@pytest.mark.parametrize(
    ("restore_scroll_region", "expected"),
    [(False, None), (True, b"\x1b[r")],
)
def test_run_interactive_shell_restores_requested_scroll_region(
    restore_scroll_region: bool,
    expected: bytes | None,
) -> None:
    import errno
    import os
    import pty

    pid, master_fd = pty.fork()
    if pid == 0:
        result = run_interactive_shell_command(
            _python_shell_command("pass"),
            show_output=False,
            echo_command=False,
            restore_scroll_region=restore_scroll_region,
        )
        os._exit(result.exit_code)

    output = b""
    while True:
        try:
            chunk = os.read(master_fd, 1024)
        except OSError as exc:
            if exc.errno == errno.EIO:
                break
            raise
        if not chunk:
            break
        output += chunk

    _, wait_status = os.waitpid(pid, 0)
    os.close(master_fd)

    assert os.waitstatus_to_exitcode(wait_status) == 0
    assert output == (expected or b"")
