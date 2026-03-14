from __future__ import annotations

import shlex
import sys

from fast_agent.ui.interactive_shell import (
    _PtyCleanupState,
    _update_alt_screen_state,
    run_interactive_shell_command,
)


def _python_shell_command(script: str) -> str:
    return f"{shlex.quote(sys.executable)} -c {shlex.quote(script)}"


def test_run_interactive_shell_command_captures_output() -> None:
    command = _python_shell_command("print('hello from shell')")

    result = run_interactive_shell_command(command, show_output=False)

    assert result.return_code == 0
    assert "hello from shell" in result.output


def test_run_interactive_shell_command_truncates_captured_output() -> None:
    command = _python_shell_command("print('x' * 32)")

    result = run_interactive_shell_command(
        command,
        max_output_chars=8,
        show_output=False,
    )

    assert result.return_code == 0
    assert len(result.output) == 8
    assert result.output == "xxxxxxx\n"


def test_update_alt_screen_state_tracks_enter_and_exit_sequences() -> None:
    cleanup_state = _PtyCleanupState()

    _update_alt_screen_state(cleanup_state, b"\x1b[?1049hhello")
    assert "1049" in cleanup_state.alt_screen_modes
    assert cleanup_state.needs_scroll_reset is True

    _update_alt_screen_state(cleanup_state, b"\x1b[?1049l")
    assert "1049" not in cleanup_state.alt_screen_modes
