from __future__ import annotations

import errno
import os
import signal
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any

from rich import print as rich_print

from fast_agent.config import get_settings
from fast_agent.constants import FAST_AGENT_SHELL_CHILD_ENV


@dataclass
class ShellExecutionResult:
    return_code: int
    output: str


@dataclass(slots=True)
class _CapturedShellOutput:
    max_output_chars: int
    output: str = ""

    def append(self, chunk: str) -> None:
        self.output += chunk
        if len(self.output) > self.max_output_chars:
            self.output = self.output[-self.max_output_chars :]


@dataclass(slots=True)
class _TerminalTargets:
    tty_fd: int | None = None
    tty_in_fd: int | None = None
    tty_out_fd: int | None = None
    opened_tty: bool = False


@dataclass(slots=True)
class _PtyCleanupState:
    master_fd: int | None = None
    old_tty: list[int] | None = None
    termios_module: Any | None = None
    needs_scroll_reset: bool = False
    scan_tail: bytes = b""
    alt_screen_modes: set[str] = field(default_factory=set)


def _build_interactive_shell_env() -> dict[str, str]:
    shell_env = os.environ.copy()
    shell_env[FAST_AGENT_SHELL_CHILD_ENV] = "1"
    return shell_env


def _interactive_shell_prefers_pty() -> bool:
    if os.name == "nt":
        return False

    settings = get_settings()
    shell_settings = getattr(settings, "shell_execution", None)
    if shell_settings is None:
        return True
    return bool(getattr(shell_settings, "interactive_use_pty", True))


def _resolve_interactive_tty_targets() -> tuple[bool, _TerminalTargets]:
    targets = _TerminalTargets()
    if not _interactive_shell_prefers_pty():
        return False, targets

    try:
        targets.tty_fd = os.open("/dev/tty", os.O_RDWR | os.O_NOCTTY)
    except OSError:
        targets.tty_fd = None

    if targets.tty_fd is not None and os.isatty(targets.tty_fd):
        targets.tty_in_fd = targets.tty_fd
        targets.tty_out_fd = targets.tty_fd
        targets.opened_tty = True
        return True, targets

    if sys.stdin.isatty() and sys.stdout.isatty():
        targets.tty_in_fd = sys.stdin.fileno()
        targets.tty_out_fd = sys.stdout.fileno()
        return True, targets

    return False, targets


def _copy_tty_window_size_to_pty_slave(
    *,
    tty_in_fd: int | None,
    slave_fd: int,
    fcntl_module: Any,
    struct_module: Any,
    termios_module: Any,
) -> None:
    if tty_in_fd is None or not os.isatty(tty_in_fd):
        return

    try:
        packed = fcntl_module.ioctl(
            tty_in_fd,
            termios_module.TIOCGWINSZ,
            struct_module.pack("HHHH", 0, 0, 0, 0),
        )
        rows, cols, xpixels, ypixels = struct_module.unpack("HHHH", packed)
        if rows and cols:
            fcntl_module.ioctl(
                slave_fd,
                termios_module.TIOCSWINSZ,
                struct_module.pack("HHHH", rows, cols, xpixels, ypixels),
            )
    except OSError:
        pass


def _configure_pty_child(slave_fd: int, fcntl_module: Any, termios_module: Any):
    def _configure_child() -> None:
        try:
            os.setsid()
        except OSError:
            pass
        try:
            fcntl_module.ioctl(slave_fd, termios_module.TIOCSCTTY, 0)
        except OSError:
            pass

    return _configure_child


def _set_tty_raw_mode(
    *,
    targets: _TerminalTargets,
    cleanup_state: _PtyCleanupState,
    termios_module: Any,
    tty_module: Any,
) -> None:
    tty_in_fd = targets.tty_in_fd
    if tty_in_fd is None or not os.isatty(tty_in_fd):
        return

    cleanup_state.old_tty = termios_module.tcgetattr(tty_in_fd)
    cleanup_state.termios_module = termios_module
    tty_module.setraw(tty_in_fd)


def _launch_pty_shell_process(
    command: str,
    *,
    shell_env: dict[str, str],
    targets: _TerminalTargets,
) -> tuple[subprocess.Popen[bytes], _PtyCleanupState]:
    import fcntl
    import pty
    import struct
    import termios
    import tty

    cleanup_state = _PtyCleanupState()
    master_fd, slave_fd = pty.openpty()
    cleanup_state.master_fd = master_fd

    _copy_tty_window_size_to_pty_slave(
        tty_in_fd=targets.tty_in_fd,
        slave_fd=slave_fd,
        fcntl_module=fcntl,
        struct_module=struct,
        termios_module=termios,
    )

    proc = subprocess.Popen(
        command,
        shell=True,
        stdin=slave_fd,
        stdout=slave_fd,
        stderr=slave_fd,
        close_fds=True,
        preexec_fn=_configure_pty_child(slave_fd, fcntl, termios),
        env=shell_env,
    )
    os.close(slave_fd)

    _set_tty_raw_mode(
        targets=targets,
        cleanup_state=cleanup_state,
        termios_module=termios,
        tty_module=tty,
    )
    return proc, cleanup_state


def _update_alt_screen_state(cleanup_state: _PtyCleanupState, data: bytes) -> None:
    scan_data = cleanup_state.scan_tail + data
    if b"\x1b[?1049h" in scan_data:
        cleanup_state.alt_screen_modes.add("1049")
        cleanup_state.needs_scroll_reset = True
    if b"\x1b[?1047h" in scan_data:
        cleanup_state.alt_screen_modes.add("1047")
        cleanup_state.needs_scroll_reset = True
    if b"\x1b[?47h" in scan_data:
        cleanup_state.alt_screen_modes.add("47")
        cleanup_state.needs_scroll_reset = True
    if b"\x1b[?1049l" in scan_data:
        cleanup_state.alt_screen_modes.discard("1049")
        cleanup_state.needs_scroll_reset = True
    if b"\x1b[?1047l" in scan_data:
        cleanup_state.alt_screen_modes.discard("1047")
        cleanup_state.needs_scroll_reset = True
    if b"\x1b[?47l" in scan_data:
        cleanup_state.alt_screen_modes.discard("47")
        cleanup_state.needs_scroll_reset = True
    cleanup_state.scan_tail = scan_data[-16:]


def _write_shell_output_bytes(
    data: bytes,
    *,
    show_output: bool,
    tty_out_fd: int | None,
) -> None:
    if not show_output:
        return

    if tty_out_fd is not None and os.isatty(tty_out_fd):
        os.write(tty_out_fd, data)
        return

    sys.stdout.buffer.write(data)
    sys.stdout.flush()


def _handle_pty_output_ready(
    *,
    cleanup_state: _PtyCleanupState,
    targets: _TerminalTargets,
    show_output: bool,
    output_capture: _CapturedShellOutput,
) -> bool:
    master_fd = cleanup_state.master_fd
    if master_fd is None:
        return False

    try:
        data = os.read(master_fd, 1024)
    except OSError as exc:
        if exc.errno == errno.EIO:
            return False
        raise

    if not data:
        return False

    _update_alt_screen_state(cleanup_state, data)
    _write_shell_output_bytes(
        data,
        show_output=show_output,
        tty_out_fd=targets.tty_out_fd,
    )
    output_capture.append(data.decode(errors="replace"))
    return True


def _forward_tty_input_to_pty(*, master_fd: int | None, tty_in_fd: int | None) -> None:
    if master_fd is None or tty_in_fd is None or not os.isatty(tty_in_fd):
        return

    try:
        input_data = os.read(tty_in_fd, 1024)
    except OSError:
        input_data = b""

    if input_data:
        os.write(master_fd, input_data)


def _run_pty_shell_loop(
    proc: subprocess.Popen[bytes],
    *,
    cleanup_state: _PtyCleanupState,
    targets: _TerminalTargets,
    show_output: bool,
    output_capture: _CapturedShellOutput,
) -> int:
    import select

    master_fd = cleanup_state.master_fd
    if master_fd is None:
        return proc.wait()

    while True:
        read_fds = [master_fd]
        if targets.tty_in_fd is not None and os.isatty(targets.tty_in_fd):
            read_fds.append(targets.tty_in_fd)

        ready, _, _ = select.select(read_fds, [], [], 0.1)
        if master_fd in ready and not _handle_pty_output_ready(
            cleanup_state=cleanup_state,
            targets=targets,
            show_output=show_output,
            output_capture=output_capture,
        ):
            break

        if targets.tty_in_fd is not None and targets.tty_in_fd in ready:
            _forward_tty_input_to_pty(
                master_fd=master_fd,
                tty_in_fd=targets.tty_in_fd,
            )

    return proc.wait()


def _start_pipe_shell_process(
    command: str,
    *,
    shell_env: dict[str, str],
) -> subprocess.Popen[str]:
    return subprocess.Popen(
        command,
        shell=True,
        start_new_session=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        errors="replace",
        env=shell_env,
    )


def _run_pipe_shell_loop(
    proc: subprocess.Popen[str],
    *,
    show_output: bool,
    output_capture: _CapturedShellOutput,
) -> int:
    if proc.stdout is not None:
        for line in iter(proc.stdout.readline, ""):
            if show_output:
                sys.stdout.write(line)
                sys.stdout.flush()
            output_capture.append(line)
            if proc.poll() is not None:
                break
    return proc.wait()


def _interrupt_shell_process(proc: subprocess.Popen[Any]) -> int:
    try:
        os.killpg(proc.pid, signal.SIGINT)
    except ProcessLookupError:
        pass

    try:
        return_code = proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return_code = proc.wait()

    rich_print("[yellow]Shell command interrupted[/yellow]")
    return return_code


def _restore_tty_mode(targets: _TerminalTargets, cleanup_state: _PtyCleanupState) -> None:
    if (
        cleanup_state.old_tty is None
        or targets.tty_in_fd is None
        or cleanup_state.termios_module is None
    ):
        return

    try:
        cleanup_state.termios_module.tcsetattr(
            targets.tty_in_fd,
            cleanup_state.termios_module.TCSADRAIN,
            cleanup_state.old_tty,
        )
    except Exception:
        pass


def _reset_alt_screen_modes(targets: _TerminalTargets, cleanup_state: _PtyCleanupState) -> None:
    if not cleanup_state.alt_screen_modes:
        return
    if targets.tty_out_fd is None or not os.isatty(targets.tty_out_fd):
        return

    try:
        seq = b""
        if "1049" in cleanup_state.alt_screen_modes:
            seq += b"\x1b[?1049l"
        if "1047" in cleanup_state.alt_screen_modes:
            seq += b"\x1b[?1047l"
        if "47" in cleanup_state.alt_screen_modes:
            seq += b"\x1b[?47l"
        if seq:
            os.write(targets.tty_out_fd, seq)
    except OSError:
        pass


def _reset_tty_scroll_region(targets: _TerminalTargets, cleanup_state: _PtyCleanupState) -> None:
    if not cleanup_state.needs_scroll_reset:
        return
    if targets.tty_out_fd is None or not os.isatty(targets.tty_out_fd):
        return

    try:
        os.write(targets.tty_out_fd, b"\x1b[r")
    except OSError:
        pass


def _close_interactive_shell_fds(
    targets: _TerminalTargets,
    cleanup_state: _PtyCleanupState,
) -> None:
    if cleanup_state.master_fd is not None:
        try:
            os.close(cleanup_state.master_fd)
        except OSError:
            pass

    if targets.opened_tty and targets.tty_fd is not None:
        try:
            os.close(targets.tty_fd)
        except OSError:
            pass


def _cleanup_interactive_shell(targets: _TerminalTargets, cleanup_state: _PtyCleanupState) -> None:
    _restore_tty_mode(targets, cleanup_state)
    _reset_alt_screen_modes(targets, cleanup_state)
    _reset_tty_scroll_region(targets, cleanup_state)
    _close_interactive_shell_fds(targets, cleanup_state)


def run_interactive_shell_command(
    command: str,
    *,
    max_output_chars: int = 50000,
    show_output: bool = True,
) -> ShellExecutionResult:
    output_capture = _CapturedShellOutput(max_output_chars=max_output_chars)
    return_code = 0
    proc: subprocess.Popen[str] | subprocess.Popen[bytes] | None = None
    cleanup_state = _PtyCleanupState()

    print(f"$ {command}", flush=True)

    shell_env = _build_interactive_shell_env()
    use_pty, targets = _resolve_interactive_tty_targets()

    try:
        if use_pty:
            proc, cleanup_state = _launch_pty_shell_process(
                command,
                shell_env=shell_env,
                targets=targets,
            )
            return_code = _run_pty_shell_loop(
                proc,
                cleanup_state=cleanup_state,
                targets=targets,
                show_output=show_output,
                output_capture=output_capture,
            )
        else:
            proc = _start_pipe_shell_process(command, shell_env=shell_env)
            return_code = _run_pipe_shell_loop(
                proc,
                show_output=show_output,
                output_capture=output_capture,
            )
    except KeyboardInterrupt:
        return_code = _interrupt_shell_process(proc) if proc is not None else 1
    finally:
        _cleanup_interactive_shell(targets, cleanup_state)

    return ShellExecutionResult(return_code=return_code, output=output_capture.output)
