from __future__ import annotations

import errno
import ntpath
import os
import signal
import subprocess
import sys
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any

from rich import print as rich_print

from fast_agent.config import get_settings
from fast_agent.constants import FAST_AGENT_SHELL_CHILD_ENV
from fast_agent.tools.execution_environment import ShellExecutionResult


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


@dataclass(frozen=True, slots=True)
class _ResolvedInteractiveTtyTargets:
    use_pty: bool
    targets: _TerminalTargets


@dataclass(slots=True)
class _PtyCleanupState:
    master_fd: int | None = None
    old_tty: list[int] | None = None
    termios_module: Any | None = None
    needs_scroll_reset: bool = False
    scan_tail: bytes = b""
    alt_screen_modes: set[str] = field(default_factory=set)


@dataclass(frozen=True, slots=True)
class _LaunchedPtyShellProcess:
    proc: subprocess.Popen[bytes]
    cleanup_state: _PtyCleanupState


def _build_interactive_shell_env() -> dict[str, str]:
    shell_env = os.environ.copy()
    shell_env[FAST_AGENT_SHELL_CHILD_ENV] = "1"
    return shell_env


def _interactive_shell_prefers_pty() -> bool:
    if os.name == "nt":
        return False

    return get_settings().shell_execution.interactive_use_pty


def _resolve_interactive_tty_targets() -> _ResolvedInteractiveTtyTargets:
    targets = _TerminalTargets()
    if not _interactive_shell_prefers_pty():
        return _ResolvedInteractiveTtyTargets(use_pty=False, targets=targets)

    try:
        targets.tty_fd = os.open("/dev/tty", os.O_RDWR | os.O_NOCTTY)
    except OSError:
        targets.tty_fd = None

    if targets.tty_fd is not None and os.isatty(targets.tty_fd):
        targets.tty_in_fd = targets.tty_fd
        targets.tty_out_fd = targets.tty_fd
        targets.opened_tty = True
        return _ResolvedInteractiveTtyTargets(use_pty=True, targets=targets)

    if sys.stdin.isatty() and sys.stdout.isatty():
        targets.tty_in_fd = sys.stdin.fileno()
        targets.tty_out_fd = sys.stdout.fileno()
        return _ResolvedInteractiveTtyTargets(use_pty=True, targets=targets)

    return _ResolvedInteractiveTtyTargets(use_pty=False, targets=targets)


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
        with suppress(OSError):
            os.setsid()
        with suppress(OSError):
            fcntl_module.ioctl(slave_fd, termios_module.TIOCSCTTY, 0)

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
) -> _LaunchedPtyShellProcess:
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
    return _LaunchedPtyShellProcess(proc=proc, cleanup_state=cleanup_state)


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


def _start_attached_shell_process(
    command: str,
    *,
    shell_env: dict[str, str],
) -> subprocess.Popen[Any]:
    if executable := _windows_interactive_shell_executable(command):
        return subprocess.Popen(
            [executable],
            start_new_session=True,
            env=shell_env,
        )

    return subprocess.Popen(
        command,
        shell=True,
        start_new_session=True,
        env=shell_env,
    )


def _run_attached_shell_loop(proc: subprocess.Popen[Any]) -> int:
    return proc.wait()


def _windows_shell_token(command: str) -> tuple[str, int]:
    stripped = command.strip()
    if not stripped:
        return "", 0
    if stripped[0] in {"'", '"'}:
        quote = stripped[0]
        end = stripped.find(quote, 1)
        return (stripped[1:end], end + 1) if end != -1 else (stripped[1:], len(stripped))
    token = stripped.split(maxsplit=1)[0]
    return token, len(token)


def _is_windows_interactive_shell_command(command: str) -> bool:
    return _windows_interactive_shell_executable(command) is not None


def _windows_interactive_shell_executable(command: str) -> str | None:
    if os.name != "nt":
        return None

    from fast_agent.utils.shell_detection import default_shell_command

    stripped = command.strip()
    shell = default_shell_command()
    if stripped.lower() == shell.lower():
        return shell

    token, token_end = _windows_shell_token(stripped)
    if not token:
        return None
    remainder = command.strip()[token_end:].strip()
    if remainder:
        return None

    shell_name = ntpath.basename(shell).lower()
    normalized_token = token.lower()
    token_name = ntpath.basename(normalized_token)
    shell_names = {shell_name, shell_name.removesuffix(".exe")}
    if normalized_token in {shell.lower(), *shell_names} or token_name in shell_names:
        return token
    return None


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
    if os.name == "nt":
        with suppress(ProcessLookupError, ValueError, OSError):
            proc.send_signal(signal.CTRL_BREAK_EVENT)
    else:
        with suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGINT)

    try:
        return_code = proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            with suppress(ProcessLookupError):
                proc.kill()
        else:
            with suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
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

    with suppress(Exception):
        cleanup_state.termios_module.tcsetattr(
            targets.tty_in_fd,
            cleanup_state.termios_module.TCSADRAIN,
            cleanup_state.old_tty,
        )


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

    with suppress(OSError):
        os.write(targets.tty_out_fd, b"\x1b[r")


def _close_interactive_shell_fds(
    targets: _TerminalTargets,
    cleanup_state: _PtyCleanupState,
) -> None:
    if cleanup_state.master_fd is not None:
        with suppress(OSError):
            os.close(cleanup_state.master_fd)

    if targets.opened_tty and targets.tty_fd is not None:
        with suppress(OSError):
            os.close(targets.tty_fd)


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
    echo_command: bool = True,
) -> ShellExecutionResult:
    output_capture = _CapturedShellOutput(max_output_chars=max_output_chars)
    return_code = 0
    proc: subprocess.Popen[str] | subprocess.Popen[bytes] | None = None
    cleanup_state = _PtyCleanupState()

    if echo_command:
        print(f"$ {command}", flush=True)

    shell_env = _build_interactive_shell_env()
    resolved_targets = _resolve_interactive_tty_targets()
    targets = resolved_targets.targets

    try:
        if resolved_targets.use_pty:
            launched_process = _launch_pty_shell_process(
                command,
                shell_env=shell_env,
                targets=targets,
            )
            proc = launched_process.proc
            cleanup_state = launched_process.cleanup_state
            return_code = _run_pty_shell_loop(
                proc,
                cleanup_state=cleanup_state,
                targets=targets,
                show_output=show_output,
                output_capture=output_capture,
            )
        else:
            if _is_windows_interactive_shell_command(command):
                proc = _start_attached_shell_process(command, shell_env=shell_env)
                return_code = _run_attached_shell_loop(proc)
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

    return ShellExecutionResult(stdout=output_capture.output, stderr="", exit_code=return_code)
