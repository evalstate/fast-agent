from __future__ import annotations

from dataclasses import dataclass

from fast_agent.config import get_settings
from fast_agent.constants import FAST_AGENT_SHELL_CHILD_ENV


@dataclass
class ShellExecutionResult:
    return_code: int
    output: str


def run_interactive_shell_command(
    command: str,
    *,
    max_output_chars: int = 50000,
) -> ShellExecutionResult:
    import errno
    import os
    import signal
    import subprocess
    import sys

    output_buffer = ""
    return_code = 0
    proc: subprocess.Popen[str] | subprocess.Popen[bytes] | None = None
    master_fd: int | None = None
    tty_fd: int | None = None
    tty_in_fd: int | None = None
    tty_out_fd: int | None = None
    opened_tty = False
    old_tty: list[int] | None = None
    termios_module = None
    needs_scroll_reset = False
    scan_tail = b""
    alt_screen_modes: set[str] = set()

    def _append_output(chunk: str) -> None:
        nonlocal output_buffer
        output_buffer += chunk
        if len(output_buffer) > max_output_chars:
            output_buffer = output_buffer[-max_output_chars:]

    print(f"$ {command}", flush=True)

    shell_env = os.environ.copy()
    shell_env[FAST_AGENT_SHELL_CHILD_ENV] = "1"

    try:
        settings = get_settings()
        shell_settings = getattr(settings, "shell_execution", None)
        use_pty_setting = True
        if shell_settings is not None:
            use_pty_setting = bool(
                getattr(shell_settings, "interactive_use_pty", True)
            )

        use_pty = False
        if use_pty_setting and os.name != "nt":
            try:
                tty_fd = os.open("/dev/tty", os.O_RDWR | os.O_NOCTTY)
            except OSError:
                tty_fd = None
            if tty_fd is not None and os.isatty(tty_fd):
                tty_in_fd = tty_fd
                tty_out_fd = tty_fd
                opened_tty = True
                use_pty = True
            elif sys.stdin.isatty() and sys.stdout.isatty():
                tty_in_fd = sys.stdin.fileno()
                tty_out_fd = sys.stdout.fileno()
                use_pty = True

        if use_pty:
            import fcntl
            import pty
            import select
            import struct
            import termios
            import tty

            termios_module = termios
            master_fd, slave_fd = pty.openpty()
            if tty_in_fd is not None and os.isatty(tty_in_fd):
                try:
                    packed = fcntl.ioctl(
                        tty_in_fd,
                        termios.TIOCGWINSZ,
                        struct.pack("HHHH", 0, 0, 0, 0),
                    )
                    rows, cols, xpixels, ypixels = struct.unpack("HHHH", packed)
                    if rows and cols:
                        fcntl.ioctl(
                            slave_fd,
                            termios.TIOCSWINSZ,
                            struct.pack("HHHH", rows, cols, xpixels, ypixels),
                        )
                except OSError:
                    pass

            def _configure_pty_child() -> None:
                try:
                    os.setsid()
                except OSError:
                    pass
                try:
                    fcntl.ioctl(slave_fd, termios.TIOCSCTTY, 0)
                except OSError:
                    pass

            proc = subprocess.Popen(
                command,
                shell=True,
                stdin=slave_fd,
                stdout=slave_fd,
                stderr=slave_fd,
                close_fds=True,
                preexec_fn=_configure_pty_child,
                env=shell_env,
            )
            os.close(slave_fd)

            if tty_in_fd is not None and os.isatty(tty_in_fd):
                old_tty = termios.tcgetattr(tty_in_fd)
                tty.setraw(tty_in_fd)

            while True:
                read_fds = [master_fd]
                if tty_in_fd is not None and os.isatty(tty_in_fd):
                    read_fds.append(tty_in_fd)
                ready, _, _ = select.select(read_fds, [], [], 0.1)
                if master_fd in ready:
                    try:
                        data = os.read(master_fd, 1024)
                    except OSError as exc:
                        if exc.errno == errno.EIO:
                            break
                        raise
                    if not data:
                        break
                    scan_data = scan_tail + data
                    if b"[?1049h" in scan_data:
                        alt_screen_modes.add("1049")
                        needs_scroll_reset = True
                    if b"[?1047h" in scan_data:
                        alt_screen_modes.add("1047")
                        needs_scroll_reset = True
                    if b"[?47h" in scan_data:
                        alt_screen_modes.add("47")
                        needs_scroll_reset = True
                    if b"[?1049l" in scan_data:
                        alt_screen_modes.discard("1049")
                        needs_scroll_reset = True
                    if b"[?1047l" in scan_data:
                        alt_screen_modes.discard("1047")
                        needs_scroll_reset = True
                    if b"[?47l" in scan_data:
                        alt_screen_modes.discard("47")
                        needs_scroll_reset = True
                    scan_tail = scan_data[-16:]
                    if tty_out_fd is not None and os.isatty(tty_out_fd):
                        os.write(tty_out_fd, data)
                    else:
                        sys.stdout.buffer.write(data)
                        sys.stdout.flush()
                    _append_output(data.decode(errors="replace"))

                if tty_in_fd is not None and os.isatty(tty_in_fd) and tty_in_fd in ready:
                    try:
                        input_data = os.read(tty_in_fd, 1024)
                    except OSError:
                        input_data = b""
                    if b"" in input_data and proc is not None:
                        os.killpg(proc.pid, signal.SIGINT)
                        input_data = input_data.replace(b"", b"")
                    if input_data:
                        os.write(master_fd, input_data)

            return_code = proc.wait()
        else:
            proc = subprocess.Popen(
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
            if proc.stdout is not None:
                for line in iter(proc.stdout.readline, ""):
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    _append_output(line)
                    if proc.poll() is not None:
                        break
            return_code = proc.wait()
    except KeyboardInterrupt:
        if proc is not None:
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
            from rich import print as rich_print

            rich_print("[yellow]Shell command interrupted[/yellow]")
        else:
            return_code = 1
    finally:
        if old_tty is not None and tty_in_fd is not None and termios_module:
            try:
                termios_module.tcsetattr(tty_in_fd, termios_module.TCSADRAIN, old_tty)
            except Exception:
                pass
        if alt_screen_modes and tty_out_fd is not None and os.isatty(tty_out_fd):
            try:
                seq = b""
                if "1049" in alt_screen_modes:
                    seq += b"[?1049l"
                if "1047" in alt_screen_modes:
                    seq += b"[?1047l"
                if "47" in alt_screen_modes:
                    seq += b"[?47l"
                if seq:
                    os.write(tty_out_fd, seq)
            except OSError:
                pass
        if needs_scroll_reset and tty_out_fd is not None and os.isatty(tty_out_fd):
            try:
                os.write(tty_out_fd, b"[r")
            except OSError:
                pass
        if master_fd is not None:
            try:
                os.close(master_fd)
            except OSError:
                pass
        if opened_tty and tty_fd is not None:
            try:
                os.close(tty_fd)
            except OSError:
                pass

    return ShellExecutionResult(return_code=return_code, output=output_buffer)
