from __future__ import annotations

import io
import os
import sys

from rich.progress import Progress

from fast_agent.ui.console import (
    SurrogateSafeConsole,
    _redirect_standard_stream_to_blocking_tty,
)


class _CountingStream(io.StringIO):
    def __init__(self) -> None:
        super().__init__()
        self.flush_count = 0

    def flush(self) -> None:
        self.flush_count += 1
        super().flush()


def _strict_utf8_stream() -> tuple[io.BytesIO, io.TextIOWrapper]:
    buffer = io.BytesIO()
    stream = io.TextIOWrapper(buffer, encoding="utf-8", errors="strict")
    return buffer, stream


def _stream_text(buffer: io.BytesIO, stream: io.TextIOWrapper) -> str:
    stream.flush()
    return buffer.getvalue().decode("utf-8")


def test_console_escapes_surrogates_and_preserves_valid_unicode() -> None:
    buffer, stream = _strict_utf8_stream()
    console = SurrogateSafeConsole(file=stream, color_system=None)

    console.print("valid 😀 malformed \ud83d\ude00 \ud800 \udfff")

    assert _stream_text(buffer, stream).splitlines() == ["valid 😀 malformed 😀 \\ud800 \\udfff"]


def test_console_file_direct_write_is_surrogate_safe() -> None:
    buffer, stream = _strict_utf8_stream()
    console = SurrogateSafeConsole(file=stream, color_system=None)

    written = console.file.write("streamed \ud800 text")

    assert written == len("streamed \ud800 text")
    assert _stream_text(buffer, stream) == "streamed \\ud800 text"


def test_console_reuses_adapter_without_implicit_flushes() -> None:
    stream = _CountingStream()
    console = SurrogateSafeConsole(file=stream, color_system=None)

    first = console.file
    second = console.file
    first.write("streamed \ud800 text")

    assert first is second
    assert stream.flush_count == 0
    assert stream.getvalue() == "streamed \\ud800 text"


def test_progress_stop_does_not_crash_on_surrogate_description() -> None:
    buffer, stream = _strict_utf8_stream()
    console = SurrogateSafeConsole(
        file=stream,
        force_terminal=True,
        color_system=None,
        width=80,
    )
    progress = Progress(console=console, auto_refresh=False)
    progress.add_task("invalid \ud83d\ude00", total=None)

    progress.start()
    progress.refresh()
    progress.stop()

    assert "😀" in _stream_text(buffer, stream)


def test_original_stdout_is_redirected_to_blocking_stream(monkeypatch) -> None:
    source_read, source_write = os.pipe()
    target_read, target_write = os.pipe()
    source = os.fdopen(source_write, "w", buffering=1, encoding="utf-8", closefd=False)
    target = os.fdopen(target_write, "w", buffering=1, encoding="utf-8", closefd=False)
    os.set_blocking(source_write, False)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(sys, "__stdout__", source)
            _redirect_standard_stream_to_blocking_tty(source_write, target)

            assert os.get_blocking(source_write)
            active_stdout = sys.__stdout__
            assert active_stdout is not None
            active_stdout.write("kitty")
            active_stdout.flush()

        assert os.read(target_read, 5) == b"kitty"
    finally:
        source.close()
        target.close()
        os.close(source_read)
        os.close(source_write)
        os.close(target_read)
        os.close(target_write)
