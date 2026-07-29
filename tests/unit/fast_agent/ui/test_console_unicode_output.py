from __future__ import annotations

import io

from rich.progress import Progress

from fast_agent.ui.console import SurrogateSafeConsole


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

    assert _stream_text(buffer, stream).splitlines() == [
        "valid 😀 malformed 😀 \\ud800 \\udfff"
    ]


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
