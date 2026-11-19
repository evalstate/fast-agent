import re
import time

from fast_agent.config import Settings
from fast_agent.llm.stream_types import StreamChunk
from fast_agent.ui.console_display import ConsoleDisplay
from fast_agent.ui.streaming import StreamContextBaseline, StreamingMessageHandle
from fast_agent.ui.streaming.display import _TokenRate


def _rate(label: str) -> float:
    return float(label.removeprefix("~").removesuffix(" tok/s"))


def test_token_rate_eases_through_silence_and_bursts() -> None:
    rate = _TokenRate()
    assert rate.label(0.0) == ""

    now = 0.0
    for _ in range(20):  # steady 100 tok/s
        rate.record(200, now)
        now += 0.5
        steady = _rate(rate.label(now))
    assert 95 <= steady <= 105

    quiet = _rate(rate.label(now + 0.5))
    assert 0 < quiet < steady  # decays rather than snapping to zero

    rate.record(40_000, now + 1.0)  # 10k tokens released in one burst
    assert _rate(rate.label(now + 1.5)) < 20_000 / 2


def test_header_shows_model_then_live_context_then_rate() -> None:
    handle = StreamingMessageHandle(
        display=ConsoleDisplay(Settings()),
        header_right="[dim]claude[/dim]",
        context_baseline=StreamContextBaseline(tokens=1_000, window=100_000),
    )
    assert handle._build_header().plain.rstrip().endswith("claude (1.00%)")

    handle._token_rate.record(4_000, time.monotonic() - 1.0)  # backdated so the rate ticks
    assert re.search(r"claude \(2\.00%\) ~\d+ tok/s\s*$", handle._build_header().plain)

    handle._handle_stream_chunk(StreamChunk("x" * 4_000))
    assert "claude (3.00%)" in handle._build_header().plain

    handle.update_model("claude (2.50%)")
    header = handle._build_header().plain
    assert "claude (2.50%) ~" in header and "3.00%" not in header


def test_header_shows_queued_messages_only_while_streaming(monkeypatch) -> None:
    from fast_agent.ui.streaming import display as streaming_display

    handle = StreamingMessageHandle(
        display=ConsoleDisplay(Settings()),
        header_right="[dim]claude[/dim]",
    )
    monkeypatch.setattr(streaming_display, "queued_input_count", lambda: 0)
    assert "queued" not in handle._build_header().plain
    monkeypatch.setattr(streaming_display, "queued_input_count", lambda: 2)
    assert re.search(r"claude ⧗ 2 queued\s*$", handle._build_header().plain)
    # The final frame can stay in scrollback: it must not carry the count.
    handle._show_stream_cursor = False
    assert "queued" not in handle._build_header().plain


def test_narrow_width_header_fits_single_line(monkeypatch) -> None:
    from rich.console import Console

    from fast_agent.ui import console as ui_console

    narrow_console = Console(width=32, height=20, force_terminal=True)
    monkeypatch.setattr(ui_console, "console", narrow_console)

    handle = StreamingMessageHandle(
        display=ConsoleDisplay(Settings()),
        header_left="[bold green]▎[/bold green][dim green]◀[/dim green] [green]assistant[/green]",
        header_right="[dim]claude-sonnet-4-5-20250929-long-model[/dim]",
        context_baseline=StreamContextBaseline(tokens=12_000, window=100_000),
    )
    handle._scroll_indicator_visible = True
    handle._token_rate.record(4_000, time.monotonic() - 1.0)

    header = handle._build_header()
    assert header.cell_len <= 32
    lines = narrow_console.render_lines(header, narrow_console.options, pad=False)
    assert len(lines) == 1
