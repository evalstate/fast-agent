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
