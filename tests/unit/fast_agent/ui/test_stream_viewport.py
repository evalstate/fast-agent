from rich.console import Console

from fast_agent.ui.plain_text_truncator import PlainTextTruncator
from fast_agent.ui.stream_segments import StreamSegment
from fast_agent.ui.stream_viewport import StreamViewport


class _FakeMarkdownTruncator:
    def __init__(
        self,
        *,
        estimate_height: int,
        measured_heights: dict[str, int],
        truncated_text: str = "trimmed",
    ) -> None:
        self._estimate_height = estimate_height
        self._measured_heights = measured_heights
        self._truncated_text = truncated_text
        self.measure_calls = 0
        self.truncate_calls = 0

    def estimate_rendered_height(self, _text: str, _terminal_width: int) -> int:
        return self._estimate_height

    def measure_rendered_height(
        self,
        text: str,
        _console: Console,
        code_theme: str = "monokai",
    ) -> int:
        del code_theme
        self.measure_calls += 1
        return self._measured_heights[text]

    def truncate_to_height(
        self,
        text: str,
        *,
        terminal_height: int,
        console: Console | None,
        code_theme: str = "monokai",
    ) -> str:
        del terminal_height, console, code_theme
        self.truncate_calls += 1
        return self._truncated_text


def test_markdown_viewport_measures_precisely_before_skipping_truncation() -> None:
    truncator = _FakeMarkdownTruncator(
        estimate_height=2,
        measured_heights={"hello": 31, "trimmed": 10},
    )
    viewport = StreamViewport(
        markdown_truncator=truncator,  # type: ignore[arg-type]
        plain_truncator=PlainTextTruncator(),
    )
    console = Console(width=80)

    segments, heights = viewport.slice_segments_with_heights(
        [StreamSegment(kind="markdown", text="hello")],
        terminal_height=20,
        console=console,
        target_ratio=0.93,
    )

    assert len(segments) == 1
    assert segments[0].text == "trimmed"
    assert heights == [10]
    assert truncator.measure_calls == 2
    assert truncator.truncate_calls == 1


def test_markdown_viewport_measures_precisely_near_budget() -> None:
    truncator = _FakeMarkdownTruncator(
        estimate_height=13,
        measured_heights={"hello": 12},
    )
    viewport = StreamViewport(
        markdown_truncator=truncator,  # type: ignore[arg-type]
        plain_truncator=PlainTextTruncator(),
    )
    console = Console(width=80)

    segments, heights = viewport.slice_segments_with_heights(
        [StreamSegment(kind="markdown", text="hello")],
        terminal_height=20,
        console=console,
        target_ratio=0.93,
    )

    assert len(segments) == 1
    assert heights == [12]
    assert truncator.measure_calls == 1
