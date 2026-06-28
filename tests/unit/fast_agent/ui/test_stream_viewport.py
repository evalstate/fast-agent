from rich.console import Console

from fast_agent.ui.markdown_truncator import MarkdownTruncator
from fast_agent.ui.plain_text_truncator import PlainTextTruncator
from fast_agent.ui.stream_segments import StreamSegment
from fast_agent.ui.stream_viewport import StreamViewport, estimate_plain_text_height


class _FakeMarkdownTruncator(MarkdownTruncator):
    def __init__(
        self,
        *,
        measured_heights: dict[str, int],
        truncated_text: str = "trimmed",
    ) -> None:
        super().__init__()
        self._measured_heights = measured_heights
        self._truncated_text = truncated_text
        self.measure_calls = 0
        self.truncate_calls = 0

    def measure_rendered_height(
        self,
        text: str,
        console: Console,
        code_theme: str = "monokai",
    ) -> int:
        del console, code_theme
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


def test_estimate_plain_text_height_counts_wrapped_and_blank_lines() -> None:
    assert estimate_plain_text_height("", 10) == 0
    assert estimate_plain_text_height("abcd", 2) == 2
    assert estimate_plain_text_height("abcd\n\nx", 2) == 4


def test_estimate_plain_text_height_expands_tabs_and_clamps_width() -> None:
    assert estimate_plain_text_height("a\tb", 4) == 3
    assert estimate_plain_text_height("ab", 0) == 2


def test_markdown_viewport_measures_precisely_before_skipping_truncation() -> None:
    truncator = _FakeMarkdownTruncator(
        measured_heights={"hello": 31, "trimmed": 10},
    )
    viewport = StreamViewport(
        markdown_truncator=truncator,
        plain_truncator=PlainTextTruncator(),
    )
    console = Console(width=80)

    window = viewport.slice_segments_with_heights(
        [StreamSegment(kind="markdown", text="hello")],
        terminal_height=20,
        console=console,
        target_ratio=0.93,
    )

    assert len(window.segments) == 1
    assert window.segments[0].text == "trimmed"
    assert window.heights == [10]
    assert truncator.measure_calls == 2
    assert truncator.truncate_calls == 1


def test_markdown_viewport_measures_precisely_near_budget() -> None:
    truncator = _FakeMarkdownTruncator(
        measured_heights={"hello": 12},
    )
    viewport = StreamViewport(
        markdown_truncator=truncator,
        plain_truncator=PlainTextTruncator(),
    )
    console = Console(width=80)

    window = viewport.slice_segments_with_heights(
        [StreamSegment(kind="markdown", text="hello")],
        terminal_height=20,
        console=console,
        target_ratio=0.93,
    )

    assert len(window.segments) == 1
    assert window.heights == [12]
    assert truncator.measure_calls == 1


def test_reasoning_viewport_uses_markdown_measurement() -> None:
    truncator = _FakeMarkdownTruncator(
        measured_heights={"thinking": 12},
    )
    viewport = StreamViewport(
        markdown_truncator=truncator,
        plain_truncator=PlainTextTruncator(),
    )
    console = Console(width=80)

    window = viewport.slice_segments_with_heights(
        [StreamSegment(kind="reasoning", text="thinking")],
        terminal_height=20,
        console=console,
        target_ratio=0.93,
    )

    assert len(window.segments) == 1
    assert window.heights == [12]
    assert truncator.measure_calls == 1
