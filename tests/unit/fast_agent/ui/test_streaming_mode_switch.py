from typing import Literal

from fast_agent.config import Settings
from fast_agent.llm.stream_types import StreamChunk
from fast_agent.ui import console
from fast_agent.ui import streaming as streaming_module
from fast_agent.ui.console_display import ConsoleDisplay, _StreamingMessageHandle
from fast_agent.ui.stream_segments import StreamSegmentAssembler


def _set_console_size(width: int = 80, height: int = 24) -> tuple[object | None, object | None]:
    original_width = getattr(console.console, "_width", None)
    original_height = getattr(console.console, "_height", None)
    console.console._width = width
    console.console._height = height
    return original_width, original_height


def _restore_console_size(original_width: object | None, original_height: object | None) -> None:
    if original_width is None:
        if hasattr(console.console, "_width"):
            delattr(console.console, "_width")
    else:
        console.console._width = original_width
    if original_height is None:
        if hasattr(console.console, "_height"):
            delattr(console.console, "_height")
    else:
        console.console._height = original_height


def _make_handle(
    streaming_mode: Literal["markdown", "plain", "none"] = "markdown",
) -> _StreamingMessageHandle:
    settings = Settings()
    settings.logger.streaming = streaming_mode
    display = ConsoleDisplay(settings)
    return _StreamingMessageHandle(
        display=display,
        bottom_items=None,
        highlight_index=None,
        max_item_length=None,
        use_plain_text=streaming_mode == "plain",
        header_left="",
        header_right="",
        progress_display=None,
    )


def test_reasoning_stream_switches_back_to_markdown() -> None:
    assembler = StreamSegmentAssembler(base_kind="markdown", tool_prefix="->")

    assembler.handle_stream_chunk(StreamChunk("Intro"))
    assembler.handle_stream_chunk(StreamChunk("Thinking", is_reasoning=True))
    assembler.handle_stream_chunk(StreamChunk("Answer"))

    text = "".join(segment.text for segment in assembler.segments)
    intro_idx = text.find("Intro")
    thinking_idx = text.find("Thinking")
    answer_idx = text.find("Answer")
    assert intro_idx != -1
    assert thinking_idx != -1
    assert answer_idx != -1
    assert "\n" in text[intro_idx + len("Intro") : thinking_idx]
    assert "\n\n" in text[thinking_idx + len("Thinking") : answer_idx]


def test_reasoning_stream_handles_multiple_blocks() -> None:
    assembler = StreamSegmentAssembler(base_kind="markdown", tool_prefix="->")

    assembler.handle_stream_chunk(StreamChunk("Think1", is_reasoning=True))
    assembler.handle_stream_chunk(StreamChunk("Answer1"))
    assembler.handle_stream_chunk(StreamChunk("Think2", is_reasoning=True))
    assembler.handle_stream_chunk(StreamChunk("Answer2"))

    text = "".join(segment.text for segment in assembler.segments)
    assert "Think1" in text
    assert "Answer1" in text
    assert "Think2" in text
    assert "Answer2" in text


def test_reasoning_gap_merges_into_markdown_segment() -> None:
    assembler = StreamSegmentAssembler(base_kind="markdown", tool_prefix="->")

    assembler.handle_stream_chunk(StreamChunk("Intro"))
    assembler.handle_stream_chunk(StreamChunk("Thinking", is_reasoning=True))
    assembler.handle_stream_chunk(StreamChunk("Answer"))

    segments = assembler.segments
    assert [segment.kind for segment in segments] == ["markdown", "reasoning", "markdown"]
    assert segments[-1].text.startswith("\n\n")


def test_streaming_live_starts_without_initial_renderable() -> None:
    handle = _make_handle("markdown")
    assert handle._live is not None
    assert getattr(handle._live, "_renderable", "missing") is None


def test_sync_streaming_markdown_unthrottled_before_scroll(monkeypatch) -> None:
    handle = _make_handle("markdown")
    assert handle._async_mode is False

    render_calls: list[None] = []
    monkeypatch.setattr(handle, "_render_current_buffer", lambda: render_calls.append(None))

    handle.update("first")
    handle.update("second")
    handle.update("third")

    assert len(render_calls) == 3


def test_sync_streaming_respects_render_interval_after_scroll(monkeypatch) -> None:
    handle = _make_handle("markdown")
    assert handle._async_mode is False
    assert handle._min_render_interval is not None
    handle._scrolling_started = True

    render_calls: list[None] = []
    monkeypatch.setattr(handle, "_render_current_buffer", lambda: render_calls.append(None))

    interval = handle._min_render_interval or 0.25
    monotonic_values = [0.0, 0.0, interval / 2, interval + 0.01, interval + 0.01]

    def _fake_monotonic() -> float:
        if monotonic_values:
            return monotonic_values.pop(0)
        return interval + 0.01

    monkeypatch.setattr(streaming_module.time, "monotonic", _fake_monotonic)

    handle.update("first")
    handle.update("second")
    handle.update("third")

    assert len(render_calls) == 2


def test_sync_streaming_plain_respects_render_interval(monkeypatch) -> None:
    handle = _make_handle("plain")
    assert handle._async_mode is False
    assert handle._min_render_interval is not None

    render_calls: list[None] = []
    monkeypatch.setattr(handle, "_render_current_buffer", lambda: render_calls.append(None))

    interval = handle._min_render_interval or 0.05
    monotonic_values = [0.0, 0.0, interval / 2, interval + 0.01, interval + 0.01]

    def _fake_monotonic() -> float:
        if monotonic_values:
            return monotonic_values.pop(0)
        return interval + 0.01

    monkeypatch.setattr(streaming_module.time, "monotonic", _fake_monotonic)

    handle.update("first")
    handle.update("second")
    handle.update("third")

    assert len(render_calls) == 2


def test_resolve_progress_resume_debounce_seconds_from_env(monkeypatch) -> None:
    monkeypatch.setenv("FAST_AGENT_PROGRESS_RESUME_DEBOUNCE_SECONDS", "0.05")
    assert streaming_module._resolve_progress_resume_debounce_seconds() == 0.05

    monkeypatch.setenv("FAST_AGENT_PROGRESS_RESUME_DEBOUNCE_SECONDS", "-1")
    assert streaming_module._resolve_progress_resume_debounce_seconds() == 0.0

    monkeypatch.setenv("FAST_AGENT_PROGRESS_RESUME_DEBOUNCE_SECONDS", "invalid")
    assert streaming_module._resolve_progress_resume_debounce_seconds() == 0.12


def test_stream_cursor_suffix_only_targets_last_segment() -> None:
    handle = _make_handle("plain")

    assert handle._cursor_suffix(segment_index=0, total_segments=2) == ""
    assert handle._cursor_suffix(segment_index=1, total_segments=2) == streaming_module.STREAM_CURSOR_BLOCK

    handle._show_stream_cursor = False
    assert handle._cursor_suffix(segment_index=1, total_segments=2) == ""


def test_preserve_final_frame_requires_rendered_content() -> None:
    handle = _make_handle("markdown")
    assert handle.preserve_final_frame() is False


def test_preserve_final_frame_allows_pending_reasoning_content() -> None:
    handle = _make_handle("markdown")
    handle.update_chunk(StreamChunk("<think>"))
    assert handle.preserve_final_frame() is True


def test_preserve_final_frame_sets_live_non_transient() -> None:
    class _FakeLive:
        def __init__(self) -> None:
            self.transient = True
            self.exited = False

        def __exit__(self, *_args: object) -> None:
            self.exited = True

    handle = _make_handle("markdown")
    fake_live = _FakeLive()
    handle._live = fake_live
    handle._live_started = True

    assert handle.preserve_final_frame() is True
    handle._shutdown_live_resources()

    assert fake_live.transient is False
    assert fake_live.exited is True


def test_preserve_final_frame_finalize_disables_padding(monkeypatch) -> None:
    handle = _make_handle("markdown")
    handle._preserve_final_frame = True
    handle._max_render_height = 99
    handle._height_fudge = 3
    handle._handle_chunk("short response")

    monkeypatch.setattr(handle._segment_assembler, "flush", lambda: False)

    captured: list[tuple[int, int]] = []

    def _capture_render() -> None:
        captured.append((handle._max_render_height, handle._height_fudge))

    monkeypatch.setattr(handle, "_render_current_buffer", _capture_render)

    handle.finalize("short response")

    assert captured == [(0, 0)]
