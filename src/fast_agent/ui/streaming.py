from __future__ import annotations

import asyncio
import os
import sys
import time
from dataclasses import dataclass
from typing import IO, TYPE_CHECKING, Any, Callable, Mapping, Protocol, TextIO, cast

from rich.console import Console, Group, RenderHook
from rich.control import Control
from rich.file_proxy import FileProxy
from rich.live import Live
from rich.markdown import Markdown
from rich.segment import ControlType, Segment
from rich.syntax import Syntax
from rich.text import Text

from fast_agent.core.logging.logger import get_logger
from fast_agent.llm.stream_types import StreamChunk
from fast_agent.ui import console
from fast_agent.ui.apply_patch_preview import style_apply_patch_preview_text
from fast_agent.ui.markdown_helpers import prepare_markdown_content
from fast_agent.ui.markdown_renderables import (
    build_markdown_renderable,
    close_incomplete_code_blocks,
)
from fast_agent.ui.markdown_truncator import MarkdownTruncator
from fast_agent.ui.plain_text_truncator import PlainTextTruncator
from fast_agent.ui.stream_segments import StreamSegmentAssembler
from fast_agent.ui.stream_viewport import StreamViewport

if TYPE_CHECKING:
    from rich.console import ConsoleRenderable, RenderableType

    from fast_agent.mcp.prompt_message_extended import PromptMessageExtended
    from fast_agent.ui.console_display import ConsoleDisplay
    from fast_agent.ui.stream_segments import StreamSegment


logger = get_logger(__name__)

MARKDOWN_STREAM_TARGET_RATIO = 0.93
MARKDOWN_STREAM_REFRESH_PER_SECOND = 16
MARKDOWN_STREAM_PRE_SCROLL_THROTTLE_RATIO = 0.7
STREAM_RENDER_WIDTH_GUTTER = 1
# Keep only a small anti-flicker pad now that scroll-indicator churn is debounced.
MARKDOWN_STREAM_HEIGHT_FUDGE = 1
PLAIN_STREAM_TARGET_RATIO = 0.95
PLAIN_STREAM_REFRESH_PER_SECOND = 20
PLAIN_STREAM_HEIGHT_FUDGE = 1
STREAM_BATCH_PERIOD = 1 / 100
STREAM_BATCH_MAX_DURATION = 1 / 60
STREAM_CURSOR_BLOCK = "●"
SCROLL_INDICATOR_DEBOUNCE_SECONDS = 0.2
# Lines reserved for the stream header (header text + spacing newline) plus a
# safety margin that absorbs rendering differences (e.g. inter-paragraph spacing
# introduced when consecutive markdown segments are coalesced into a single
# Markdown renderable).
_STREAM_HEADER_AND_MARGIN_LINES = 3


def _resolve_progress_resume_debounce_seconds() -> float:
    """Return debounce duration for progress resume after streaming closes."""
    raw_value = os.getenv("FAST_AGENT_PROGRESS_RESUME_DEBOUNCE_SECONDS", "0.12").strip()
    try:
        parsed = float(raw_value)
    except ValueError:
        return 0.12
    return max(0.0, parsed)


STREAM_PROGRESS_RESUME_DEBOUNCE_SECONDS = _resolve_progress_resume_debounce_seconds()


def _alt_screen_streaming_enabled() -> bool:
    raw_value = os.getenv("FAST_AGENT_STREAM_ALT_SCREEN", "").strip().lower()
    return raw_value in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class _ToolStreamEvent:
    event_type: str
    info: dict[str, Any] | None


@dataclass(frozen=True)
class _QueuedItem:
    payload: object
    enqueued_at: float


@dataclass(frozen=True)
class _RenderedLine:
    text: str
    cell_length: int


class _DiffLive(RenderHook):
    """Minimal live-region renderer that updates changed lines in place."""

    def __init__(
        self,
        renderable: RenderableType | None = None,
        *,
        console: Console,
        screen: bool = False,
        auto_refresh: bool = True,
        refresh_per_second: float = 4,
        transient: bool = False,
        redirect_stdout: bool = True,
        redirect_stderr: bool = True,
        vertical_overflow: str = "ellipsis",
        get_renderable: Callable[[], RenderableType] | None = None,
    ) -> None:
        del screen, auto_refresh, refresh_per_second, vertical_overflow
        self.console = console
        self.transient = transient
        self._renderable = renderable
        self._get_renderable = get_renderable
        self._started = False
        self._lines: list[_RenderedLine] = []
        self._is_interactive = self.console.is_terminal
        self._nested = False
        self._redirect_stdout = redirect_stdout
        self._redirect_stderr = redirect_stderr
        self._restore_stdout: IO[str] | None = None
        self._restore_stderr: IO[str] | None = None
        self._console_state_active = False
        self._cursor_below_frame = False
        self._frame_truncated = False

    def __enter__(self) -> "_DiffLive":
        if not self._started:
            self._started = True
            if self._is_interactive:
                if not self.console.set_live(self):
                    self._nested = True
                    return self
                console.ensure_blocking_console()
                self.console.show_cursor(False)
                self._enable_redirect_io()
                self.console.push_render_hook(self)
                self._console_state_active = True
        return self

    def __exit__(self, *_args: object) -> None:
        self.stop()

    def update(self, renderable: RenderableType, *, refresh: bool = False) -> None:
        self._renderable = renderable
        if refresh:
            self.refresh()

    def refresh(self) -> None:
        if not self._started:
            self.__enter__()
        renderable = self.get_renderable()
        if renderable is None:
            return
        if not self._is_interactive:
            return
        lines = self._render_lines(renderable)

        # Safety: clamp both the new frame and the stored old frame to the
        # visible terminal height.  When a frame exceeds the visible area,
        # _scroll_newline() at the bottom row triggers implicit terminal
        # scrolls that shift all content upward.  Our relative-cursor
        # bookkeeping cannot detect these shifts, so subsequent diffs would
        # mis-position the cursor and duplicate or overwrite rows.  Keeping
        # both old and new within bounds guarantees that cursor-up commands
        # can always reach the first stored line.
        max_visible = self.console.size.height
        self._frame_truncated = False
        if max_visible > 0:
            if len(lines) > max_visible:
                self._frame_truncated = True
                lines = lines[-max_visible:]
            if len(self._lines) > max_visible:
                self._lines = self._lines[-max_visible:]

        if not self._lines:
            self._write_initial(lines)
        else:
            self._write_diff(lines)
        self._lines = lines

    def stop(self) -> None:
        if not self._started:
            return
        try:
            self._started = False
            if not self._is_interactive:
                renderable = self.get_renderable()
                if renderable is not None:
                    self.console.print(renderable)
                return
            self.console.clear_live()
            if self._nested:
                if not self.transient:
                    renderable = self.get_renderable()
                    if renderable is not None:
                        self.console.print(renderable)
                return
            if self._lines:
                if self.transient:
                    self._clear_region()
                elif self._frame_truncated:
                    renderable = self.get_renderable()
                    self._clear_region()
                    if renderable is not None:
                        self.console.print(renderable)
                    else:
                        self._write("\n")
                elif not self._cursor_below_frame:
                    self._write("\n")
        finally:
            if self._is_interactive and self._console_state_active:
                self._disable_redirect_io()
                self.console.pop_render_hook()
                self.console.show_cursor(True)
                self._console_state_active = False
            self._nested = False
            self._frame_truncated = False

    def get_renderable(self) -> RenderableType | None:
        if self._get_renderable is not None:
            return self._get_renderable()
        return self._renderable

    def process_renderables(
        self,
        renderables: list["ConsoleRenderable"],
    ) -> list["ConsoleRenderable"]:
        if not self._is_interactive or not self._started or self._nested:
            return renderables
        renderable = self.get_renderable()
        if renderable is None:
            return renderables
        if isinstance(renderable, str):
            current_renderable: ConsoleRenderable = self.console.render_str(renderable)
        else:
            current_renderable = cast("ConsoleRenderable", renderable)
        # Rich prints a trailing newline for console output, so once we re-render
        # the live frame as part of that output the cursor ends one line below it.
        self._cursor_below_frame = True
        return [self._position_cursor_control(), *renderables, current_renderable]

    def _render_lines(self, renderable: RenderableType) -> list[_RenderedLine]:
        options = self.console.options.update(width=self.console.size.width)
        rendered_lines = self.console.render_lines(renderable, options=options, pad=False)
        return [
            _RenderedLine(
                text=self.console._render_buffer(line),
                cell_length=Segment.get_line_length(line),
            )
            for line in rendered_lines
        ]

    def _write_initial(self, lines: list[_RenderedLine]) -> None:
        if not lines:
            return
        payload_parts: list[str] = []
        for index, line in enumerate(lines):
            if index:
                payload_parts.append(self._scroll_newline())
            payload_parts.append(line.text)
        payload_parts.append(str(Control.move_to_column(0)))
        payload = "".join(payload_parts)
        self._write(payload)
        self._cursor_below_frame = False

    def _write_diff(self, lines: list[_RenderedLine]) -> None:
        old_lines = self._lines
        first_diff = 0
        shared = min(len(old_lines), len(lines))
        while first_diff < shared and old_lines[first_diff] == lines[first_diff]:
            first_diff += 1
        if first_diff == len(old_lines) == len(lines):
            return
        if first_diff == len(old_lines) and len(lines) > len(old_lines):
            self._write_appended_lines(lines[first_diff:])
            return
        if old_lines and len(lines) > len(old_lines) and first_diff == len(old_lines) - 1:
            self._rewrite_growing_last_line(old_lines[-1], lines[first_diff:])
            return

        max_height = max(len(old_lines), len(lines))
        current_row = len(old_lines) - 1
        if self._cursor_below_frame:
            current_row += 1
        payload: list[str] = [self._move_to_line_start(first_diff - current_row)]
        last_old_row = len(old_lines) - 1

        for row in range(first_diff, max_height):
            new_line = lines[row] if row < len(lines) else None
            old_line = old_lines[row] if row < len(old_lines) else None
            if new_line is not None:
                payload.append(new_line.text)
                if old_line is not None and old_line.cell_length > new_line.cell_length:
                    payload.append(str(Control((ControlType.ERASE_IN_LINE, 0))))
            else:
                payload.append(str(Control((ControlType.ERASE_IN_LINE, 2))))

            if row < max_height - 1:
                next_row = row + 1
                if row >= last_old_row or next_row > last_old_row:
                    payload.append(self._scroll_newline())
                else:
                    payload.append(self._move_to_line_start(1))

        target_row = len(lines) - 1
        payload.append(self._move_to_line_start(target_row - (max_height - 1)))
        self._write("".join(payload))
        self._cursor_below_frame = False

    def _write_appended_lines(self, lines: list[_RenderedLine]) -> None:
        if not lines:
            return
        payload_parts: list[str] = []
        if self._cursor_below_frame:
            payload_parts.append(str(Control.move_to_column(0)))
            payload_parts.append(lines[0].text)
            remaining_lines = lines[1:]
        else:
            remaining_lines = lines
        for line in remaining_lines:
            payload_parts.append(self._scroll_newline())
            payload_parts.append(line.text)
        payload_parts.append(str(Control.move_to_column(0)))
        payload = "".join(payload_parts)
        self._write(payload)
        self._cursor_below_frame = False

    def _rewrite_growing_last_line(
        self,
        old_last_line: _RenderedLine,
        new_lines: list[_RenderedLine],
    ) -> None:
        if not new_lines:
            return
        row_delta = -1 if self._cursor_below_frame else 0
        payload = [self._move_to_line_start(row_delta), new_lines[0].text]
        if old_last_line.cell_length > new_lines[0].cell_length:
            payload.append(str(Control((ControlType.ERASE_IN_LINE, 0))))
        for line in new_lines[1:]:
            payload.append(self._scroll_newline())
            payload.append(line.text)
        payload.append(str(Control.move_to_column(0)))
        self._write("".join(payload))
        self._cursor_below_frame = False

    def _clear_region(self) -> None:
        height = len(self._lines)
        if height <= 0:
            return
        cursor_row = height - 1
        if self._cursor_below_frame:
            cursor_row += 1
        payload = [self._move_to_line_start(-cursor_row)]
        for row in range(height):
            payload.append(str(Control((ControlType.ERASE_IN_LINE, 2))))
            if row < height - 1:
                payload.append(self._move_to_line_start(1))
        payload.append(self._move_to_line_start(-(height - 1)))
        self._write("".join(payload))
        self._lines = []
        self._cursor_below_frame = False

    def _move_to_line_start(self, row_delta: int) -> str:
        return str(Control.move_to_column(0, y=row_delta))

    def _scroll_newline(self) -> str:
        """Advance to the next physical row in a way that is safe after right-edge writes."""
        return f"{Control.move_to_column(0)}\n"

    def _position_cursor_control(self) -> Control:
        height = len(self._lines)
        if height <= 0:
            return Control()
        lines_to_rewind = height - 1
        if self._cursor_below_frame:
            lines_to_rewind += 1
        return Control(
            ControlType.CARRIAGE_RETURN,
            (ControlType.ERASE_IN_LINE, 2),
            *(((ControlType.CURSOR_UP, 1), (ControlType.ERASE_IN_LINE, 2)) * lines_to_rewind),
        )

    def _write(self, text: str) -> None:
        if not text:
            return
        with self.console._lock:
            self.console.file.write(text)
            self.console.file.flush()

    def _enable_redirect_io(self) -> None:
        if not self._is_interactive:
            return
        if self._redirect_stdout and not isinstance(sys.stdout, FileProxy):
            self._restore_stdout = sys.stdout
            sys.stdout = cast("TextIO", FileProxy(self.console, sys.stdout))
        if self._redirect_stderr and not isinstance(sys.stderr, FileProxy):
            self._restore_stderr = sys.stderr
            sys.stderr = cast("TextIO", FileProxy(self.console, sys.stderr))

    def _disable_redirect_io(self) -> None:
        if self._restore_stdout:
            sys.stdout = cast("TextIO", self._restore_stdout)
            self._restore_stdout = None
        if self._restore_stderr:
            sys.stderr = cast("TextIO", self._restore_stderr)
            self._restore_stderr = None


class NullStreamingHandle:
    """No-op streaming handle used when streaming is disabled."""

    def update(self, _chunk: str) -> None:
        return

    def update_chunk(self, _chunk: StreamChunk) -> None:
        return

    def finalize(self, _message: "PromptMessageExtended | str") -> None:
        return

    def close(self) -> None:
        return

    def handle_tool_event(self, _event_type: str, info: dict[str, Any] | None = None) -> None:
        return

    def has_scrolled(self) -> bool:
        return False

    def preserve_final_frame(self) -> bool:
        return False

    async def wait_for_drain(self) -> None:
        return


class StreamingMessageHandle:
    """Helper that manages live rendering for streaming assistant responses."""

    def __init__(
        self,
        *,
        display: "ConsoleDisplay",
        bottom_items: list[str] | None,
        highlight_index: int | None,
        max_item_length: int | None,
        use_plain_text: bool = False,
        header_left: str = "",
        header_right: str = "",
        tool_header_name: str | None = None,
        tool_metadata_resolver: Callable[[str], Mapping[str, Any] | None] | None = None,
        progress_display: Any = None,
        performance_hook: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self._display = display
        self._bottom_items = bottom_items
        self._highlight_index = highlight_index
        self._max_item_length = max_item_length
        self._use_plain_text = use_plain_text
        self._header_left = header_left
        self._header_right = header_right
        self._tool_header_prefix: Text | None = None
        self._tool_header_prefix_plain = ""
        self._tool_header_color: str | None = None
        self._progress_display = progress_display
        self._progress_paused = False
        self._plain_text_style: str | None = None
        base_kind = "plain" if use_plain_text else "markdown"
        self._render_reasoning_markdown = not use_plain_text
        self._set_tool_header_prefix(tool_header_name)
        self._segment_assembler = StreamSegmentAssembler(
            base_kind=base_kind,
            tool_prefix=self._tool_header_prefix_plain,
            tool_metadata_resolver=tool_metadata_resolver,
            apply_patch_preview_max_lines=self._display.apply_patch_preview_max_lines,
        )
        self._markdown_truncator = MarkdownTruncator(
            target_height_ratio=1.0,
            code_word_wrap=self._display.code_word_wrap,
            render_fences_with_syntax=self._display.render_fences_with_syntax,
        )
        self._plain_truncator = PlainTextTruncator(target_height_ratio=1.0)
        self._viewport = StreamViewport(
            markdown_truncator=self._markdown_truncator,
            plain_truncator=self._plain_truncator,
            code_theme=self._display.code_style,
        )
        self._stream_target_ratio = (
            PLAIN_STREAM_TARGET_RATIO if use_plain_text else MARKDOWN_STREAM_TARGET_RATIO
        )
        self._height_fudge = (
            PLAIN_STREAM_HEIGHT_FUDGE if use_plain_text else MARKDOWN_STREAM_HEIGHT_FUDGE
        )
        refresh_rate = (
            PLAIN_STREAM_REFRESH_PER_SECOND
            if self._use_plain_text
            else MARKDOWN_STREAM_REFRESH_PER_SECOND
        )
        self._min_render_interval = 1.0 / refresh_rate if refresh_rate else None
        self._last_render_time = 0.0
        self._performance_hook = performance_hook
        self._batch_period = STREAM_BATCH_PERIOD
        self._batch_max_duration = STREAM_BATCH_MAX_DURATION
        self._pending_batch_meta: dict[str, Any] | None = None
        self._scrolling_started = False
        self._scroll_start_time: float | None = None
        self._pre_scroll_throttle_started = False
        self._display_truncated = False
        self._scroll_indicator_visible = False
        self._scroll_indicator_pending_since: float | None = None
        self._header_cache: dict[tuple[int, bool], Text] = {}
        self._next_render_deadline: float | None = None
        self._alt_screen_streaming = _alt_screen_streaming_enabled()
        try:
            self._loop: asyncio.AbstractEventLoop | None = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None
        self._async_mode = self._loop is not None
        self._queue: asyncio.Queue[object] | None = asyncio.Queue() if self._async_mode else None
        self._stop_sentinel: object = object()
        self._worker_task: asyncio.Task[None] | None = None
        self._live: Any | None
        if self._alt_screen_streaming:
            self._live = Live(
                None,
                console=console.console,
                screen=True,
                vertical_overflow="ellipsis",
                refresh_per_second=refresh_rate,
                auto_refresh=False,
                transient=True,
                redirect_stdout=True,
                redirect_stderr=True,
            )
        else:
            self._live = _DiffLive(
                None,
                console=console.console,
                vertical_overflow="ellipsis",
                refresh_per_second=refresh_rate,
                auto_refresh=False,
                transient=True,
            )
        self._live_started = False
        self._active = True
        self._finalized = False
        self._show_stream_cursor = True
        self._max_render_height = 0
        self._preserve_final_frame = False
        self._suppress_tail_padding_for_final_frame = False

        if self._async_mode and self._loop and self._queue is not None:
            self._worker_task = self._loop.create_task(self._render_worker())

    def _set_tool_header_prefix(self, tool_header_name: str | None) -> None:
        from fast_agent.ui.message_primitives import MESSAGE_CONFIGS, MessageType

        config = MESSAGE_CONFIGS[MessageType.TOOL_CALL]
        self._tool_header_color = config["block_color"]

        header_markup = self._display.build_header_left(
            block_color=config["block_color"],
            arrow=config["arrow"],
            arrow_style=config["arrow_style"],
            name=tool_header_name,
            is_error=False,
            show_hook_indicator=False,
        )
        header_text = Text.from_markup(header_markup)
        self._tool_header_prefix = header_text
        self._tool_header_prefix_plain = header_text.plain

    def update(self, chunk: str) -> None:
        if not self._active or not chunk:
            return

        if self._async_mode and self._queue is not None:
            self._enqueue_chunk(chunk)
            return

        if self._handle_chunk(chunk):
            now = time.perf_counter()
            self._pending_batch_meta = {
                "batch_size": 1,
                "queue_depth": 0,
                "oldest_enqueued_at": now,
                "newest_enqueued_at": now,
                "batch_chars": len(chunk),
            }
            self._render_sync_if_due()

    def update_chunk(self, chunk: StreamChunk) -> None:
        """Structured streaming update with an explicit reasoning flag."""
        if not self._active or not chunk or not chunk.text:
            return

        if self._async_mode and self._queue is not None:
            self._enqueue_chunk(chunk)
            return

        if self._handle_stream_chunk(chunk):
            now = time.perf_counter()
            self._pending_batch_meta = {
                "batch_size": 1,
                "queue_depth": 0,
                "oldest_enqueued_at": now,
                "newest_enqueued_at": now,
                "batch_chars": len(chunk.text),
            }
            self._render_sync_if_due()

    def _build_header(self) -> Text:
        width = console.console.size.width
        cache_key = (width, self._scroll_indicator_visible)
        cached = self._header_cache.get(cache_key)
        if cached is not None:
            return cached

        right_content = self._header_right.strip()
        if self._scroll_indicator_visible:
            indicator = "[black on blue]scrolling[/black on blue]"
            right_content = f"{right_content} {indicator}" if right_content else indicator

        combined = self._display._format_header_line(self._header_left, right_content)
        self._header_cache[cache_key] = combined
        if len(self._header_cache) > 8:
            self._header_cache.clear()
        return combined

    def _pause_progress_display(self) -> None:
        if self._progress_display and not self._progress_paused:
            try:
                self._progress_display.pause()
                self._progress_paused = True
            except Exception:
                self._progress_paused = False

    def _resume_progress_display(self) -> None:
        if self._progress_display and self._progress_paused:
            try:
                self._progress_display.resume(
                    debounce_seconds=STREAM_PROGRESS_RESUME_DEBOUNCE_SECONDS
                )
            except TypeError:
                # Backward compatibility for non-standard displays in tests/experiments.
                self._progress_display.resume()
            except Exception:
                pass
            finally:
                self._progress_paused = False

    def _ensure_started(self) -> None:
        if not self._live or self._live_started:
            return

        self._pause_progress_display()

        if self._live and not self._live_started:
            self._live.__enter__()
            self._live_started = True

    def _close_incomplete_code_blocks(self, text: str) -> str:
        return close_incomplete_code_blocks(text)

    def _set_scroll_indicator_visible(self, visible: bool) -> None:
        if self._scroll_indicator_visible == visible:
            return
        self._scroll_indicator_visible = visible
        self._header_cache.clear()

    def _reset_scroll_indicator(self) -> None:
        self._scroll_indicator_pending_since = None
        self._set_scroll_indicator_visible(False)

    def _update_scroll_status(self, *, is_truncated: bool, now: float) -> None:
        self._display_truncated = is_truncated
        if is_truncated and not self._scrolling_started:
            self._scrolling_started = True
            self._scroll_start_time = now

        if not is_truncated:
            self._scroll_indicator_pending_since = None
            return

        if self._scroll_indicator_visible:
            return

        if self._scroll_indicator_pending_since is None:
            self._scroll_indicator_pending_since = now
            return

        if now - self._scroll_indicator_pending_since >= SCROLL_INDICATOR_DEBOUNCE_SECONDS:
            self._set_scroll_indicator_visible(True)

    def finalize(self, _message: "PromptMessageExtended | str") -> None:
        if not self._active or self._finalized:
            return

        # Remove the transient cursor in the final frame before closing Live rendering.
        self._show_stream_cursor = False
        if not self._preserve_final_frame:
            self._reset_scroll_indicator()

        # Flush any buffered reasoning content before closing the live view
        if self._segment_assembler.flush():
            self._suppress_tail_padding_for_final_frame = self._preserve_final_frame
            self._render_current_buffer()
        elif self._segment_assembler.segments:
            self._suppress_tail_padding_for_final_frame = self._preserve_final_frame
            self._render_current_buffer()

        self._finalized = True
        self.close()

    def close(self) -> None:
        if not self._active:
            return

        self._active = False
        if self._async_mode:
            if self._queue and self._loop:
                try:
                    current_loop = asyncio.get_running_loop()
                except RuntimeError:
                    current_loop = None

                try:
                    if current_loop is self._loop:
                        self._queue.put_nowait(self._stop_sentinel)
                    else:
                        self._loop.call_soon_threadsafe(self._queue.put_nowait, self._stop_sentinel)
                except RuntimeError as exc:
                    logger.debug(
                        "RuntimeError while closing streaming display (expected during shutdown)",
                        data={"error": str(exc)},
                    )
                except Exception as exc:
                    logger.warning(
                        "Unexpected error while closing streaming display",
                        exc_info=True,
                        data={"error": str(exc)},
                    )
            if self._worker_task:
                self._worker_task.cancel()
                self._worker_task = None
        self._shutdown_live_resources()
        self._max_render_height = 0
        self._next_render_deadline = None

    async def _wait_for_render_slot(self) -> bool:
        """Sleep until the next render deadline to keep frame cadence steady."""
        interval = self._min_render_interval
        if not interval:
            return True

        if not self._render_throttle_active:
            # Keep markdown updates immediate while content still fits the viewport.
            self._next_render_deadline = None
            return True

        now = time.monotonic()
        deadline = self._next_render_deadline
        if deadline is None:
            self._next_render_deadline = now
            return True

        delay = deadline - now
        if delay <= 0:
            return True

        try:
            await asyncio.sleep(delay)
        except asyncio.CancelledError:
            return False
        return True

    def _advance_render_deadline(self) -> None:
        """Advance render deadline without drifting under variable render cost."""
        interval = self._min_render_interval
        if not interval:
            return

        if not self._render_throttle_active:
            self._next_render_deadline = None
            return

        now = time.monotonic()
        deadline = self._next_render_deadline
        if deadline is None:
            self._next_render_deadline = now + interval
            return

        next_deadline = deadline + interval
        if next_deadline <= now:
            skipped_slots = int((now - deadline) // interval) + 1
            next_deadline = deadline + (skipped_slots * interval)
        self._next_render_deadline = next_deadline

    def _sync_render_due(self) -> bool:
        """Return True when sync mode is allowed to render the next frame."""
        interval = self._min_render_interval
        if not interval:
            return True

        if not self._render_throttle_active:
            self._next_render_deadline = None
            return True

        now = time.monotonic()
        deadline = self._next_render_deadline
        if deadline is None:
            self._next_render_deadline = now
            return True
        return now >= deadline

    @property
    def _render_throttle_active(self) -> bool:
        """Whether frame pacing should currently throttle renders.

        Plain-text mode keeps its fixed cadence. Markdown starts unthrottled,
        but once content begins truncating or grows tall enough to be highly
        reflow-sensitive in the current viewport, we switch to the configured
        cadence for the rest of the stream.
        """
        if self._use_plain_text:
            return True
        return self._scrolling_started or self._pre_scroll_throttle_started

    def _update_pre_scroll_throttle(self, *, content_height: int, max_allowed_height: int) -> None:
        if self._use_plain_text or self._pre_scroll_throttle_started:
            return
        if max_allowed_height <= 0:
            return
        threshold = max(6, int(max_allowed_height * MARKDOWN_STREAM_PRE_SCROLL_THROTTLE_RATIO))
        if content_height >= threshold:
            self._pre_scroll_throttle_started = True

    def _render_sync_if_due(self) -> None:
        """Render in sync mode while respecting frame-rate limits."""
        if not self._sync_render_due():
            return
        self._render_current_buffer()
        self._advance_render_deadline()

    def _enqueue_chunk(self, chunk: object) -> None:
        if not self._queue or not self._loop:
            return

        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        queued = (
            chunk
            if isinstance(chunk, _QueuedItem)
            else _QueuedItem(payload=chunk, enqueued_at=time.perf_counter())
        )
        if current_loop is self._loop:
            try:
                self._queue.put_nowait(queued)
            except asyncio.QueueFull:
                pass
        else:
            try:
                self._loop.call_soon_threadsafe(self._queue.put_nowait, queued)
            except RuntimeError as exc:
                logger.debug(
                    "RuntimeError while enqueuing chunk (expected during shutdown)",
                    data={"error": str(exc), "chunk_repr": repr(chunk)},
                )
            except Exception as exc:
                logger.warning(
                    "Unexpected error while enqueuing chunk",
                    exc_info=True,
                    data={"error": str(exc), "chunk_repr": repr(chunk)},
                )

    def _handle_stream_chunk(self, chunk: StreamChunk) -> bool:
        """Process a typed stream chunk with explicit reasoning flag."""
        return self._segment_assembler.handle_stream_chunk(chunk)

    def _handle_chunk(self, chunk: str) -> bool:
        return self._segment_assembler.handle_text(chunk)

    def _cursor_suffix(self, *, segment_index: int, total_segments: int) -> str:
        if not self._show_stream_cursor:
            return ""
        if total_segments <= 0:
            return ""
        if segment_index != total_segments - 1:
            return ""
        return STREAM_CURSOR_BLOCK

    def has_scrolled(self) -> bool:
        """Return whether viewport truncation/scrolling has started."""
        return self._scrolling_started

    def preserve_final_frame(self) -> bool:
        """Request that closing leaves the final streamed frame visible.

        Returns True when the request can be honored for this stream.
        """
        if not self._live:
            return False
        if self._alt_screen_streaming:
            return False
        if (
            not self._live_started
            and not self._segment_assembler.segments
            and not self._segment_assembler.has_pending_content()
        ):
            return False
        self._preserve_final_frame = True
        return True

    async def wait_for_drain(self) -> None:
        """Wait until all queued stream updates have been rendered."""
        if not self._async_mode or self._queue is None:
            return
        if self._worker_task is None or self._worker_task.done():
            return
        await self._queue.join()

    def _render_current_buffer(self) -> None:
        if not self._active:
            return

        segments = self._segment_assembler.segments
        if not segments:
            return

        self._ensure_started()

        if not self._live:
            return
        width_override = self._effective_stream_width()
        width_attr_present = hasattr(console.console, "_width")
        original_width_attr = getattr(console.console, "_width", None)
        try:
            if width_override is not None:
                console.console._width = width_override

            header = self._build_header()
            # Reserve lines for the header (text + spacing newline) and a
            # safety margin that absorbs height differences introduced when
            # consecutive markdown segments are coalesced into one Markdown
            # renderable (Rich adds inter-paragraph spacing that the
            # per-segment height estimates do not account for).
            max_allowed_height = max(
                1, console.console.size.height - _STREAM_HEADER_AND_MARGIN_LINES
            )
            window_segments, window_heights = self._viewport.slice_segments_with_heights(
                segments,
                terminal_height=max_allowed_height,
                console=console.console,
                target_ratio=self._stream_target_ratio,
            )
            if not window_segments:
                return
            is_truncated = len(window_segments) < len(segments) or (
                bool(window_segments) and bool(segments) and window_segments[0] is not segments[0]
            )
            now = time.monotonic()
            self._update_scroll_status(is_truncated=is_truncated, now=now)
            self._segment_assembler.compact(window_segments)

            renderables: list[RenderableType] = []
            content_height = sum(window_heights)
            self._update_pre_scroll_throttle(
                content_height=content_height,
                max_allowed_height=max_allowed_height,
            )
            width = console.console.size.width
            render_start = time.perf_counter()

            display_segments = self._coalesce_display_segments(window_segments)
            total_segments = len(display_segments)
            for segment_index, segment in enumerate(display_segments):
                cursor_suffix = self._cursor_suffix(
                    segment_index=segment_index,
                    total_segments=total_segments,
                )
                if segment.kind == "markdown":
                    renderables.append(
                        build_markdown_renderable(
                            segment.text,
                            code_theme=self._display.code_style,
                            escape_xml=self._display._escape_xml,
                            cursor_suffix=cursor_suffix,
                            close_incomplete_fences=True,
                            render_fences_with_syntax=self._display.render_fences_with_syntax,
                            code_word_wrap=self._display.code_word_wrap,
                        )
                    )
                elif segment.kind == "reasoning":
                    if self._render_reasoning_markdown:
                        prepared = prepare_markdown_content(segment.text, self._display._escape_xml)
                        prepared_for_display = close_incomplete_code_blocks(prepared)
                        if cursor_suffix:
                            prepared_for_display += cursor_suffix
                        markdown = Markdown(
                            prepared_for_display,
                            code_theme=self._display.code_style,
                            style="dim italic",
                        )
                        renderables.append(markdown)
                    else:
                        renderables.append(
                            Text(f"{segment.text}{cursor_suffix}", style="dim italic")
                        )
                else:
                    if segment.kind == "tool":
                        renderables.append(
                            self._render_tool_segment(segment, cursor_suffix=cursor_suffix)
                        )
                    else:
                        renderables.append(Text(f"{segment.text}{cursor_suffix}"))

            self._max_render_height = min(self._max_render_height, max_allowed_height)
            budget_height = min(content_height + self._height_fudge, max_allowed_height)

            if budget_height > self._max_render_height:
                self._max_render_height = budget_height

            padding_lines = (
                0
                if self._suppress_tail_padding_for_final_frame
                else max(0, self._max_render_height - content_height)
            )
            # Ensure content + padding cannot exceed the content budget so the
            # total frame (header + content + padding) stays within the terminal.
            if content_height + padding_lines > max_allowed_height:
                padding_lines = max(0, max_allowed_height - content_height)
            if padding_lines:
                # Text("\n" * n) renders n+1 lines, so subtract one for exact padding.
                renderables.append(Text("\n" * max(0, padding_lines - 1)))

            content = (
                Group(*renderables)
                if len(renderables) > 1
                else (renderables[0] if renderables else Text(""))
            )

            header_with_spacing = header.copy()
            header_with_spacing.append("\n", style="default")

            combined = Group(header_with_spacing, content)
            render_interval_ms = (
                (now - self._last_render_time) * 1000 if self._last_render_time else None
            )
            try:
                self._live.update(combined, refresh=True)
                self._last_render_time = time.monotonic()
            except Exception as exc:
                logger.warning(
                    "Error updating live display during streaming",
                    exc_info=True,
                    data={"error": str(exc)},
                )
            finally:
                if self._performance_hook:
                    render_ms = (time.perf_counter() - render_start) * 1000
                    batch_meta = self._pending_batch_meta or {}
                    oldest_enqueued = batch_meta.get("oldest_enqueued_at")
                    newest_enqueued = batch_meta.get("newest_enqueued_at")
                    queue_age_ms = (
                        (render_start - oldest_enqueued) * 1000
                        if isinstance(oldest_enqueued, (int, float))
                        else None
                    )
                    batch_span_ms = (
                        (newest_enqueued - oldest_enqueued) * 1000
                        if isinstance(oldest_enqueued, (int, float))
                        and isinstance(newest_enqueued, (int, float))
                        else None
                    )
                    scroll_age_ms = (
                        (now - self._scroll_start_time) * 1000
                        if self._scroll_start_time is not None
                        else None
                    )
                    try:
                        self._performance_hook(
                            {
                                "render_ms": render_ms,
                                "content_height": content_height,
                                "max_allowed_height": max_allowed_height,
                                "max_render_height": self._max_render_height,
                                "segment_count": len(segments),
                                "window_segment_count": len(window_segments),
                                "width": width,
                                "height": console.console.size.height,
                                "batch_size": batch_meta.get("batch_size"),
                                "queue_depth": batch_meta.get("queue_depth"),
                                "queue_age_ms": queue_age_ms,
                                "batch_span_ms": batch_span_ms,
                                "batch_window_ms": batch_meta.get("batch_window_ms"),
                                "batch_chars": batch_meta.get("batch_chars"),
                                "render_interval_ms": render_interval_ms,
                                "phase": "scrolling" if self._scrolling_started else "pre_scroll",
                                "is_truncated": is_truncated,
                                "scroll_age_ms": scroll_age_ms,
                            }
                        )
                    except Exception:
                        pass
                    self._pending_batch_meta = None
        finally:
            self._suppress_tail_padding_for_final_frame = False
            if width_override is not None:
                if width_attr_present:
                    console.console._width = original_width_attr
                elif hasattr(console.console, "_width"):
                    delattr(console.console, "_width")

    def _effective_stream_width(self) -> int | None:
        """Return an optional live-render width override for normal-screen markdown."""
        if self._use_plain_text or self._alt_screen_streaming:
            return None
        actual_width = console.console.size.width
        if actual_width <= 20:
            return None
        return max(20, actual_width - STREAM_RENDER_WIDTH_GUTTER)

    def _coalesce_display_segments(self, segments: list["StreamSegment"]) -> list["StreamSegment"]:
        if not segments:
            return []

        merged: list["StreamSegment"] = []
        for segment in segments:
            if merged and segment.kind == "markdown" and merged[-1].kind == "markdown":
                merged[-1] = merged[-1].copy_with_text(merged[-1].text + segment.text)
                continue
            merged.append(segment)
        return merged

    def _tool_header_text(self, segment: "StreamSegment") -> Text:
        header_text = self._tool_header_prefix.copy() if self._tool_header_prefix is not None else Text()
        tool_name = segment.tool_name or "tool"
        if tool_name:
            if header_text.plain:
                header_text.append(" ")
            header_text.append(tool_name, style=self._tool_header_color or "")
        return header_text

    def _render_tool_segment(
        self,
        segment: "StreamSegment",
        *,
        cursor_suffix: str,
    ) -> "RenderableType":
        preview = segment.code_preview
        if preview is not None and preview.code.strip():
            code_text = preview.code + cursor_suffix if cursor_suffix else preview.code
            return Group(
                self._tool_header_text(segment),
                Syntax(
                    code_text,
                    preview.language,
                    theme=self._display.code_style,
                    line_numbers=False,
                    word_wrap=self._display.code_word_wrap,
                ),
            )

        header_text = self._tool_header_text(segment)
        tool_text = header_text
        if segment.text:
            _, _, args_text = segment.text.partition("\n")
            if args_text:
                tool_text.append("\n")
                if segment.apply_patch_preview or "apply_patch preview:" in args_text:
                    tool_text.append_text(
                        style_apply_patch_preview_text(args_text, default_style="dim")
                    )
                else:
                    tool_text.append(args_text)
        if cursor_suffix:
            tool_text.append(cursor_suffix, style="dim")
        return tool_text

    async def _render_worker(self) -> None:
        assert self._queue is not None
        try:
            while True:
                try:
                    item = await self._queue.get()
                except asyncio.CancelledError:
                    break

                if item is self._stop_sentinel:
                    self._queue.task_done()
                    break

                stop_requested = False
                chunks = [item]
                batch_start = time.monotonic()
                while True:
                    try:
                        next_item = self._queue.get_nowait()
                    except asyncio.QueueEmpty:
                        elapsed = time.monotonic() - batch_start
                        if elapsed >= self._batch_max_duration:
                            break
                        try:
                            timeout = min(self._batch_period, self._batch_max_duration - elapsed)
                            next_item = await asyncio.wait_for(self._queue.get(), timeout=timeout)
                        except asyncio.TimeoutError:
                            break
                    if next_item is self._stop_sentinel:
                        stop_requested = True
                        chunks.append(next_item)
                        break
                    chunks.append(next_item)

                should_render = False
                queued_items: list[_QueuedItem] = []
                batch_chars = 0
                for chunk in chunks:
                    if chunk is self._stop_sentinel:
                        continue
                    payload = chunk
                    if isinstance(chunk, _QueuedItem):
                        queued_items.append(chunk)
                        payload = chunk.payload
                    if isinstance(payload, StreamChunk):
                        if payload.text:
                            batch_chars += len(payload.text)
                        should_render = self._handle_stream_chunk(payload) or should_render
                    elif isinstance(payload, str):
                        batch_chars += len(payload)
                        should_render = self._handle_chunk(payload) or should_render
                    elif isinstance(payload, _ToolStreamEvent):
                        should_render = (
                            self._segment_assembler.handle_tool_event(
                                payload.event_type, payload.info
                            )
                            or should_render
                        )

                if should_render:
                    if not await self._wait_for_render_slot():
                        for _ in chunks:
                            self._queue.task_done()
                        break
                    oldest_enqueued_at = None
                    newest_enqueued_at = None
                    if queued_items:
                        oldest_enqueued_at = min(item.enqueued_at for item in queued_items)
                        newest_enqueued_at = max(item.enqueued_at for item in queued_items)
                    self._pending_batch_meta = {
                        "batch_size": len(chunks),
                        "queue_depth": self._queue.qsize() if self._queue else 0,
                        "oldest_enqueued_at": oldest_enqueued_at,
                        "newest_enqueued_at": newest_enqueued_at,
                        "batch_window_ms": (time.monotonic() - batch_start) * 1000,
                        "batch_chars": batch_chars,
                    }
                    self._render_current_buffer()
                    self._advance_render_deadline()

                for _ in chunks:
                    self._queue.task_done()

                if stop_requested:
                    break
        except asyncio.CancelledError:
            pass
        finally:
            self._shutdown_live_resources()

    def _shutdown_live_resources(self) -> None:
        if self._live and self._live_started:
            if self._preserve_final_frame:
                try:
                    self._live.transient = False
                except Exception:
                    pass
            try:
                self._live.__exit__(None, None, None)
            except Exception:
                pass
            self._live = None
            self._live_started = False
        self._preserve_final_frame = False

        self._resume_progress_display()
        self._active = False

    def handle_tool_event(self, event_type: str, info: dict[str, Any] | None = None) -> None:
        try:
            if not self._active:
                return

            event = _ToolStreamEvent(event_type=event_type, info=info)
            if self._async_mode and self._queue is not None:
                self._enqueue_chunk(event)
                return

            if self._segment_assembler.handle_tool_event(event_type, info):
                now = time.perf_counter()
                self._pending_batch_meta = {
                    "batch_size": 1,
                    "queue_depth": 0,
                    "oldest_enqueued_at": now,
                    "newest_enqueued_at": now,
                    "batch_chars": 0,
                }
                self._render_sync_if_due()
        except Exception as exc:
            logger.warning(
                "Error handling tool event",
                exc_info=True,
                data={
                    "event_type": event_type,
                    "error": str(exc),
                },
            )


__all__ = [
    "NullStreamingHandle",
    "StreamingMessageHandle",
    "StreamingHandle",
    "MARKDOWN_STREAM_TARGET_RATIO",
    "MARKDOWN_STREAM_REFRESH_PER_SECOND",
    "MARKDOWN_STREAM_HEIGHT_FUDGE",
    "PLAIN_STREAM_TARGET_RATIO",
    "PLAIN_STREAM_REFRESH_PER_SECOND",
    "PLAIN_STREAM_HEIGHT_FUDGE",
]


class StreamingHandle(Protocol):
    def update(self, chunk: str) -> None: ...
    def update_chunk(self, chunk: StreamChunk) -> None: ...

    def finalize(self, message: "PromptMessageExtended | str") -> None: ...

    def close(self) -> None: ...

    def handle_tool_event(self, event_type: str, info: dict[str, Any] | None = None) -> None: ...

    def has_scrolled(self) -> bool: ...

    def preserve_final_frame(self) -> bool: ...

    async def wait_for_drain(self) -> None: ...
