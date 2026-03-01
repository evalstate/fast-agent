from __future__ import annotations

import asyncio
import os
import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Protocol

from rich.console import Group
from rich.live import Live
from rich.markdown import Markdown
from rich.text import Text

from fast_agent.core.logging.logger import get_logger
from fast_agent.llm.stream_types import StreamChunk
from fast_agent.ui import console
from fast_agent.ui.markdown_helpers import prepare_markdown_content
from fast_agent.ui.markdown_truncator import MarkdownTruncator
from fast_agent.ui.plain_text_truncator import PlainTextTruncator
from fast_agent.ui.stream_segments import StreamSegmentAssembler
from fast_agent.ui.stream_viewport import StreamViewport

if TYPE_CHECKING:
    from rich.console import RenderableType

    from fast_agent.mcp.prompt_message_extended import PromptMessageExtended
    from fast_agent.ui.console_display import ConsoleDisplay


logger = get_logger(__name__)

MARKDOWN_STREAM_TARGET_RATIO = 0.93
MARKDOWN_STREAM_REFRESH_PER_SECOND = 4
MARKDOWN_STREAM_HEIGHT_FUDGE = 3
PLAIN_STREAM_TARGET_RATIO = 0.95
PLAIN_STREAM_REFRESH_PER_SECOND = 20
PLAIN_STREAM_HEIGHT_FUDGE = 3
STREAM_BATCH_PERIOD = 1 / 100
STREAM_BATCH_MAX_DURATION = 1 / 60
STREAM_CURSOR_BLOCK = "●"


def _resolve_progress_resume_debounce_seconds() -> float:
    """Return debounce duration for progress resume after streaming closes."""
    raw_value = os.getenv("FAST_AGENT_PROGRESS_RESUME_DEBOUNCE_SECONDS", "0.12").strip()
    try:
        parsed = float(raw_value)
    except ValueError:
        return 0.12
    return max(0.0, parsed)


STREAM_PROGRESS_RESUME_DEBOUNCE_SECONDS = _resolve_progress_resume_debounce_seconds()

_FENCE_OPEN_RE = re.compile(r"^```", re.MULTILINE)
_FENCE_CLOSE_RE = re.compile(r"^```\s*$", re.MULTILINE)
_FENCE_TRAILING_CLOSE_RE = re.compile(r"```\s*$")


@dataclass(frozen=True)
class _ToolStreamEvent:
    event_type: str
    info: dict[str, Any] | None


@dataclass(frozen=True)
class _QueuedItem:
    payload: object
    enqueued_at: float


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
        )
        self._markdown_truncator = MarkdownTruncator(target_height_ratio=1.0)
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
        self._display_truncated = False
        self._header_cache: dict[tuple[int, bool], Text] = {}
        self._next_render_deadline: float | None = None
        try:
            self._loop: asyncio.AbstractEventLoop | None = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None
        self._async_mode = self._loop is not None
        self._queue: asyncio.Queue[object] | None = asyncio.Queue() if self._async_mode else None
        self._stop_sentinel: object = object()
        self._worker_task: asyncio.Task[None] | None = None
        self._live: Live | None = Live(
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
        cache_key = (width, self._display_truncated)
        cached = self._header_cache.get(cache_key)
        if cached is not None:
            return cached

        right_content = self._header_right.strip()
        if self._display_truncated:
            indicator = "[black on blue]display truncated[/black on blue]"
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
        if "```" not in text:
            return text

        opening_fences = len(_FENCE_OPEN_RE.findall(text))
        closing_fences = len(_FENCE_CLOSE_RE.findall(text))

        if opening_fences > closing_fences:
            if not _FENCE_TRAILING_CLOSE_RE.search(text):
                return text + "\n```\n"

        return text

    def finalize(self, _message: "PromptMessageExtended | str") -> None:
        if not self._active or self._finalized:
            return

        # Remove the transient cursor in the final frame before closing Live rendering.
        self._show_stream_cursor = False

        if self._preserve_final_frame:
            # Avoid carrying the anti-flicker height padding into the persisted
            # final frame, otherwise short responses keep visible blank lines.
            self._max_render_height = 0
            self._height_fudge = 0

        # Flush any buffered reasoning content before closing the live view
        if self._segment_assembler.flush():
            self._render_current_buffer()
        elif self._segment_assembler.segments:
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

        Markdown renders are unthrottled until truncation begins (first scroll),
        then we switch to the configured cadence. Plain-text mode keeps its
        existing fixed cadence.
        """
        if self._use_plain_text:
            return True
        return self._scrolling_started

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

        header = self._build_header()
        max_allowed_height = max(1, console.console.size.height - 2)
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
        self._display_truncated = is_truncated
        if is_truncated and not self._scrolling_started:
            self._scrolling_started = True
            self._scroll_start_time = time.monotonic()
        self._segment_assembler.compact(window_segments)

        renderables: list[RenderableType] = []
        content_height = sum(window_heights)
        width = console.console.size.width
        render_start = time.perf_counter()

        total_segments = len(window_segments)
        for segment_index, segment in enumerate(window_segments):
            cursor_suffix = self._cursor_suffix(
                segment_index=segment_index,
                total_segments=total_segments,
            )
            if segment.kind == "markdown":
                prepared = prepare_markdown_content(segment.text, self._display._escape_xml)
                prepared_for_display = self._close_incomplete_code_blocks(prepared)
                if cursor_suffix:
                    prepared_for_display += cursor_suffix
                if prepared_for_display:
                    renderables.append(
                        Markdown(prepared_for_display, code_theme=self._display.code_style)
                    )
                else:
                    renderables.append(Text(""))
            elif segment.kind == "reasoning":
                if self._render_reasoning_markdown:
                    prepared = prepare_markdown_content(segment.text, self._display._escape_xml)
                    prepared_for_display = self._close_incomplete_code_blocks(prepared)
                    if cursor_suffix:
                        prepared_for_display += cursor_suffix
                    markdown = Markdown(
                        prepared_for_display,
                        code_theme=self._display.code_style,
                        style="dim italic",
                    )
                    renderables.append(markdown)
                else:
                    renderables.append(Text(f"{segment.text}{cursor_suffix}", style="dim italic"))
            else:
                if segment.kind == "tool":
                    header_text = (
                        self._tool_header_prefix.copy()
                        if self._tool_header_prefix is not None
                        else Text()
                    )
                    tool_name = segment.tool_name or "tool"
                    if tool_name:
                        if header_text.plain:
                            header_text.append(" ")
                        header_text.append(tool_name, style=self._tool_header_color or "")

                    tool_text = header_text
                    if segment.text:
                        _, _, args_text = segment.text.partition("\n")
                        if args_text:
                            tool_text.append("\n")
                            tool_text.append(args_text)
                    if cursor_suffix:
                        tool_text.append(cursor_suffix, style="dim")
                    renderables.append(tool_text)
                else:
                    renderables.append(Text(f"{segment.text}{cursor_suffix}"))

        self._max_render_height = min(self._max_render_height, max_allowed_height)
        budget_height = min(content_height + self._height_fudge, max_allowed_height)

        if budget_height > self._max_render_height:
            self._max_render_height = budget_height

        padding_lines = max(0, self._max_render_height - content_height)
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
        now = time.monotonic()
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
