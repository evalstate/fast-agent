from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

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
from fast_agent.ui.stream_viewport import StreamViewport, estimate_plain_text_height

if TYPE_CHECKING:
    from rich.console import RenderableType

    from fast_agent.mcp.prompt_message_extended import PromptMessageExtended
    from fast_agent.ui.console_display import ConsoleDisplay


logger = get_logger(__name__)

MARKDOWN_STREAM_TARGET_RATIO = 0.85
MARKDOWN_STREAM_REFRESH_PER_SECOND = 4
MARKDOWN_STREAM_HEIGHT_FUDGE = 2
PLAIN_STREAM_TARGET_RATIO = 0.92
PLAIN_STREAM_REFRESH_PER_SECOND = 20
PLAIN_STREAM_HEIGHT_FUDGE = 2


@dataclass(frozen=True)
class _ToolStreamEvent:
    event_type: str
    info: dict[str, Any] | None


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
        progress_display: Any = None,
    ) -> None:
        self._display = display
        self._bottom_items = bottom_items
        self._highlight_index = highlight_index
        self._max_item_length = max_item_length
        self._use_plain_text = use_plain_text
        self._header_left = header_left
        self._header_right = header_right
        self._progress_display = progress_display
        self._progress_paused = False
        self._plain_text_style: str | None = None
        base_kind = "plain" if use_plain_text else "markdown"
        self._render_reasoning_markdown = not use_plain_text
        self._segment_assembler = StreamSegmentAssembler(
            base_kind=base_kind,
            tool_prefix="→",
        )
        self._markdown_truncator = MarkdownTruncator(target_height_ratio=1.0)
        self._plain_truncator = PlainTextTruncator(target_height_ratio=1.0)
        self._viewport = StreamViewport(
            markdown_truncator=self._markdown_truncator,
            plain_truncator=self._plain_truncator,
        )
        self._stream_target_ratio = (
            PLAIN_STREAM_TARGET_RATIO if use_plain_text else MARKDOWN_STREAM_TARGET_RATIO
        )
        self._height_fudge = (
            PLAIN_STREAM_HEIGHT_FUDGE if use_plain_text else MARKDOWN_STREAM_HEIGHT_FUDGE
        )
        initial_renderable = (
            Text("", style=self._plain_text_style or "") if self._use_plain_text else Markdown("")
        )
        refresh_rate = (
            PLAIN_STREAM_REFRESH_PER_SECOND
            if self._use_plain_text
            else MARKDOWN_STREAM_REFRESH_PER_SECOND
        )
        self._min_render_interval = 1.0 / refresh_rate if refresh_rate else None
        self._last_render_time = 0.0
        try:
            self._loop: asyncio.AbstractEventLoop | None = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None
        self._async_mode = self._loop is not None
        self._queue: asyncio.Queue[object] | None = asyncio.Queue() if self._async_mode else None
        self._stop_sentinel: object = object()
        self._worker_task: asyncio.Task[None] | None = None
        self._live: Live | None = Live(
            initial_renderable,
            console=console.console,
            vertical_overflow="ellipsis",
            refresh_per_second=refresh_rate,
            transient=True,
        )
        self._live_started = False
        self._active = True
        self._finalized = False
        self._max_render_height = 0

        if self._async_mode and self._loop and self._queue is not None:
            self._worker_task = self._loop.create_task(self._render_worker())

    def update(self, chunk: str) -> None:
        if not self._active or not chunk:
            return

        if self._async_mode and self._queue is not None:
            self._enqueue_chunk(chunk)
            return

        if self._handle_chunk(chunk):
            self._render_current_buffer()

    def update_chunk(self, chunk: StreamChunk) -> None:
        """Structured streaming update with an explicit reasoning flag."""
        if not self._active or not chunk or not chunk.text:
            return

        if self._async_mode and self._queue is not None:
            self._enqueue_chunk(chunk)
            return

        if self._handle_stream_chunk(chunk):
            self._render_current_buffer()

    def _build_header(self) -> Text:
        width = console.console.size.width
        left_text = Text.from_markup(self._header_left)

        if self._header_right and self._header_right.strip():
            right_text = Text()
            right_text.append("[", style="dim")
            right_text.append_text(Text.from_markup(self._header_right))
            right_text.append("]", style="dim")
            separator_count = width - left_text.cell_len - right_text.cell_len
            if separator_count < 1:
                separator_count = 1
        else:
            right_text = Text("")
            separator_count = width - left_text.cell_len

        combined = Text()
        combined.append_text(left_text)
        combined.append(" ", style="default")
        combined.append("─" * (separator_count - 1), style="dim")
        combined.append_text(right_text)
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
        import re

        opening_fences = len(re.findall(r"^```", text, re.MULTILINE))
        closing_fences = len(re.findall(r"^```\s*$", text, re.MULTILINE))

        if opening_fences > closing_fences:
            if not re.search(r"```\s*$", text):
                return text + "\n```\n"

        return text

    def finalize(self, _message: "PromptMessageExtended | str") -> None:
        if not self._active or self._finalized:
            return

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

    def _enqueue_chunk(self, chunk: object) -> None:
        if not self._queue or not self._loop:
            return

        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        if current_loop is self._loop:
            try:
                self._queue.put_nowait(chunk)
            except asyncio.QueueFull:
                pass
        else:
            try:
                self._loop.call_soon_threadsafe(self._queue.put_nowait, chunk)
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

    def _render_current_buffer(self) -> None:
        segments = self._segment_assembler.segments
        if not segments:
            return

        self._ensure_started()

        if not self._live:
            return

        header = self._build_header()
        max_allowed_height = max(1, console.console.size.height - 2)
        window_segments = self._viewport.slice_segments(
            segments,
            terminal_height=max_allowed_height,
            console=console.console,
            target_ratio=self._stream_target_ratio,
        )
        if not window_segments:
            return
        self._segment_assembler.compact(window_segments)

        renderables: list[RenderableType] = []
        content_height = 0
        width = console.console.size.width

        for segment in window_segments:
            if segment.kind == "markdown":
                prepared = prepare_markdown_content(segment.text, self._display._escape_xml)
                prepared_for_display = self._close_incomplete_code_blocks(prepared)
                if prepared_for_display:
                    renderables.append(
                        Markdown(prepared_for_display, code_theme=self._display.code_style)
                    )
                    content_height += self._markdown_truncator.measure_rendered_height(
                        prepared_for_display,
                        console.console,
                        self._display.code_style,
                    )
                else:
                    renderables.append(Text(""))
            elif segment.kind == "reasoning":
                if self._render_reasoning_markdown:
                    prepared = prepare_markdown_content(segment.text, self._display._escape_xml)
                    prepared_for_display = self._close_incomplete_code_blocks(prepared)
                    markdown = Markdown(
                        prepared_for_display,
                        code_theme=self._display.code_style,
                        style="dim italic",
                    )
                    renderables.append(markdown)
                    content_height += self._markdown_truncator.measure_rendered_height(
                        prepared_for_display,
                        console.console,
                        self._display.code_style,
                    )
                else:
                    renderables.append(Text(segment.text, style="dim italic"))
                    content_height += estimate_plain_text_height(segment.text, width)
            else:
                renderables.append(Text(segment.text))
                content_height += estimate_plain_text_height(segment.text, width)

        self._max_render_height = min(self._max_render_height, max_allowed_height)
        budget_height = min(content_height + self._height_fudge, max_allowed_height)

        if budget_height > self._max_render_height:
            self._max_render_height = budget_height

        padding_lines = max(0, self._max_render_height - content_height)
        if padding_lines:
            renderables.append(Text("\n" * padding_lines))

        content = (
            Group(*renderables)
            if len(renderables) > 1
            else (renderables[0] if renderables else Text(""))
        )

        header_with_spacing = header.copy()
        header_with_spacing.append("\n", style="default")

        combined = Group(header_with_spacing, content)
        try:
            self._live.update(combined)
            self._last_render_time = time.monotonic()
        except Exception as exc:
            logger.warning(
                "Error updating live display during streaming",
                exc_info=True,
                data={"error": str(exc)},
            )

    async def _render_worker(self) -> None:
        assert self._queue is not None
        try:
            while True:
                try:
                    item = await self._queue.get()
                except asyncio.CancelledError:
                    break

                if item is self._stop_sentinel:
                    break

                stop_requested = False
                chunks = [item]
                while True:
                    try:
                        next_item = self._queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    if next_item is self._stop_sentinel:
                        stop_requested = True
                        break
                    chunks.append(next_item)

                should_render = False
                for chunk in chunks:
                    if isinstance(chunk, StreamChunk):
                        should_render = self._handle_stream_chunk(chunk) or should_render
                    elif isinstance(chunk, str):
                        should_render = self._handle_chunk(chunk) or should_render
                    elif isinstance(chunk, _ToolStreamEvent):
                        should_render = (
                            self._segment_assembler.handle_tool_event(chunk.event_type, chunk.info)
                            or should_render
                        )

                if should_render:
                    self._render_current_buffer()
                    if self._min_render_interval:
                        try:
                            await asyncio.sleep(self._min_render_interval)
                        except asyncio.CancelledError:
                            break

                if stop_requested:
                    break
        except asyncio.CancelledError:
            pass
        finally:
            self._shutdown_live_resources()

    def _shutdown_live_resources(self) -> None:
        if self._live and self._live_started:
            try:
                self._live.__exit__(None, None, None)
            except Exception:
                pass
            self._live = None
            self._live_started = False

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
                self._render_current_buffer()
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
