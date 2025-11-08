from __future__ import annotations

import asyncio
import math
import time
from typing import TYPE_CHECKING, Any, Protocol

from rich.console import Group
from rich.live import Live
from rich.markdown import Markdown
from rich.text import Text

from fast_agent.core.logging.logger import get_logger
from fast_agent.ui import console
from fast_agent.ui.markdown_helpers import prepare_markdown_content
from fast_agent.ui.markdown_truncator import MarkdownTruncator
from fast_agent.ui.plain_text_truncator import PlainTextTruncator

if TYPE_CHECKING:
    from fast_agent.mcp.prompt_message_extended import PromptMessageExtended
    from fast_agent.ui.console_display import ConsoleDisplay


logger = get_logger(__name__)

MARKDOWN_STREAM_TARGET_RATIO = 0.75
MARKDOWN_STREAM_REFRESH_PER_SECOND = 4
MARKDOWN_STREAM_HEIGHT_FUDGE = 1
PLAIN_STREAM_TARGET_RATIO = 0.9
PLAIN_STREAM_REFRESH_PER_SECOND = 20
PLAIN_STREAM_HEIGHT_FUDGE = 1


class NullStreamingHandle:
    """No-op streaming handle used when streaming is disabled."""

    def update(self, _chunk: str) -> None:
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
        self._buffer: list[str] = []
        self._plain_text_style: str | None = None
        self._convert_literal_newlines = False
        self._pending_literal_backslashes = ""
        initial_renderable = (
            Text("", style=self._plain_text_style) if self._use_plain_text else Markdown("")
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
        self._in_table = False
        self._pending_table_row = ""
        self._truncator = MarkdownTruncator(target_height_ratio=MARKDOWN_STREAM_TARGET_RATIO)
        self._plain_truncator = (
            PlainTextTruncator(target_height_ratio=PLAIN_STREAM_TARGET_RATIO)
            if self._use_plain_text
            else None
        )
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

    def _trim_to_displayable(self, text: str) -> str:
        if not text:
            return text

        terminal_height = console.console.size.height - 1

        if self._use_plain_text and self._plain_truncator:
            terminal_width = console.console.size.width
            return self._plain_truncator.truncate(
                text,
                terminal_height=terminal_height,
                terminal_width=terminal_width,
            )

        return self._truncator.truncate(
            text,
            terminal_height=terminal_height,
            console=console.console,
            code_theme=self._display.code_style,
            prefer_recent=True,
        )

    def _switch_to_plain_text(self) -> None:
        if not self._use_plain_text:
            self._use_plain_text = True
        if not self._plain_truncator:
            self._plain_truncator = PlainTextTruncator(
                target_height_ratio=PLAIN_STREAM_TARGET_RATIO
            )
        self._plain_text_style = "dim"
        self._convert_literal_newlines = True

    def finalize(self, _message: "PromptMessageExtended | str") -> None:
        if not self._active or self._finalized:
            return

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

    def _extract_trailing_paragraph(self, text: str) -> str:
        if not text:
            return ""
        double_break = text.rfind("\n\n")
        if double_break != -1:
            candidate = text[double_break + 2 :]
        else:
            candidate = text
        if "\n" in candidate:
            candidate = candidate.split("\n")[-1]
        return candidate

    def _wrap_plain_chunk(self, chunk: str) -> str:
        width = max(1, console.console.size.width)
        if not chunk or width <= 1:
            return chunk

        result_segments: list[str] = []
        start = 0
        length = len(chunk)

        while start < length:
            newline_pos = chunk.find("\n", start)
            if newline_pos == -1:
                line = chunk[start:]
                delimiter = ""
                start = length
            else:
                line = chunk[start:newline_pos]
                delimiter = "\n"
                start = newline_pos + 1

            if len(line.expandtabs()) > width:
                wrapped = self._wrap_plain_line(line, width)
                result_segments.append("\n".join(wrapped))
            else:
                result_segments.append(line)

            result_segments.append(delimiter)

        return "".join(result_segments)

    @staticmethod
    def _wrap_plain_line(line: str, width: int) -> list[str]:
        if not line:
            return [""]

        segments: list[str] = []
        remaining = line

        while len(remaining) > width:
            break_at = remaining.rfind(" ", 0, width)
            if break_at == -1 or break_at < width // 2:
                break_at = width
                segments.append(remaining[:break_at])
                remaining = remaining[break_at:]
            else:
                segments.append(remaining[:break_at])
                remaining = remaining[break_at + 1 :]
        segments.append(remaining)
        return segments

    def _decode_literal_newlines(self, chunk: str) -> str:
        if not chunk:
            return chunk

        text = chunk
        if self._pending_literal_backslashes:
            text = self._pending_literal_backslashes + text
            self._pending_literal_backslashes = ""

        result: list[str] = []
        length = len(text)
        index = 0

        while index < length:
            char = text[index]
            if char == "\\":
                start = index
                while index < length and text[index] == "\\":
                    index += 1
                count = index - start

                if index >= length:
                    self._pending_literal_backslashes = "\\" * count
                    break

                next_char = text[index]
                if next_char == "n" and count % 2 == 1:
                    if count > 1:
                        result.append("\\" * (count - 1))
                    result.append("\n")
                    index += 1
                else:
                    result.append("\\" * count)
                    continue
            else:
                result.append(char)
                index += 1

        return "".join(result)

    def _estimate_plain_render_height(self, text: str) -> int:
        if not text:
            return 0

        width = max(1, console.console.size.width)
        lines = text.split("\n")
        total = 0
        for line in lines:
            expanded_len = len(line.expandtabs())
            total += max(1, math.ceil(expanded_len / width)) if expanded_len else 1
        return total

    def _enqueue_chunk(self, chunk: str) -> None:
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
                    data={"error": str(exc), "chunk_length": len(chunk)},
                )
            except Exception as exc:
                logger.warning(
                    "Unexpected error while enqueuing chunk",
                    exc_info=True,
                    data={"error": str(exc), "chunk_length": len(chunk)},
                )

    def _handle_chunk(self, chunk: str) -> bool:
        if not chunk:
            return False

        if self._use_plain_text:
            if self._convert_literal_newlines:
                chunk = self._decode_literal_newlines(chunk)
                if not chunk:
                    if self._pending_table_row:
                        self._buffer.append(self._pending_table_row)
                        self._pending_table_row = ""
                    return False
            chunk = self._wrap_plain_chunk(chunk)
            if self._pending_table_row:
                self._buffer.append(self._pending_table_row)
                self._pending_table_row = ""
        else:
            text_so_far = "".join(self._buffer)
            lines = text_so_far.strip().split("\n")
            last_line = lines[-1] if lines else ""
            currently_in_table = last_line.strip().startswith("|")

            if currently_in_table and "\n" not in chunk:
                self._pending_table_row += chunk
                return False

            if self._pending_table_row:
                self._buffer.append(self._pending_table_row)
                self._pending_table_row = ""

        self._buffer.append(chunk)
        return True

    def _render_current_buffer(self) -> None:
        if not self._buffer:
            return

        self._ensure_started()

        if not self._live:
            return

        # Consolidate buffer if it gets fragmented (>10 items)
        # Then check if we need to truncate to keep only recent content
        if len(self._buffer) > 10:
            text = "".join(self._buffer)
            trimmed = self._trim_to_displayable(text)
            # Only update buffer if truncation actually reduced content
            # This keeps buffer size manageable for continuous scrolling
            if len(trimmed) < len(text):
                self._buffer = [trimmed]
            else:
                self._buffer = [text]

        text = "".join(self._buffer)

        # Check if trailing paragraph is too long and needs trimming
        trailing_paragraph = self._extract_trailing_paragraph(text)
        if trailing_paragraph and "\n" not in trailing_paragraph:
            width = max(1, console.console.size.width)
            target_ratio = (
                PLAIN_STREAM_TARGET_RATIO if self._use_plain_text else MARKDOWN_STREAM_TARGET_RATIO
            )
            target_rows = max(1, int(console.console.size.height * target_ratio) - 1)
            estimated_rows = math.ceil(len(trailing_paragraph.expandtabs()) / width)
            if estimated_rows > target_rows:
                trimmed = self._trim_to_displayable(text)
                if len(trimmed) < len(text):
                    text = trimmed
                    self._buffer = [trimmed]

        header = self._build_header()
        max_allowed_height = max(1, console.console.size.height - 2)
        self._max_render_height = min(self._max_render_height, max_allowed_height)

        if self._use_plain_text:
            content_height = self._estimate_plain_render_height(text)
            budget_height = min(content_height + PLAIN_STREAM_HEIGHT_FUDGE, max_allowed_height)

            if budget_height > self._max_render_height:
                self._max_render_height = budget_height

            padding_lines = max(0, self._max_render_height - content_height)
            display_text = text + ("\n" * padding_lines if padding_lines else "")
            content = (
                Text(display_text, style=self._plain_text_style)
                if self._plain_text_style
                else Text(display_text)
            )
        else:
            prepared = prepare_markdown_content(text, self._display._escape_xml)
            prepared_for_display = self._close_incomplete_code_blocks(prepared)

            content_height = self._truncator.measure_rendered_height(
                prepared_for_display, console.console, self._display.code_style
            )
            budget_height = min(content_height + MARKDOWN_STREAM_HEIGHT_FUDGE, max_allowed_height)

            if budget_height > self._max_render_height:
                self._max_render_height = budget_height

            padding_lines = max(0, self._max_render_height - content_height)
            if padding_lines:
                prepared_for_display = prepared_for_display + ("\n" * padding_lines)

            content = Markdown(prepared_for_display, code_theme=self._display.code_style)

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
                    if isinstance(chunk, str):
                        should_render = self._handle_chunk(chunk) or should_render

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

            streams_arguments = info.get("streams_arguments", False) if info else False
            tool_name = info.get("tool_name", "unknown") if info else "unknown"

            if event_type == "start":
                if streams_arguments:
                    self._switch_to_plain_text()
                    self.update(f"\n→ Calling {tool_name}\n")
                else:
                    self._pause_progress_display()
                    self._switch_to_plain_text()
                    self.update(f"\n→ Calling {tool_name}\n")
                return
            if event_type == "delta":
                if streams_arguments and info and "chunk" in info:
                    self.update(info["chunk"])
            elif event_type == "text":
                self._pause_progress_display()
            elif event_type == "stop":
                if streams_arguments:
                    self.update("\n")
                    self.close()
                else:
                    self.update("\n")
                    self.close()
        except Exception as exc:
            logger.warning(
                "Error handling tool event",
                exc_info=True,
                data={
                    "event_type": event_type,
                    "streams_arguments": info.get("streams_arguments") if info else None,
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

    def finalize(self, message: "PromptMessageExtended | str") -> None: ...

    def close(self) -> None: ...

    def handle_tool_event(self, event_type: str, info: dict[str, Any] | None = None) -> None: ...
