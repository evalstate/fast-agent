"""Segmented streaming buffer for assistant output and tool events."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from fast_agent.utils.reasoning_stream_parser import ReasoningSegment, ReasoningStreamParser

if TYPE_CHECKING:
    from fast_agent.llm.stream_types import StreamChunk

SegmentKind = Literal["markdown", "plain", "reasoning", "tool"]


@dataclass
class StreamSegment:
    """A contiguous chunk of streamed content with a single rendering mode."""

    kind: SegmentKind
    text: str
    tool_name: str | None = None
    tool_use_id: str | None = None

    def append(self, text: str) -> None:
        self.text += text

    def copy_with_text(self, text: str) -> "StreamSegment":
        return StreamSegment(
            kind=self.kind,
            text=text,
            tool_name=self.tool_name,
            tool_use_id=self.tool_use_id,
        )


class LiteralNewlineDecoder:
    """Convert escaped newline sequences while preserving trailing backslashes."""

    def __init__(self) -> None:
        self._pending_backslashes = ""

    def decode(self, chunk: str) -> str:
        if not chunk:
            return chunk

        text = chunk
        if self._pending_backslashes:
            text = self._pending_backslashes + text
            self._pending_backslashes = ""

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
                    self._pending_backslashes = "\\" * count
                    break

                next_char = text[index]
                if next_char == "n" and count % 2 == 1:
                    if count > 1:
                        result.append("\\" * (count - 1))
                    result.append("\n")
                    index += 1
                else:
                    result.append("\\" * count)
            else:
                result.append(char)
                index += 1

        return "".join(result)


class StreamSegmentBuffer:
    """Collect streaming content while keeping markdown/table boundaries intact."""

    def __init__(self, base_kind: SegmentKind) -> None:
        if base_kind not in ("markdown", "plain"):
            raise ValueError("base_kind must be 'markdown' or 'plain'")
        self._base_kind = base_kind
        self._segments: list[StreamSegment] = []
        self._pending_table_row = ""
        self._reasoning_separator_pending = False
        self._plain_decoder = LiteralNewlineDecoder()
        self._reasoning_decoder = LiteralNewlineDecoder()

    @property
    def segments(self) -> list[StreamSegment]:
        return self._segments

    @property
    def pending_table_row(self) -> str:
        return self._pending_table_row

    def mark_reasoning_boundary(self) -> None:
        self._reasoning_separator_pending = True

    def ensure_separator(self) -> None:
        """Insert a newline before switching into a plain segment if needed."""
        if self._pending_table_row:
            return
        if not self._segments:
            return
        if self._segments[-1].text.endswith("\n"):
            return
        self._append_to_segment(self._base_kind, "\n")

    def append_content(self, text: str) -> bool:
        if self._base_kind == "plain":
            return self._append_plain(text, kind="plain", decoder=self._plain_decoder)
        return self._append_markdown(text)

    def append_reasoning(self, text: str) -> bool:
        return self._append_plain(text, kind="reasoning", decoder=self._reasoning_decoder)

    def append_segment(self, segment: StreamSegment) -> None:
        self._segments.append(segment)

    def consume_reasoning_gap(self) -> None:
        gap = self._consume_reasoning_gap()
        if gap:
            self._append_to_segment("plain", gap)

    def _append_plain(
        self,
        text: str,
        *,
        kind: SegmentKind,
        decoder: LiteralNewlineDecoder,
    ) -> bool:
        if not text:
            return False
        processed = decoder.decode(text)
        if not processed:
            return False
        if kind != "reasoning":
            self.consume_reasoning_gap()
        self._append_to_segment(kind, processed)
        return True

    def _append_markdown(self, text: str) -> bool:
        if not text:
            return False
        self.consume_reasoning_gap()

        if self._pending_table_row:
            if "\n" not in text:
                self._pending_table_row += text
                return False
            text = self._pending_table_row + text
            self._pending_table_row = ""

        last_segment = self._last_segment(kind="markdown")
        text_so_far = last_segment.text if last_segment else ""
        ends_with_newline = text_so_far.endswith("\n")
        last_line = "" if ends_with_newline else (text_so_far.split("\n")[-1] if text_so_far else "")
        currently_in_table = bool(last_segment) and last_line.strip().startswith("|")
        starts_table_row = text.lstrip().startswith("|")

        if "\n" not in text and (currently_in_table or starts_table_row):
            pending_seed = ""
            if currently_in_table and last_segment:
                split_index = text_so_far.rfind("\n")
                if split_index == -1:
                    pending_seed = text_so_far
                    last_segment.text = ""
                else:
                    pending_seed = text_so_far[split_index + 1 :]
                    last_segment.text = text_so_far[: split_index + 1]
                if last_segment.text == "":
                    self._segments.pop()
            self._pending_table_row = pending_seed + text
            return False

        if self._pending_table_row:
            self._append_to_segment("markdown", self._pending_table_row)
            self._pending_table_row = ""

        self._append_to_segment("markdown", text)
        return True

    def _consume_reasoning_gap(self) -> str:
        if not self._reasoning_separator_pending:
            return ""
        if self._pending_table_row:
            self._reasoning_separator_pending = False
            return ""
        if not self._segments:
            self._reasoning_separator_pending = False
            return ""

        last_text = self._segments[-1].text
        if not last_text:
            self._reasoning_separator_pending = False
            return ""

        last_line = last_text.split("\n")[-1]
        if last_line.strip().startswith("|"):
            self._reasoning_separator_pending = False
            return ""

        if last_text.endswith("\n\n"):
            gap = ""
        elif last_text.endswith("\n"):
            gap = "\n"
        else:
            gap = "\n\n"

        self._reasoning_separator_pending = False
        return gap

    def _append_to_segment(self, kind: SegmentKind, text: str) -> None:
        if not text:
            return
        last_segment = self._last_segment(kind=kind)
        if last_segment is not None:
            last_segment.append(text)
        else:
            self._segments.append(StreamSegment(kind=kind, text=text))

    def _last_segment(self, *, kind: SegmentKind) -> StreamSegment | None:
        if not self._segments:
            return None
        last_segment = self._segments[-1]
        if last_segment.kind != kind:
            return None
        return last_segment


@dataclass
class ToolStreamState:
    tool_use_id: str
    tool_name: str
    segment_index: int
    raw_text: str = ""
    display_text: str = ""
    completed: bool = False
    decoder: LiteralNewlineDecoder = field(default_factory=LiteralNewlineDecoder)

    def append(self, chunk: str) -> None:
        if not chunk:
            return
        self.raw_text += chunk
        self.display_text += self.decoder.decode(chunk)

    def render_text(self, *, prefix: str, pretty: bool) -> str:
        header = f"{prefix} Calling {self.tool_name}\n"
        if not self.display_text and not pretty:
            return header

        args_text = self.display_text
        if pretty and self.raw_text.strip():
            formatted = _format_json(self.raw_text)
            if formatted is not None:
                args_text = formatted

        if pretty and not args_text:
            return header + "\n"
        if args_text and pretty and not args_text.endswith("\n"):
            args_text += "\n"
        return header + (args_text or "")


def _format_json(raw_text: str) -> str | None:
    if not raw_text:
        return None
    try:
        parsed = json.loads(raw_text)
    except Exception:
        return None
    return json.dumps(parsed, indent=2, ensure_ascii=True)


class StreamSegmentAssembler:
    """Route streamed chunks into markdown/reasoning/tool segments."""

    def __init__(self, *, base_kind: SegmentKind, tool_prefix: str) -> None:
        self._buffer = StreamSegmentBuffer(base_kind)
        self._reasoning_parser = ReasoningStreamParser()
        self._reasoning_active = False
        self._tool_prefix = tool_prefix
        self._tool_states: dict[str, ToolStreamState] = {}
        self._fallback_tool_counter = 0
        self._last_tool_id: str | None = None

    @property
    def segments(self) -> list[StreamSegment]:
        return self._buffer.segments

    @property
    def pending_table_row(self) -> str:
        return self._buffer.pending_table_row

    def handle_stream_chunk(self, chunk: StreamChunk) -> bool:
        if not chunk.text:
            return False

        if not chunk.is_reasoning and self._process_reasoning_tags(chunk.text):
            return True

        if chunk.is_reasoning:
            if not self._reasoning_active:
                self._buffer.ensure_separator()
                self._reasoning_active = True
            return self._buffer.append_reasoning(chunk.text)

        if self._reasoning_active:
            self._reasoning_active = False
            self._buffer.mark_reasoning_boundary()

        return self._buffer.append_content(chunk.text)

    def handle_text(self, chunk: str) -> bool:
        if not chunk:
            return False
        if self._process_reasoning_tags(chunk):
            return True
        if self._reasoning_active:
            self._reasoning_active = False
            self._buffer.mark_reasoning_boundary()
        return self._buffer.append_content(chunk)

    def flush(self) -> bool:
        if not self._reasoning_parser.in_think:
            return False
        segments = self._reasoning_parser.flush()
        return self._handle_reasoning_segments(segments)

    def handle_tool_event(self, event_type: str, info: dict[str, Any] | None) -> bool:
        tool_name = str(info.get("tool_name") or "tool") if info else "tool"
        tool_use_id = str(info.get("tool_use_id")) if info and info.get("tool_use_id") else ""

        if not tool_use_id:
            if event_type == "start":
                tool_use_id = self._fallback_tool_id()
            else:
                tool_use_id = self._last_tool_id or self._fallback_tool_id()
        self._last_tool_id = tool_use_id

        state = self._tool_states.get(tool_use_id)
        if state is not None and tool_name and state.tool_name != tool_name:
            state.tool_name = tool_name

        if event_type == "start":
            if state is None:
                state = self._start_tool(tool_use_id, tool_name)
            state.completed = False
            self._update_tool_segment(state, pretty=False)
            return True

        if event_type == "delta":
            if state is None:
                state = self._start_tool(tool_use_id, tool_name)
            chunk = str(info.get("chunk") or "") if info else ""
            state.append(chunk)
            self._update_tool_segment(state, pretty=False)
            return True

        if event_type == "stop":
            if state is None:
                state = self._start_tool(tool_use_id, tool_name)
            state.completed = True
            self._update_tool_segment(state, pretty=True)
            self._tool_states.pop(tool_use_id, None)
            if self._last_tool_id == tool_use_id:
                self._last_tool_id = None
            return True

        return False

    def compact(self, window_segments: list[StreamSegment]) -> None:
        if not window_segments or self._tool_states:
            return
        segments = self._buffer.segments
        if not segments:
            return
        filtered = [(idx, segment) for idx, segment in enumerate(segments) if segment.text]
        if not filtered:
            return
        last_window = window_segments[-1]
        last_pos = next(
            (pos for pos, (_, segment) in enumerate(filtered) if segment is last_window),
            None,
        )
        if last_pos is None:
            last_pos = len(filtered) - 1
            last_index = filtered[last_pos][0]
            last_segment = segments[last_index]
            if (
                last_segment.kind != last_window.kind
                or last_segment.tool_use_id != last_window.tool_use_id
                or not last_segment.text.endswith(last_window.text)
            ):
                return
        start_pos = last_pos - (len(window_segments) - 1)
        if start_pos < 0:
            return
        if start_pos >= len(filtered):
            return
        start_index = filtered[start_pos][0]
        first_window = window_segments[0]
        original_first = segments[start_index]
        if first_window is not original_first:
            original_first.text = first_window.text
        if start_index > 0:
            del segments[:start_index]

    def _start_tool(self, tool_use_id: str, tool_name: str) -> ToolStreamState:
        self._buffer.consume_reasoning_gap()
        self._buffer.ensure_separator()
        segment = StreamSegment(kind="tool", text="", tool_name=tool_name, tool_use_id=tool_use_id)
        self._buffer.append_segment(segment)
        state = ToolStreamState(
            tool_use_id=tool_use_id,
            tool_name=tool_name,
            segment_index=len(self._buffer.segments) - 1,
        )
        self._tool_states[tool_use_id] = state
        return state

    def _update_tool_segment(self, state: ToolStreamState, *, pretty: bool) -> None:
        segment = self._buffer.segments[state.segment_index]
        segment.text = state.render_text(prefix=self._tool_prefix, pretty=pretty)

    def _fallback_tool_id(self) -> str:
        self._fallback_tool_counter += 1
        return f"tool-{self._fallback_tool_counter}"

    def _process_reasoning_tags(self, chunk: str) -> bool:
        should_process = (
            self._reasoning_parser.in_think or "<think>" in chunk or "</think>" in chunk
        )
        if not should_process:
            return False
        segments = self._reasoning_parser.feed(chunk)
        return self._handle_reasoning_segments(segments)

    def _handle_reasoning_segments(self, segments: list[ReasoningSegment]) -> bool:
        if not segments:
            return False
        handled = False
        emitted_non_reasoning = False

        for segment in segments:
            if segment.is_thinking:
                if not self._reasoning_active:
                    self._buffer.ensure_separator()
                    self._reasoning_active = True
                handled = self._buffer.append_reasoning(segment.text) or handled
            else:
                if self._reasoning_active:
                    self._reasoning_active = False
                    self._buffer.mark_reasoning_boundary()
                emitted_non_reasoning = True
                handled = self._buffer.append_content(segment.text) or handled

        if (
            self._reasoning_active
            and not self._reasoning_parser.in_think
            and not emitted_non_reasoning
        ):
            self._reasoning_active = False
            self._buffer.mark_reasoning_boundary()

        return handled


__all__ = [
    "SegmentKind",
    "StreamSegment",
    "StreamSegmentAssembler",
    "StreamSegmentBuffer",
]
