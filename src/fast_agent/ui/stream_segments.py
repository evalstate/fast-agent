"""Segmented streaming buffer for assistant output and tool events."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from fast_agent.tools.apply_patch_tool import is_apply_patch_tool_name
from fast_agent.ui.apply_patch_preview import (
    build_apply_patch_preview,
    build_apply_patch_preview_from_input,
    build_partial_apply_patch_preview,
    extract_non_command_args,
    format_apply_patch_preview,
    format_partial_apply_patch_preview,
    is_shell_execution_tool,
    shell_syntax_language,
)
from fast_agent.utils.reasoning_stream_parser import ReasoningSegment, ReasoningStreamParser

if TYPE_CHECKING:
    from fast_agent.llm.stream_types import StreamChunk

SegmentKind = Literal["markdown", "plain", "reasoning", "tool"]
_JSON_PARSE_FAILED = object()
_FENCE_OPEN_LINE_RE = re.compile(r"^\s{0,3}(?P<delim>`{3,}|~{3,})(?P<info>.*)$")
_CONTAINER_BLOCK_LINE_RE = re.compile(r"^\s{0,3}(?:>|\d+[.)][ \t]+|[*+-][ \t]+)")


@dataclass
class StreamSegment:
    """A contiguous chunk of streamed content with a single rendering mode."""

    kind: SegmentKind
    text: str
    tool_name: str | None = None
    tool_use_id: str | None = None
    frozen: bool = False
    code_preview: "ToolCodePreview | None" = None
    apply_patch_preview: bool = False

    def append(self, text: str) -> None:
        self.text += text

    def copy_with_text(self, text: str) -> "StreamSegment":
        return StreamSegment(
            kind=self.kind,
            text=text,
            tool_name=self.tool_name,
            tool_use_id=self.tool_use_id,
            frozen=self.frozen,
            code_preview=self.code_preview,
            apply_patch_preview=self.apply_patch_preview,
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
            target_kind: SegmentKind = "markdown" if self._base_kind == "markdown" else "plain"
            self._append_to_segment(target_kind, gap)

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
        self._freeze_completed_markdown_tail()
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
        if last_segment.kind != kind or last_segment.frozen:
            return None
        return last_segment

    def _freeze_completed_markdown_tail(self) -> None:
        segment = self._last_segment(kind="markdown")
        if segment is None or not segment.text:
            return
        # Freeze only stable block prefixes so cached measurement/rendering can
        # reuse earlier markdown while the active tail keeps growing.
        split_at = self._stable_markdown_prefix_length(segment.text)
        if split_at <= 0:
            return

        frozen_text = segment.text[:split_at]
        if not frozen_text.strip():
            return
        tail_text = segment.text[split_at:]
        frozen_segment = StreamSegment(kind="markdown", text=frozen_text, frozen=True)
        last_index = len(self._segments) - 1
        if tail_text:
            segment.text = tail_text
            self._segments.insert(last_index, frozen_segment)
            return
        self._segments[last_index] = frozen_segment

    def _stable_markdown_prefix_length(self, text: str) -> int:
        if not text:
            return 0

        boundary = 0
        current_block_safe: bool | None = None
        in_fence = False
        fence_char = ""
        fence_len = 0
        offset = 0

        for raw_line in text.splitlines(keepends=True):
            line = raw_line[:-1] if raw_line.endswith("\n") else raw_line
            stripped = line.lstrip(" ")

            if in_fence:
                if stripped and stripped[0] == fence_char:
                    index = 0
                    while index < len(stripped) and stripped[index] == fence_char:
                        index += 1
                    if index >= fence_len and stripped[index:].strip() == "":
                        in_fence = False
                        if current_block_safe and raw_line.endswith("\n"):
                            boundary = offset + len(raw_line)
                        # A closed fence is a complete block; the next line starts fresh.
                        current_block_safe = None
                offset += len(raw_line)
                continue

            if not line.strip():
                if current_block_safe:
                    boundary = offset + len(raw_line)
                current_block_safe = None
                offset += len(raw_line)
                continue

            line_safe = self._is_freeze_safe_markdown_line(line)
            if current_block_safe is None:
                current_block_safe = line_safe
            else:
                current_block_safe = current_block_safe and line_safe

            opening = _FENCE_OPEN_LINE_RE.match(line)
            if opening:
                delimiter = opening.group("delim")
                info = opening.group("info")
                if delimiter[0] != "`" or "`" not in info:
                    in_fence = True
                    fence_char = delimiter[0]
                    fence_len = len(delimiter)

            offset += len(raw_line)

        return boundary

    def _is_freeze_safe_markdown_line(self, line: str) -> bool:
        stripped = line.lstrip(" ")
        indent = len(line) - len(stripped)
        if indent != 0:
            return False
        return _CONTAINER_BLOCK_LINE_RE.match(line) is None


@dataclass
class ToolStreamState:
    tool_use_id: str
    tool_name: str
    segment_index: int | None
    tool_metadata: Mapping[str, Any] | None = None
    apply_patch_preview_max_lines: int | None = None
    raw_text: str = ""
    display_text: str = ""
    completed: bool = False
    decoder: LiteralNewlineDecoder = field(default_factory=LiteralNewlineDecoder)

    def append(self, chunk: str) -> None:
        if not chunk:
            return
        self.raw_text += chunk
        self.display_text += self.decoder.decode(chunk)

    def code_preview(self) -> "ToolCodePreview | None":
        preview_spec = _tool_code_preview_spec(self.tool_metadata)
        if preview_spec is None:
            return None
        field_name, language = preview_spec
        extracted = extract_partial_json_string_field(self.raw_text, field_name=field_name)
        if extracted is None or not extracted.value:
            return None
        if field_name == "command" and build_partial_apply_patch_preview(extracted.value) is not None:
            return None
        return ToolCodePreview(
            code=extracted.value,
            language=language,
            complete=extracted.complete,
        )

    def has_apply_patch_preview(self) -> bool:
        tool_name = self.tool_name or "tool"
        stripped_text = self.raw_text.strip()
        if not stripped_text:
            return False

        if is_apply_patch_tool_name(tool_name):
            return build_apply_patch_preview_from_input(
                stripped_text,
                max_lines=self.apply_patch_preview_max_lines,
            ) is not None or (
                stripped_text.lstrip().startswith("*** Begin Patch")
            )

        if not is_shell_execution_tool(tool_name):
            return False

        parsed_args = _parse_json_value(self.raw_text)
        if parsed_args is not _JSON_PARSE_FAILED:
            if not isinstance(parsed_args, dict):
                return False
            command = parsed_args.get("command")
            return isinstance(command, str) and (
                build_apply_patch_preview(
                    command,
                    max_lines=self.apply_patch_preview_max_lines,
                )
                is not None
                or build_partial_apply_patch_preview(
                    command,
                    max_lines=self.apply_patch_preview_max_lines,
                )
                is not None
            )

        extracted = extract_partial_json_string_field(self.raw_text, field_name="command")
        return extracted is not None and bool(extracted.value) and (
            build_partial_apply_patch_preview(
                extracted.value,
                max_lines=self.apply_patch_preview_max_lines,
            )
            is not None
        )

    def render_text(self, *, prefix: str, pretty: bool) -> str:
        tool_name = self.tool_name or "tool"
        header_prefix = prefix.strip()
        if header_prefix:
            header = f"{header_prefix} {tool_name}\n"
        else:
            header = f"{tool_name}\n"

        args_text = self.display_text
        if is_apply_patch_tool_name(tool_name):
            stripped_text = self.raw_text.strip()
            if stripped_text:
                preview = build_apply_patch_preview_from_input(
                    stripped_text,
                    max_lines=self.apply_patch_preview_max_lines,
                )
                if preview is not None:
                    args_text = format_apply_patch_preview(preview)
                elif stripped_text.lstrip().startswith("*** Begin Patch"):
                    args_text = format_partial_apply_patch_preview(
                        stripped_text,
                        max_lines=self.apply_patch_preview_max_lines,
                    )

        if self.raw_text.strip():
            parsed_args = _parse_json_value(self.raw_text)
            if parsed_args is not _JSON_PARSE_FAILED:
                if pretty:
                    args_text = json.dumps(parsed_args, indent=2, ensure_ascii=True)
                if isinstance(parsed_args, dict) and is_shell_execution_tool(tool_name):
                    command = parsed_args.get("command")
                    if isinstance(command, str):
                        preview = build_apply_patch_preview(
                            command,
                            max_lines=self.apply_patch_preview_max_lines,
                        )
                        if preview is not None:
                            args_text = format_apply_patch_preview(
                                preview,
                                other_args=extract_non_command_args(parsed_args),
                            )
                        else:
                            partial_preview = build_partial_apply_patch_preview(
                                command,
                                other_args=extract_non_command_args(parsed_args),
                                max_lines=self.apply_patch_preview_max_lines,
                            )
                            if partial_preview is not None:
                                args_text = partial_preview
            elif is_shell_execution_tool(tool_name):
                extracted = extract_partial_json_string_field(self.raw_text, field_name="command")
                if extracted is not None and extracted.value:
                    partial_preview = build_partial_apply_patch_preview(
                        extracted.value,
                        max_lines=self.apply_patch_preview_max_lines,
                    )
                    if partial_preview is not None:
                        args_text = partial_preview

        if args_text and pretty and not args_text.endswith("\n"):
            args_text += "\n"
        return header + (args_text or "")


@dataclass(frozen=True)
class PartialJsonStringField:
    key: str
    value: str
    complete: bool


@dataclass(frozen=True)
class ToolCodePreview:
    code: str
    language: str
    complete: bool


def _tool_code_preview_spec(metadata: Mapping[str, Any] | None) -> tuple[str, str] | None:
    if not metadata:
        return None

    variant = metadata.get("variant")
    if variant == "code":
        code_arg = metadata.get("code_arg") or "code"
        language = metadata.get("language") or "python"
    elif variant == "shell":
        code_arg = "command"
        shell_path = metadata.get("shell_path")
        language = shell_syntax_language(
            metadata.get("shell_name"),
            shell_path=shell_path if isinstance(shell_path, str) else None,
        )
    else:
        return None

    if not isinstance(code_arg, str) or not code_arg:
        return None
    if not isinstance(language, str) or not language:
        return None
    return code_arg, language


def _decode_json_string_at(raw_text: str, start_index: int) -> tuple[str, int, bool]:
    if start_index >= len(raw_text) or raw_text[start_index] != '"':
        return "", start_index, False

    result: list[str] = []
    index = start_index + 1
    length = len(raw_text)

    while index < length:
        char = raw_text[index]
        if char == '"':
            return "".join(result), index + 1, True
        if char != "\\":
            result.append(char)
            index += 1
            continue
        if index + 1 >= length:
            return "".join(result), length, False

        escape = raw_text[index + 1]
        simple_escapes = {
            '"': '"',
            "\\": "\\",
            "/": "/",
            "b": "\b",
            "f": "\f",
            "n": "\n",
            "r": "\r",
            "t": "\t",
        }
        replacement = simple_escapes.get(escape)
        if replacement is not None:
            result.append(replacement)
            index += 2
            continue
        if escape == "u":
            if index + 5 >= length:
                return "".join(result), length, False
            hex_digits = raw_text[index + 2 : index + 6]
            try:
                result.append(chr(int(hex_digits, 16)))
            except ValueError:
                result.append("\\u" + hex_digits)
            index += 6
            continue

        result.append(escape)
        index += 2

    return "".join(result), length, False


def _skip_json_value(raw_text: str, start_index: int) -> int:
    length = len(raw_text)
    if start_index >= length:
        return -1

    char = raw_text[start_index]
    if char == '"':
        _, end_index, complete = _decode_json_string_at(raw_text, start_index)
        return end_index if complete else -1

    if char in "[{":
        stack = [char]
        index = start_index + 1
        in_string = False
        escape = False
        matching = {"{": "}", "[": "]"}

        while index < length:
            current = raw_text[index]
            if in_string:
                if escape:
                    escape = False
                elif current == "\\":
                    escape = True
                elif current == '"':
                    in_string = False
                index += 1
                continue

            if current == '"':
                in_string = True
            elif current in "[{":
                stack.append(current)
            elif current in "]}":
                if not stack or matching[stack[-1]] != current:
                    return -1
                stack.pop()
                if not stack:
                    return index + 1
            index += 1

        return -1

    index = start_index
    while index < length and raw_text[index] not in ",}":
        index += 1
    return index


def extract_partial_json_string_field(
    raw_text: str,
    *,
    field_name: str,
) -> PartialJsonStringField | None:
    length = len(raw_text)
    if length == 0:
        return None

    index = 0
    while index < length and raw_text[index].isspace():
        index += 1
    if index >= length or raw_text[index] != "{":
        return None
    index += 1

    while index < length:
        while index < length and raw_text[index].isspace():
            index += 1
        if index >= length:
            return None
        if raw_text[index] == "}":
            return None
        if raw_text[index] == ",":
            index += 1
            continue
        if raw_text[index] != '"':
            return None

        key, index, key_complete = _decode_json_string_at(raw_text, index)
        if not key_complete:
            return None

        while index < length and raw_text[index].isspace():
            index += 1
        if index >= length or raw_text[index] != ":":
            return None
        index += 1

        while index < length and raw_text[index].isspace():
            index += 1
        if index >= length:
            return None

        if key != field_name:
            index = _skip_json_value(raw_text, index)
            if index < 0:
                return None
            continue

        if raw_text[index] != '"':
            return None

        value, _end_index, complete = _decode_json_string_at(raw_text, index)
        return PartialJsonStringField(
            key=field_name,
            value=value,
            complete=complete,
        )

    return None


def _parse_json_value(raw_text: str) -> Any:
    if not raw_text:
        return _JSON_PARSE_FAILED
    try:
        return json.loads(raw_text)
    except Exception:
        return _JSON_PARSE_FAILED


def _normalize_tool_name(tool_name: str) -> str:
    if tool_name in {"web_search", "web_search_call"}:
        return "Searching the web"
    return tool_name


def _status_chunk(status: str) -> str:
    normalized = status.strip().lower()
    if not normalized:
        return ""

    known_chunks = {
        "in_progress": "starting...",
        "queued": "queued...",
        "started": "started...",
        "searching": "searching...",
        "completed": "completed",
        "failed": "failed",
        "cancelled": "cancelled",
        "incomplete": "incomplete",
    }
    return known_chunks.get(normalized, normalized.replace("_", " "))


class StreamSegmentAssembler:
    """Route streamed chunks into markdown/reasoning/tool segments."""

    def __init__(
        self,
        *,
        base_kind: SegmentKind,
        tool_prefix: str,
        tool_metadata_resolver: Callable[[str], Mapping[str, Any] | None] | None = None,
        apply_patch_preview_max_lines: int | None = None,
    ) -> None:
        self._buffer = StreamSegmentBuffer(base_kind)
        self._reasoning_parser = ReasoningStreamParser()
        self._reasoning_active = False
        self._tool_prefix = tool_prefix
        self._tool_metadata_resolver = tool_metadata_resolver
        self._apply_patch_preview_max_lines = apply_patch_preview_max_lines
        self._tool_states: dict[str, ToolStreamState] = {}
        self._fallback_tool_counter = 0
        self._last_tool_id: str | None = None

    @property
    def segments(self) -> list[StreamSegment]:
        return self._buffer.segments

    @property
    def pending_table_row(self) -> str:
        return self._buffer.pending_table_row

    def has_pending_content(self) -> bool:
        """Return True when buffered stream state can still emit content on flush."""
        if self._buffer.pending_table_row:
            return True
        if self._reasoning_parser.in_think:
            return True
        for state in self._tool_states.values():
            if state.raw_text or state.display_text:
                return True
        return False

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
        lookup_tool_name = str(info.get("tool_name") or "tool") if info else "tool"
        tool_name = (
            str(info.get("tool_display_name") or lookup_tool_name or "tool") if info else "tool"
        )
        tool_name = _normalize_tool_name(tool_name)
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
        tool_metadata = self._resolve_tool_metadata(lookup_tool_name, info)
        if state is not None and tool_metadata is not None:
            state.tool_metadata = tool_metadata

        if event_type == "start":
            if state is None:
                state = self._start_tool(
                    tool_use_id,
                    tool_name,
                    tool_metadata=tool_metadata,
                    create_segment=False,
                )
            state.completed = False
            chunk = str(info.get("chunk") or "") if info else ""
            if not chunk:
                return False
            state.append(chunk)
            self._update_tool_segment(state, pretty=False)
            return True

        if event_type == "delta":
            chunk = str(info.get("chunk") or "") if info else ""
            if not chunk:
                return False
            if state is None:
                state = self._start_tool(
                    tool_use_id,
                    tool_name,
                    tool_metadata=tool_metadata,
                    create_segment=False,
                )
            state.append(chunk)
            self._update_tool_segment(state, pretty=False)
            return True

        if event_type == "replace":
            chunk = str(info.get("chunk") or "") if info else ""
            if not chunk:
                return False
            if state is None:
                state = self._start_tool(
                    tool_use_id,
                    tool_name,
                    tool_metadata=tool_metadata,
                    create_segment=False,
                )
            state.raw_text = ""
            state.display_text = ""
            state.decoder = LiteralNewlineDecoder()
            state.append(chunk)
            self._update_tool_segment(state, pretty=False)
            return True

        if event_type == "status":
            chunk = str(info.get("chunk") or "") if info else ""
            if not chunk and info:
                raw_status = info.get("status")
                if isinstance(raw_status, str):
                    chunk = _status_chunk(raw_status)
            if not chunk:
                return False
            if state is None:
                state = self._start_tool(
                    tool_use_id,
                    tool_name,
                    tool_metadata=tool_metadata,
                    create_segment=False,
                )
            state.raw_text = ""
            state.display_text = ""
            state.decoder = LiteralNewlineDecoder()
            state.append(chunk)
            self._update_tool_segment(state, pretty=False)
            return True

        if event_type == "stop":
            if state is None:
                return False
            state.completed = True
            if not state.raw_text and not state.display_text:
                self._tool_states.pop(tool_use_id, None)
                if self._last_tool_id == tool_use_id:
                    self._last_tool_id = None
                return False
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

    def _start_tool(
        self,
        tool_use_id: str,
        tool_name: str,
        *,
        tool_metadata: Mapping[str, Any] | None = None,
        create_segment: bool = True,
    ) -> ToolStreamState:
        segment_index: int | None = None
        if create_segment:
            self._buffer.consume_reasoning_gap()
            self._buffer.ensure_separator()
            segment = StreamSegment(
                kind="tool", text="", tool_name=tool_name, tool_use_id=tool_use_id
            )
            self._buffer.append_segment(segment)
            segment_index = len(self._buffer.segments) - 1
        state = ToolStreamState(
            tool_use_id=tool_use_id,
            tool_name=tool_name,
            segment_index=segment_index,
            tool_metadata=tool_metadata,
            apply_patch_preview_max_lines=self._apply_patch_preview_max_lines,
        )
        self._tool_states[tool_use_id] = state
        return state

    def _resolve_tool_metadata(
        self,
        tool_name: str,
        info: Mapping[str, Any] | None,
    ) -> Mapping[str, Any] | None:
        if info:
            metadata = info.get("tool_metadata")
            if isinstance(metadata, Mapping):
                return metadata
        if self._tool_metadata_resolver is None or not tool_name:
            return None
        return self._tool_metadata_resolver(tool_name)

    def _update_tool_segment(self, state: ToolStreamState, *, pretty: bool) -> None:
        if state.segment_index is None or state.segment_index >= len(self._buffer.segments):
            self._buffer.consume_reasoning_gap()
            self._buffer.ensure_separator()
            segment = StreamSegment(
                kind="tool",
                text="",
                tool_name=state.tool_name,
                tool_use_id=state.tool_use_id,
            )
            self._buffer.append_segment(segment)
            state.segment_index = len(self._buffer.segments) - 1
        segment = self._buffer.segments[state.segment_index]
        segment.text = state.render_text(prefix=self._tool_prefix, pretty=pretty)
        segment.code_preview = state.code_preview()
        segment.apply_patch_preview = state.has_apply_patch_preview()

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
