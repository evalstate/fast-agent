from dataclasses import dataclass

_OPENING_TAG = "<think>"
_CLOSING_TAG = "</think>"


@dataclass(frozen=True, slots=True)
class ReasoningSegment:
    """Represents a slice of streamed text and whether it's a reasoning chunk."""

    text: str
    is_thinking: bool


class ReasoningStreamParser:
    """Incrementally split streamed text into thought vs final answer segments."""

    def __init__(self) -> None:
        self._buffer = ""
        self._in_think = False

    @property
    def in_think(self) -> bool:
        """Whether the parser is currently inside a <think>...</think> block."""
        return self._in_think

    @property
    def has_pending_text(self) -> bool:
        """Whether a possible partial tag is waiting for more input."""
        return bool(self._buffer)

    def feed(self, chunk: str) -> list[ReasoningSegment]:
        """Consume a new chunk and return parsed segments."""
        if not chunk:
            return []

        self._buffer += chunk
        return self._extract_segments()

    def flush(self) -> list[ReasoningSegment]:
        """Return any remaining buffered text as a final segment."""
        if not self._buffer:
            return []
        remaining = ReasoningSegment(text=self._buffer, is_thinking=self._in_think)
        self._buffer = ""
        return [remaining]

    def _extract_segments(self) -> list[ReasoningSegment]:
        segments: list[ReasoningSegment] = []

        while self._buffer:
            tag = _CLOSING_TAG if self._in_think else _OPENING_TAG
            tag_index = self._buffer.find(tag)
            if tag_index == -1:
                safe_text, self._buffer = _split_possible_tag_suffix(self._buffer, tag)
                _append_segment(segments, safe_text, is_thinking=self._in_think)
                break

            if tag_index > 0:
                _append_segment(
                    segments,
                    self._buffer[:tag_index],
                    is_thinking=self._in_think,
                )

            self._buffer = self._buffer[tag_index + len(tag) :]
            self._in_think = not self._in_think

        return segments


def _append_segment(
    segments: list[ReasoningSegment],
    text: str,
    *,
    is_thinking: bool,
) -> None:
    if text:
        segments.append(ReasoningSegment(text=text, is_thinking=is_thinking))


def _split_possible_tag_suffix(text: str, tag: str) -> tuple[str, str]:
    """Keep a trailing partial tag buffered until the next chunk arrives."""
    for suffix_length in range(min(len(text), len(tag) - 1), 0, -1):
        suffix = text[-suffix_length:]
        if tag.startswith(suffix):
            return text[:-suffix_length], suffix
    return text, ""
