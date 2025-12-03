from dataclasses import dataclass
from typing import List


@dataclass
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
        segments: List[ReasoningSegment] = []

        while self._buffer:
            if self._in_think:
                closing_index = self._buffer.find("</think>")
                if closing_index == -1:
                    segments.append(ReasoningSegment(text=self._buffer, is_thinking=True))
                    self._buffer = ""
                    break

                if closing_index > 0:
                    segments.append(
                        ReasoningSegment(text=self._buffer[:closing_index], is_thinking=True)
                    )

                self._buffer = self._buffer[closing_index + len("</think>") :]
                self._in_think = False
            else:
                opening_index = self._buffer.find("<think>")
                if opening_index == -1:
                    segments.append(ReasoningSegment(text=self._buffer, is_thinking=False))
                    self._buffer = ""
                    break

                if opening_index > 0:
                    segments.append(
                        ReasoningSegment(
                            text=self._buffer[:opening_index],
                            is_thinking=False,
                        )
                    )

                self._buffer = self._buffer[opening_index + len("<think>") :]
                self._in_think = True

        return [segment for segment in segments if segment.text]
