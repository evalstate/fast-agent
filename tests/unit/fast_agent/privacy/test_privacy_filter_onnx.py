from __future__ import annotations

from dataclasses import dataclass

from fast_agent.privacy.privacy_filter_onnx import OpenAIPrivacyFilterOnnxSanitizer
from fast_agent.privacy.sanitizer import RedactionSpan


@dataclass(slots=True)
class _Encoding:
    offsets: list[tuple[int, int]]


class _WhitespaceTokenizer:
    def encode(self, text: str) -> _Encoding:
        offsets: list[tuple[int, int]] = [(0, 0)]
        cursor = 0
        for part in text.split(" "):
            start = cursor
            end = start + len(part)
            offsets.append((start, end))
            cursor = end + 1
        offsets.append((0, 0))
        return _Encoding(offsets=offsets)


class _ChunkRecordingSanitizer(OpenAIPrivacyFilterOnnxSanitizer):
    def __init__(self) -> None:
        self._tokenizer = _WhitespaceTokenizer()
        self._max_window_tokens = 4
        self._window_overlap_tokens = 1
        self._progress_callback = None
        self.calls: list[tuple[str, int]] = []

    def _detect_spans_single(self, text: str, *, char_offset: int) -> list[RedactionSpan]:
        self.calls.append((text, char_offset))
        marker = "Alice"
        index = text.find(marker)
        if index < 0:
            return []
        return [
            RedactionSpan(
                label="private_person",
                start=char_offset + index,
                end=char_offset + index + len(marker),
            )
        ]


def test_onnx_sanitizer_chunks_long_text_with_overlap() -> None:
    sanitizer = _ChunkRecordingSanitizer()

    spans = sanitizer.detect_spans("one two three Alice five six seven")

    assert sanitizer.calls == [
        ("one two three Alice", 0),
        ("Alice five six seven", 14),
    ]
    assert spans == [RedactionSpan(label="private_person", start=14, end=19)]
