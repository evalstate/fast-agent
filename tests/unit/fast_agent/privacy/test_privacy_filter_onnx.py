from __future__ import annotations

from dataclasses import dataclass

import pytest

from fast_agent.privacy.privacy_filter_onnx import (
    OpenAIPrivacyFilterOnnxSanitizer,
    _merge_spans,
    _provider_status_message,
    _replace_spans,
    _resolve_onnx_execution_providers,
)
from fast_agent.privacy.sanitizer import RedactionSpan
from fast_agent.session.trace_export_errors import SessionExportPrivacyFilterError


@dataclass(slots=True)
class _Encoding:
    ids: list[int]
    attention_mask: list[int]
    offsets: list[tuple[int, int]]


class _WhitespaceTokenizer:
    """Whitespace tokenizer with leading/trailing special tokens at offset (0, 0)."""

    def encode(self, text: str) -> _Encoding:
        offsets: list[tuple[int, int]] = [(0, 0)]
        cursor = 0
        for part in text.split(" "):
            start = cursor
            end = start + len(part)
            offsets.append((start, end))
            cursor = end + 1
        offsets.append((0, 0))
        ids = list(range(len(offsets)))
        attention_mask = [1] * len(offsets)
        return _Encoding(ids=ids, attention_mask=attention_mask, offsets=offsets)


class _RunWindowRecordingSanitizer(OpenAIPrivacyFilterOnnxSanitizer):
    """Stand-in sanitizer that records each window passed to `_run_window`."""

    def __init__(self) -> None:
        self._tokenizer = _WhitespaceTokenizer()
        self._max_window_tokens = 4
        self._window_overlap_tokens = 1
        self._progress_callback = None
        self._show_redactions = False
        self.calls: list[tuple[int, int]] = []

    def _run_window(
        self,
        *,
        ids: list[int],
        attention: list[int],
        offsets: list[tuple[int, int]],
        text: str,
    ) -> list[RedactionSpan]:
        # Record the window's covered char range using only real (non-special) tokens.
        real = [(start, end) for (start, end) in offsets if end > start]
        if not real:
            return []
        char_start = real[0][0]
        char_end = real[-1][1]
        self.calls.append((char_start, char_end))
        marker = "Alice"
        index = text.find(marker, char_start, char_end)
        if index < 0:
            return []
        return [
            RedactionSpan(
                label="private_person",
                start=index,
                end=index + len(marker),
            )
        ]


def test_onnx_sanitizer_chunks_long_text_with_overlap() -> None:
    sanitizer = _RunWindowRecordingSanitizer()

    spans = sanitizer.detect_spans("one two three Alice five six seven")

    # Two windows: first covers tokens [one..Alice] = chars [0..19]; the
    # overlap of 1 token slides by step=3, so window two starts at the
    # 4th real token ("Alice") and runs to the end ("seven", char 34).
    assert sanitizer.calls == [(0, 19), (14, 34)]
    assert spans == [RedactionSpan(label="private_person", start=14, end=19)]


def test_merge_spans_preserves_adjacent_entities() -> None:
    spans = _merge_spans(
        [
            RedactionSpan(label="private_email", start=0, end=5),
            RedactionSpan(label="private_phone", start=5, end=10),
        ]
    )

    assert spans == [
        RedactionSpan(label="private_email", start=0, end=5),
        RedactionSpan(label="private_phone", start=5, end=10),
    ]


def test_merge_spans_unions_same_label_overlap() -> None:
    spans = _merge_spans(
        [
            RedactionSpan(label="private_person", start=0, end=5),
            RedactionSpan(label="private_person", start=3, end=9),
        ]
    )

    assert spans == [RedactionSpan(label="private_person", start=0, end=9)]


def test_merge_spans_drops_overlapping_distinct_label() -> None:
    # Replacement requires non-overlapping spans; deterministically keep the
    # earlier detection rather than silently mislabeling a merged span.
    spans = _merge_spans(
        [
            RedactionSpan(label="private_email", start=0, end=10),
            RedactionSpan(label="private_phone", start=5, end=12),
        ]
    )

    assert spans == [RedactionSpan(label="private_email", start=0, end=10)]


def test_replace_spans_handles_many_redactions_in_linear_pass() -> None:
    text = "a Alice b Alice c Alice d"
    spans = [
        RedactionSpan(label="private_person", start=2, end=7),
        RedactionSpan(label="private_person", start=10, end=15),
        RedactionSpan(label="private_person", start=18, end=23),
    ]

    redacted = _replace_spans(text, spans)

    assert redacted == "a <PRIVATE_PERSON> b <PRIVATE_PERSON> c <PRIVATE_PERSON> d"


def test_resolve_onnx_execution_providers_prefers_cuda_for_auto() -> None:
    providers = _resolve_onnx_execution_providers(
        available_providers=["CPUExecutionProvider", "CUDAExecutionProvider"],
        device="auto",
        cuda_device_id=2,
    )

    assert providers == [
        ("CUDAExecutionProvider", {"device_id": "2"}),
        "CPUExecutionProvider",
    ]


def test_resolve_onnx_execution_providers_allows_cpu_override() -> None:
    providers = _resolve_onnx_execution_providers(
        available_providers=["CPUExecutionProvider", "CUDAExecutionProvider"],
        device="cpu",
    )

    assert providers == ["CPUExecutionProvider"]


def test_resolve_onnx_execution_providers_requires_cuda_when_requested() -> None:
    with pytest.raises(SessionExportPrivacyFilterError):
        _resolve_onnx_execution_providers(
            available_providers=["CPUExecutionProvider"],
            device="cuda",
        )


def test_provider_status_reports_cuda_fallback() -> None:
    message = _provider_status_message(
        device="auto",
        available_providers=["CPUExecutionProvider", "CUDAExecutionProvider"],
        requested_providers=[
            ("CUDAExecutionProvider", {"device_id": "0"}),
            "CPUExecutionProvider",
        ],
        active_providers=["CPUExecutionProvider"],
    )

    assert message == (
        "Privacy filter: provider CPUExecutionProvider "
        "(CUDA was available but failed to initialize; using CPU fallback)."
    )


def test_provider_status_reports_cuda_active() -> None:
    message = _provider_status_message(
        device="auto",
        available_providers=["CPUExecutionProvider", "CUDAExecutionProvider"],
        requested_providers=[
            ("CUDAExecutionProvider", {"device_id": "0"}),
            "CPUExecutionProvider",
        ],
        active_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    assert message == (
        "Privacy filter: provider CUDAExecutionProvider (GPU; fallback: CPUExecutionProvider)."
    )
