from __future__ import annotations

from dataclasses import dataclass

import pytest

from fast_agent.privacy.privacy_filter_onnx import (
    OpenAIPrivacyFilterOnnxSanitizer,
    _merge_spans,
    _provider_status_message,
    _resolve_onnx_execution_providers,
)
from fast_agent.privacy.sanitizer import RedactionSpan
from fast_agent.session.trace_export_errors import SessionExportPrivacyFilterError


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
