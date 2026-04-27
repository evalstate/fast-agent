"""OpenAI Privacy Filter ONNX Runtime sanitizer."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING, Any, Callable

from fast_agent.privacy.model_resolver import (
    DEFAULT_PRIVACY_FILTER_REPO,
    DEFAULT_PRIVACY_FILTER_REVISION,
    DEFAULT_PRIVACY_FILTER_VARIANT,
)
from fast_agent.privacy.sanitizer import (
    PrivacyFilterModelInfo,
    RedactionSpan,
    SanitizedText,
    TraceSanitizer,
)
from fast_agent.privacy.viterbi import constrained_viterbi, token_spans_from_path
from fast_agent.session.trace_export_errors import SessionExportPrivacyFilterError

if TYPE_CHECKING:
    from pathlib import Path

_PLACEHOLDERS = {
    "account_number": "<ACCOUNT_NUMBER>",
    "private_address": "<PRIVATE_ADDRESS>",
    "private_date": "<PRIVATE_DATE>",
    "private_email": "<PRIVATE_EMAIL>",
    "private_person": "<PRIVATE_PERSON>",
    "private_phone": "<PRIVATE_PHONE>",
    "private_url": "<PRIVATE_URL>",
    "secret": "<SECRET>",
}
_DEFAULT_MAX_WINDOW_TOKENS = 1024
_DEFAULT_WINDOW_OVERLAP_TOKENS = 32


@dataclass(frozen=True, slots=True)
class _ModelFiles:
    config: Path
    tokenizer: Path
    model: Path


class OpenAIPrivacyFilterOnnxSanitizer(TraceSanitizer):
    """Local ONNX Runtime wrapper around OpenAI Privacy Filter."""

    def __init__(
        self,
        model_dir: Path,
        *,
        variant: str = DEFAULT_PRIVACY_FILTER_VARIANT,
        progress_callback: Callable[[str], None] | None = None,
        show_redactions: bool = False,
    ) -> None:
        self._model_dir = model_dir
        self._variant = variant
        self._progress_callback = progress_callback
        self._show_redactions = show_redactions
        self._files = _model_files(model_dir, variant=variant)
        self._config = _load_json(self._files.config)
        self._labels = _load_labels(self._config)
        self._tokenizer, self._session, self._np = self._load_runtime()
        self._max_window_tokens = _env_int(
            "FAST_AGENT_PRIVACY_FILTER_MAX_WINDOW_TOKENS",
            default=_DEFAULT_MAX_WINDOW_TOKENS,
            minimum=128,
        )
        self._window_overlap_tokens = _env_int(
            "FAST_AGENT_PRIVACY_FILTER_WINDOW_OVERLAP_TOKENS",
            default=_DEFAULT_WINDOW_OVERLAP_TOKENS,
            minimum=0,
        )
        if self._window_overlap_tokens >= self._max_window_tokens:
            self._window_overlap_tokens = max(0, self._max_window_tokens // 8)
        self._input_names = {item.name for item in self._session.get_inputs()}
        if not {"input_ids", "attention_mask"}.issubset(self._input_names):
            raise SessionExportPrivacyFilterError(
                "Privacy filter ONNX model must accept input_ids and attention_mask."
            )

    @property
    def model_info(self) -> PrivacyFilterModelInfo:
        return PrivacyFilterModelInfo(
            backend="onnxruntime",
            repo_id=DEFAULT_PRIVACY_FILTER_REPO,
            revision=DEFAULT_PRIVACY_FILTER_REVISION,
            variant=self._variant,
        )

    def sanitize_text(self, text: str) -> SanitizedText:
        if not text:
            return SanitizedText(text=text)
        spans = self.detect_spans(text)
        if self._show_redactions:
            self._emit_redactions(text, spans)
        return SanitizedText(
            text=_replace_spans(text, spans),
            spans=tuple(spans),
        )

    def detect_spans(self, text: str) -> list[RedactionSpan]:
        encoding = self._tokenizer.encode(text)
        real_token_indices = _real_token_indices(list(encoding.offsets))
        if len(real_token_indices) <= self._max_window_tokens:
            return self._detect_spans_single(text, char_offset=0)

        spans: list[RedactionSpan] = []
        step = self._max_window_tokens - self._window_overlap_tokens
        window_starts = list(range(0, len(real_token_indices), step))
        total_windows = len(window_starts)
        self._emit_progress(
            f"Privacy filter scanning {len(text):,} characters in {total_windows} windows..."
        )
        for window_number, start in enumerate(window_starts, start=1):
            window_indices = real_token_indices[start : start + self._max_window_tokens]
            if not window_indices:
                continue
            if _should_emit_window_progress(window_number, total_windows):
                self._emit_progress(
                    f"Privacy filter window {window_number}/{total_windows} "
                    f"({_percent(window_number, total_windows)}%)..."
                )
            char_start = encoding.offsets[window_indices[0]][0]
            char_end = encoding.offsets[window_indices[-1]][1]
            if char_start >= char_end:
                continue
            chunk = text[char_start:char_end]
            spans.extend(self._detect_spans_single(chunk, char_offset=char_start))
            if start + self._max_window_tokens >= len(real_token_indices):
                break
        return _merge_spans(spans)

    def _emit_progress(self, message: str) -> None:
        if self._progress_callback is not None:
            self._progress_callback(message)

    def _emit_redactions(self, text: str, spans: list[RedactionSpan]) -> None:
        for span in spans:
            snippet = _redaction_snippet(text[span.start : span.end])
            self._emit_progress(
                f"Privacy filter redaction: {span.label} {span.start}:{span.end} {snippet!r}"
            )

    def _detect_spans_single(self, text: str, *, char_offset: int) -> list[RedactionSpan]:
        encoding = self._tokenizer.encode(text)
        ids = list(encoding.ids)
        offsets = list(encoding.offsets)
        attention = list(encoding.attention_mask)
        if not ids:
            return []

        input_ids = self._np.asarray([ids], dtype=self._np.int64)
        attention_mask = self._np.asarray([attention], dtype=self._np.int64)
        outputs = self._session.run(
            None,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            },
        )
        logits = outputs[0]
        if len(logits.shape) != 3 or logits.shape[0] != 1:
            raise SessionExportPrivacyFilterError(
                f"Unexpected privacy filter logits shape: {logits.shape}"
            )
        if logits.shape[2] != len(self._labels):
            raise SessionExportPrivacyFilterError(
                "Privacy filter label count does not match ONNX logits dimension."
            )
        log_probs = _log_softmax(self._np, logits[0]).tolist()
        path = constrained_viterbi(log_probs, self._labels)
        spans: list[RedactionSpan] = []
        for token_span in token_spans_from_path(path, self._labels):
            start, _ = offsets[token_span.start]
            _, end = offsets[token_span.end - 1]
            if start == end:
                continue
            trimmed = _trim_span(text, start, end)
            if trimmed is None:
                continue
            spans.append(
                RedactionSpan(
                    label=token_span.label,
                    start=char_offset + trimmed[0],
                    end=char_offset + trimmed[1],
                )
            )
        return _merge_spans(spans)

    def _load_runtime(self) -> tuple[Any, Any, Any]:
        try:
            np = import_module("numpy")
            ort = import_module("onnxruntime")
            tokenizers = import_module("tokenizers")
        except Exception as exc:
            raise SessionExportPrivacyFilterError(
                "Privacy filtering requires optional dependencies. "
                'Install with `fast-agent-mcp[privacy]`.'
            ) from exc

        tokenizer = tokenizers.Tokenizer.from_file(str(self._files.tokenizer))
        options = ort.SessionOptions()
        options.intra_op_num_threads = _env_int(
            "FAST_AGENT_PRIVACY_FILTER_INTRA_OP_THREADS",
            default=1,
            minimum=1,
        )
        options.inter_op_num_threads = _env_int(
            "FAST_AGENT_PRIVACY_FILTER_INTER_OP_THREADS",
            default=1,
            minimum=1,
        )
        session = ort.InferenceSession(
            str(self._files.model),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        return tokenizer, session, np


def _model_files(model_dir: Path, *, variant: str) -> _ModelFiles:
    if variant != "q4":
        raise SessionExportPrivacyFilterError(
            f"Unsupported privacy filter variant '{variant}'. Supported variants: q4."
        )
    return _ModelFiles(
        config=model_dir / "config.json",
        tokenizer=model_dir / "tokenizer.json",
        model=model_dir / "onnx" / "model_q4.onnx",
    )


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SessionExportPrivacyFilterError(f"Failed to read {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SessionExportPrivacyFilterError(f"Expected object JSON in {path}.")
    return payload


def _load_labels(config: dict[str, Any]) -> list[str]:
    id2label = config.get("id2label")
    if isinstance(id2label, list):
        labels = [label for label in id2label if isinstance(label, str) and label]
        if len(labels) != len(id2label):
            raise SessionExportPrivacyFilterError(
                "Privacy filter config has invalid id2label entries."
            )
        return _validate_labels(labels)
    if not isinstance(id2label, dict):
        raise SessionExportPrivacyFilterError("Privacy filter config is missing id2label.")
    labels: list[str] = []
    for index in range(len(id2label)):
        label = id2label.get(str(index), id2label.get(index))
        if not isinstance(label, str) or not label:
            raise SessionExportPrivacyFilterError(
                f"Privacy filter config has invalid id2label entry for index {index}."
            )
        labels.append(label)
    return _validate_labels(labels)


def _validate_labels(labels: list[str]) -> list[str]:
    if "O" not in labels:
        raise SessionExportPrivacyFilterError("Privacy filter labels must include O.")
    return labels


def _log_softmax(np: Any, logits: Any) -> Any:
    maximum = np.max(logits, axis=-1, keepdims=True)
    shifted = logits - maximum
    return shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))


def _real_token_indices(offsets: list[tuple[int, int]]) -> list[int]:
    return [index for index, (start, end) in enumerate(offsets) if end > start]


def _should_emit_window_progress(window_number: int, total_windows: int) -> bool:
    if total_windows <= 20:
        return True
    return window_number == 1 or window_number == total_windows or window_number % 10 == 0


def _percent(value: int, total: int) -> int:
    if total <= 0:
        return 100
    return min(100, round((value / total) * 100))


def _replace_spans(text: str, spans: list[RedactionSpan]) -> str:
    redacted = text
    for span in sorted(spans, key=lambda item: item.start, reverse=True):
        placeholder = _PLACEHOLDERS.get(span.label, f"<{span.label.upper()}>")
        redacted = redacted[: span.start] + placeholder + redacted[span.end :]
    return redacted


def _trim_span(text: str, start: int, end: int) -> tuple[int, int] | None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if start >= end:
        return None
    return start, end


def _merge_spans(spans: list[RedactionSpan]) -> list[RedactionSpan]:
    if not spans:
        return []
    ordered = sorted(spans, key=lambda span: (span.start, span.end))
    merged: list[RedactionSpan] = []
    for span in ordered:
        if not merged or span.start > merged[-1].end:
            merged.append(span)
            continue
        previous = merged[-1]
        merged[-1] = RedactionSpan(
            label=previous.label,
            start=previous.start,
            end=max(previous.end, span.end),
        )
    return merged


def _redaction_snippet(text: str, *, limit: int = 160) -> str:
    snippet = text.replace("\n", "\\n").replace("\r", "\\r")
    if len(snippet) <= limit:
        return snippet
    return f"{snippet[: limit - 1]}…"


def _env_int(name: str, *, default: int, minimum: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(minimum, value)
