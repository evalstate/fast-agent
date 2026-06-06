"""OpenAI Privacy Filter ONNX Runtime sanitizer."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING, Any, Literal

from fast_agent.privacy.dependencies import (
    PRIVACY_EXTRA_INSTALL_MESSAGE,
    PRIVACY_EXTRA_REQUIREMENTS,
    format_missing_privacy_dependencies,
    missing_privacy_dependencies,
)
from fast_agent.privacy.model_resolver import (
    DEFAULT_PRIVACY_FILTER_REPO,
    DEFAULT_PRIVACY_FILTER_REVISION,
    DEFAULT_PRIVACY_FILTER_VARIANT,
    PRIVACY_FILTER_VARIANTS,
)
from fast_agent.privacy.sanitizer import (
    PrivacyFilterModelInfo,
    RedactionSpan,
    SanitizedText,
    TraceSanitizer,
)
from fast_agent.privacy.viterbi import (
    VITERBI_TRANSITION_BIAS_KEYS,
    ZERO_TRANSITION_BIASES,
    ViterbiTables,
    build_viterbi_tables,
    constrained_viterbi_np,
    token_spans_from_path,
)
from fast_agent.session.trace_export_errors import SessionExportPrivacyFilterError
from fast_agent.utils.text import strip_casefold

if TYPE_CHECKING:
    from collections.abc import Callable
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
# CPU trace export intentionally uses a much smaller inference chunk than the
# model's advertised default_n_ctx (currently 128k). 4096 keeps ORT memory and
# latency manageable while still dwarfing the model's local attention radius.
_DEFAULT_INFERENCE_WINDOW_TOKENS = 4096
_DEFAULT_WINDOW_OVERLAP_TOKENS = 128
_DEFAULT_DEVICE: Literal["auto"] = "auto"
_SUPPORTED_DEVICES = ("auto", "cpu", "cuda")


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
        device: str | None = None,
        progress_callback: Callable[[str], None] | None = None,
        show_redactions: bool = False,
    ) -> None:
        self._model_dir = model_dir
        self._variant = variant
        self._device = _privacy_filter_device(device)
        self._cuda_device_id = _env_int(
            "FAST_AGENT_PRIVACY_FILTER_CUDA_DEVICE_ID",
            default=0,
            minimum=0,
        )
        self._progress_callback = progress_callback
        self._show_redactions = show_redactions
        self._files = _model_files(model_dir, variant=variant)
        self._config = _load_json(self._files.config)
        self._labels = _load_labels(self._config)
        self._tokenizer, self._session, self._np = self._load_runtime()
        transition_biases = _load_viterbi_transition_biases(
            self._model_dir / "viterbi_calibration.json"
        )
        self._viterbi_tables: ViterbiTables = build_viterbi_tables(
            self._labels,
            self._np,
            transition_biases=transition_biases,
        )
        session_providers = list(self._session.get_providers())
        self._active_provider = session_providers[0] if session_providers else None
        self._max_window_tokens = _env_int(
            "FAST_AGENT_PRIVACY_FILTER_MAX_WINDOW_TOKENS",
            default=_DEFAULT_INFERENCE_WINDOW_TOKENS,
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
        backend = (
            "onnxruntime-cuda"
            if self._active_provider == "CUDAExecutionProvider"
            else "onnxruntime"
        )
        return PrivacyFilterModelInfo(
            backend=backend,
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
        if not text:
            return []
        encoding = self._tokenizer.encode(text)
        ids = list(encoding.ids)
        attention = list(encoding.attention_mask)
        offsets = list(encoding.offsets)
        if not ids:
            return []
        real_token_indices = _real_token_indices(offsets)
        if not real_token_indices:
            return []

        # Single-window fast path.
        if len(real_token_indices) <= self._max_window_tokens:
            return self._run_window(
                ids=ids,
                attention=attention,
                offsets=offsets,
                text=text,
            )

        # Slide over real-token positions; slice the *already tokenized* sequence
        # rather than re-encoding substrings (which doubles tokenizer cost and
        # changes BPE boundaries at window seams).
        spans: list[RedactionSpan] = []
        step = self._max_window_tokens - self._window_overlap_tokens
        window_starts = list(range(0, len(real_token_indices), step))
        total_windows = len(window_starts)
        self._emit_progress(
            f"Privacy filter: scanning large text ({len(text):,} chars, "
            f"{total_windows:,} windows)..."
        )
        for window_number, start_real in enumerate(window_starts, start=1):
            end_real = min(start_real + self._max_window_tokens, len(real_token_indices))
            if start_real >= end_real:
                continue
            if _should_emit_window_progress(window_number, total_windows):
                self._emit_progress(
                    f"Privacy filter: large text window {window_number:,}/{total_windows:,} "
                    f"({_percent(window_number, total_windows)}%)..."
                )
            # Map real-token indices to absolute positions in the tokenized
            # sequence. Include any leading/trailing special tokens at the
            # boundaries of the full encoding so the model still sees them.
            token_start = real_token_indices[start_real] if start_real > 0 else 0
            token_end = (
                real_token_indices[end_real - 1] + 1
                if end_real < len(real_token_indices)
                else len(ids)
            )
            spans.extend(
                self._run_window(
                    ids=ids[token_start:token_end],
                    attention=attention[token_start:token_end],
                    offsets=offsets[token_start:token_end],
                    text=text,
                )
            )
            if end_real == len(real_token_indices):
                break
        return _merge_spans(spans)

    def _run_window(
        self,
        *,
        ids: list[int],
        attention: list[int],
        offsets: list[tuple[int, int]],
        text: str,
    ) -> list[RedactionSpan]:
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
        if logits.shape[1] != len(ids):
            raise SessionExportPrivacyFilterError(
                "Privacy filter logits length does not match window token count "
                f"(got {logits.shape[1]}, expected {len(ids)})."
            )
        # Softmax normalization is unnecessary for Viterbi: subtracting the
        # per-token logsumexp adds the same constant to every label score for
        # that token and cannot change the best path.
        path = constrained_viterbi_np(logits[0], self._viterbi_tables, self._np)
        spans: list[RedactionSpan] = []
        for token_span in token_spans_from_path(path, self._labels):
            start_char, _ = offsets[token_span.start]
            _, end_char = offsets[token_span.end - 1]
            if start_char >= end_char:
                # Span landed on special / zero-offset tokens; skip.
                continue
            trimmed = _trim_span(text, start_char, end_char)
            if trimmed is None:
                continue
            spans.append(
                RedactionSpan(
                    label=token_span.label,
                    start=trimmed[0],
                    end=trimmed[1],
                )
            )
        return _merge_spans(spans)

    def _emit_progress(self, message: str) -> None:
        if self._progress_callback is not None:
            self._progress_callback(message)

    def _emit_redactions(self, text: str, spans: list[RedactionSpan]) -> None:
        if self._progress_callback is None:
            return
        for span in spans:
            snippet = _redaction_snippet(text[span.start : span.end])
            self._emit_progress(
                f"Privacy filter: redaction {span.label} {span.start}:{span.end} {snippet!r}"
            )

    def _load_runtime(self) -> tuple[Any, Any, Any]:
        try:
            modules = {
                module_name: import_module(module_name)
                for module_name in PRIVACY_EXTRA_REQUIREMENTS
            }
            np = modules["numpy"]
            ort = modules["onnxruntime"]
            tokenizers = modules["tokenizers"]
        except Exception as exc:
            missing = missing_privacy_dependencies()
            message = (
                format_missing_privacy_dependencies(missing)
                if missing
                else (
                    "Privacy filtering requires optional dependencies, but one failed "
                    f"to load. {PRIVACY_EXTRA_INSTALL_MESSAGE}"
                )
            )
            raise SessionExportPrivacyFilterError(message) from exc

        # Only try to preload CUDA / cuDNN DLLs when CUDA is actually in play.
        # On CPU-only "auto" runs this avoids spurious ORT warnings about
        # missing libcudart / libcudnn.
        available_providers = list(ort.get_available_providers())
        if self._device == "cuda" or "CUDAExecutionProvider" in available_providers:
            ort.preload_dlls(cuda=True, cudnn=True, msvc=True)
            available_providers = list(ort.get_available_providers())

        tokenizer = tokenizers.Tokenizer.from_file(str(self._files.tokenizer))
        options = ort.SessionOptions()
        options.log_severity_level = _env_int(
            "FAST_AGENT_PRIVACY_FILTER_LOG_SEVERITY",
            default=3,
            minimum=0,
        )
        options.intra_op_num_threads = _env_int(
            "FAST_AGENT_PRIVACY_FILTER_INTRA_OP_THREADS",
            default=0,
            minimum=1,
        )
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        requested_providers = _resolve_onnx_execution_providers(
            available_providers=available_providers,
            device=self._device,
            cuda_device_id=self._cuda_device_id,
        )
        session = ort.InferenceSession(
            str(self._files.model),
            sess_options=options,
            providers=requested_providers,
        )
        self._emit_progress(
            _provider_status_message(
                device=self._device,
                available_providers=available_providers,
                requested_providers=requested_providers,
                active_providers=list(session.get_providers()),
            )
        )
        return tokenizer, session, np


def _model_files(model_dir: Path, *, variant: str) -> _ModelFiles:
    if variant not in PRIVACY_FILTER_VARIANTS:
        supported = ", ".join(PRIVACY_FILTER_VARIANTS)
        raise SessionExportPrivacyFilterError(
            f"Unsupported privacy filter variant '{variant}'. Supported variants: {supported}."
        )
    model_name = {
        "q4": "model_q4.onnx",
        "q4f16": "model_q4f16.onnx",
        "q8": "model_quantized.onnx",
        "fp16": "model_fp16.onnx",
    }[variant]
    return _ModelFiles(
        config=model_dir / "config.json",
        tokenizer=model_dir / "tokenizer.json",
        model=model_dir / "onnx" / model_name,
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
        labels: list[str] = []
        for entry in id2label:
            if not isinstance(entry, str) or not entry:
                raise SessionExportPrivacyFilterError(
                    "Privacy filter config has invalid id2label entries."
                )
            labels.append(entry)
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


def _load_viterbi_transition_biases(path: Path) -> dict[str, float]:
    """Load optional calibrated BIOES transition biases.

    The current OpenAI Privacy Filter calibration file contains all zeroes, so
    this is usually behavior-preserving. Keeping support here lets newer model
    revisions tune precision/recall without code changes.
    """

    if not path.is_file():
        return dict(ZERO_TRANSITION_BIASES)
    payload = _load_json(path)
    raw_biases: object = payload
    operating_points = payload.get("operating_points")
    if operating_points is not None:
        if not isinstance(operating_points, dict):
            raise SessionExportPrivacyFilterError(
                f"Invalid Viterbi calibration operating_points in {path}."
            )
        default = operating_points.get("default")
        raw_biases = default.get("biases") if isinstance(default, dict) else None

    if raw_biases in (None, {}):
        return dict(ZERO_TRANSITION_BIASES)
    if not isinstance(raw_biases, dict):
        raise SessionExportPrivacyFilterError(f"Invalid Viterbi calibration biases in {path}.")

    biases = dict(ZERO_TRANSITION_BIASES)
    for key in VITERBI_TRANSITION_BIAS_KEYS:
        raw_value = raw_biases.get(key, 0.0)
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            raise SessionExportPrivacyFilterError(
                f"Invalid Viterbi calibration value {key!r} in {path}."
            )
        biases[key] = float(raw_value)
    return biases


def _real_token_indices(offsets: list[tuple[int, int]]) -> list[int]:
    return [index for index, (start, end) in enumerate(offsets) if end > start]


def _should_emit_window_progress(window_number: int, total_windows: int) -> bool:
    if total_windows <= 5:
        return True
    step = max(1, total_windows // 10)
    return window_number in (1, total_windows) or window_number % step == 0


def _percent(value: int, total: int) -> int:
    if total <= 0:
        return 100
    return min(100, round((value / total) * 100))


def _replace_spans(text: str, spans: list[RedactionSpan]) -> str:
    if not spans:
        return text
    parts: list[str] = []
    cursor = 0
    for span in sorted(spans, key=lambda item: item.start):
        if span.start < cursor:
            # Overlap with a previous span (different-label cases handled in
            # _merge_spans, but defend in case callers feed raw spans).
            continue
        if span.start > cursor:
            parts.append(text[cursor : span.start])
        placeholder = _PLACEHOLDERS.get(span.label, f"<{span.label.upper()}>")
        parts.append(placeholder)
        cursor = span.end
    parts.append(text[cursor:])
    return "".join(parts)


def _trim_span(text: str, start: int, end: int) -> tuple[int, int] | None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if start >= end:
        return None
    return start, end


def _merge_spans(spans: list[RedactionSpan]) -> list[RedactionSpan]:
    """Merge overlapping same-label spans; keep distinct-label spans separate.

    Different-label overlaps drop the later span — replacement requires
    non-overlapping intervals, and `(start, end)` ordering picks the earlier
    detection deterministically.
    """

    if not spans:
        return []
    ordered = sorted(spans, key=lambda span: (span.start, span.end))
    merged: list[RedactionSpan] = []
    for span in ordered:
        if not merged:
            merged.append(span)
            continue
        previous = merged[-1]
        if span.start >= previous.end:
            merged.append(span)
            continue
        # Overlap.
        if span.label == previous.label:
            merged[-1] = RedactionSpan(
                label=previous.label,
                start=previous.start,
                end=max(previous.end, span.end),
            )
        # else: drop the overlapping later span.
    return merged


def _redaction_snippet(text: str, *, limit: int = 160) -> str:
    snippet = text.replace("\n", "\\n").replace("\r", "\\r")
    if len(snippet) <= limit:
        return snippet
    return f"{snippet[: limit - 1]}…"


def _privacy_filter_device(value: str | None) -> Literal["auto", "cpu", "cuda"]:
    if value is None:
        value = os.getenv("FAST_AGENT_PRIVACY_FILTER_DEVICE", _DEFAULT_DEVICE)
    normalized = strip_casefold(value)
    if normalized == "auto":
        return "auto"
    if normalized == "cpu":
        return "cpu"
    if normalized == "cuda":
        return "cuda"
    supported = ", ".join(_SUPPORTED_DEVICES)
    raise SessionExportPrivacyFilterError(
        f"Unsupported privacy filter device '{value}'. Supported devices: {supported}."
    )


def _resolve_onnx_execution_providers(
    *,
    available_providers: list[str],
    device: Literal["auto", "cpu", "cuda"],
    cuda_device_id: int = 0,
) -> list[Any]:
    if device == "cpu":
        return ["CPUExecutionProvider"]
    if "CUDAExecutionProvider" in set(available_providers):
        return [
            ("CUDAExecutionProvider", {"device_id": str(cuda_device_id)}),
            "CPUExecutionProvider",
        ]
    if device == "cuda":
        raise SessionExportPrivacyFilterError(
            "CUDA was requested for the privacy filter, but ONNX Runtime does not "
            "expose CUDAExecutionProvider. Install `fast-agent-mcp[privacy-gpu]` "
            "or a compatible `onnxruntime-gpu`, then ensure CUDA/cuDNN are available; "
            "or set FAST_AGENT_PRIVACY_FILTER_DEVICE=cpu."
        )
    return ["CPUExecutionProvider"]


def _provider_status_message(
    *,
    device: Literal["auto", "cpu", "cuda"],
    available_providers: list[str],
    requested_providers: list[Any],
    active_providers: list[str],
) -> str:
    active_text = ", ".join(active_providers) or "unknown"
    if "CUDAExecutionProvider" in active_providers:
        return (
            "Privacy filter: provider CUDAExecutionProvider (GPU; fallback: CPUExecutionProvider)."
        )
    if _requested_cuda_provider(requested_providers):
        return (
            "Privacy filter: provider "
            f"{active_text} (CUDA was available but failed to initialize; using CPU fallback)."
        )
    if device == "auto" and "CUDAExecutionProvider" not in available_providers:
        return f"Privacy filter: provider {active_text} (CUDA provider not available)."
    return f"Privacy filter: provider {active_text}."


def _requested_cuda_provider(providers: list[Any]) -> bool:
    for provider in providers:
        if provider == "CUDAExecutionProvider":
            return True
        if isinstance(provider, tuple) and provider and provider[0] == "CUDAExecutionProvider":
            return True
    return False


def _env_int(name: str, *, default: int, minimum: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(minimum, value)
