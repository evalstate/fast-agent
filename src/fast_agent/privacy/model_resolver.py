"""Resolve OpenAI Privacy Filter model files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from fast_agent.session.trace_export_errors import SessionExportPrivacyFilterError

if TYPE_CHECKING:
    from collections.abc import Sequence

DEFAULT_PRIVACY_FILTER_REPO = "openai/privacy-filter"
DEFAULT_PRIVACY_FILTER_REVISION = "7ffa9a043d54d1be65afb281eddf0ffbe629385b"
# Default to int8 (q8). On CPU — the dominant deployment for trace export —
# ORT's int8 GEMM kernels are typically faster than the q4 MatMulNBits path.
# CUDA users should pick `q4f16` or `fp16` explicitly.
DEFAULT_PRIVACY_FILTER_VARIANT = "q8"
PRIVACY_FILTER_VARIANTS = ("q4", "q4f16", "q8", "fp16")

# When the user does not specify a variant, prefer these in order if cached.
# `q8` first matches the documented CPU default; `q4` is a common pre-existing
# cache from earlier releases; remaining variants act as best-effort fallbacks.
_VARIANT_FALLBACK_ORDER = ("q8", "q4", "q4f16", "fp16")

COMMON_FILES = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "viterbi_calibration.json",
]

VARIANT_FILES = {
    "q4": [
        "onnx/model_q4.onnx",
        "onnx/model_q4.onnx_data",
    ],
    "q4f16": [
        "onnx/model_q4f16.onnx",
        "onnx/model_q4f16.onnx_data",
    ],
    "q8": [
        "onnx/model_quantized.onnx",
        "onnx/model_quantized.onnx_data",
    ],
    "fp16": [
        "onnx/model_fp16.onnx",
        "onnx/model_fp16.onnx_data",
        "onnx/model_fp16.onnx_data_1",
    ],
}


def resolve_privacy_filter_model_dir(
    *,
    model_path: Path | None,
    repo_id: str = DEFAULT_PRIVACY_FILTER_REPO,
    revision: str = DEFAULT_PRIVACY_FILTER_REVISION,
    variant: str = DEFAULT_PRIVACY_FILTER_VARIANT,
    allow_download: bool = False,
    variant_explicit: bool = True,
) -> tuple[Path, str]:
    """Resolve and validate a privacy-filter model directory.

    Returns ``(model_dir, effective_variant)``. When ``variant_explicit`` is
    ``False`` and the requested variant is not cached, falls back to any other
    cached variant before considering a download — this avoids forcing an
    unnecessary re-download when the default variant changes between releases.
    """

    if model_path is not None:
        return _validate_model_dir(model_path.expanduser(), variant=variant), variant

    candidates: list[str] = [variant]
    if not variant_explicit:
        for fallback in _VARIANT_FALLBACK_ORDER:
            if fallback != variant and fallback not in candidates:
                candidates.append(fallback)

    last_error: Exception | None = None
    for candidate in candidates:
        try:
            cached = _snapshot_download(
                repo_id=repo_id,
                revision=revision,
                allow_patterns=COMMON_FILES + _variant_files(candidate),
                local_files_only=True,
            )
            return _validate_model_dir(Path(cached), variant=candidate), candidate
        except Exception as exc:
            last_error = exc
            continue

    if not allow_download:
        raise SessionExportPrivacyFilterError(_uncached_model_message()) from last_error

    try:
        downloaded = _snapshot_download(
            repo_id=repo_id,
            revision=revision,
            allow_patterns=COMMON_FILES + _variant_files(variant),
            local_files_only=False,
        )
    except Exception as exc:
        raise SessionExportPrivacyFilterError(
            f"Failed to download privacy filter model '{repo_id}' at revision '{revision}': {exc}"
        ) from exc
    return _validate_model_dir(Path(downloaded), variant=variant), variant


def _snapshot_download(
    *,
    repo_id: str,
    revision: str,
    allow_patterns: Sequence[str],
    local_files_only: bool,
) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id=repo_id,
        revision=revision,
        allow_patterns=list(allow_patterns),
        local_files_only=local_files_only,
    )


def _variant_files(variant: str) -> list[str]:
    files = VARIANT_FILES.get(variant)
    if files is None:
        supported = ", ".join(PRIVACY_FILTER_VARIANTS)
        raise SessionExportPrivacyFilterError(
            f"Unsupported privacy filter variant '{variant}'. Supported variants: {supported}."
        )
    return files


def _validate_model_dir(model_dir: Path, *, variant: str) -> Path:
    model_dir = model_dir.resolve()
    if not model_dir.is_dir():
        raise SessionExportPrivacyFilterError(
            f"Privacy filter model path is not a directory: {model_dir}"
        )

    missing = [
        relative
        for relative in COMMON_FILES + _variant_files(variant)
        if not (model_dir / relative).is_file()
    ]
    if missing:
        missing_lines = "\n".join(f"  - {relative}" for relative in missing)
        raise SessionExportPrivacyFilterError(
            f"Privacy filter model directory is missing required files:\n{missing_lines}"
        )

    config_path = model_dir / "config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SessionExportPrivacyFilterError(
            f"Failed to read privacy filter config from {config_path}: {exc}"
        ) from exc
    model_type = config.get("model_type")
    if model_type != "openai_privacy_filter":
        raise SessionExportPrivacyFilterError(
            "Privacy filter model config is not an OpenAI Privacy Filter model "
            f"(model_type={model_type!r})."
        )
    return model_dir


def _uncached_model_message() -> str:
    return (
        "Privacy filter model is not cached.\n\n"
        "The default model is:\n"
        f"  {DEFAULT_PRIVACY_FILTER_REPO} @ {DEFAULT_PRIVACY_FILTER_REVISION}, "
        f"variant {DEFAULT_PRIVACY_FILTER_VARIANT}\n\n"
        "Required download is approximately 1 GB.\n\n"
        "Run again with:\n"
        "  fast-agent export latest --privacy-filter --download-privacy-filter\n\n"
        "or provide a local model directory:\n"
        "  fast-agent export latest --privacy-filter --privacy-filter-path /path/to/model"
    )
