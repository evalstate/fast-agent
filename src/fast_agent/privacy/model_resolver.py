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
DEFAULT_PRIVACY_FILTER_VARIANT = "q4"

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
}


def resolve_privacy_filter_model_dir(
    *,
    model_path: Path | None,
    repo_id: str = DEFAULT_PRIVACY_FILTER_REPO,
    revision: str = DEFAULT_PRIVACY_FILTER_REVISION,
    variant: str = DEFAULT_PRIVACY_FILTER_VARIANT,
    allow_download: bool = False,
) -> Path:
    """Resolve and validate a privacy-filter model directory."""

    if model_path is not None:
        return _validate_model_dir(model_path.expanduser(), variant=variant)

    allow_patterns = COMMON_FILES + _variant_files(variant)
    try:
        cached = _snapshot_download(
            repo_id=repo_id,
            revision=revision,
            allow_patterns=allow_patterns,
            local_files_only=True,
        )
        return _validate_model_dir(Path(cached), variant=variant)
    except Exception as cached_exc:
        if not allow_download:
            raise SessionExportPrivacyFilterError(_uncached_model_message()) from cached_exc

    try:
        downloaded = _snapshot_download(
            repo_id=repo_id,
            revision=revision,
            allow_patterns=allow_patterns,
            local_files_only=False,
        )
    except Exception as exc:
        raise SessionExportPrivacyFilterError(
            f"Failed to download privacy filter model '{repo_id}' at revision '{revision}': {exc}"
        ) from exc
    return _validate_model_dir(Path(downloaded), variant=variant)


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
        supported = ", ".join(sorted(VARIANT_FILES))
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
