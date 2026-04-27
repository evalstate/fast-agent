from __future__ import annotations

import json
from typing import TYPE_CHECKING

from fast_agent.privacy import model_resolver

if TYPE_CHECKING:
    from pathlib import Path


def _write_common_model_files(model_dir: Path) -> None:
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": "openai_privacy_filter"}),
        encoding="utf-8",
    )
    (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (model_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (model_dir / "viterbi_calibration.json").write_text("{}", encoding="utf-8")


def test_resolve_privacy_filter_model_dir_falls_back_when_default_variant_incomplete(
    monkeypatch,
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "privacy-filter"
    _write_common_model_files(model_dir)
    onnx_dir = model_dir / "onnx"
    onnx_dir.mkdir()
    (onnx_dir / "model_q4.onnx").write_bytes(b"q4")
    (onnx_dir / "model_q4.onnx_data").write_bytes(b"q4-data")

    calls: list[list[str]] = []

    def fake_snapshot_download(
        *,
        repo_id: str,
        revision: str,
        allow_patterns: list[str],
        local_files_only: bool,
    ) -> str:
        del repo_id, revision, local_files_only
        calls.append(allow_patterns)
        return str(model_dir)

    monkeypatch.setattr(model_resolver, "_snapshot_download", fake_snapshot_download)

    resolved_dir, variant = model_resolver.resolve_privacy_filter_model_dir(
        model_path=None,
        variant="q8",
        allow_download=False,
        variant_explicit=False,
    )

    assert resolved_dir == model_dir.resolve()
    assert variant == "q4"
    assert calls == [
        model_resolver.COMMON_FILES + model_resolver.VARIANT_FILES["q8"],
        model_resolver.COMMON_FILES + model_resolver.VARIANT_FILES["q4"],
    ]
