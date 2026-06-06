from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from fast_agent.session.trace_export_errors import SessionExportUploadError
from fast_agent.session.trace_export_hf import HuggingFaceDatasetTraceUploader

if TYPE_CHECKING:
    from pathlib import Path


class _ApiStub:
    def __init__(self) -> None:
        self.create_repo_calls: list[tuple[str, str, bool]] = []
        self.upload_file_calls: list[tuple[str, str, str, str, str]] = []

    def create_repo(self, *, repo_id: str, repo_type: str, exist_ok: bool) -> None:
        self.create_repo_calls.append((repo_id, repo_type, exist_ok))

    def upload_file(
        self,
        *,
        path_or_fileobj: str,
        path_in_repo: str,
        repo_id: str,
        repo_type: str,
        commit_message: str,
    ) -> str:
        self.upload_file_calls.append(
            (path_or_fileobj, path_in_repo, repo_id, repo_type, commit_message)
        )
        return "https://huggingface.co/datasets/owner/dataset/commit/main"


def test_hf_dataset_uploader_defaults_to_repo_root(tmp_path: Path) -> None:
    api = _ApiStub()
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text("{}", encoding="utf-8")

    result = HuggingFaceDatasetTraceUploader(api=api).upload(
        dataset_repo="owner/dataset",
        trace_path=trace_path,
    )

    assert api.create_repo_calls == [("owner/dataset", "dataset", True)]
    assert api.upload_file_calls == [
        (
            str(trace_path),
            "trace.jsonl",
            "owner/dataset",
            "dataset",
            "Upload fast-agent trace trace.jsonl",
        )
    ]
    assert result.path_in_repo == "trace.jsonl"


def test_hf_dataset_uploader_appends_filename_for_folder_path(tmp_path: Path) -> None:
    api = _ApiStub()
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text("{}", encoding="utf-8")

    result = HuggingFaceDatasetTraceUploader(api=api).upload(
        dataset_repo="owner/dataset",
        trace_path=trace_path,
        dataset_path="exports/",
    )

    assert api.upload_file_calls[0][1] == "exports/trace.jsonl"
    assert result.path_in_repo == "exports/trace.jsonl"


def test_hf_dataset_uploader_appends_filename_for_padded_folder_path(tmp_path: Path) -> None:
    api = _ApiStub()
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text("{}", encoding="utf-8")

    result = HuggingFaceDatasetTraceUploader(api=api).upload(
        dataset_repo="owner/dataset",
        trace_path=trace_path,
        dataset_path="  /exports/  ",
    )

    assert api.upload_file_calls[0][1] == "exports/trace.jsonl"
    assert result.path_in_repo == "exports/trace.jsonl"


def test_hf_dataset_uploader_treats_blank_dataset_path_as_repo_root(tmp_path: Path) -> None:
    api = _ApiStub()
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text("{}", encoding="utf-8")

    result = HuggingFaceDatasetTraceUploader(api=api).upload(
        dataset_repo="owner/dataset",
        trace_path=trace_path,
        dataset_path=" / ",
    )

    assert api.upload_file_calls[0][1] == "trace.jsonl"
    assert result.path_in_repo == "trace.jsonl"


def test_hf_dataset_uploader_reports_missing_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    def _missing_module(module_name: str) -> object:
        raise ModuleNotFoundError(module_name)

    monkeypatch.setattr("fast_agent.session.trace_export_hf.import_module", _missing_module)

    with pytest.raises(SessionExportUploadError, match="requires `huggingface_hub`"):
        HuggingFaceDatasetTraceUploader()
