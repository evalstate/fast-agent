from __future__ import annotations

from io import BytesIO
from typing import TYPE_CHECKING, BinaryIO
from urllib.parse import quote

import pytest
from huggingface_hub.utils import parse_hf_uri

from fast_agent.session.trace_export_errors import SessionExportUploadError
from fast_agent.session.trace_export_hf import HuggingFaceDatasetTraceUploader

if TYPE_CHECKING:
    from pathlib import Path
    from types import TracebackType


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


class _WritableHfFile(BytesIO):
    def __enter__(self) -> "_WritableHfFile":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        return None


class _FileSystemStub:
    def __init__(self) -> None:
        self.opened: list[tuple[str, str]] = []
        self.files: list[_WritableHfFile] = []

    def open(self, path: str, mode: str = "rb") -> BinaryIO:
        self.opened.append((path, mode))
        file = _WritableHfFile()
        self.files.append(file)
        return file


def test_hf_dataset_uploader_defaults_to_repo_root(tmp_path: Path) -> None:
    api = _ApiStub()
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text("{}", encoding="utf-8")

    filesystem = _FileSystemStub()
    result = HuggingFaceDatasetTraceUploader(api=api, filesystem=filesystem).upload(
        dataset_repo="owner/dataset",
        trace_path=trace_path,
    )

    assert api.create_repo_calls == [("owner/dataset", "dataset", True)]
    assert api.upload_file_calls == []
    assert filesystem.opened == [("hf://datasets/owner/dataset/trace.jsonl", "wb")]
    assert result.path_in_repo == "trace.jsonl"


def test_hf_dataset_uploader_appends_filename_for_folder_path(tmp_path: Path) -> None:
    api = _ApiStub()
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text("{}", encoding="utf-8")

    filesystem = _FileSystemStub()
    result = HuggingFaceDatasetTraceUploader(api=api, filesystem=filesystem).upload(
        dataset_repo="owner/dataset",
        trace_path=trace_path,
        dataset_path="exports/",
    )

    assert filesystem.opened[0][0] == "hf://datasets/owner/dataset/exports/trace.jsonl"
    assert result.path_in_repo == "exports/trace.jsonl"


def test_hf_dataset_uploader_appends_filename_for_padded_folder_path(tmp_path: Path) -> None:
    api = _ApiStub()
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text("{}", encoding="utf-8")

    filesystem = _FileSystemStub()
    result = HuggingFaceDatasetTraceUploader(api=api, filesystem=filesystem).upload(
        dataset_repo="owner/dataset",
        trace_path=trace_path,
        dataset_path="  /exports/  ",
    )

    assert filesystem.opened[0][0] == "hf://datasets/owner/dataset/exports/trace.jsonl"
    assert result.path_in_repo == "exports/trace.jsonl"


def test_hf_dataset_uploader_treats_blank_dataset_path_as_repo_root(tmp_path: Path) -> None:
    api = _ApiStub()
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text("{}", encoding="utf-8")

    filesystem = _FileSystemStub()
    result = HuggingFaceDatasetTraceUploader(api=api, filesystem=filesystem).upload(
        dataset_repo="owner/dataset",
        trace_path=trace_path,
        dataset_path=" / ",
    )

    assert filesystem.opened[0][0] == "hf://datasets/owner/dataset/trace.jsonl"
    assert result.path_in_repo == "trace.jsonl"


def test_hf_url_uploader_writes_to_bucket_url(tmp_path: Path) -> None:
    api = _ApiStub()
    filesystem = _FileSystemStub()
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text("{}", encoding="utf-8")

    result = HuggingFaceDatasetTraceUploader(api=api, filesystem=filesystem).upload(
        trace_path=trace_path,
        hf_url="hf://buckets/evalstate/traces/",
    )

    assert api.create_repo_calls == []
    assert filesystem.opened == [("hf://buckets/evalstate/traces/trace.jsonl", "wb")]
    assert filesystem.files[0].getvalue() == b"{}"
    assert result.file_url == "hf://buckets/evalstate/traces/trace.jsonl"


def test_hf_url_uploader_appends_filename_for_bucket_root(tmp_path: Path) -> None:
    api = _ApiStub()
    filesystem = _FileSystemStub()
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text("{}", encoding="utf-8")

    result = HuggingFaceDatasetTraceUploader(api=api, filesystem=filesystem).upload(
        trace_path=trace_path,
        hf_url="hf://buckets/evalstate/openclaw-data",
    )

    assert filesystem.opened == [("hf://buckets/evalstate/openclaw-data/trace.jsonl", "wb")]
    assert result.file_url == "hf://buckets/evalstate/openclaw-data/trace.jsonl"


def test_hf_url_uploader_appends_filename_for_dataset_root(tmp_path: Path) -> None:
    api = _ApiStub()
    filesystem = _FileSystemStub()
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text("{}", encoding="utf-8")

    result = HuggingFaceDatasetTraceUploader(api=api, filesystem=filesystem).upload(
        trace_path=trace_path,
        hf_url="hf://datasets/owner/dataset",
    )

    assert api.create_repo_calls == [("owner/dataset", "dataset", True)]
    assert filesystem.opened == [("hf://datasets/owner/dataset/trace.jsonl", "wb")]
    assert result.file_url == "https://huggingface.co/datasets/owner/dataset/blob/main/trace.jsonl"


@pytest.mark.parametrize("revision", ["review", "refs/pr/12", "refs%2Fpr%2F12"])
@pytest.mark.parametrize("path_in_url", ["", "/exports/trace.jsonl"])
def test_hf_dataset_uploader_preserves_revision(
    tmp_path: Path, revision: str, path_in_url: str
) -> None:
    api = _ApiStub()
    filesystem = _FileSystemStub()
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text("{}", encoding="utf-8")
    hf_url = f"hf://datasets/owner/dataset@{revision}{path_in_url}"

    result = HuggingFaceDatasetTraceUploader(api=api, filesystem=filesystem).upload(
        trace_path=trace_path,
        hf_url=hf_url,
    )

    destination = f"{hf_url}/trace.jsonl" if not path_in_url else hf_url
    assert filesystem.opened == [(destination, "wb")]
    parsed = parse_hf_uri(destination)
    assert api.create_repo_calls == [(parsed.id, "dataset", True)]
    assert result.repo_id == parsed.id
    assert result.path_in_repo == parsed.path_in_repo
    assert result.file_url == (
        f"https://huggingface.co/datasets/{parsed.id}/blob/"
        f"{quote(parsed.revision or 'main', safe='')}/{parsed.path_in_repo}"
    )
    assert filesystem.files[0].getvalue() == trace_path.read_bytes()


def test_hf_dataset_uploader_reports_invalid_destination(tmp_path: Path) -> None:
    api = _ApiStub()
    filesystem = _FileSystemStub()
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text("{}", encoding="utf-8")

    with pytest.raises(SessionExportUploadError, match="Failed to upload trace"):
        HuggingFaceDatasetTraceUploader(api=api, filesystem=filesystem).upload(
            trace_path=trace_path,
            hf_url="hf://datasets/owner",
        )

    assert api.create_repo_calls == []
    assert filesystem.opened == []


def test_hf_dataset_uploader_reports_missing_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    def _missing_module(module_name: str) -> object:
        raise ModuleNotFoundError(module_name)

    monkeypatch.setattr("fast_agent.session.trace_export_hf.import_module", _missing_module)

    with pytest.raises(SessionExportUploadError, match="requires `huggingface_hub`"):
        HuggingFaceDatasetTraceUploader()
