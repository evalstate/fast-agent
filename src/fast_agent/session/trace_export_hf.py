"""Upload exported session traces to Hugging Face datasets."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol
from urllib.parse import quote

from huggingface_hub import HfApi

from fast_agent.session.trace_export_errors import SessionExportUploadError
from fast_agent.session.trace_export_models import DatasetUploadResult

if TYPE_CHECKING:
    from pathlib import Path


def _build_dataset_file_url(repo_id: str, path_in_repo: str) -> str:
    return f"https://huggingface.co/datasets/{repo_id}/blob/main/{quote(path_in_repo, safe='/')}"


def _resolve_path_in_repo(trace_path: Path, dataset_path: str | None) -> str:
    if dataset_path is None:
        return trace_path.name
    normalized = dataset_path.strip().strip("/")
    if not normalized:
        return trace_path.name
    if dataset_path.endswith("/"):
        return f"{normalized}/{trace_path.name}"
    return normalized


class HuggingFaceDatasetTraceUploader:
    """Upload exported traces to a dataset repository on the Hugging Face Hub."""

    def __init__(self, *, api: HubApiProtocol | None = None) -> None:
        self._api = api or HfApi()

    def upload(
        self,
        *,
        dataset_repo: str,
        trace_path: Path,
        dataset_path: str | None = None,
    ) -> DatasetUploadResult:
        path_in_repo = _resolve_path_in_repo(trace_path, dataset_path)
        commit_message = f"Upload fast-agent trace {trace_path.name}"
        try:
            self._api.create_repo(
                repo_id=dataset_repo,
                repo_type="dataset",
                exist_ok=True,
            )
            commit = self._api.upload_file(
                path_or_fileobj=str(trace_path),
                path_in_repo=path_in_repo,
                repo_id=dataset_repo,
                repo_type="dataset",
                commit_message=commit_message,
            )
        except Exception as exc:
            raise SessionExportUploadError(
                f"Failed to upload trace to Hugging Face dataset '{dataset_repo}': {exc}"
            ) from exc

        return DatasetUploadResult(
            repo_id=dataset_repo,
            path_in_repo=path_in_repo,
            commit_url=str(commit),
            file_url=_build_dataset_file_url(dataset_repo, path_in_repo),
        )


class DatasetTraceUploader(Protocol):
    """Minimal uploader interface for exported trace uploads."""

    def upload(
        self,
        *,
        dataset_repo: str,
        trace_path: Path,
        dataset_path: str | None = None,
    ) -> DatasetUploadResult: ...


class HubApiProtocol(Protocol):
    """Subset of Hugging Face Hub API methods used for trace uploads."""

    def create_repo(
        self,
        *,
        repo_id: str,
        repo_type: str,
        exist_ok: bool,
    ) -> object: ...

    def upload_file(
        self,
        *,
        path_or_fileobj: str,
        path_in_repo: str,
        repo_id: str,
        repo_type: str,
        commit_message: str,
    ) -> object: ...
