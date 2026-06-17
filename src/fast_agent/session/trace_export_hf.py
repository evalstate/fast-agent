"""Upload exported session traces to Hugging Face Hub URLs."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Protocol
from urllib.parse import quote, urlparse

from fast_agent.session.trace_export_errors import SessionExportUploadError
from fast_agent.session.trace_export_models import DatasetUploadResult

if TYPE_CHECKING:
    from pathlib import Path
    from typing import BinaryIO


def _build_dataset_file_url(repo_id: str, path_in_repo: str) -> str:
    return f"https://huggingface.co/datasets/{repo_id}/blob/main/{quote(path_in_repo, safe='/')}"


def _resolve_path_in_repo(trace_path: Path, dataset_path: str | None) -> str:
    if dataset_path is None:
        return trace_path.name
    stripped = dataset_path.strip()
    normalized = stripped.strip("/")
    if not normalized:
        return trace_path.name
    if stripped.endswith("/"):
        return f"{normalized}/{trace_path.name}"
    return normalized


def dataset_upload_url(*, dataset_repo: str, trace_path: Path, dataset_path: str | None) -> str:
    """Build the compatibility hf:// URL for a dataset upload request."""
    return f"hf://datasets/{dataset_repo}/{_resolve_path_in_repo(trace_path, dataset_path)}"


def resolve_hf_upload_url(*, hf_url: str, trace_path: Path) -> str:
    """Resolve an hf:// destination, appending the local filename for folder URLs."""
    stripped = hf_url.strip()
    normalized = stripped.rstrip("/")
    if stripped.endswith("/") or _is_hf_storage_root(normalized):
        return f"{normalized}/{trace_path.name}"
    return stripped


def _is_hf_storage_root(hf_url: str) -> bool:
    parsed = urlparse(hf_url)
    if parsed.scheme != "hf" or parsed.netloc not in {"buckets", "datasets"}:
        return False
    parts = [part for part in parsed.path.split("/") if part]
    return len(parts) == 2


def _parse_dataset_hf_url(hf_url: str) -> tuple[str, str] | None:
    parsed = urlparse(hf_url)
    if parsed.scheme != "hf" or parsed.netloc != "datasets":
        return None
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 3:
        return None
    return f"{parts[0]}/{parts[1]}", "/".join(parts[2:])


def _parse_hf_url_parts(hf_url: str) -> tuple[str, str]:
    parsed = urlparse(hf_url)
    parts = [part for part in parsed.path.split("/") if part]
    if parsed.scheme != "hf" or not parsed.netloc or not parts:
        return hf_url, ""
    if parsed.netloc == "datasets" and len(parts) >= 3:
        return f"{parts[0]}/{parts[1]}", "/".join(parts[2:])
    return f"hf://{parsed.netloc}/{parts[0]}", "/".join(parts[1:])


class HuggingFaceDatasetTraceUploader:
    """Upload exported traces to Hugging Face Hub storage."""

    def __init__(
        self,
        *,
        api: HubApiProtocol | None = None,
        filesystem: HubFileSystemProtocol | None = None,
    ) -> None:
        self._api = api or _create_hf_api()
        self._filesystem = filesystem or _create_hf_filesystem()

    def upload(
        self,
        *,
        trace_path: Path,
        hf_url: str | None = None,
        dataset_repo: str | None = None,
        dataset_path: str | None = None,
    ) -> DatasetUploadResult:
        if hf_url is None:
            if dataset_repo is None:
                raise SessionExportUploadError("A Hugging Face upload URL is required.")
            hf_url = dataset_upload_url(
                dataset_repo=dataset_repo,
                trace_path=trace_path,
                dataset_path=dataset_path,
            )
        resolved_url = resolve_hf_upload_url(hf_url=hf_url, trace_path=trace_path)
        dataset_target = _parse_dataset_hf_url(resolved_url)
        try:
            if dataset_target is not None:
                self._api.create_repo(
                    repo_id=dataset_target[0],
                    repo_type="dataset",
                    exist_ok=True,
                )
            with trace_path.open("rb") as source, self._filesystem.open(resolved_url, "wb") as dest:
                dest.write(source.read())
        except Exception as exc:
            raise SessionExportUploadError(
                f"Failed to upload trace to Hugging Face URL '{resolved_url}': {exc}"
            ) from exc

        repo_id, path_in_repo = _parse_hf_url_parts(resolved_url)
        file_url = resolved_url
        if dataset_target is not None:
            file_url = _build_dataset_file_url(*dataset_target)
        return DatasetUploadResult(
            repo_id=repo_id,
            path_in_repo=path_in_repo,
            commit_url="",
            file_url=file_url,
            destination_url=resolved_url,
            destination_label="url",
        )


class DatasetTraceUploader(Protocol):
    """Minimal uploader interface for exported trace uploads."""

    def upload(
        self,
        *,
        trace_path: Path,
        hf_url: str | None = None,
        dataset_repo: str | None = None,
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


class HubFileSystemProtocol(Protocol):
    """Subset of Hugging Face filesystem methods used for trace uploads."""

    def open(self, path: str, mode: str = "rb") -> "BinaryIO": ...


def _create_hf_api() -> HubApiProtocol:
    try:
        module = import_module("huggingface_hub")
        api_class = module.HfApi
    except Exception as exc:
        raise SessionExportUploadError(
            "Uploading traces to Hugging Face datasets requires `huggingface_hub`. "
            "Install it first, then retry the export."
        ) from exc
    return api_class()


def _create_hf_filesystem() -> HubFileSystemProtocol:
    try:
        module = import_module("huggingface_hub")
        filesystem_class = module.HfFileSystem
    except Exception as exc:
        raise SessionExportUploadError(
            "Uploading traces to Hugging Face URLs requires `huggingface_hub`. "
            "Install it first, then retry the export."
        ) from exc
    return filesystem_class()
