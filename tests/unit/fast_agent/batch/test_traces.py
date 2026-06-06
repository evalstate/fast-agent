from pathlib import Path
from typing import cast

from fast_agent.batch.traces import BatchTraceRecorder, HuggingFaceDatasetFolderUploader
from fast_agent.interfaces import AgentProtocol


class RecordingHubApi:
    def __init__(self) -> None:
        self.created: dict[str, object] | None = None
        self.uploaded: dict[str, object] | None = None

    def create_repo(
        self,
        *,
        repo_id: str,
        repo_type: str,
        exist_ok: bool,
    ) -> object:
        self.created = {
            "repo_id": repo_id,
            "repo_type": repo_type,
            "exist_ok": exist_ok,
        }
        return object()

    def upload_folder(
        self,
        *,
        folder_path: str,
        path_in_repo: str,
        repo_id: str,
        repo_type: str,
        commit_message: str,
    ) -> object:
        self.uploaded = {
            "folder_path": folder_path,
            "path_in_repo": path_in_repo,
            "repo_id": repo_id,
            "repo_type": repo_type,
            "commit_message": commit_message,
        }
        return "https://huggingface.co/datasets/owner/dataset/commit/abc"


def _agent_stub() -> AgentProtocol:
    return cast("AgentProtocol", object())


def test_trace_recorder_initialize_truncates_manifest_for_new_run(tmp_path: Path) -> None:
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    manifest = trace_dir / "manifest.jsonl"
    manifest.write_text('{"row_number":1}\n', encoding="utf-8")

    recorder = BatchTraceRecorder(trace_dir=trace_dir, agent=_agent_stub(), run_metadata={})
    recorder.initialize()

    assert manifest.read_text(encoding="utf-8") == ""


def test_trace_recorder_initialize_preserves_manifest_for_resume(tmp_path: Path) -> None:
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    manifest = trace_dir / "manifest.jsonl"
    manifest.write_text('{"row_number":1}', encoding="utf-8")

    recorder = BatchTraceRecorder(trace_dir=trace_dir, agent=_agent_stub(), run_metadata={})
    recorder.initialize(resume=True)

    assert manifest.read_text(encoding="utf-8") == '{"row_number":1}\n'


def test_hf_dataset_folder_uploader_uploads_directory_once(tmp_path: Path) -> None:
    api = RecordingHubApi()
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    (trace_dir / "manifest.jsonl").write_text("{}", encoding="utf-8")

    result = HuggingFaceDatasetFolderUploader(api=api).upload_folder(
        dataset_repo="owner/dataset",
        folder_path=trace_dir,
        dataset_path="runs/",
        run_id="run-123",
    )

    assert api.created == {
        "repo_id": "owner/dataset",
        "repo_type": "dataset",
        "exist_ok": True,
    }
    assert api.uploaded == {
        "folder_path": str(trace_dir),
        "path_in_repo": "runs/run-123",
        "repo_id": "owner/dataset",
        "repo_type": "dataset",
        "commit_message": "Upload fast-agent batch traces run-123",
    }
    assert result.path_in_repo == "runs/run-123"


def test_hf_dataset_folder_uploader_appends_run_id_for_padded_folder_path(
    tmp_path: Path,
) -> None:
    api = RecordingHubApi()
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    (trace_dir / "manifest.jsonl").write_text("{}", encoding="utf-8")

    result = HuggingFaceDatasetFolderUploader(api=api).upload_folder(
        dataset_repo="owner/dataset",
        folder_path=trace_dir,
        dataset_path=" /runs/ ",
        run_id="run-123",
    )

    assert api.uploaded is not None
    assert api.uploaded["path_in_repo"] == "runs/run-123"
    assert result.path_in_repo == "runs/run-123"
