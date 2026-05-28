"""Trace artifact support for batch runs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import import_module
from typing import TYPE_CHECKING, Protocol
from uuid import uuid4

from fast_agent.agents.tool_runner import ToolRunnerHooks
from fast_agent.interfaces import ToolRunnerHookCapable
from fast_agent.mcp.prompt import Prompt
from fast_agent.session.snapshot import (
    JsonValue,
    SessionAgentSnapshot,
    SessionContinuationSnapshot,
    SessionMetadataSnapshot,
    SessionSnapshot,
)
from fast_agent.session.trace_export_codex import CodexTraceWriter
from fast_agent.session.trace_export_errors import SessionExportUploadError
from fast_agent.session.trace_export_models import ResolvedSessionExport

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from fast_agent.agents.tool_runner import ToolRunner
    from fast_agent.interfaces import AgentProtocol
    from fast_agent.mcp.prompt_message_extended import PromptMessageExtended


@dataclass(frozen=True)
class BatchTraceOptions:
    export_traces_path: Path | None = None
    hf_dataset: str | None = None
    hf_dataset_path: str | None = None


@dataclass(frozen=True)
class BatchTraceUploadResult:
    repo_id: str
    path_in_repo: str
    commit_url: str
    file_url: str


@dataclass(frozen=True)
class _RowTraceContext:
    row_number: int
    identity: str | int
    rendered: str


class BatchTraceRecorder:
    """Capture row-local tool-loop messages and write Codex-format traces."""

    def __init__(
        self,
        *,
        trace_dir: Path,
        agent: AgentProtocol,
        run_metadata: dict[str, object],
        progress_callback: Callable[[str], None] | None = None,
    ) -> None:
        self.trace_dir = trace_dir
        self.run_id = _new_run_id()
        self._agent = agent
        self._run_metadata = run_metadata
        self._progress_callback = progress_callback
        self._active: _RowTraceContext | None = None
        self._captured: list[PromptMessageExtended] | None = None
        self._manifest_path = self.trace_dir / "manifest.jsonl"

    def initialize(self) -> None:
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        with self._manifest_path.open("w", encoding="utf-8") as handle:
            handle.write("")

    def install_hook(self) -> None:
        if not isinstance(self._agent, ToolRunnerHookCapable):
            return

        existing = self._agent.tool_runner_hooks or ToolRunnerHooks()
        existing_after_turn = existing.after_turn_complete

        async def after_turn_complete(
            runner: ToolRunner,
            message: PromptMessageExtended,
        ) -> None:
            if existing_after_turn is not None:
                await existing_after_turn(runner, message)
            self.capture_turn(runner, message)

        self._agent.tool_runner_hooks = ToolRunnerHooks(
            before_llm_call=existing.before_llm_call,
            after_llm_call=existing.after_llm_call,
            before_tool_call=existing.before_tool_call,
            after_tool_call=existing.after_tool_call,
            after_turn_complete=after_turn_complete,
        )

    def start_row(self, *, row_number: int, identity: str | int, rendered: str) -> None:
        self._active = _RowTraceContext(
            row_number=row_number,
            identity=identity,
            rendered=rendered,
        )
        self._captured = None

    def capture_turn(self, runner: ToolRunner, message: PromptMessageExtended) -> None:
        if self._active is None:
            return
        messages = [item.model_copy(deep=True) for item in runner.delta_messages]
        messages.append(message.model_copy(deep=True))
        self._captured = messages

    def finish_row(
        self,
        *,
        ok: bool,
        response: PromptMessageExtended | None = None,
        error_type: str | None = None,
        error_message: str | None = None,
    ) -> None:
        active = self._active
        if active is None:
            return
        try:
            messages = self._captured
            if messages is None and response is not None:
                messages = [
                    Prompt.user(active.rendered),
                    response.model_copy(deep=True),
                ]

            trace_relpath = None
            record_count = 0
            if messages:
                trace_path = self._trace_path(active)
                result = CodexTraceWriter(progress_callback=self._progress_callback).write(
                    self._resolved_export(active, messages, trace_path.name),
                    trace_path,
                )
                trace_relpath = trace_path.name
                record_count = result.record_count

            self._write_manifest(
                active=active,
                ok=ok,
                trace=trace_relpath,
                record_count=record_count,
                error_type=error_type,
                error_message=error_message,
            )
        finally:
            self._active = None
            self._captured = None

    def record_row_without_trace(
        self,
        *,
        row_number: int,
        identity: str | int,
        ok: bool,
        error_type: str | None = None,
        error_message: str | None = None,
    ) -> None:
        self._write_manifest(
            active=_RowTraceContext(row_number=row_number, identity=identity, rendered=""),
            ok=ok,
            trace=None,
            record_count=0,
            error_type=error_type,
            error_message=error_message,
        )

    def upload_to_hf_dataset(
        self,
        *,
        dataset_repo: str,
        dataset_path: str | None,
    ) -> BatchTraceUploadResult:
        return HuggingFaceDatasetFolderUploader().upload_folder(
            dataset_repo=dataset_repo,
            folder_path=self.trace_dir,
            dataset_path=dataset_path,
            run_id=self.run_id,
        )

    def _resolved_export(
        self,
        active: _RowTraceContext,
        history: list[PromptMessageExtended],
        history_file: str,
    ) -> ResolvedSessionExport:
        created_at = _first_timestamp(history) or datetime.now(UTC)
        session_id = f"{self.run_id}-row-{active.row_number:06d}"
        snapshot = SessionSnapshot(
            session_id=session_id,
            created_at=created_at,
            last_activity=datetime.now(UTC),
            metadata=SessionMetadataSnapshot(
                extras={
                    **_json_metadata(self._run_metadata),
                    "batch_run_id": self.run_id,
                    "batch_row_number": active.row_number,
                    "batch_row_id": str(active.identity),
                }
            ),
            continuation=SessionContinuationSnapshot(
                active_agent=self._agent.name,
                agents={
                    self._agent.name: SessionAgentSnapshot(
                        history_file=history_file,
                        resolved_prompt=self._agent.instruction,
                        model=_agent_model(self._agent),
                        provider=_agent_provider(self._agent),
                    )
                },
            ),
        )
        return ResolvedSessionExport(
            session_id=session_id,
            session_dir=self.trace_dir,
            snapshot=snapshot,
            agent_name=self._agent.name,
            history_path=self.trace_dir / history_file,
            history=history,
            message_timestamps=tuple(message.timestamp for message in history),
        )

    def _trace_path(self, active: _RowTraceContext) -> Path:
        identity = _sanitize_filename_component(str(active.identity))[:80]
        agent = _sanitize_filename_component(self._agent.name)
        return self.trace_dir / f"row-{active.row_number:06d}__id-{identity}__{agent}.codex.jsonl"

    def _write_manifest(
        self,
        *,
        active: _RowTraceContext,
        ok: bool,
        trace: str | None,
        record_count: int,
        error_type: str | None,
        error_message: str | None,
    ) -> None:
        record: dict[str, object] = {
            "run_id": self.run_id,
            "row_number": active.row_number,
            "id": active.identity,
            "ok": ok,
            "trace": trace,
            "record_count": record_count,
            "agent": self._agent.name,
        }
        if error_type is not None or error_message is not None:
            record["error"] = {
                "type": error_type,
                "message": error_message,
            }
        with self._manifest_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


class HuggingFaceDatasetFolderUploader:
    """Upload a batch trace directory to a Hugging Face dataset in one commit."""

    def __init__(self, *, api: HubApiProtocol | None = None) -> None:
        self._api = api or _create_hf_api()

    def upload_folder(
        self,
        *,
        dataset_repo: str,
        folder_path: Path,
        dataset_path: str | None,
        run_id: str,
    ) -> BatchTraceUploadResult:
        path_in_repo = _resolve_dataset_path(run_id, dataset_path)
        try:
            self._api.create_repo(
                repo_id=dataset_repo,
                repo_type="dataset",
                exist_ok=True,
            )
            commit = self._api.upload_folder(
                folder_path=str(folder_path),
                path_in_repo=path_in_repo,
                repo_id=dataset_repo,
                repo_type="dataset",
                commit_message=f"Upload fast-agent batch traces {run_id}",
            )
        except Exception as exc:
            raise SessionExportUploadError(
                f"Failed to upload batch traces to Hugging Face dataset '{dataset_repo}': {exc}"
            ) from exc

        return BatchTraceUploadResult(
            repo_id=dataset_repo,
            path_in_repo=path_in_repo,
            commit_url=str(commit),
            file_url=f"https://huggingface.co/datasets/{dataset_repo}/tree/main/{path_in_repo}",
        )


class HubApiProtocol(Protocol):
    def create_repo(
        self,
        *,
        repo_id: str,
        repo_type: str,
        exist_ok: bool,
    ) -> object: ...

    def upload_folder(
        self,
        *,
        folder_path: str,
        path_in_repo: str,
        repo_id: str,
        repo_type: str,
        commit_message: str,
    ) -> object: ...


def _new_run_id() -> str:
    timestamp = datetime.now(UTC).strftime("%y%m%d-%H%M%S")
    return f"{timestamp}-{uuid4().hex[:8]}"


def _first_timestamp(messages: list[PromptMessageExtended]) -> datetime | None:
    for message in messages:
        if message.timestamp is not None:
            return message.timestamp
    return None


def _sanitize_filename_component(value: str) -> str:
    sanitized = "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in value)
    return sanitized or "row"


def _json_metadata(metadata: dict[str, object]) -> dict[str, JsonValue]:
    return {
        key: value
        for key, value in metadata.items()
        if value is None or isinstance(value, bool | int | float | str)
    }


def _agent_model(agent: AgentProtocol) -> str | None:
    llm = agent.llm
    if llm is not None and llm.model_name is not None:
        return llm.model_name
    return agent.config.model


def _agent_provider(agent: AgentProtocol) -> str | None:
    llm = agent.llm
    return llm.provider.config_name if llm is not None else None


def _resolve_dataset_path(run_id: str, dataset_path: str | None) -> str:
    if dataset_path is None:
        return f"traces/{run_id}"
    normalized = dataset_path.strip().strip("/")
    if not normalized:
        return f"traces/{run_id}"
    if dataset_path.endswith("/"):
        return f"{normalized}/{run_id}"
    return normalized


def _create_hf_api() -> HubApiProtocol:
    try:
        module = import_module("huggingface_hub")
        api_class = module.HfApi
    except Exception as exc:
        raise SessionExportUploadError(
            "Uploading batch traces to Hugging Face datasets requires `huggingface_hub`. "
            "Install it first, then retry the batch run."
        ) from exc
    return api_class()
