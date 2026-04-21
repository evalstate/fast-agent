"""Session trace export service."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import ValidationError

from fast_agent.core.exceptions import AgentConfigError
from fast_agent.mcp.prompt_serialization import load_messages
from fast_agent.session.snapshot import SessionSnapshot, load_session_snapshot
from fast_agent.session.trace_export_codex import CodexTraceWriter
from fast_agent.session.trace_export_errors import (
    InvalidSessionExportTargetError,
    SessionExportAgentNotFoundError,
    SessionExportAmbiguousAgentError,
    SessionExportNoAgentsError,
    SessionExportNotFoundError,
    SessionExportReadError,
    SessionExportWriteError,
    UnsupportedTraceExportFormatError,
)
from fast_agent.session.trace_export_hf import (
    DatasetTraceUploader,
    HuggingFaceDatasetTraceUploader,
)
from fast_agent.session.trace_export_models import (
    ExportRequest,
    ExportResult,
    ResolvedSessionExport,
)

if TYPE_CHECKING:
    from fast_agent.mcp.prompt_message_extended import PromptMessageExtended
    from fast_agent.session.session_manager import SessionManager


def _sanitize_filename_component(value: str) -> str:
    sanitized = "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in value)
    return sanitized or "agent"


class SessionTraceExporter:
    """Resolve persisted sessions and export them in trace formats."""

    def __init__(
        self,
        *,
        session_manager: SessionManager,
        dataset_uploader: DatasetTraceUploader | None = None,
    ) -> None:
        self._session_manager = session_manager
        self._dataset_uploader = dataset_uploader

    def export(self, request: ExportRequest) -> ExportResult:
        resolved = self._resolve_request(request)
        output_path = self._resolve_output_path(request, resolved)
        writer = self._writer_for_format(request.format)
        try:
            result = writer.write(resolved, output_path)
        except OSError as exc:
            raise SessionExportWriteError(
                f"Failed to write trace export to {output_path}: {exc}"
            ) from exc
        if request.hf_dataset is None:
            return result
        uploader = self._dataset_uploader or HuggingFaceDatasetTraceUploader()
        upload = uploader.upload(
            dataset_repo=request.hf_dataset,
            trace_path=result.output_path,
            dataset_path=request.hf_dataset_path,
        )
        return ExportResult(
            session_id=result.session_id,
            agent_name=result.agent_name,
            format=result.format,
            output_path=result.output_path,
            record_count=result.record_count,
            upload=upload,
        )

    def _resolve_request(self, request: ExportRequest) -> ResolvedSessionExport:
        session_dir = self._resolve_session_dir(
            target=request.target,
            current_session_id=request.current_session_id,
        )
        snapshot = self._load_snapshot(session_dir)
        agent_name, history_path = self._resolve_agent(
            snapshot=snapshot,
            session_dir=session_dir,
            requested_agent=request.agent_name,
        )
        try:
            history, message_timestamps = self._load_history(history_path)
        except (AgentConfigError, OSError, ValueError, ValidationError) as exc:
            raise SessionExportReadError(
                f"Failed to load session history for agent '{agent_name}' from {history_path}: {exc}"
            ) from exc
        return ResolvedSessionExport(
            session_id=snapshot.session_id,
            session_dir=session_dir,
            snapshot=snapshot,
            agent_name=agent_name,
            history_path=history_path,
            history=history,
            message_timestamps=message_timestamps,
        )

    def _resolve_session_dir(
        self,
        *,
        target: str | Path | None,
        current_session_id: str | None,
    ) -> Path:
        if isinstance(target, Path):
            return self._resolve_session_path_target(target)

        target_text = target.strip() if isinstance(target, str) else None
        if target_text:
            if target_text == "latest":
                return self._resolve_latest_session_dir()
            candidate_path = Path(target_text).expanduser()
            if candidate_path.exists():
                return self._resolve_session_path_target(candidate_path)
            return self._resolve_named_session_dir(target_text)

        if current_session_id:
            return self._resolve_named_session_dir(current_session_id)
        return self._resolve_latest_session_dir()

    def _resolve_latest_session_dir(self) -> Path:
        sessions = self._session_manager.list_sessions()
        if not sessions:
            raise SessionExportNotFoundError("No sessions found.")
        return self._resolve_named_session_dir(sessions[0].name)

    def _resolve_named_session_dir(self, target: str) -> Path:
        session_name = self._session_manager.resolve_session_name(target)
        if session_name is None:
            raise SessionExportNotFoundError(f"Session not found: {target}")
        session_dir = self._session_manager.base_dir / session_name
        metadata_path = session_dir / "session.json"
        if not session_dir.is_dir() or not metadata_path.exists():
            raise SessionExportNotFoundError(f"Session not found: {target}")
        return session_dir

    def _resolve_session_path_target(self, target_path: Path) -> Path:
        path = target_path.resolve()
        if path.is_file():
            if path.name != "session.json":
                raise InvalidSessionExportTargetError(
                    "Session export target file must be a session.json snapshot."
                )
            session_dir = path.parent
        else:
            session_dir = path
        metadata_path = session_dir / "session.json"
        if not session_dir.is_dir() or not metadata_path.exists():
            raise SessionExportNotFoundError(f"Session not found: {target_path}")
        return session_dir

    def _load_snapshot(self, session_dir: Path) -> SessionSnapshot:
        metadata_path = session_dir / "session.json"
        try:
            with metadata_path.open(encoding="utf-8") as handle:
                payload = json.load(handle)
            return load_session_snapshot(payload)
        except (OSError, ValueError, ValidationError, json.JSONDecodeError) as exc:
            raise SessionExportReadError(
                f"Failed to load session snapshot from {metadata_path}: {exc}"
            ) from exc

    def _resolve_agent(
        self,
        *,
        snapshot: SessionSnapshot,
        session_dir: Path,
        requested_agent: str | None,
    ) -> tuple[str, Path]:
        exportable: dict[str, Path] = {}
        for agent_name, agent_snapshot in snapshot.continuation.agents.items():
            history_file = agent_snapshot.history_file
            if history_file is None:
                continue
            history_path = session_dir / history_file
            if history_path.exists():
                exportable[agent_name] = history_path

        if requested_agent is not None:
            history_path = exportable.get(requested_agent)
            if history_path is None:
                raise SessionExportAgentNotFoundError(
                    f"Agent '{requested_agent}' has no exportable history in session '{snapshot.session_id}'."
                )
            return requested_agent, history_path

        if not exportable:
            raise SessionExportNoAgentsError(
                f"Session '{snapshot.session_id}' has no exportable agent histories."
            )
        if len(exportable) > 1:
            available = ", ".join(sorted(exportable))
            raise SessionExportAmbiguousAgentError(
                f"Session '{snapshot.session_id}' has multiple exportable agents: {available}. "
                "Please specify --agent."
            )

        agent_name = next(iter(exportable))
        return agent_name, exportable[agent_name]

    def _resolve_output_path(
        self,
        request: ExportRequest,
        resolved: ResolvedSessionExport,
    ) -> Path:
        workspace_dir = self._session_manager.workspace_dir.resolve()
        output_path = request.output_path
        if output_path is None:
            filename = (
                f"{resolved.session_id}__{_sanitize_filename_component(resolved.agent_name)}__"
                f"{request.format}.jsonl"
            )
            return (workspace_dir / filename).resolve()
        output_path = output_path.expanduser()
        if not output_path.is_absolute():
            output_path = (workspace_dir / output_path).resolve()
        else:
            output_path = output_path.resolve()
        return output_path

    def _load_history(
        self, history_path: Path
    ) -> tuple[list[PromptMessageExtended], tuple[datetime | None, ...]]:
        history = load_messages(str(history_path))
        model_timestamps = tuple(
            _normalize_utc(message.timestamp) if message.timestamp is not None else None
            for message in history
        )
        raw_timestamps = self._load_message_timestamps(history_path)
        if raw_timestamps is not None and len(raw_timestamps) == len(history):
            message_timestamps = tuple(
                model_timestamp or raw_timestamp
                for model_timestamp, raw_timestamp in zip(
                    model_timestamps,
                    raw_timestamps,
                    strict=True,
                )
            )
        else:
            message_timestamps = model_timestamps
        return history, message_timestamps

    def _load_message_timestamps(self, history_path: Path) -> tuple[datetime | None, ...] | None:
        if history_path.suffix.lower() != ".json":
            return None

        with history_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)

        if not isinstance(payload, dict):
            return None
        messages = payload.get("messages")
        if not isinstance(messages, list):
            return None

        timestamps: list[datetime | None] = []
        for item in messages:
            if not isinstance(item, dict):
                return None
            timestamps.append(_message_timestamp(item))
        return tuple(timestamps)

    def _writer_for_format(self, export_format: str):
        if export_format == "codex":
            return CodexTraceWriter()
        raise UnsupportedTraceExportFormatError(
            f"Unsupported session export format: {export_format}"
        )


def _message_timestamp(message: dict[object, object]) -> datetime | None:
    for key in ("timestamp", "created_at", "started_at", "completed_at"):
        value = message.get(key)
        timestamp = _parse_timestamp(value)
        if timestamp is not None:
            return timestamp
    return None


def _parse_timestamp(value: object) -> datetime | None:
    if isinstance(value, datetime):
        return _normalize_utc(value)
    if isinstance(value, str):
        normalized = value.replace("Z", "+00:00")
        try:
            return _normalize_utc(datetime.fromisoformat(normalized))
        except ValueError:
            return None
    if isinstance(value, int | float) and not isinstance(value, bool):
        return datetime.fromtimestamp(value, tz=timezone.utc)
    return None


def _normalize_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
