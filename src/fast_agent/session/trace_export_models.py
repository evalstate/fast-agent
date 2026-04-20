"""Shared models for session trace export."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from datetime import datetime
    from pathlib import Path

    from fast_agent.mcp.prompt_message_extended import PromptMessageExtended
    from fast_agent.session.snapshot import SessionSnapshot

ExportFormat = Literal["codex"]


@dataclass(frozen=True, slots=True)
class ExportRequest:
    """Typed input for session trace export."""

    target: str | Path | None
    agent_name: str | None
    output_path: Path | None
    hf_dataset: str | None = None
    hf_dataset_path: str | None = None
    format: str = "codex"
    current_session_id: str | None = None


@dataclass(frozen=True, slots=True)
class DatasetUploadResult:
    """Remote dataset upload details for an exported trace."""

    repo_id: str
    path_in_repo: str
    commit_url: str
    file_url: str


@dataclass(frozen=True, slots=True)
class ResolvedSessionExport:
    """Resolved session export payload ready for a format writer."""

    session_id: str
    session_dir: Path
    snapshot: SessionSnapshot
    agent_name: str
    history_path: Path
    history: list[PromptMessageExtended]
    message_timestamps: tuple[datetime | None, ...]


@dataclass(frozen=True, slots=True)
class ExportResult:
    """Result from a completed session trace export."""

    session_id: str
    agent_name: str
    format: ExportFormat
    output_path: Path
    record_count: int
    upload: DatasetUploadResult | None = None
