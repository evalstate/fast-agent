"""Trajectory persistence for stateless agent invocations.

Phase 1 stores one self-contained JSON file per invocation. This is intentionally
separate from session history: histories are conversational snapshots for resume,
while trajectories are full-fidelity execution/audit records for agents that keep
``use_history=False``.

The long-term direction is an append-only event stream. The per-invocation file
format keeps the first implementation simple and naturally safe for parallel
sub-agent calls; event streams can later become the canonical source and derive
these files as views.
"""

from __future__ import annotations

import json
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fast_agent.utils.async_utils import run_in_thread
from fast_agent.utils.filename import sanitize_filename_component

if TYPE_CHECKING:
    from fast_agent.session.session_manager import Session
    from fast_agent.types import PromptMessageExtended

TRAJECTORY_SCHEMA_VERSION = 1
TRAJECTORIES_DIR = "trajectories"


@dataclass(frozen=True, slots=True)
class TrajectoryRecord:
    """Replay-oriented record for one stateless agent invocation."""

    trajectory_id: str
    session_id: str
    parent_agent_name: str | None
    agent_name: str
    template_agent_name: str | None
    tool_name: str | None
    parent_tool_call_id: str | None
    use_history: bool
    started_at: str
    completed_at: str
    tool_input_schema: dict[str, Any] | None
    tool_arguments: dict[str, Any] | None
    effective_tool_arguments: dict[str, Any] | None
    rendered_child_input: str | None
    messages: list["PromptMessageExtended"]
    usage_summary: dict[str, Any] | None = None
    schema_version: int = TRAJECTORY_SCHEMA_VERSION

    def model_dump(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "trajectory_id": self.trajectory_id,
            "session_id": self.session_id,
            "parent_agent_name": self.parent_agent_name,
            "agent_name": self.agent_name,
            "template_agent_name": self.template_agent_name,
            "tool_name": self.tool_name,
            "parent_tool_call_id": self.parent_tool_call_id,
            "use_history": self.use_history,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "tool_input_schema": self.tool_input_schema,
            "tool_arguments": self.tool_arguments,
            "effective_tool_arguments": self.effective_tool_arguments,
            "rendered_child_input": self.rendered_child_input,
            "usage_summary": self.usage_summary,
            "messages": [
                message.model_dump(mode="json", exclude_none=True) for message in self.messages
            ],
        }


def new_trajectory_id() -> str:
    return f"traj_{uuid.uuid4().hex}"


async def save_trajectory_record(session: "Session", record: TrajectoryRecord) -> Path:
    """Atomically save a trajectory record in ``<session>/trajectories``."""

    return await run_in_thread(_save_trajectory_record_sync, session, record)


def _save_trajectory_record_sync(session: "Session", record: TrajectoryRecord) -> Path:
    directory = session.directory / TRAJECTORIES_DIR
    directory.mkdir(parents=True, exist_ok=True)
    path = _trajectory_path(directory, record)
    payload = record.model_dump()

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=directory,
        prefix=f".{path.stem}.",
        suffix=".tmp",
        delete=False,
    ) as tmp:
        tmp_path = Path(tmp.name)
        json.dump(payload, tmp, ensure_ascii=False, indent=2)
        tmp.write("\n")

    tmp_path.replace(path)
    return path


def _trajectory_path(directory: Path, record: TrajectoryRecord) -> Path:
    label_parts = [
        record.parent_tool_call_id or record.trajectory_id,
        record.agent_name,
    ]
    label = "__".join(part for part in label_parts if part)
    filename = sanitize_filename_component(label, fallback=record.trajectory_id)
    candidate = directory / f"{filename}.json"
    if not candidate.exists():
        return candidate

    suffix = sanitize_filename_component(record.trajectory_id, fallback=uuid.uuid4().hex)
    return directory / f"{filename}__{suffix}.json"
