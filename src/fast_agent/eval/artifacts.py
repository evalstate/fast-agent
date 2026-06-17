"""Candidate run and artifact helpers for evaluator loops."""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


def _utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class CommandResult:
    command: list[str]
    cwd: Path | None
    returncode: int
    timed_out: bool
    duration_seconds: float
    stdout_path: Path
    stderr_path: Path

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out

    def to_dict(self) -> dict[str, Any]:
        return {
            "command": self.command,
            "cwd": str(self.cwd) if self.cwd is not None else None,
            "returncode": self.returncode,
            "timed_out": self.timed_out,
            "duration_seconds": self.duration_seconds,
            "stdout": str(self.stdout_path),
            "stderr": str(self.stderr_path),
        }


@dataclass(frozen=True)
class EvalScore:
    score: float
    side_info: Any
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "side_info": self.side_info,
            "metadata": self.metadata,
        }


class CandidateRun:
    """Inspectable directory for one candidate evaluation."""

    def __init__(self, path: str | Path, *, index: int | None = None):
        self.path = Path(path)
        self.index = index
        self.artifacts_dir = self.path / "artifacts"
        self.reports_dir = self.path / "reports"
        self.logs_dir = self.path / "logs"
        self.prompts_dir = self.path / "prompts"

    def create(self) -> "CandidateRun":
        for path in (
            self.path,
            self.artifacts_dir,
            self.reports_dir,
            self.logs_dir,
            self.prompts_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
        return self

    def write_json(self, relative_path: str | Path, payload: Mapping[str, Any]) -> Path:
        path = self.path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n", "utf-8")
        return path

    def write_text(self, relative_path: str | Path, text: str) -> Path:
        path = self.path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def materialize_candidate(
        self,
        candidate: Mapping[str, Any],
        *,
        variables: Mapping[str, str] | None = None,
    ) -> None:
        self.write_json("candidate.json", candidate)
        if variables is not None:
            self.write_json("variables.json", variables)

    def copy_tree(self, source: str | Path, relative_dest: str | Path) -> Path:
        destination = self.path / relative_dest
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(source, destination)
        return destination

    def run_command(
        self,
        command: Sequence[str],
        *,
        cwd: str | Path | None = None,
        timeout_seconds: float | None = None,
        log_prefix: str = "command",
        env: Mapping[str, str] | None = None,
    ) -> CommandResult:
        command_list = list(command)
        started = time.monotonic()
        stdout_path = self.logs_dir / f"{log_prefix}.stdout.txt"
        stderr_path = self.logs_dir / f"{log_prefix}.stderr.txt"
        command_path = self.logs_dir / f"{log_prefix}.json"
        cwd_path = Path(cwd) if cwd is not None else None
        timed_out = False
        returncode = 0
        try:
            completed = subprocess.run(
                command_list,
                cwd=cwd_path,
                env=dict(env) if env is not None else None,
                timeout=timeout_seconds,
                capture_output=True,
                text=True,
                check=False,
            )
            stdout = completed.stdout
            stderr = completed.stderr
            returncode = completed.returncode
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            returncode = -1
            stdout = _expired_output_to_text(exc.stdout)
            stderr = _expired_output_to_text(exc.stderr)
        duration = time.monotonic() - started
        stdout_path.write_text(stdout, encoding="utf-8")
        stderr_path.write_text(stderr, encoding="utf-8")
        result = CommandResult(
            command=command_list,
            cwd=cwd_path,
            returncode=returncode,
            timed_out=timed_out,
            duration_seconds=duration,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )
        command_path.write_text(
            json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return result

    def write_score(
        self,
        score: float,
        side_info: Any,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> EvalScore:
        eval_score = EvalScore(score=score, side_info=side_info, metadata=dict(metadata or {}))
        self.write_json("score.json", eval_score.to_dict())
        return eval_score


class EvalRun:
    """Allocator for candidate artifact directories."""

    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.run_dir / "run.json"
        if not self.metadata_path.exists():
            self.metadata_path.write_text(
                json.dumps({"created_at": _utc_now_iso()}, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

    def candidate(self, index: int | None = None) -> CandidateRun:
        candidate_index = index if index is not None else self._next_index()
        return CandidateRun(
            self.run_dir / f"candidate-{candidate_index:04d}",
            index=candidate_index,
        ).create()

    def _next_index(self) -> int:
        existing = [
            int(path.name.removeprefix("candidate-"))
            for path in self.run_dir.glob("candidate-[0-9][0-9][0-9][0-9]")
            if path.name.removeprefix("candidate-").isdigit()
        ]
        return max(existing, default=0) + 1


ArtifactRun = EvalRun


def _expired_output_to_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value
