"""Signal Store — file-based completion signals for agent coordination.

Each agent writes a signal file when it completes/errors/times out.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CompletionSignal:
    """Signal written when an agent completes."""

    run_id: str
    role: str
    status: str  # completed | error | timeout | killed
    result_summary: str = ""
    error: str = ""
    output_files: list[str] = field(default_factory=list)
    completed_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CompletionSignal:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in known})


class SignalStore:
    """File-based signal store for agent completion tracking."""

    def __init__(self, signals_dir: str | Path) -> None:
        self._dir = Path(signals_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    @property
    def signals_dir(self) -> Path:
        return self._dir

    def _signal_path(self, role: str, run_id: str, status: str) -> Path:
        short_id = run_id[:8] if len(run_id) > 8 else run_id
        return self._dir / f"{role}_{short_id}.{status}.json"

    def write_signal(self, signal: CompletionSignal) -> Path:
        """Write a completion signal to disk (atomic via rename)."""
        path = self._signal_path(signal.role, signal.run_id, signal.status)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(signal.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        tmp.rename(path)
        logger.info("Signal written: %s", path.name)
        return path

    def read_signal(self, role: str, run_id: str) -> CompletionSignal | None:
        """Read a completion signal for a specific role and run_id."""
        short_id = run_id[:8] if len(run_id) > 8 else run_id
        prefix = f"{role}_{short_id}."
        for f in self._dir.iterdir():
            if f.name.startswith(prefix) and f.suffix == ".json":
                try:
                    data = json.loads(f.read_text(encoding="utf-8"))
                    return CompletionSignal.from_dict(data)
                except (json.JSONDecodeError, KeyError):
                    continue
        return None

    def list_signals(self, status: str | None = None) -> list[CompletionSignal]:
        """List all signals, optionally filtered by status."""
        signals: list[CompletionSignal] = []
        for f in sorted(self._dir.glob("*.json")):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                sig = CompletionSignal.from_dict(data)
                if status is None or sig.status == status:
                    signals.append(sig)
            except (json.JSONDecodeError, KeyError):
                continue
        return signals

    def get_completed_roles(self) -> set[str]:
        return {s.role for s in self.list_signals() if s.status == "completed"}

    def get_failed_roles(self) -> dict[str, str]:
        return {
            s.role: s.error
            for s in self.list_signals()
            if s.status in ("error", "timeout", "killed")
        }

    def is_role_done(self, role: str) -> bool:
        prefix = f"{role}_"
        for f in self._dir.iterdir():
            if f.name.startswith(prefix) and f.suffix == ".json":
                return True
        return False

    def wait_for_role(
        self,
        role: str,
        timeout_seconds: float = 300,
        poll_interval: float = 2.0,
    ) -> CompletionSignal | None:
        """Poll for a role's completion signal. Returns None on timeout."""
        start = time.time()
        while time.time() - start < timeout_seconds:
            for f in self._dir.iterdir():
                if f.name.startswith(f"{role}_") and f.suffix == ".json":
                    try:
                        data = json.loads(f.read_text(encoding="utf-8"))
                        return CompletionSignal.from_dict(data)
                    except (json.JSONDecodeError, KeyError):
                        continue
            time.sleep(poll_interval)
        return None

    def clear(self) -> int:
        """Delete all signal files. Returns count deleted."""
        count = 0
        for f in self._dir.glob("*.json"):
            f.unlink()
            count += 1
        return count
