"""Resume-state helpers for batch runs."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def load_completed_ids(path: Path) -> set[str]:
    """Load IDs for existing successful output records."""
    completed: set[str] = set()
    if not path.exists():
        return completed

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL in existing output at line {line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Invalid existing output at line {line_number}: expected object")
            if record.get("ok") is True and "id" in record:
                completed.add(str(record["id"]))
    return completed
