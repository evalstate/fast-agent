"""Resume-state helpers for batch runs."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

CanonicalBatchId = str | int


def canonical_batch_id(value: Any, *, source: str) -> CanonicalBatchId:
    """Validate an output/resume ID without collapsing distinct JSON values."""
    if isinstance(value, bool) or value is None:
        raise ValueError(f"{source} must be a non-empty string or integer")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        if value == "":
            raise ValueError(f"{source} must be a non-empty string or integer")
        return value
    raise ValueError(f"{source} must be a non-empty string or integer")


def load_completed_ids(path: Path) -> set[CanonicalBatchId]:
    """Load IDs for existing successful output records."""
    completed: set[CanonicalBatchId] = set()
    if not path.exists():
        return completed

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL in existing output at line {line_number}: {exc}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(f"Invalid existing output at line {line_number}: expected object")
            if record.get("ok") is True and "id" in record:
                completed.add(
                    canonical_batch_id(
                        record["id"],
                        source=f"Existing output id at line {line_number}",
                    )
                )
    return completed
