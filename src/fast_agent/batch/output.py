"""JSONL output envelope helpers for batch runs."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, TextIO

if TYPE_CHECKING:
    from pathlib import Path

    from fast_agent.batch.input import RowError


def success_envelope(
    *,
    identity: str | int,
    row_number: int,
    result: Any,
    row: dict[str, Any] | None,
    include_input: bool,
) -> dict[str, Any]:
    envelope: dict[str, Any] = {
        "id": identity,
        "row_number": row_number,
        "ok": True,
        "result": result,
        "error": None,
    }
    if include_input:
        envelope["input"] = row
    return envelope


def error_envelope(
    *,
    identity: str | int,
    row_number: int,
    error: RowError,
    row: dict[str, Any] | None,
    include_input: bool,
) -> dict[str, Any]:
    envelope: dict[str, Any] = {
        "id": identity,
        "row_number": row_number,
        "ok": False,
        "result": None,
        "error": {
            "type": error.type,
            "message": error.message,
        },
    }
    if include_input:
        envelope["input"] = row
    return envelope


def ensure_parent(path: Path) -> None:
    parent = path.parent
    if str(parent):
        parent.mkdir(parents=True, exist_ok=True)


def write_jsonl_record(handle: TextIO, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    handle.flush()


def extract_structured_output(row: dict[str, Any]) -> Any | None:
    """Return the normalized structured result from a batch output envelope."""
    result = row.get("result")
    if isinstance(result, dict) and "parsed" in result:
        return result["parsed"]
    return result


def extract_text_output(row: dict[str, Any]) -> str:
    """Return the best text representation from a batch output envelope."""
    result = row.get("result")
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        for key in ("text", "output", "raw", "raw_text"):
            value = result.get(key)
            if isinstance(value, str):
                return value
    value = row.get("text")
    return value if isinstance(value, str) else ""
