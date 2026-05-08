"""Input row loading and selection for batch runs."""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class RowError:
    type: str
    message: str


@dataclass(frozen=True)
class RowCandidate:
    row_number: int
    row: dict[str, Any] | None
    error: RowError | None = None


def iter_jsonl_rows(path: Path) -> Iterable[RowCandidate]:
    """Yield JSON object rows, preserving invalid lines as row-error candidates."""
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                yield RowCandidate(
                    row_number=line_number,
                    row=None,
                    error=RowError("InvalidJSON", f"Line {line_number}: {exc.msg}"),
                )
                continue

            if not isinstance(payload, dict):
                yield RowCandidate(
                    row_number=line_number,
                    row=None,
                    error=RowError(
                        "InvalidRow",
                        f"Line {line_number}: expected a JSON object, got {type(payload).__name__}",
                    ),
                )
                continue

            yield RowCandidate(row_number=line_number, row=payload)


def iter_csv_rows(path: Path) -> Iterable[RowCandidate]:
    """Yield CSV rows as dictionaries keyed by header name."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row_number, row in enumerate(reader, start=1):
            yield RowCandidate(row_number=row_number, row=dict(row))


def iter_input_rows(path: Path) -> Iterable[RowCandidate]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return iter_jsonl_rows(path)
    if suffix == ".csv":
        return iter_csv_rows(path)
    raise ValueError(f"Unsupported input format for {path}; expected .jsonl or .csv")


def select_rows(
    rows: Iterable[RowCandidate],
    *,
    offset: int | None = None,
    sample: int | None = None,
    seed: int | None = None,
    limit: int | None = None,
) -> list[RowCandidate]:
    """Apply offset, deterministic sample, input-order restoration, and limit."""
    candidates = list(rows)
    if offset is not None and offset > 0:
        candidates = candidates[offset:]

    if sample is not None:
        if sample < len(candidates):
            rng = random.Random(0 if seed is None else seed)
            indexed = list(enumerate(candidates))
            sampled = rng.sample(indexed, sample)
            candidates = [candidate for _, candidate in sorted(sampled, key=lambda item: item[0])]

    if limit is not None:
        candidates = candidates[:limit]

    return candidates
