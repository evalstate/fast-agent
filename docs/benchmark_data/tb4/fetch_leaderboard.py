#!/usr/bin/env python3
"""Fetch the Terminal-Bench 4.0 leaderboard and every trial of every row.

Writes to .cache/ (git-ignored) next to this file:

    leaderboard.json         the leaderboard and its rows, as published now (+ "fetched" date)
    rows/<row_id>.json       {row, jobs, trial_ids, trials}: every trial of the row
    jobs/<job_id>.json       trial listings, cached per job
    digests.json             [[row_id, task, digest], ...] for the subset tasks

Rows and digests already in the cache are kept, so a re-run only fetches new rows.
Needs the `harbor` CLI. Then run import_leaderboard.py and calibration.py.

    uv run --no-project python docs/benchmark_data/tb4/fetch_leaderboard.py
"""

from __future__ import annotations

import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
CACHE = HERE / ".cache"
LEADERBOARD_ID = "9f966760-00f1-424e-90f5-c964fb6f6091"


def harbor(*args: str) -> Any:
    out = subprocess.run(
        ["harbor", "hub", *args, "--json"], capture_output=True, text=True, check=True
    ).stdout
    return json.loads(out)


def paged(*args: str, limit: int = 1000) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    page = 1
    while True:
        result = harbor(*args, "--limit", str(limit), "--page", str(page))
        items += result["items"]
        total = result.get("total_pages") or (page + 1 if len(result["items"]) == limit else 1)
        if page >= total or not result["items"]:
            return items
        page += 1


def job_trials(job: str) -> list[dict[str, Any]]:
    path = CACHE / "jobs" / f"{job}.json"
    if path.exists():
        return json.loads(path.read_text())
    items = paged("job", "trials", job, "--include-retries")
    path.write_text(json.dumps(items))
    return items


def fetch_row(row: dict[str, Any]) -> str:
    """Resolve the row's trial ids to full trial records via the jobs they belong to."""
    path = CACHE / "rows" / f"{row['id']}.json"
    if path.exists():
        return row["id"]
    ids = [t["trial_id"] for t in paged("leaderboard", "row", "trial", "list", row["id"])]
    found: dict[str, dict[str, Any]] = {}
    jobs: list[str] = []
    missing = list(ids)
    while missing:
        job = harbor("trial", "show", missing[0])["job_id"]
        if job in jobs:
            raise RuntimeError(f"trial {missing[0]} not listed in its own job {job}")
        jobs.append(job)
        found.update({t["id"]: t for t in job_trials(job)})
        missing = [i for i in ids if i not in found]
    record = {"row": row, "jobs": jobs, "trial_ids": ids, "trials": [found[i] for i in ids]}
    path.write_text(json.dumps(record))
    return row["id"]


def digest(item: tuple[str, str, str]) -> list[str]:
    row_id, task, trial = item
    lock = harbor("trial", "show", trial)["lock"]
    return [row_id, task, lock["task"]["digest"].removeprefix("sha256:")]


def main() -> None:
    (CACHE / "rows").mkdir(parents=True, exist_ok=True)
    (CACHE / "jobs").mkdir(exist_ok=True)
    board = harbor("leaderboard", "show", LEADERBOARD_ID)
    board["fetched"] = date.today().isoformat()
    (CACHE / "leaderboard.json").write_text(json.dumps(board, indent=1))
    with ThreadPoolExecutor(8) as pool:
        for row_id in pool.map(fetch_row, board["rows"]):
            print("row", row_id, flush=True)

    # One trial per (row, subset task) is enough to see which task revision the row ran.
    subset: list[str] = json.loads((HERE / "subset.json").read_text())
    digests_path = CACHE / "digests.json"
    seen: list[list[str]] = json.loads(digests_path.read_text()) if digests_path.exists() else []
    have = {(r, t) for r, t, _ in seen}
    todo: list[tuple[str, str, str]] = []
    for path in sorted((CACHE / "rows").glob("*.json")):
        export = json.loads(path.read_text())
        row_id = export["row"]["id"]
        for trial in export["trials"]:
            task = trial["task_name"].split("/")[-1]
            if task in subset and (row_id, task) not in have:
                have.add((row_id, task))
                todo.append((row_id, task, trial["id"]))
    with ThreadPoolExecutor(8) as pool:
        seen += list(pool.map(digest, todo))
    digests_path.write_text(json.dumps(seen))
    print(f"{len(board['rows'])} rows; {len(todo)} new digests")


if __name__ == "__main__":
    main()
