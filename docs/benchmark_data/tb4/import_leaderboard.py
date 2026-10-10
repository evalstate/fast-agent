#!/usr/bin/env python3
"""Import Terminal-Bench 4.0 leaderboard rows as comparator runs.

Reads the per-row trial exports written by fetch_leaderboard.py (one JSON per
leaderboard row with its 330 trials) and writes one compact run file per row to
runs/<id>.json. Every row keeps its full 66-task result; the page derives the
19-task subset from subset.json.

    uv run --no-project python docs/benchmark_data/tb4/import_leaderboard.py [--source DIR]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
LEADERBOARD_URL = (
    "https://hub.harborframework.com/datasets/terminal-bench/terminal-bench/leaderboards/4-0-0"
)


def cell(trial: dict[str, Any]) -> str:
    """Same codes as TB2.1: 1 pass, 0 fail, t timeout, e other error, - missing."""
    if (trial.get("reward") or 0) > 0:
        return "1"
    error = trial.get("error_type")
    if error == "AgentTimeoutError":
        return "t"
    if error:
        return "e"
    return "0" if trial.get("reward") is not None else "-"


def slug(*parts: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", "-".join(parts).lower()).strip("-")


def convert(export: dict[str, Any], tasks: list[str]) -> dict[str, Any]:
    row = export["row"]
    meta = row["metadata"]
    metrics = row["metrics"]
    by_task: dict[str, list[dict[str, Any]]] = {t: [] for t in tasks}
    for trial in export["trials"]:
        by_task[trial["task_name"].split("/")[-1]].append(trial)
    cells: dict[str, str] = {}
    costs: dict[str, float | None] = {}
    for task, trials in by_task.items():
        trials.sort(key=lambda t: t.get("started_at") or "")
        cells[task] = "".join(cell(t) for t in trials).ljust(5, "-")[:5]
        known = [t["cost_usd"] for t in trials if t.get("cost_usd") is not None]
        costs[task] = round(sum(known), 6) if len(known) == len(trials) else None

    recorded_passes = sum(c == "1" for c in "".join(cells.values()))
    harness = meta["agent_display"]["label"]
    model = meta["model_display"]["label"]
    effort = meta["reasoning_effort"]
    notes = []
    if recorded_passes != metrics["successes"]:
        notes.append(
            f"Trial records show {recorded_passes} passes; the leaderboard publishes "
            f"{metrics['successes']}. Cells are shown as recorded."
        )
    return {
        "id": slug("tb4", harness, model, effort),
        "benchmark": "terminal-bench-4.0",
        "harness": harness,
        "harness_version": "",
        "model": model,
        "effort": effort,
        "model_org": meta["model_org"]["label"],
        "date": meta["date"],
        "listed": row["created_at"][:10],
        "leaderboard_row": row["id"],
        "jobs": [
            {"id": j, "url": f"https://hub.harborframework.com/jobs/{j}"} for j in export["jobs"]
        ],
        "published": {
            "score": metrics["accuracy"],
            "passes": metrics["successes"],
            "total_cost": metrics.get("total_cost_usd"),
            "ci95": metrics.get("accuracy_ci95_half_width"),
            "source_url": f"{LEADERBOARD_URL}/rows/{row['id']}",
        },
        "reconciled": recorded_passes == metrics["successes"],
        "tasks": cells,
        "task_costs": costs,
        "notes": notes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--source",
        type=Path,
        default=HERE / ".cache",
    )
    args = parser.parse_args()
    tasks: list[str] = json.loads((HERE / "tasks.json").read_text())
    out = HERE / "runs"
    out.mkdir(exist_ok=True)
    for path in sorted((args.source / "rows").glob("*.json")):
        run = convert(json.loads(path.read_text()), tasks)
        (out / f"{run['id']}.json").write_text(json.dumps(run, indent=1) + "\n")
        print(run["id"], run["published"]["score"], "" if run["reconciled"] else "(not reconciled)")


if __name__ == "__main__":
    main()
