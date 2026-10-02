#!/usr/bin/env python3
"""Build the benchmarks page data from docs/benchmark_data.

Inputs:  benchmark_data/catalog.json (benchmarks, families, curation, claims,
         comparisons, sample runs), and per benchmark a tasks.json plus
         runs/*.json of per-trial facts (fetch_runs.py for TB2.1,
         tb4/import_leaderboard.py for TB4).
Output:  docs/javascripts/benchmark-data.js (window.faBench).

Run with: uv run --no-project python docs/generate_benchmark_data.py
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

DOCS_ROOT = Path(__file__).resolve().parent
DATA_DIR = DOCS_ROOT / "benchmark_data"
OUTPUT = DOCS_ROOT / "docs" / "javascripts" / "benchmark-data.js"


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _task_scores(cells: str, attempts: int) -> list[float]:
    return [
        sum(c == "1" for c in cells[i : i + attempts]) / attempts
        for i in range(0, len(cells), attempts)
    ]


def _mean_rates(cells_list: list[str], attempts: int) -> list[float]:
    """Mean pass rate per task across runs."""
    per_run = [_task_scores(cells, attempts) for cells in cells_list]
    return [round(sum(s[i] for s in per_run) / len(per_run), 4) for i in range(len(per_run[0]))]


def _summary(cells: str, attempts: int) -> dict[str, Any]:
    """Score and task-clustered standard error (percentage points) for a cell string."""
    scores = _task_scores(cells, attempts)
    n = len(scores)
    mean = sum(scores) / n
    var = sum((s - mean) ** 2 for s in scores) / (n - 1)
    passes = cells.count("1")
    return {
        "passes": passes,
        "slots": len(cells),
        "score": round(passes / len(cells) * 100, 2),
        "se": round(math.sqrt(var / n) * 100, 2),
    }


def _scan_summary(scan: dict[str, Any] | None, tasks: list[str]) -> dict[str, Any] | None:
    """Run-level atif-scan summary plus one code per trial, aligned with ``cells``.

    Codes: c/h/m/l = highest priority (critical, high, medium, low or none);
    upper case = the trial used a model other than the run's; ? = no scan result.
    """
    if not scan:
        return None
    findings = scan["findings_medium_plus"]
    review = scan["review"]
    return {
        "version": scan["version"],
        "trials": scan["trials"],
        "rewarded": scan["rewarded"],
        "findings": {
            "trials": findings["trials"],
            "rewarded": findings["rewarded"],
            "byPriority": findings["by_highest_priority"],
            "top": findings["top"][:6],
        },
        "review": {
            "highRewarded": review["high_or_critical_rewarded"],
            "otherModelTrials": review["other_model_trials"],
            "incompleteRewarded": review["incomplete_evidence_rewarded"],
            "scoreIfFlaggedFailed": review["accuracy_if_flagged_failed"],
        },
        "awareness": scan["awareness"],
        "evidence": scan["evidence"],
        "cost": scan["cost"],
        "walltimeHours": scan["walltime_hours"]["trial_sum"],
        "cells": "".join(scan["cells"][task] for task in tasks),
    }


def _family(catalog: dict[str, Any], model: str) -> str:
    for prefix, family in catalog["familyByModel"].items():
        if model.startswith(prefix):
            return family
    raise ValueError(f"no family for model {model!r}; add it to familyByModel")


def _tb21_runs(catalog: dict[str, Any], bench: dict[str, Any]) -> list[dict[str, Any]]:
    tasks: list[str] = _load(DATA_DIR / bench["dir"] / "tasks.json")
    bench["tasks"] = tasks
    attempts = bench["attempts"]
    pricing = catalog["pricing"]
    runs = []
    for run_id, curation in catalog["runs"].items():
        raw = _load(DATA_DIR / bench["dir"] / "runs" / f"{run_id}.json")
        if set(raw["tasks"]) != set(tasks):
            raise ValueError(f"{run_id}: task set does not match tasks.json")
        cells = "".join(raw["tasks"][task] for task in tasks)
        if len(cells) != len(tasks) * attempts:
            raise ValueError(f"{run_id}: expected {len(tasks) * attempts} cells")
        if raw.get("scan") and len("".join(raw["scan"]["cells"].values())) != len(cells):
            raise ValueError(f"{run_id}: scan cells do not align with trial cells")

        recorded = raw["cost"]["total_usd_exact"]
        published = raw["published"]["total_cost"]
        cost: dict[str, Any] = {
            "recorded": round(recorded, 2),
            "published": round(published, 2),
            "coverage": raw["cost"]["trials_with_cost"],
        }
        if curation.get("costFrom") == "published":
            cost["total"] = round(published, 2)
            cost["basis"] = "published leaderboard total"
        elif price_id := curation.get("pricing"):
            cost["total"] = round(recorded * pricing[price_id]["multiplier"], 2)
            cost["basis"] = pricing[price_id]["label"]
            cost["estimate"] = True
            cost["pricing"] = price_id
        else:
            cost["total"] = round(recorded, 2)
            cost["basis"] = "recorded Harbor cost"
            cost["estimate"] = raw["cost"]["trials_with_cost"] < raw["slots"]

        runs.append(
            {
                "id": run_id,
                "benchmark": bench["id"],
                "tier": "ours" if raw["harness"] == "fast-agent" else "leaderboard",
                "family": curation["family"],
                "harness": raw["harness"],
                "harnessVersion": curation.get("harnessVersion", raw["harness_version"]),
                "model": curation.get("model", raw["model"]),
                "effort": raw["effort"],
                "modelString": raw["model_string"],
                "date": raw["date"],
                "timeout": raw["timeout"],
                "status": curation.get("status"),
                "reconciled": raw["reconciled"],
                "cells": cells,
                **_summary(cells, attempts),
                # Published passes win over raw cells (leaderboard judge DQs are marked "x").
                "passes": raw["passes"],
                "score": round(raw["passes"] / raw["slots"] * 100, 2),
                "cost": cost,
                "perTrialCost": raw["per_trial_cost"],
                "tokens": {k: raw["tokens"][k] for k in ("input", "cached", "output")},
                "errors": raw["errors"],
                "jobs": [{"id": j["id"], "name": j["name"], "url": j["url"]} for j in raw["jobs"]],
                "excluded": raw["excluded_trials"],
                "disqualified": len(raw["disqualified_trials"]),
                "source": raw["published"]["source_url"],
                # "Reconciliation:" lines are fetch diagnostics that restate the human notes.
                "notes": [n for n in raw["notes"] if not n.startswith("Reconciliation:")],
                "scan": _scan_summary(raw.get("scan"), tasks),
            }
        )
    return runs


def _subset_runs(catalog: dict[str, Any], bench: dict[str, Any]) -> list[dict[str, Any]]:
    """Leaderboard rows cut to the subset tasks, keeping the full run alongside."""
    folder = DATA_DIR / bench["dir"]
    full_tasks: list[str] = _load(folder / "tasks.json")
    subset: list[str] = _load(folder / bench["subset"])
    bench["tasks"] = subset
    bench["full"]["tasks"] = full_tasks
    attempts = bench["attempts"]
    runs = []
    for path in sorted((folder / "runs").glob("*.json")):
        raw = _load(path)
        cells = "".join(raw["tasks"][t] for t in subset)
        full_cells = "".join(raw["tasks"][t] for t in full_tasks)
        task_costs = [raw["task_costs"][t] for t in subset]
        known = [c for c in task_costs if c is not None]
        published = raw["published"]
        notes = list(raw["notes"])
        if len(known) < len(subset):
            notes.append(
                f"{len(subset) - len(known)} subset task(s) include a trial without a recorded cost; "
                "the subset cost counts complete tasks only."
            )
        runs.append(
            {
                "id": raw["id"],
                "benchmark": bench["id"],
                "tier": "leaderboard",
                "family": _family(catalog, raw["model"]),
                "harness": raw["harness"],
                "harnessVersion": raw["harness_version"],
                "model": raw["model"],
                "effort": raw["effort"],
                "modelString": raw["model_org"] + " · " + raw["model"],
                "date": raw["date"],
                "timeout": "standard",
                "status": None,
                "reconciled": raw["reconciled"],
                "cells": cells,
                **_summary(cells, attempts),
                "cost": {
                    "total": round(sum(known), 2),
                    "basis": "recorded trial costs on the subset tasks",
                    "estimate": len(known) < len(subset),
                    "taskCosts": task_costs,
                },
                "full": {
                    "cells": full_cells,
                    **_summary(full_cells, attempts),
                    "publishedScore": published["score"],
                    "publishedPasses": published["passes"],
                    "cost": published["total_cost"],
                },
                "jobs": [{"id": j["id"], "name": j["id"], "url": j["url"]} for j in raw["jobs"]],
                "excluded": [],
                "source": published["source_url"],
                "notes": notes,
                "scan": None,
            }
        )
    return runs


def _samples(
    catalog: dict[str, Any], runs: list[dict[str, Any]], benches: dict[str, Any]
) -> list[dict[str, Any]]:
    """Synthesise clearly flagged sample runs for layout review. Never real results."""
    by_id = {r["id"]: r for r in runs}
    out = []
    for spec in catalog.get("samples", []):
        base = by_id[spec["basedOn"]]
        attempts = benches[spec["benchmark"]]["attempts"]
        rng = random.Random(spec["seed"])
        cells = ""
        for rate in _task_scores(base["cells"], attempts):
            p = min(0.97, max(0.02, rate + spec["lift"]))
            cells += "".join(
                "1" if rng.random() < p else rng.choice("000t") for _ in range(attempts)
            )
        out.append(
            {
                "id": spec["id"],
                "benchmark": spec["benchmark"],
                "tier": "ours",
                "sample": True,
                "family": _family(catalog, spec["model"]),
                "harness": spec["harness"],
                "harnessVersion": spec["harnessVersion"],
                "model": spec["model"],
                "effort": spec["effort"],
                "modelString": "sample data",
                "date": spec["date"],
                "timeout": "standard",
                "status": "Sample data",
                "reconciled": True,
                "cells": cells,
                **_summary(cells, attempts),
                "cost": {
                    "total": round(base["cost"]["total"] * spec["costFactor"], 2),
                    "basis": "synthesised",
                    "estimate": True,
                },
                "jobs": [],
                "excluded": [],
                "source": None,
                "notes": [
                    (
                        f"Sample data for layout review, synthesised from {base['harness']} · "
                        f"{base['model']} {base['effort']}. Not a real run."
                    )
                ],
                "scan": None,
            }
        )
    return out


def build() -> dict[str, Any]:
    catalog = _load(DATA_DIR / "catalog.json")
    benches = {b["id"]: b for b in catalog["benchmarks"]}
    runs: list[dict[str, Any]] = []
    for bench in catalog["benchmarks"]:
        runs += _subset_runs(catalog, bench) if "subset" in bench else _tb21_runs(catalog, bench)
    runs += _samples(catalog, runs, benches)

    for bench in catalog["benchmarks"]:
        real = [r for r in runs if r["benchmark"] == bench["id"] and not r.get("sample")]
        bench["difficulty"] = _mean_rates([r["cells"] for r in real], bench["attempts"])
        if "full" in bench:
            bench["full"]["difficulty"] = _mean_rates(
                [r["full"]["cells"] for r in real], bench["attempts"]
            )
        del bench["dir"]
        bench.pop("subset", None)

    pricing = catalog["pricing"]
    default_bench = catalog["benchmarks"][0]["id"]
    claims = []
    for claim in catalog["claims"]:
        bench = benches[claim.get("benchmark", default_bench)]
        per_task = claim["costPerTask"]
        if price_id := claim.get("pricing"):
            per_task *= pricing[price_id]["multiplier"]
        claims.append(
            {
                **claim,
                "benchmark": bench["id"],
                "tier": "claim",
                "cost": {
                    "total": round(per_task * len(bench["tasks"]) * bench["attempts"], 2),
                    "perTask": round(per_task, 4),
                    "original": claim["costPerTask"],
                    "estimate": bool(claim.get("pricing")),
                },
            }
        )

    return {
        "updated": catalog["updated"],
        "benchmarks": catalog["benchmarks"],
        "families": catalog["families"],
        "pricing": pricing,
        "runs": runs,
        "claims": claims,
        "comparisons": [
            {**c, "benchmark": c.get("benchmark", default_bench)} for c in catalog["comparisons"]
        ],
    }


def main() -> None:
    data = build()
    body = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    OUTPUT.write_text(
        "/* Generated by docs/generate_benchmark_data.py from docs/benchmark_data. Do not edit. */\n"
        f"window.faBench = {body};\n",
        encoding="utf-8",
    )
    counts = {
        b["id"]: sum(r["benchmark"] == b["id"] for r in data["runs"]) for b in data["benchmarks"]
    }
    print(f"wrote {OUTPUT.relative_to(DOCS_ROOT)} ({len(body) // 1024} KB, runs: {counts})")


if __name__ == "__main__":
    main()
