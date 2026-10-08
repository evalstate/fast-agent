#!/usr/bin/env python3
"""Build the benchmarks page data from docs/benchmark_data.

Inputs:  benchmark_data/catalog.json (benchmarks, families, curation, claims,
         comparisons, sample runs), and per benchmark a tasks.json plus
         runs/*.json of per-trial facts (fetch_runs.py for TB2.1,
         tb4/import_leaderboard.py for TB4).
Output:  docs/javascripts/benchmark-data.js (window.faBench), and the build-time
         ledger per benchmark in docs/_generated/benchmarks/ (benchmark_ledger.py).

Run with: uv run --no-project python docs/generate_benchmark_data.py
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

from benchmark_ledger import ledger

DOCS_ROOT = Path(__file__).resolve().parent
DATA_DIR = DOCS_ROOT / "benchmark_data"
OUTPUT = DOCS_ROOT / "docs" / "javascripts" / "benchmark-data.js"
LEDGER_DIR = DOCS_ROOT / "docs" / "_generated" / "benchmarks"
# Site root relative to the page that includes the ledgers (benchmarks/ledger-prototype/).
LEDGER_ROOT = "../../"


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


def _scan_level(scan: dict[str, Any]) -> dict[str, Any]:
    """How deep the scan went, from the flags it ran with (``scan.scanner.args``).

    basic: atif-scan's deterministic detectors only, no model calls. + images: images
    that blocked a check were transcribed by a model. full: the LLM trace questions
    (``atif-scan hunt``) were answered and applied (``--answers``). Scans from before
    the scanner was recorded are basic.
    """
    args = (scan.get("scanner") or {}).get("args") or []

    def value(flag: str) -> str | None:
        return (
            args[args.index(flag) + 1]
            if flag in args and args.index(flag) + 1 < len(args)
            else None
        )

    image_model = value("--image-model")
    if "--answers" in args:
        name, label = "full", "full"
    elif image_model:
        name, label = "images", "basic + images"
    else:
        name, label = "basic", "basic"
    return {"name": name, "label": label, "imageModel": image_model}


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
        "scope": scan.get("scope"),
        "level": _scan_level(scan),
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


def _review(raw: dict[str, Any]) -> dict[str, Any] | None:
    """Our own review outcome: recorded vs reviewed score and the disqualified trials."""
    review = raw.get("review")
    if not review:
        return None

    def tally(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        counts: dict[str, int] = {}
        for item in items:
            counts[item["reason"]] = counts.get(item["reason"], 0) + 1
        return [{"reason": r, "trials": n} for r, n in counts.items()]

    return {
        "note": review.get("note"),
        "coverage": review.get("coverage"),
        "recordedPasses": review["recorded_passes"],
        "recordedScore": review["recorded_score"],
        "disqualified": len(review["disqualified"]),
        "reasons": tally(review["disqualified"]),
        "cleared": len(review.get("cleared", [])),
        "clearedReasons": tally(review.get("cleared", [])),
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
        safety = _safety_cells(run_id, raw, tasks, attempts, cells)

        recorded = raw["cost"]["total_usd_exact"]
        published = (raw.get("published") or {}).get("total_cost")
        cost: dict[str, Any] = {
            "recorded": round(recorded, 2),
            "published": round(published, 2) if published is not None else None,
            "coverage": raw["cost"]["trials_with_cost"],
        }
        if raw["cost"].get("computed"):
            # No cost recorded at run time: computed from tokens at stated rates.
            pricing_doc = raw["cost"]["pricing"]
            cost["total"] = round(recorded, 2)
            cost["basis"] = "computed from recorded tokens"
            cost["estimate"] = True
            cost["lowerBound"] = raw["cost"].get("lower_bound", False)
            cost["computed"] = {
                "rates": pricing_doc["rates_per_mtok"],
                "source": pricing_doc.get("source"),
                "note": pricing_doc.get("note"),
                "lowerBoundNote": pricing_doc.get("lower_bound_note"),
                "usageIncomplete": raw["cost"].get("usage_incomplete_trials", []),
            }
        elif curation.get("costFrom") == "published":
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
            cost["lowerBound"] = raw["cost"].get("lower_bound", False)
            cost["estimate"] = cost["lowerBound"] or raw["cost"]["trials_with_cost"] < raw["slots"]

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
                "source": (raw.get("published") or {}).get("source_url"),
                "review": _review(raw),
                "asRun": raw.get("as_run"),
                # "Reconciliation:" lines are fetch diagnostics that restate the human notes.
                "notes": [n for n in raw["notes"] if not n.startswith("Reconciliation:")],
                "scan": _scan_summary(raw.get("scan"), tasks),
                "safety": {"refused": safety} if safety else None,
            }
        )
    return runs


SAFETY_ERRORS = ("AgentSafetyRefusalError", "AgentSafetyStopError")


def _safety_cells(
    run_id: str, raw: dict[str, Any], tasks: list[str], attempts: int, cells: str
) -> list[int]:
    """Cell indices of unrewarded trials a provider safety stop or refusal ended."""
    if "safety_cells" not in raw:
        if any(e in raw["errors"] for e in SAFETY_ERRORS):
            raise ValueError(f"{run_id}: safety errors but no safety_cells; rerun fetch_runs.py")
        return []
    out = [
        tasks.index(task) * attempts + i
        for task, idx in raw["safety_cells"].items()
        for i in idx
    ]
    if any(cells[i] in "1x" for i in out):
        raise ValueError(f"{run_id}: a safety cell is a pass")
    return sorted(out)


def _safety_allowance(runs: list[dict[str, Any]], bench: dict[str, Any]) -> dict[str, Any] | None:
    """The safety-allowance scenario (safety-allowance.json) for one benchmark.

    Each run has one reference model: the first reference listing the run's family,
    else the default. A task is easy for that run when the reference passed at least
    ``minPasses`` attempts (a site run's "1" cells, after review, or an external per-task
    pass table). Each run gets the refused cells on its easy tasks as
    ``safety.allowance``; the page counts them only when the scenario is switched on.
    """
    path = DATA_DIR / "safety-allowance.json"
    if not path.exists():
        return None
    cfg = _load(path)
    if cfg["benchmark"] != bench["id"]:
        return None
    tasks, attempts, need = bench["tasks"], bench["attempts"], cfg["minPasses"]
    by_id = {r["id"]: r for r in runs}
    refs = []
    for ref in cfg["references"]:
        if "runs" in ref:
            missing = [rid for rid in ref["runs"] if rid not in by_id]
            if missing:
                raise ValueError(f"safety allowance: reference runs not on the page: {missing}")
            passes = [
                sum(
                    by_id[rid]["cells"][t * attempts : (t + 1) * attempts].count("1")
                    for rid in ref["runs"]
                )
                for t in range(len(tasks))
            ]
            total = attempts * len(ref["runs"])
        else:
            if set(ref["passes"]) != set(tasks):
                raise ValueError(f"safety allowance: {ref['id']} does not cover the task list")
            passes = [ref["passes"][t] for t in tasks]
            total = ref["attempts"]
        if total != attempts:
            raise ValueError(f"safety allowance: {ref['id']} must have {attempts} attempts per task")
        refs.append(
            {
                "id": ref["id"],
                "label": ref["label"],
                "families": ref.get("families", []),
                "default": bool(ref.get("default")),
                "note": ref.get("note"),
                "source": ref.get("source"),
                "runs": ref.get("runs"),
                "attempts": total,
                "passes": passes,
                "easy": [t for t in range(len(tasks)) if passes[t] >= need],
            }
        )
    defaults = [r for r in refs if r["default"]]
    if len(defaults) != 1:
        raise ValueError("safety allowance: exactly one reference must be the default")
    for run in runs:
        if run["benchmark"] != bench["id"] or not run.get("safety"):
            continue
        ref = next((r for r in refs if run["family"] in r["families"]), defaults[0])
        easy = set(ref["easy"])
        run["safety"]["reference"] = ref["id"]
        run["safety"]["allowance"] = [
            i for i in run["safety"]["refused"] if i // attempts in easy
        ]
    return {
        "label": cfg["label"],
        "rule": cfg["rule"],
        "minPasses": need,
        "errorTypes": cfg["errorTypes"],
        "references": refs,
    }


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
    # Announced benchmarks with no data yet carry no "dir"; they render as "Coming soon".
    measured = [b for b in catalog["benchmarks"] if "dir" in b]
    runs: list[dict[str, Any]] = []
    for bench in measured:
        runs += _subset_runs(catalog, bench) if "subset" in bench else _tb21_runs(catalog, bench)
    runs += _samples(catalog, runs, benches)

    for bench in measured:
        real = [r for r in runs if r["benchmark"] == bench["id"] and not r.get("sample")]
        bench["difficulty"] = _mean_rates([r["cells"] for r in real], bench["attempts"])
        if "full" in bench:
            bench["full"]["difficulty"] = _mean_rates(
                [r["full"]["cells"] for r in real], bench["attempts"]
            )
        del bench["dir"]
        bench.pop("subset", None)
        if allowance := _safety_allowance(real, bench):
            bench["safetyAllowance"] = allowance

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

    LEDGER_DIR.mkdir(parents=True, exist_ok=True)
    for bench in (b for b in data["benchmarks"] if "tasks" in b and not b.get("comingSoon")):
        path = LEDGER_DIR / f"ledger-{bench['slug']}.html"
        path.write_text(
            "<!-- Generated by docs/generate_benchmark_data.py. Do not edit. -->\n"
            f"{ledger(data, bench['id'], LEDGER_ROOT)}\n",
            encoding="utf-8",
        )
        print(f"wrote {path.relative_to(DOCS_ROOT)} ({path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
