"""Fetch Terminal-Bench 2.1 per-trial data for the benchmarks page.

Reads ``manifest.json`` next to this file and writes ``runs/<run_id>.json`` and
``tasks.json``. Data comes from the ``harbor`` CLI (Hub jobs, trials and the
TB2.1 leaderboard), the ``gh`` CLI (leaderboard submission files) and,
optionally, ``atif-scan`` for an integrity summary.

    uv run python docs/benchmark_data/fetch_runs.py                 # all runs
    uv run python docs/benchmark_data/fetch_runs.py --run luna-max-6h
    uv run python docs/benchmark_data/fetch_runs.py --scan          # also run atif-scan

Raw CLI responses are cached under ``--cache-dir`` (default /tmp/bench/cache);
pass ``--refresh`` to re-fetch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
HUB = "https://hub.harborframework.com/jobs/"
TRIAL_PAGE = 500
ROW_PAGE = 1000
SEVERITY_RANK = {"medium": 2, "high": 3, "critical": 4}

Json = dict[str, Any]


class Fetcher:
    def __init__(self, cache_dir: Path, refresh: bool) -> None:
        self.cache_dir = cache_dir
        self.refresh = refresh
        cache_dir.mkdir(parents=True, exist_ok=True)

    def _cached(
        self, key: str, cmd: list[str], *, raw: bool = False, cwd: Path | None = None
    ) -> Any:
        path = self.cache_dir / f"{key}.json"
        if path.exists() and not self.refresh:
            return json.loads(path.read_text())
        print("  $", " ".join(cmd), file=sys.stderr)
        env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, env=env, check=False)
        if proc.returncode not in (0, 2) or not proc.stdout.strip():
            raise RuntimeError(f"{' '.join(cmd)} failed ({proc.returncode}): {proc.stderr[-500:]}")
        data = json.loads(proc.stdout)
        path.write_text(proc.stdout if raw else json.dumps(data))
        return data

    def job_show(self, job_id: str) -> Json:
        return self._cached(f"show_{job_id}", ["harbor", "hub", "job", "show", job_id, "--json"])

    def job_trials(self, job_id: str) -> list[Json]:
        items: list[Json] = []
        page = 1
        while True:
            data = self._cached(
                f"trials_{job_id}_p{page}",
                [
                    "harbor",
                    "hub",
                    "job",
                    "trials",
                    job_id,
                    "--limit",
                    str(TRIAL_PAGE),
                    "--page",
                    str(page),
                    "--json",
                ],
            )
            items.extend(data["items"])
            if page >= (data["total_pages"] or 1):
                break
            page += 1
        return items

    def trial_show(self, trial_id: str) -> Json:
        return self._cached(
            f"trial_{trial_id}", ["harbor", "hub", "trial", "show", trial_id, "--json"]
        )

    def leaderboard(self, slug: str) -> Json:
        key = "leaderboard_" + slug.replace("/", "_")
        return self._cached(key, ["harbor", "hub", "leaderboard", "show", slug, "--json"])

    def row_trial_ids(self, row_id: str) -> list[str]:
        ids: list[str] = []
        page = 1
        while True:
            data = self._cached(
                f"row_{row_id}_p{page}",
                [
                    "harbor",
                    "hub",
                    "leaderboard",
                    "row",
                    "trial",
                    "list",
                    row_id,
                    "--limit",
                    str(ROW_PAGE),
                    "--page",
                    str(page),
                    "--json",
                ],
            )
            ids.extend(item["trial_id"] for item in data["items"])
            if page >= (data["total_pages"] or 1):
                break
            page += 1
        return ids

    def submission(self, repo: str, path: str) -> Json:
        key = "submission_" + hashlib.sha1(path.encode()).hexdigest()[:12]
        return self._cached(
            key,
            [
                "gh",
                "api",
                f"repos/{repo}/contents/{path}",
                "-H",
                "Accept: application/vnd.github.raw",
            ],
        )

    def scan(self, key: str, job_ids: list[str], atif_dir: Path, *, brief: bool) -> Json:
        cmd = [
            "uv",
            "run",
            "atif-scan",
            *(f"harbor://jobs/{j}" for j in job_ids),
            "--expect-tasks",
            "89",
            "--format",
            "json",
        ]
        if brief:
            cmd.append("--brief")
        return self._cached(
            f"scan_{key}_{'brief' if brief else 'full'}", cmd, raw=True, cwd=atif_dir
        )


def task_of(trial: Json) -> str:
    return trial["task_name"].removeprefix("terminal-bench/")


def rewarded(trial: Json) -> bool:
    return (trial["reward"] or 0) > 0


def cell(trial: Json) -> str:
    if rewarded(trial):
        return "1"
    if trial["error_type"] == "AgentTimeoutError":
        return "t"
    if trial["error_type"]:
        return "e"
    return "0"


def canonical_tasks(fetcher: Fetcher, job_ids: list[str], expected: int) -> list[str]:
    names: set[str] = set()
    for job_id in job_ids:
        for dataset in fetcher.job_show(job_id)["config"]["datasets"]:
            names.update(n.removeprefix("terminal-bench/") for n in dataset["task_names"])
    if len(names) != expected:
        raise SystemExit(f"expected {expected} tasks from {job_ids}, found {len(names)}")
    return sorted(names)


def select_leaderboard_trials(fetcher: Fetcher, row_id: str) -> tuple[list[Json], list[str]]:
    """Resolve a leaderboard row's trial associations to Hub trial records."""
    wanted = set(fetcher.row_trial_ids(row_id))
    found: dict[str, Json] = {}
    job_ids: list[str] = []
    while missing := sorted(wanted - found.keys()):
        job_id = fetcher.trial_show(missing[0])["job_id"]
        if job_id in job_ids:
            raise SystemExit(f"row {row_id}: trial {missing[0]} not listed by its job {job_id}")
        job_ids.append(job_id)
        found.update({t["id"]: t for t in fetcher.job_trials(job_id) if t["id"] in wanted})
    return list(found.values()), job_ids


def model_string(show: Json, model_name: str) -> str:
    for agent in show["config"]["agents"]:
        if agent.get("model_name", "").split("/")[-1] == model_name:
            kwargs = agent.get("kwargs") or {}
            return kwargs.get("fast_agent_model") or agent["model_name"]
    raise SystemExit(f"no agent config for model {model_name}")


def pct(values: list[float], q: float) -> float:
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return ordered[idx]


def _bare_model(name: str) -> str:
    """Compare model names without provider prefix or case (as atif-scan does)."""
    return name.split("/")[-1].lower()


def scan_cells(full: Json, ordered: dict[str, list[Json]], run_model: str) -> dict[str, str]:
    """One character per selected trial, aligned with ``tasks`` cells.

    c/h/m/l: the trial's highest unexcused atif-scan priority (l covers low, info
    and none). Upper case: the trial's steps used a model other than the run's.
    ?: no scan result for the trial.
    """
    by_id = {i["hub_trial_id"]: i for i in full["inputs"] if i.get("hub_trial_id")}
    model = _bare_model(run_model)
    out: dict[str, str] = {}
    for task, trials in ordered.items():
        chars = []
        for trial in trials:
            item = by_id.get(trial["id"])
            if item is None or item["input_status"] != "available":
                chars.append("?")
                continue
            char = {"critical": "c", "high": "h", "medium": "m"}.get(item["severity"], "l")
            used = {_bare_model(m) for m in item.get("step_models") or {} if not m.startswith("<")}
            chars.append(char.upper() if used - {model} else char)
        out[task] = "".join(chars).ljust(len(trials), "?")
    return out


def summarize_scan(brief: Json, full: Json, excluded_ids: set[str]) -> Json:
    ov = brief["overview"]
    findings = brief["findings"]
    dq = ov["disqualification"] or {}
    inputs = [i for i in full["inputs"] if i.get("hub_trial_id") not in excluded_ids]
    flagged: dict[str, Json] = {}
    for item in inputs:
        sev = item["severity"]
        if sev not in SEVERITY_RANK:
            continue
        entry = flagged.setdefault(item["task"], {"trials": 0, "rewarded": 0, "max_priority": sev})
        entry["trials"] += 1
        entry["rewarded"] += int((item["reward"] or 0) > 0)
        if SEVERITY_RANK[sev] > SEVERITY_RANK[entry["max_priority"]]:
            entry["max_priority"] = sev
    top = [
        {
            "check": check,
            "title": brief["titles"].get(check),
            "priority": info["severity"],
            "trials": info["traces"],
            "rewarded": info["rewarded"],
        }
        for check, info in findings["checks"].items()
        if info["severity"] in SEVERITY_RANK
    ][:10]
    awareness = brief["awareness"]
    unmetered = brief["unmetered_work"] or {}
    unpriced = brief["cost_estimate"] or {}
    recorded = ov["cost"]["total_usd"]
    extra = (unmetered.get("estimate_usd") or 0) + (unpriced.get("estimate_usd") or 0)
    walltime = ov["walltime"]
    return {
        "version": brief["scanner_version"],
        "scope": "all trials in the run's jobs (includes excluded trials)"
        if excluded_ids
        else "all trials in the run's jobs",
        "trials": ov["trials"]["present"],
        "rewarded": sum(1 for i in full["inputs"] if (i["reward"] or 0) > 0),
        "accuracy": ov["accuracy"][0],
        "accuracy_se": ov["accuracy"][1],
        "findings_medium_plus": {
            "trials": findings["medium_plus_trials"],
            "rewarded": findings["medium_plus_rewarded"],
            "by_highest_priority": findings["traces_by_highest_severity"],
            "top": top,
        },
        "review": {
            "dq_threshold": brief["dq_threshold"],
            "high_or_critical_rewarded": dq.get("by_findings", 0),
            "other_model_trials": dq.get("by_model_only", 0),
            "incomplete_evidence_rewarded": dq.get("rewarded_not_cleared", 0),
            "accuracy_if_flagged_failed": (dq.get("accuracy_if_disqualified") or [None])[0],
        },
        "awareness": {
            "trials": awareness["trials"],
            "stages": {
                s["stage"]: {"trials": s["trials"], "rewarded": s["rewarded"]}
                for s in awareness["stages"]
            },
            "verifier_talk": {
                "trials": awareness["verifier_talk"]["trials"],
                "rewarded": awareness["verifier_talk"]["rewarded"],
            },
        },
        "evidence": {
            "planned": ov["trials"]["planned"],
            "errored": ov["trials"]["errored"],
            "error_types": ov["trials"]["error_types"],
            "compacted_history": ov["trials"]["compacted"],
            "without_trajectory": ov["trials"]["without_trajectory"],
            "incomplete_scans": ov["trials"]["incomplete_scans"],
            "unmetered_trials": unmetered.get("trials", 0),
        },
        "cost": {
            "recorded": recorded,
            "trials_missing_cost": ov["cost"]["missing"],
            "estimated_unpriced": unpriced.get("estimate_usd"),
            "estimated_unmetered": unmetered.get("estimate_usd"),
            "estimated_total": round(recorded + extra, 2),
        },
        "tokens": ov["tokens"],
        "walltime_hours": {
            "trial_sum": round(walltime["trial"]["seconds"] / 3600, 1)
            if walltime["trial"]["seconds"]
            else None,
            "agent_sum": round(walltime["agent"]["seconds"] / 3600, 1)
            if walltime["agent"]["seconds"]
            else None,
        },
        "overrides": ov["overrides"],
        "flagged_tasks": dict(sorted(flagged.items())),
    }


def build_run(
    run: Json,
    manifest: Json,
    fetcher: Fetcher,
    tasks: list[str],
    lb_rows: dict[str, Json],
    scan: bool,
    atif_dir: Path,
) -> Json:
    notes: list[str] = list(run.get("notes", []))
    per_task = manifest["attempts_per_task"]
    excluded_cfg = {e["id"]: e["reason"] for e in run.get("exclude_trials", [])}
    published = run.get("published")
    date = run.get("date")
    submission: Json | None = None

    if row_id := run.get("leaderboard_row"):
        selected, job_ids = select_leaderboard_trials(fetcher, row_id)
        row = lb_rows[row_id]
        published = {
            "score": row["metrics"]["accuracy"],
            "total_cost": row["metrics"]["total_cost_usd"],
            "source_url": row["metadata"]["pr_url"]["url"],
        }
        date = row["metadata"]["date"]
        excluded: list[Json] = []
    else:
        job_ids = run["jobs"]
        all_trials = [t for j in job_ids for t in fetcher.job_trials(j)]
        excluded = [t for t in all_trials if t["id"] in excluded_cfg]
        selected = [t for t in all_trials if t["id"] not in excluded_cfg]
        if missing := excluded_cfg.keys() - {t["id"] for t in excluded}:
            raise SystemExit(f"{run['id']}: excluded trials not found: {sorted(missing)}")

    if path := run.get("submission_file"):
        submission = fetcher.submission(manifest["leaderboard_repo"], path)

    shows = {j: fetcher.job_show(j) for j in job_ids}
    models = sorted({t["model_name"] for t in selected})
    if len(models) != 1:
        notes.append(f"Selected trials report several models: {models}.")
    model_str = model_string(next(s for s in shows.values() if s.get("config")), models[0])
    versions = sorted({t["agent_version"] for t in selected})

    by_task: dict[str, list[Json]] = {task: [] for task in tasks}
    for trial in selected:
        by_task.setdefault(task_of(trial), []).append(trial)
    unknown = sorted(by_task.keys() - set(tasks))
    reconciled = not unknown
    if unknown:
        notes.append(f"Trials for tasks outside the canonical list: {unknown}.")

    cells: dict[str, list[str]] = {}
    ordered: dict[str, list[Json]] = {}
    for task in tasks:
        trials = sorted(by_task[task], key=lambda t: t["started_at"] or "")
        ordered[task] = trials[:per_task]
        cells[task] = [cell(t) for t in trials[:per_task]] + ["-"] * max(0, per_task - len(trials))
        if len(trials) > per_task:
            reconciled = False
            notes.append(
                f"{task}: {len(trials)} selected trials, only the first {per_task} by start time are shown."
            )
    if missing_slots := sum(c.count("-") for c in cells.values()):
        reconciled = False
        notes.append(f"{missing_slots} missing trial slots (scored 0).")

    disqualified: list[Json] = []
    for dq in (submission or {}).get("disqualified_trials", []):
        task = dq["judge_trial"].rstrip("/").split("/")[-1]
        row_cells = cells[task]
        idx = row_cells.index("1")
        row_cells[idx] = "x"
        disqualified.append(
            {
                "leaderboard_trial_id": dq["trial_id"],
                "task": task,
                "reason": dq["reason"],
                "cell_index": idx,
            }
        )
    if disqualified:
        notes.append(
            "Disqualified trials are identified by leaderboard-clone trial id, which does not map to the public "
            "scrubbed Hub trial ids; the page marks the first passing attempt of that task, which may not be the disqualified one."
        )

    passes = sum(c.count("1") for c in cells.values())
    slots = len(tasks) * per_task
    costed = selected + excluded
    costs = [t["cost_usd"] for t in costed if t["cost_usd"] is not None]
    selected_costed = sum(1 for t in selected if t["cost_usd"] is not None)
    total_cost = round(sum(costs), 6)

    expected = run.get("expected", {})
    checks: list[str] = []
    if "passes" in expected and expected["passes"] != passes:
        reconciled = False
        checks.append(f"passes {passes} != expected {expected['passes']}")
    if "trials_with_cost" in expected and expected["trials_with_cost"] != selected_costed:
        checks.append(
            f"trials_with_cost {selected_costed} != expected {expected['trials_with_cost']}"
        )
    errors = dict(sorted(Counter(t["error_type"] for t in selected if t["error_type"]).items()))
    if "errors" in expected and expected["errors"] != errors:
        reconciled = False
        checks.append(f"error mix {errors} != submitted {expected['errors']}")
    if "harness_version" in expected and versions != [expected["harness_version"]]:
        reconciled = False
        checks.append(f"harness versions {versions} != submitted {expected['harness_version']}")
    cost_matches = None
    if "total_cost_usd" in expected:
        cost_matches = abs(total_cost - expected["total_cost_usd"]) < 0.01
        if not cost_matches:
            checks.append(f"cost ${total_cost:.2f} != expected ${expected['total_cost_usd']:.2f}")
    if checks:
        notes.append("Reconciliation: " + "; ".join(checks) + ".")

    def token_sum(key: str) -> int | None:
        vals = [t[key] for t in costed if t.get(key) is not None]
        return sum(vals) if vals else None

    out: Json = {
        "id": run["id"],
        "benchmark": manifest["benchmark"],
        "harness": run["harness"],
        "harness_version": ", ".join(versions),
        "model": run["model"],
        "effort": run["effort"],
        "model_string": model_str,
        "date": date,
        "timeout": run["timeout"],
        "jobs": [{"id": j, "name": shows[j].get("name"), "url": HUB + j} for j in job_ids],
        "excluded_trials": [
            {
                "id": t["id"],
                "task": task_of(t),
                "error_type": t["error_type"],
                "cost_usd": t["cost_usd"],
                "reason": excluded_cfg[t["id"]],
            }
            for t in excluded
        ],
        "disqualified_trials": disqualified,
        "slots": slots,
        "passes": passes,
        "score": round(100 * passes / slots, 2),
        "tasks": {task: "".join(c) for task, c in cells.items()},
        "errors": errors,
        "cost": {
            "total_usd": round(total_cost, 2),
            "total_usd_exact": total_cost,
            "trials_with_cost": selected_costed,
            "trials": len(selected),
            "excluded_trials_with_cost": len(costs) - selected_costed,
            "basis": "sum of Harbor recorded cost_usd over selected trials"
            + (" plus excluded (replaced) trials" if excluded else ""),
            "matches_published": cost_matches if run.get("cost_reconciles", True) else False,
        },
        "tokens": {
            "input": token_sum("input_tokens"),
            "cached": token_sum("cache_tokens"),
            "output": token_sum("output_tokens"),
            "basis": "Harbor per-trial input_tokens (includes cached), cache_tokens, output_tokens",
        },
        "per_trial_cost": {
            "median": round(statistics.median(costs), 4) if costs else None,
            "p90": round(pct(costs, 0.9), 4) if costs else None,
            "max": round(max(costs), 4) if costs else None,
        },
        "published": published,
        "reconciled": reconciled,
        "notes": notes,
        "scan": None,
    }
    if run.get("pr"):
        out["pr"] = f"https://github.com/{manifest['leaderboard_repo']}/pull/{run['pr']}"

    if scan and run.get("scan"):
        try:
            brief = fetcher.scan(run["id"], job_ids, atif_dir, brief=True)
            full = fetcher.scan(run["id"], job_ids, atif_dir, brief=False)
            out["scan"] = summarize_scan(brief, full, set(excluded_cfg))
            out["scan"]["cells"] = scan_cells(full, ordered, models[0])
        except (RuntimeError, KeyError, json.JSONDecodeError) as exc:
            out["notes"].append(f"atif-scan failed: {type(exc).__name__}: {str(exc)[:200]}")
    elif run.get("scan"):
        previous = HERE / "runs" / f"{run['id']}.json"
        if previous.exists():
            out["scan"] = json.loads(previous.read_text()).get("scan")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--manifest", type=Path, default=HERE / "manifest.json")
    parser.add_argument("--out-dir", type=Path, default=HERE)
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/bench/cache"))
    parser.add_argument("--refresh", action="store_true", help="ignore cached CLI responses")
    parser.add_argument("--run", action="append", help="only these run ids (repeatable)")
    parser.add_argument("--scan", action="store_true", help="run atif-scan for runs with scan=true")
    parser.add_argument("--atif-scan-dir", type=Path, default=Path.home() / "source" / "atif-scan")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    fetcher = Fetcher(args.cache_dir, args.refresh)
    tasks = canonical_tasks(fetcher, manifest["tasks_from_jobs"], manifest["task_count"])
    (args.out_dir / "tasks.json").write_text(json.dumps(tasks, indent=2) + "\n")

    lb_rows: dict[str, Json] = {}
    if any(r.get("leaderboard_row") for r in manifest["runs"]):
        lb_rows = {r["id"]: r for r in fetcher.leaderboard(manifest["leaderboard"])["rows"]}

    runs_dir = args.out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    for run in manifest["runs"]:
        if args.run and run["id"] not in args.run:
            continue
        print(f"{run['id']}", file=sys.stderr)
        out = build_run(run, manifest, fetcher, tasks, lb_rows, args.scan, args.atif_scan_dir)
        (runs_dir / f"{run['id']}.json").write_text(json.dumps(out, indent=2) + "\n")
        print(
            f"  passes {out['passes']}/{out['slots']}  cost ${out['cost']['total_usd']}"
            f" ({out['cost']['trials_with_cost']} costed)  reconciled={out['reconciled']}"
            f"  scan={'yes' if out['scan'] else 'no'}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
