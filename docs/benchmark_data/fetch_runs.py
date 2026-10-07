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

    def scan(
        self,
        key: str,
        inputs: list[str],
        atif_dir: Path,
        *,
        brief: bool,
        extra: tuple[str, ...] = (),
    ) -> Json:
        cmd = [
            "uv",
            "run",
            "atif-scan",
            *inputs,
            "--expect-tasks",
            "89",
            *extra,
            "--format",
            "json",
        ]
        if brief:
            cmd.append("--brief")
        # Key on the whole command (inputs, flags) and the scanner's source, so a
        # changed scanner, flag or input path never reuses a stale scan.
        rules = [file_sha256(Path(extra[i + 1])) for i, a in enumerate(extra) if a == "--rules"]
        stamp = hashlib.sha1(
            json.dumps([cmd, scanner_source(atif_dir), rules]).encode()
        ).hexdigest()[:12]
        return self._cached(
            f"scan_{key}_{'brief' if brief else 'full'}_{stamp}", cmd, raw=True, cwd=atif_dir
        )


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decide_flags(
    run: Json, out: Json, full: Json, selected: list[Json], review_cfg: Json
) -> None:
    """Every reported pass atif-scan flags high or critical needs a review decision.

    ``review.clear_when_only`` maps publish-side rule ids to a reason: a flagged pass
    whose only unexcused high/critical findings are among them is cleared with that
    reason (``auto``). Any other flagged pass must be listed in ``disqualified`` or
    ``cleared``; a missing decision stops the build.
    """
    auto = review_cfg.get("clear_when_only") or {}
    review = out.get("review")
    decided = set()
    if review:
        decided = {d["trial"] for d in review["disqualified"]} | {
            c["trial"] for c in review["cleared"]
        }
    by_key = {t.get("scan_key", t["id"]): t for t in selected}
    undecided = []
    for item in full["inputs"]:
        trial = by_key.get(scan_key(item) or "")
        if trial is None or not rewarded(trial):
            continue
        flags = {
            a["id"]
            for a in item["assessments"]
            if a["status"] == "match"
            and a["severity"] in ("high", "critical")
            and not a.get("expected_by")
        }
        if not flags or {trial["id"], trial.get("trial_name")} & decided:
            continue
        if review is not None and flags <= auto.keys():
            rule = sorted(flags)[0]
            review["cleared"].append(
                {"trial": trial.get("trial_name") or trial["id"], "task": task_of(trial),
                 "reason": auto[rule], "auto": rule}
            )
            continue
        undecided.append(f"{trial.get('trial_name') or trial['id']} {sorted(flags)}")
    if undecided and review is not None:
        raise SystemExit(
            f"{run['id']}: flagged passes without a review decision: " + "; ".join(undecided)
        )


def scanner_source(atif_dir: Path) -> Json:
    """The atif-scan checkout behind a scan: commit, and a digest of any local changes."""

    def git(*args: str) -> str:
        proc = subprocess.run(
            ["git", "-C", str(atif_dir), *args], capture_output=True, text=True, check=False
        )
        return proc.stdout.strip()

    diff = git("diff", "HEAD") + git("ls-files", "--others", "--exclude-standard")
    return {
        "commit": git("rev-parse", "HEAD") or None,
        "dirty": bool(diff),
        "diff_sha1": hashlib.sha1(diff.encode()).hexdigest()[:12] if diff else None,
    }


def bucket_job_dir(root: Path, spec: Json, run_id: str) -> Path:
    """Local mirror of one harbor-hf run's job folder (atif-scan's sync layout).

    ``spec["path"]`` is the folder holding the runs inside the bucket (default
    ``runs``, the harbor-hf layout)."""
    return root / spec["repo"] / spec.get("path", "runs") / run_id / "job"


def _trial_usage(trial_dir: Path, agent_result: Json) -> tuple[Json, bool | None]:
    """(input/cache/output tokens, complete) for one bucket trial.

    result.json is preferred. When it has no usage (Harbor drops the counts if any
    model call went unmetered), the trajectory's totals are used if fast-agent says
    every call was metered, else its observed lower bounds (complete=False).
    """
    usage = {
        "input_tokens": agent_result.get("n_input_tokens"),
        "cache_tokens": agent_result.get("n_cache_tokens"),
        "output_tokens": agent_result.get("n_output_tokens"),
    }
    if usage["input_tokens"] is not None:
        return usage, True
    path = trial_dir / "agent" / "trajectory.json"
    if not path.exists():
        return usage, None
    metrics = json.loads(path.read_text()).get("final_metrics") or {}
    extra = metrics.get("extra") or {}
    if extra.get("llm_usage_calls_complete") and metrics.get("total_prompt_tokens") is not None:
        return {
            "input_tokens": metrics["total_prompt_tokens"],
            "cache_tokens": metrics.get("total_cached_tokens") or 0,
            "output_tokens": metrics.get("total_completion_tokens") or 0,
        }, True
    if extra.get("observed_prompt_tokens_lower_bound") is None:
        return usage, None
    return {
        "input_tokens": extra["observed_prompt_tokens_lower_bound"],
        "cache_tokens": extra.get("observed_cached_tokens_lower_bound") or 0,
        "output_tokens": extra.get("observed_completion_tokens_lower_bound") or 0,
    }, False


def _priced(usage: Json, rates: Json) -> float | None:
    if usage["input_tokens"] is None:
        return None
    uncached = usage["input_tokens"] - usage["cache_tokens"]
    return (
        uncached * rates["input"]
        + usage["cache_tokens"] * rates["cached"]
        + usage["output_tokens"] * rates["output"]
    ) / 1e6


def bucket_trials(job_dir: Path, rates: Json | None) -> list[Json]:
    """Hub-shaped trial records from a harbor-hf job folder.

    ``scan_key`` is the trial folder name (how atif-scan identifies local inputs).
    Cost is computed from tokens at ``rates`` (the harbor-hf runner records none).
    """
    trials: list[Json] = []
    for result_path in sorted(job_dir.glob("*__*/result.json")):
        trial_dir = result_path.parent
        result = json.loads(result_path.read_text())
        agent_result = result.get("agent_result") or {}
        rewards = (result.get("verifier_result") or {}).get("rewards") or {}
        usage, complete = _trial_usage(trial_dir, agent_result)
        trajectory = trial_dir / "agent" / "trajectory.json"
        agent = (
            (json.loads(trajectory.read_text()).get("agent") or {}) if trajectory.exists() else {}
        )
        trials.append(
            {
                "id": result["id"],
                "scan_key": trial_dir.name,
                "trial_name": trial_dir.name,
                "task_name": result["task_name"],
                "reward": rewards.get("reward"),
                "error_type": (result.get("exception_info") or {}).get("exception_type"),
                "started_at": result.get("started_at"),
                "agent_version": agent.get("version"),
                "model_name": agent.get("model_name") or result["config"]["agent"]["model_name"],
                "usage_complete": complete,
                "cost_usd": _priced(usage, rates) if rates else agent_result.get("cost_usd"),
                **usage,
            }
        )
    return trials


def select_release_trials(
    run: Json, root: Path
) -> tuple[list[Json], list[Json], dict[str, str], list[Json]]:
    """A bench-run release in a bucket: ``release`` names its manifest
    (``bench-run.release/v1``) under ``path``; job folders sit in ``path/jobs``.

    The manifest's trials are the reported set (replacements substituted); its lineage
    names the replaced originals, which are excluded (kept as evidence, cost counted).
    Every trial in the job folders must be one or the other.
    """
    spec = run["bucket"]
    base = root / spec["repo"] / spec["path"]
    manifest = json.loads((base / spec["release"]).read_text())
    if manifest.get("schema") != "bench-run.release/v1" or len(manifest["cohorts"]) != 1:
        raise SystemExit(f"{run['id']}: expected a one-cohort bench-run.release/v1 manifest")
    cohort = manifest["cohorts"][0]
    reported = {t["id"] for t in cohort["trials"]}
    # Every link's replaced trial is excluded: an original, or the trial of a replacement
    # that was itself replaced (a superseded link). The manifest's trials are the ends.
    lineage = {
        r["replaced_trial"]: r for r in cohort["lineage"] if r["state"] in ("finalized", "superseded")
    }
    replacement_jobs = {t["job"] for t in cohort["trials"] if t.get("replacement")}
    rates = (run.get("pricing") or {}).get("rates_per_mtok")
    trials: list[Json] = []
    jobs: list[Json] = []
    for job_dir in sorted(p for p in (base / "jobs").iterdir() if p.is_dir()):
        found = bucket_trials(job_dir, rates)
        for trial in found:
            trial["bucket_run"] = job_dir.name
            trial["main_run"] = job_dir.name not in replacement_jobs
        trials += found
        jobs.append(
            {
                "id": job_dir.name,
                "name": f"{job_dir.name} ({len(found)} trial{'s' if len(found) != 1 else ''})",
                "url": f"https://huggingface.co/buckets/{spec['repo']}/tree/{spec['path']}/jobs/{job_dir.name}",
                "path": str(job_dir),
            }
        )
    reasons = {
        t["id"]: (
            f"{lineage[t['trial_name']]['replaced_error'] or 'error'} (infrastructure); replaced by "
            f"{lineage[t['trial_name']]['id']}"
            + (" (itself later replaced)" if lineage[t["trial_name"]]["state"] == "superseded" else "")
            + ". Not scored; its cost is still counted."
        )
        for t in trials
        if t["trial_name"] in lineage
    }
    selected = [t for t in trials if t["id"] in reported]
    excluded = [t for t in trials if t["id"] in reasons]
    stray = [t["trial_name"] for t in trials if t["id"] not in reported and t["id"] not in reasons]
    if len(selected) != len(reported) or len(excluded) != len(lineage) or stray:
        raise SystemExit(
            f"{run['id']}: release/job mismatch: {len(selected)}/{len(reported)} reported, "
            f"{len(excluded)}/{len(lineage)} replaced, stray {stray[:3]}"
        )
    return selected, excluded, reasons, jobs


def select_bucket_trials(
    run: Json, root: Path
) -> tuple[list[Json], list[Json], dict[str, str], list[Json]]:
    """(selected, excluded, exclusion reasons, job descriptors) for a bucket run.

    The first run is the main run; each later run's ``operator_selection`` names the
    trials of ``original_run_id`` it replaces (same task, one for one).
    """
    spec = run["bucket"]
    if "release" in spec:
        return select_release_trials(run, root)
    rates = (run.get("pricing") or {}).get("rates_per_mtok")
    by_run: dict[str, list[Json]] = {}
    jobs: list[Json] = []
    for run_id in spec["runs"]:
        job_dir = bucket_job_dir(root, spec, run_id)
        remote = f"{spec['repo']}/{spec.get('path', 'runs')}/{run_id}"
        if not job_dir.is_dir():
            raise SystemExit(
                f"{run['id']}: no local mirror at {job_dir}; sync it with "
                f"`uv run atif-scan hf://buckets/{remote}/ --sync`"
            )
        by_run[run_id] = bucket_trials(job_dir, rates)
        for trial in by_run[run_id]:
            trial["bucket_run"] = run_id
            trial["main_run"] = run_id == spec["runs"][0]
        jobs.append(
            {
                "id": run_id,
                "name": f"{run_id} ({len(by_run[run_id])} trials)",
                "url": f"https://huggingface.co/buckets/{spec['repo']}/tree/{spec.get('path', 'runs')}/{run_id}",
                "path": str(job_dir),
            }
        )
    reasons: dict[str, str] = {}
    for run_id in spec["runs"][1:]:
        meta = json.loads((bucket_job_dir(root, spec, run_id).parent / "run.json").read_text())
        selection = meta["operator_selection"]
        original = {t["id"]: t for t in by_run[selection["original_run_id"]]}
        replacements = {task_of(t): t for t in by_run[run_id]}
        for trial_id in selection["trial_ids"]:
            replaced = original[trial_id]
            task = task_of(replaced)
            if task not in replacements:
                raise SystemExit(f"{run['id']}: no replacement for {task} in {run_id}")
            reasons[trial_id] = (
                f"{replaced['error_type'] or 'error'} (infrastructure); replaced by the rerun "
                f"in {run_id}. Not scored; its cost is still counted."
            )
    trials = [t for ts in by_run.values() for t in ts]
    excluded = [t for t in trials if t["id"] in reasons]
    selected = [t for t in trials if t["id"] not in reasons]
    return selected, excluded, reasons, jobs


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
    by_id = {key: i for i in full["inputs"] if (key := scan_key(i))}
    model = _bare_model(run_model)
    out: dict[str, str] = {}
    for task, trials in ordered.items():
        chars = []
        for trial in trials:
            item = by_id.get(trial.get("scan_key", trial["id"]))
            if item is None or item["input_status"] != "available":
                chars.append("?")
                continue
            char = {"critical": "c", "high": "h", "medium": "m"}.get(item["severity"], "l")
            used = {_bare_model(m) for m in item.get("step_models") or {} if not m.startswith("<")}
            chars.append(char.upper() if used - {model} else char)
        out[task] = "".join(chars).ljust(len(trials), "?")
    return out


def scan_key(item: Json) -> str | None:
    """How a scan input matches a trial: Hub trial id, else the local trial folder."""
    if item.get("hub_trial_id"):
        return item["hub_trial_id"]
    parts = (item.get("input_id") or "").split("/")
    return parts[0] if parts[0] else None


def reported_review(brief: Json, full: Json, keys: set[str], slots: int) -> Json:
    """Scan figures over exactly the reported trials (replaced originals left out)."""
    inputs = [i for i in full["inputs"] if scan_key(i) in keys]
    rewarded = [i for i in inputs if (i["reward"] or 0) > 0]
    flagged = [i for i in rewarded if i["severity"] in ("high", "critical")]
    errors = Counter(i["error_type"] for i in inputs if i.get("error_type"))
    return {
        "trials": len(inputs),
        "rewarded": len(rewarded),
        "high_or_critical_rewarded": len(flagged),
        # atif-scan's own uncleared list, cut to the reported trials.
        "incomplete_evidence_rewarded": sum(
            1
            for x in (brief["overview"]["disqualification"] or {}).get(
                "rewarded_not_cleared_ids", []
            )
            if x.split("/")[0] in keys
        ),
        "accuracy_if_flagged_failed": round(100 * (len(rewarded) - len(flagged)) / slots, 2),
        "evidence": {
            "planned": slots,
            "errored": sum(errors.values()),
            "error_types": dict(sorted(errors.items())),
            "compacted_history": sum(1 for i in inputs if i["compacted"]),
            "without_trajectory": slots
            - sum(1 for i in inputs if i["input_status"] == "available"),
            "incomplete_scans": sum(1 for i in inputs if i["incomplete"]),
        },
    }


def summarize_scan(brief: Json, full: Json, excluded_ids: set[str]) -> Json:
    ov = brief["overview"]
    findings = brief["findings"]
    dq = ov["disqualification"] or {}
    inputs = [i for i in full["inputs"] if scan_key(i) not in excluded_ids]
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
    bucket_root: Path = Path.home() / ".cache" / "atif-scan" / "hf" / "buckets",
    scan_extra: tuple[str, ...] = (),
    scan_rules: Path | None = None,
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
    elif "bucket" in run:
        selected, excluded, bucket_reasons, bucket_jobs = select_bucket_trials(run, bucket_root)
        excluded_cfg = {**bucket_reasons, **excluded_cfg}
        job_ids = [j["id"] for j in bucket_jobs]
    else:
        job_ids = run["jobs"]
        all_trials = [t for j in job_ids for t in fetcher.job_trials(j)]
        excluded = [t for t in all_trials if t["id"] in excluded_cfg]
        selected = [t for t in all_trials if t["id"] not in excluded_cfg]
        if missing := excluded_cfg.keys() - {t["id"] for t in excluded}:
            raise SystemExit(f"{run['id']}: excluded trials not found: {sorted(missing)}")

    if path := run.get("submission_file"):
        submission = fetcher.submission(manifest["leaderboard_repo"], path)

    models = sorted({t["model_name"] for t in selected if t["model_name"]})
    if len(models) != 1:
        notes.append(f"Selected trials report several models: {models}.")
    if "bucket" in run:
        shows = {j["id"]: {"name": j["name"]} for j in bucket_jobs}
        model_str = run["model_string"]
    else:
        shows = {j: fetcher.job_show(j) for j in job_ids}
        model_str = model_string(next(s for s in shows.values() if s.get("config")), models[0])
    versions = sorted({t["agent_version"] for t in selected if t["agent_version"]})

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

    # Our own review: disqualified trials are known exactly, so mark that attempt.
    review_cfg = run.get("review") or {}
    recorded_passes = sum(c.count("1") for c in cells.values())
    review_dq: list[Json] = []
    for dq in review_cfg.get("disqualified", []):
        hits = [
            (task, i)
            for task, trials in ordered.items()
            for i, t in enumerate(trials)
            if dq["trial"] in (t["id"], t.get("trial_name"))
        ]
        if len(hits) != 1:
            raise SystemExit(
                f"{run['id']}: review trial {dq['trial']} matched {len(hits)} selected trials"
            )
        task, idx = hits[0]
        if cells[task][idx] != "1":
            raise SystemExit(f"{run['id']}: review trial {dq['trial']} is not a pass")
        cells[task][idx] = "x"
        review_dq.append(
            {"trial": dq["trial"], "task": task, "reason": dq["reason"], "cell_index": idx}
        )
    # Flagged passes the review kept: recorded with a reason, so every flag has a decision.
    review_cleared: list[Json] = []
    for item in review_cfg.get("cleared", []):
        hits = [
            t
            for trials in ordered.values()
            for t in trials
            if item["trial"] in (t["id"], t.get("trial_name"))
        ]
        if len(hits) != 1 or not rewarded(hits[0]):
            raise SystemExit(f"{run['id']}: cleared trial {item['trial']} is not one reported pass")
        if item["trial"] in {d["trial"] for d in review_dq}:
            raise SystemExit(f"{run['id']}: {item['trial']} is both disqualified and cleared")
        review_cleared.append(
            {"trial": item["trial"], "task": task_of(hits[0]), "reason": item["reason"]}
        )

    passes = sum(c.count("1") for c in cells.values())
    slots = len(tasks) * per_task
    costed = selected + excluded
    costs = [t["cost_usd"] for t in costed if t["cost_usd"] is not None]
    selected_costed = sum(1 for t in selected if t["cost_usd"] is not None)
    total_cost = round(sum(costs), 6)

    expected = run.get("expected", {})
    checks: list[str] = []
    # Expected passes reconcile the source data, so compare before our review.
    if "passes" in expected and expected["passes"] != recorded_passes:
        reconciled = False
        checks.append(f"passes {recorded_passes} != expected {expected['passes']}")
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
        "jobs": bucket_jobs
        if "bucket" in run
        else [{"id": j, "name": shows[j].get("name"), "url": HUB + j} for j in job_ids],
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
    if pricing := run.get("pricing"):
        incomplete = sorted(t["trial_name"] for t in costed if t.get("usage_complete") is False)
        out["cost"]["basis"] = (
            "computed from recorded tokens at "
            + " / ".join(f"${pricing['rates_per_mtok'][k]}" for k in ("input", "cached", "output"))
            + " per M input / cached / output tokens"
            + (" (selected plus excluded trials)" if excluded else "")
        )
        out["cost"]["pricing"] = pricing
        out["cost"]["computed"] = True
        out["cost"]["usage_incomplete_trials"] = incomplete
        out["cost"]["lower_bound"] = bool(incomplete) or bool(run.get("cost_lower_bound"))
    if run.get("cost_lower_bound"):
        # Usage the run recorded only in part (e.g. a failed stream attempt): a lower bound.
        out["cost"]["lower_bound"] = True
    if "bucket" in run:
        out["as_run"] = {
            "passes": sum(
                1 for t in selected + excluded if t["main_run"] and rewarded(t)
            ),
            "slots": slots,
        }
    if review_cfg or review_dq:
        out["review"] = {
            "note": review_cfg.get("note"),
            "coverage": review_cfg.get("coverage"),
            "recorded_passes": recorded_passes,
            "recorded_score": round(100 * recorded_passes / slots, 2),
            "disqualified": review_dq,
            "cleared": review_cleared,
        }

    if scan and run.get("scan"):
        try:
            rules = ("--rules", str(scan_rules)) if scan_rules else ()
            if "bucket" in run:
                inputs = [j["path"] for j in bucket_jobs]
                extra = (
                    "--min-trials", str(per_task), *run.get("scan_args", []), *rules, *scan_extra
                )
                excluded_keys = {t["scan_key"] for t in excluded}
            else:
                inputs = [f"harbor://jobs/{j}" for j in job_ids]
                extra = (*rules, *scan_extra)
                excluded_keys = set(excluded_cfg)
            brief = fetcher.scan(run["id"], inputs, atif_dir, brief=True, extra=extra)
            full = fetcher.scan(run["id"], inputs, atif_dir, brief=False, extra=extra)
            out["scan"] = summarize_scan(brief, full, excluded_keys)
            out["scan"]["scanner"] = {
                **scanner_source(atif_dir),
                # The rules file by its repository path, not this machine's.
                "args": [
                    f"docs/benchmark_data/{scan_rules.name}" if scan_rules and a == str(scan_rules) else a
                    for a in extra
                ],
                "rules_sha256": file_sha256(scan_rules) if scan_rules else None,
            }
            out["scan"]["cells"] = scan_cells(full, ordered, models[0])
            if "bucket" in run:
                keys = {t["scan_key"] for t in selected}
                # Report the scan over the reported trials, like the score. Replaced
                # originals are still scanned (kept as evidence) but not counted here.
                reported = reported_review(brief, full, keys, slots)
                scanned = out["scan"]["trials"]
                out["scan"]["trials"] = reported["trials"]
                out["scan"]["rewarded"] = reported["rewarded"]
                for k in (
                    "high_or_critical_rewarded",
                    "incomplete_evidence_rewarded",
                    "accuracy_if_flagged_failed",
                ):
                    out["scan"]["review"][k] = reported[k]
                out["scan"]["evidence"].update(reported["evidence"])
                extra = scanned - reported["trials"]
                out["scan"]["scope"] = f"{reported['trials']} reported trials" + (
                    f" (findings also include {extra} replaced original"
                    f"{'s' if extra != 1 else ''}, kept as evidence)"
                    if extra
                    else ""
                )
        except (RuntimeError, KeyError, json.JSONDecodeError) as exc:
            out["notes"].append(f"atif-scan failed: {type(exc).__name__}: {str(exc)[:200]}")
        else:
            decide_flags(run, out, full, selected, review_cfg)
    elif run.get("scan"):
        previous = HERE / "runs" / f"{run['id']}.json"
        if previous.exists():
            prev = json.loads(previous.read_text())
            out["scan"] = prev.get("scan")
            # Rule-cleared decisions came from that scan; keep them with it.
            if out.get("review") and prev.get("review"):
                out["review"]["cleared"] += [c for c in prev["review"]["cleared"] if c.get("auto")]
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
    parser.add_argument(
        "--bucket-root",
        type=Path,
        default=Path.home() / ".cache" / "atif-scan" / "hf" / "buckets",
        help="local mirror of Hugging Face buckets (atif-scan's sync layout) for bucket runs",
    )
    parser.add_argument(
        "--image-model",
        help="pass --image-model to atif-scan (sends images that block a check to this model; "
        "model calls happen only with this flag)",
    )
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    fetcher = Fetcher(args.cache_dir, args.refresh)
    selected_runs = [r for r in manifest["runs"] if not args.run or r["id"] in args.run]
    tasks_path = args.out_dir / "tasks.json"
    if selected_runs and all("bucket" in r for r in selected_runs) and tasks_path.exists():
        # Bucket runs need no Hub access; keep the committed canonical task list.
        tasks = json.loads(tasks_path.read_text())
    else:
        tasks = canonical_tasks(fetcher, manifest["tasks_from_jobs"], manifest["task_count"])
        tasks_path.write_text(json.dumps(tasks, indent=2) + "\n")

    lb_rows: dict[str, Json] = {}
    if any(r.get("leaderboard_row") for r in selected_runs):
        lb_rows = {r["id"]: r for r in fetcher.leaderboard(manifest["leaderboard"])["rows"]}

    scan_extra = ("--image-model", args.image_model) if args.image_model else ()
    runs_dir = args.out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    for run in selected_runs:
        print(f"{run['id']}", file=sys.stderr)
        out = build_run(
            run,
            manifest,
            fetcher,
            tasks,
            lb_rows,
            args.scan,
            args.atif_scan_dir,
            bucket_root=args.bucket_root,
            scan_extra=scan_extra,
            scan_rules=(args.manifest.parent / manifest["scan_rules"])
            if manifest.get("scan_rules")
            else None,
        )
        (runs_dir / f"{run['id']}.json").write_text(json.dumps(out, indent=2) + "\n")
        print(
            f"  passes {out['passes']}/{out['slots']}  cost ${out['cost']['total_usd']}"
            f" ({out['cost']['trials_with_cost']} costed)  reconciled={out['reconciled']}"
            f"  scan={'yes' if out['scan'] else 'no'}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
