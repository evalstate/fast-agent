"""Validate a bench-run TB4.1 handoff and emit a portable verifier evidence index.

This checks native evidence, not just the aggregate score. The output references
the original trial UUIDs and native result digests without changing either.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

Json = dict[str, Any]


def load(path: Path) -> Json:
    return json.loads(path.read_bytes())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, detail: str) -> None:
    if not condition:
        raise ValueError(detail)


def index_release(root: Path, summary_path: Path, definition_path: Path) -> Json:
    summary = load(summary_path)
    with_def = load(definition_path)
    pins = {task: data["ref"] for task, data in with_def["tasks"].items()}
    definition_sha = sha256(definition_path)
    release_path = root / "releases" / summary["release"] / "manifest.json"
    release = load(release_path)
    release_sha = sha256(release_path)
    (cohort,) = release["cohorts"]
    expected = {trial["id"]: trial for trial in cohort["trials"]}
    slots = {(t["job"], t["trial"]): t["id"] for t in expected.values()}
    count = Counter(t["task"] for t in expected.values())
    require(len(slots) == len(expected) == 95, f"{summary_path.name}: expected 95 source trials")
    require(count == {task: 5 for task in pins}, f"{summary_path.name}: task selection drift")
    require(summary["definition_sha256"] == definition_sha, "summary definition digest drift")

    primary = summary["rescore"]
    executions = [primary, *(item["run_id"] for item in summary["follow_ups"])]
    chosen = {item["graded_by"] for item in summary["rows"]}
    require(chosen <= set(executions), f"{primary}: summary uses an unknown rescore")
    native: dict[tuple[str, str], Json] = {}
    latest: dict[str, str] = {}
    sequence = 0
    for run_id in executions:
        receipt_path = root / "runs" / "rescores" / run_id / "receipt.json"
        receipt = load(receipt_path)
        require(
            receipt["release_manifest_sha256"] == release_sha, f"{run_id}: release digest drift"
        )
        require(
            receipt["definition_sha256"] == definition_sha, f"{run_id}: definition digest drift"
        )
        require(receipt["cohort"] == cohort["cohort"], f"{run_id}: foreign cohort")
        if run_id != primary:
            require(
                receipt["parent"] == primary and receipt["sequence"] > sequence,
                f"{run_id}: invalid follow-up order",
            )
            sequence = receipt["sequence"]
        require(
            sha256(receipt_path.with_name("sources.lock.json")) == receipt["sources_lock_sha256"],
            f"{run_id}: sources lock digest drift",
        )
        for name, digest in receipt["configs"].items():
            require(
                sha256(receipt_path.with_name(name)) == digest, f"{run_id}: config digest drift"
            )
        permitted: dict[str, str] = {}
        for trial in receipt["trials"]:
            uid = slots[trial["job"], trial["trial"]]
            source = expected[uid]
            digest = next(r["sha256"] for r in source["files"] if r["path"] == "result.json")
            require(trial["result_sha256"] == digest, f"{run_id}: frozen source digest drift")
            require(trial["task"] == source["task"], f"{run_id}: source task drift")
            require(uid not in permitted, f"{run_id}: duplicate receipt source {uid}")
            permitted[uid] = digest
            latest[uid] = run_id
        if run_id == primary:
            require(permitted.keys() == expected.keys(), f"{run_id}: incomplete primary selection")
        for partition in receipt["concurrency"]:
            job = root / "jobs" / f"{run_id}-{partition}"
            for result_path in sorted(job.glob("*__*/result.json")):
                result = load(result_path)
                source = result["config"]["source_trial"]
                uid = source["trial_id"]
                require(uid in permitted, f"{run_id}: result is not selected by receipt")
                require(
                    Path(source["path"]).name == expected[uid]["trial"],
                    f"{run_id}: source mismatch",
                )
                key = (run_id, uid)
                require(key not in native, f"{run_id}: duplicate result for {uid}")
                lock_path = result_path.with_name("lock.json")
                lock = load(lock_path)
                task = expected[uid]["task"]
                require(
                    lock["task"]["name"].rsplit("/", 1)[-1] == task
                    and lock["task"]["digest"] == pins[task],
                    f"{run_id}: unexpected task pin for {uid}",
                )
                native[key] = {
                    "result": result_path.relative_to(root).as_posix(),
                    "result_sha256": sha256(result_path),
                    "lock_sha256": sha256(lock_path),
                    "source_result_sha256": permitted[uid],
                    "reward": ((result.get("verifier_result") or {}).get("rewards") or {}).get(
                        "reward"
                    ),
                    "error": (result.get("exception_info") or {}).get("exception_type"),
                    "collection_failed": "collection failed in the source trial"
                    in ((result.get("exception_info") or {}).get("exception_message") or ""),
                }

    results: dict[str, Json] = {}
    for row in summary["rows"]:
        uid = slots[row["job"], row["trial"]]
        old = expected[uid]
        run_id = row["graded_by"]
        require(
            run_id == latest[uid], f"{primary}: summary did not select latest receipt for {uid}"
        )
        evidence = native[run_id, uid]
        require(uid not in results, f"{primary}: duplicate summary slot {uid}")
        require(
            row["task"] == old["task"]
            and row["partition"] == old["partition"]
            and row["reward_before"] == old["effective_reward"]
            and row["reward"] == evidence["reward"]
            and row["error"] == evidence["error"],
            f"{primary}: inconsistent summary slot {uid}",
        )
        status = "graded"
        if row["reward"] is None:
            require(row["reward_before"] == 0, f"{primary}: missing reward for source pass {uid}")
            if row["error"] == "RegradeError" and evidence["collection_failed"]:
                status = "no_collected_artifact"
            elif (
                uid == "5aff4e82-c05a-4399-9e6f-6939b77fc708"
                and row["error"] == "RewardFileNotFoundError"
                and "Criterion 'filter_blocks_xss' failed: Worker timed out"
                in (root / evidence["result"])
                .with_name("verifier")
                .joinpath("test-stdout.txt")
                .read_text()
            ):
                # Reviewed HY-4-Flash HTML slot. The task worker timed out on the
                # replay; keep the missing reward visible without inventing a verdict.
                status = "task_worker_timeout"
            else:
                raise ValueError(f"{primary}: unresolved reward for {uid}")
        results[uid] = {
            "task": row["task"],
            "source_job": row["job"],
            "source_trial": row["trial"],
            "reward_before": row["reward_before"],
            "graded_by": run_id,
            "status": status,
            **{k: v for k, v in evidence.items() if k != "collection_failed"},
        }
    require(results.keys() == expected.keys(), f"{primary}: summary does not cover all 95 slots")
    before = sum((r["reward_before"] or 0) >= 1 for r in results.values())
    after = sum((r["reward"] or 0) >= 1 for r in results.values())
    require(before == summary["passes_before"] and after == summary["passes"], "score mismatch")
    return {
        "schema": "fast-agent.tb41-rescore/v1",
        "release": summary["release"],
        "release_sha256": release_sha,
        "definition": definition_path.name,
        "definition_sha256": definition_sha,
        "primary": primary,
        "follow_ups": executions[1:],
        "passes_before": before,
        "passes": after,
        "trials": {uid: results[uid] for uid in expected},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench-run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("summaries", type=Path, nargs="+")
    args = parser.parse_args()
    definition = args.bench_run / "rescores" / "tb4-subset-4.1.0.json"
    for summary in args.summaries:
        result = index_release(args.bench_run, summary, definition)
        args.out.mkdir(parents=True, exist_ok=True)
        out = args.out / f"{result['release']}.json"
        out.write_text(json.dumps(result, indent=2) + "\n")
        print(f"{result['release']}: {result['passes_before']} -> {result['passes']}/95")


if __name__ == "__main__":
    main()
