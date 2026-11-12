#!/usr/bin/env python3
"""Recalculate a Grok Build leaderboard row's cost on the subset tasks, reasoning included.

Harbor's grok-build adapter (up to at least 1.0.34 runs) counts only the Grok CLI's
``output_tokens`` as output. The CLI reports ``reasoning_tokens`` separately, so the
recorded output tokens and cost leave reasoning out
(https://github.com/harbor-framework/harbor/pull/3805). This script reprices each of the
row's subset trials from the CLI's own per-call ``usage`` events, in the trial's public
Harbor Hub archive (``agent/grok-build.txt``), the way Harbor prices them: per call, at
LiteLLM's ``xai/grok-4.7`` rates, the long-context rates from 200k prompt tokens. Every
call is priced twice: with output as recorded, which must reproduce the Hub's recorded
cost for the trial (else the script stops), and with reasoning added to output, which is
the recalculated cost. Only numeric usage fields are read; no trace text is kept.

    uv run --no-project python docs/benchmark_data/tb4/recalculate_grok_build.py \\
        tb4-grok-build-grok-4-7-xhigh

Needs the ``harbor`` CLI. Writes ``recalculated/<run>.json`` next to this file; the run's
``recalculated_cost`` in manifest.json points fetch_runs.py at it.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
# LiteLLM model_cost["xai/grok-4.7"] (litellm 1.93.0), USD per million tokens; the same as
# xAI's list prices. Harbor applies the long-context rates from 200k prompt tokens.
RATES = {"input": 2.0, "cached": 0.5, "output": 6.0}
LONG_RATES = {"input": 4.0, "cached": 1.0, "output": 12.0}
LONG_FROM = 200_000
TOLERANCE = 1e-6

Json = dict[str, Any]


def harbor(*args: str) -> Any:
    out = subprocess.run(
        ["harbor", "hub", *args, "--json"], capture_output=True, text=True, check=True
    ).stdout
    return json.loads(out)


def paged(*args: str, limit: int = 1000) -> list[Json]:
    items: list[Json] = []
    page = 1
    while True:
        result = harbor(*args, "--limit", str(limit), "--page", str(page))
        items += result["items"]
        if page >= (result.get("total_pages") or 1) or not result["items"]:
            return items
        page += 1


def row_trials(row: str) -> list[Json]:
    """The row's trials, as their jobs list them (task, recorded cost)."""
    ids = {t["trial_id"] for t in paged("leaderboard", "row", "trial", "list", row)}
    found: dict[str, Json] = {}
    while missing := sorted(ids - found.keys()):
        job = harbor("trial", "show", missing[0])["job_id"]
        listed = {t["id"]: t for t in paged("job", "trials", job, "--include-retries")}
        if missing[0] not in listed:
            raise SystemExit(f"trial {missing[0]} not listed by its job {job}")
        found.update({i: dict(listed[i], job_id=job) for i in ids & listed.keys()})
    return list(found.values())


def usage_events(log: Path) -> list[Json]:
    calls = []
    for line in log.read_text(errors="replace").splitlines():
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if event.get("type") == "usage" and isinstance(event.get("usage"), dict):
            calls.append(event["usage"])
    return calls


def prompt(call: Json) -> int:
    keys = ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")
    return sum(call.get(k) or 0 for k in keys)


def price(call: Json, with_reasoning: bool) -> float:
    uncached = call.get("input_tokens") or 0
    cached = call.get("cache_read_input_tokens") or 0
    written = call.get("cache_creation_input_tokens") or 0  # no xAI write rate: input
    rates = LONG_RATES if prompt(call) >= LONG_FROM else RATES
    output = (call.get("output_tokens") or 0) + (
        (call.get("reasoning_tokens") or 0) if with_reasoning else 0
    )
    return (
        (uncached + written) * rates["input"] + cached * rates["cached"] + output * rates["output"]
    ) / 1e6


def recalculate(trial: Json, scratch: Path) -> Json:
    dest = scratch / trial["id"]
    subprocess.run(
        ["harbor", "hub", "trial", "download", trial["id"], "-o", str(dest)],
        capture_output=True,
        check=True,
    )
    (log,) = dest.glob("*/agent/grok-build.txt")
    calls = usage_events(log)
    recorded = sum(price(c, False) for c in calls)
    if trial["cost_usd"] is None or abs(recorded - trial["cost_usd"]) > TOLERANCE:
        raise SystemExit(
            f"{trial['id']}: usage reprices to {recorded:.6f}, the Hub records {trial['cost_usd']}"
        )
    return {
        "id": trial["id"],
        "task": trial["task_name"].split("/")[-1],
        "calls": len(calls),
        "long_context_calls": sum(1 for c in calls if prompt(c) >= LONG_FROM),
        "output_tokens": sum(c.get("output_tokens") or 0 for c in calls),
        "reasoning_tokens": sum(c.get("reasoning_tokens") or 0 for c in calls),
        "recorded_usd": round(trial["cost_usd"], 6),
        "recalculated_usd": round(sum(price(c, True) for c in calls), 6),
    }


def main() -> None:
    (run_id,) = sys.argv[1:]
    manifest = json.loads((HERE / "manifest.json").read_text())
    (run,) = [r for r in manifest["runs"] if r["id"] == run_id]
    subset = set(json.loads((HERE / "subset.json").read_text()))
    trials = [
        t for t in row_trials(run["leaderboard_row"]) if t["task_name"].split("/")[-1] in subset
    ]
    with tempfile.TemporaryDirectory(prefix="grok-build-") as tmp, ThreadPoolExecutor(4) as pool:
        records = list(pool.map(lambda t: recalculate(t, Path(tmp)), trials))
    records.sort(key=lambda r: (r["task"], r["id"]))
    doc = {
        "run": run_id,
        "leaderboard_row": run["leaderboard_row"],
        "jobs": sorted({t["job_id"] for t in trials}),
        "basis": "recalculated from the Grok CLI's per-call usage with reasoning tokens priced as "
        "output, at LiteLLM's xai/grok-4.7 rates as Harbor prices them",
        "rates_usd_per_mtok": {
            **RATES,
            "long_context": {"from_prompt_tokens": LONG_FROM, **LONG_RATES},
        },
        "defect": "https://github.com/harbor-framework/harbor/pull/3805",
        "trials": records,
    }
    out = HERE / "recalculated" / f"{run_id}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(doc, indent=1) + "\n")
    total = sum(r["recalculated_usd"] for r in records)
    recorded = sum(r["recorded_usd"] for r in records)
    print(f"{len(records)} trials: recorded ${recorded:.2f}, recalculated ${total:.2f} -> {out}")


if __name__ == "__main__":
    main()
