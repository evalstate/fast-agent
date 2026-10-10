#!/usr/bin/env python3
"""Evidence cards for a run page: the moments behind its high findings and review decisions.

Input is an atif-scan highlight export (``atif-scan … --highlights DIR``, same inputs and
options as the run's scan): short, masked excerpts of where each finding matched, as
said (reasoning before) / ran (the matched span in context) / got (what came back).

Output is ``moments/<run id>.json``: one card per trial worth showing, i.e. a reward with
a high or critical finding, or any trial our review decided on, carrying the decision, the
findings (our publish-side rules first, minus the checks they restate) and up to two
moments each, plus links to the trial folder and trajectory in the published bucket.

Excerpts are trace text. atif-scan masks secret shapes (best effort); this script also
refuses to write token-shaped text. Check the output against known credentials before
committing it (bench-run: ``release.known_credentials()``). Standard library only.

    python docs/benchmark_data/moments.py --run opus55-high-6h --highlights /tmp/hl/data.js
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
Json = dict[str, Any]

PRIORITY = {"critical": 0, "high": 1, "medium": 2}
MAX_FINDINGS = 3  # per card
MAX_MOMENTS = 2  # per finding
CLIP = 600  # characters per excerpt part
TOKEN = re.compile(
    r"\b(?:sk-[A-Za-z0-9_\-]{20,}|sk-ant-[A-Za-z0-9_\-]{20,}|hf_[A-Za-z0-9]{24,}|gh[oupsr]_[A-Za-z0-9]{20,}"
    r"|github_pat_[A-Za-z0-9_]{20,}|xai-[A-Za-z0-9]{40,}|AKIA[0-9A-Z]{16}|eyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{10,})"
)


def load_highlights(path: Path) -> Json:
    text = path.read_text(encoding="utf-8")
    doc = json.loads(text[text.index("{") : text.rindex("}") + 1])
    if doc.get("format") != "atif-scan-highlights/1":
        raise SystemExit(f"{path}: not an atif-scan highlight export")
    return doc


def _restates() -> dict[str, str]:
    """Publish-side rule id -> the single check it restates (atif-rules.json)."""
    rules = json.loads((HERE / "atif-rules.json").read_text())["rules"]
    return {
        r["id"]: r["when"]["all"][0]
        for r in rules
        if list(r.get("when", {})) == ["all"] and len(r["when"]["all"]) == 1
    }


def _clip(text: str, keep: str) -> str:
    """Shorten an excerpt part, keeping the side next to the match."""
    if len(text) <= CLIP:
        return text
    return "…" + text[-CLIP:] if keep == "end" else text[:CLIP] + "…"


def build(
    run_id: str,
    highlights: Json,
    bucket_repo: str,
    run: Json | None = None,
    alias: dict[str, str] | None = None,
) -> Json:
    """``alias`` maps atif-scan's trial folder names to the run file's trial names (Hub
    runs name trials by id)."""
    alias = alias or {}
    if run is None:
        run = json.loads((HERE / "runs" / f"{run_id}.json").read_text())
    if "trials" not in run:
        raise SystemExit(f"{run_id}: run file has no trial names; rerun fetch_runs.py")
    rows = {r["key"]: r for g in highlights["groups"] for r in g["rows"]}
    restated = set(_restates().values())

    # Trial name -> (task, attempt index in the task's cells, bucket path).
    where: dict[str, tuple[str, int, str | None]] = {}
    for task, trials in run["trials"].items():
        for i, t in enumerate(trials):
            where[t["name"]] = (task, i, t.get("path"))

    review = run.get("review") or {}
    decisions: dict[str, Json] = {}
    for d in review.get("disqualified", []):
        decisions[d["trial"]] = {"decision": "disqualified", "reason": d["reason"]}
    for c in review.get("cleared", []):
        decisions[c["trial"]] = {
            "decision": "auto-cleared" if c.get("auto") else "cleared",
            "reason": c["reason"],
        }

    # Findings per trial from the highlight rows (medium and up, not explained).
    per_trial: dict[str, dict[str, Json]] = {}
    label: dict[str, str] = {}
    for key, moments in highlights["moments"].items():
        if not key.startswith("check:"):
            continue
        row = rows.get(key, {})
        priority = row.get("priority") or moments[0].get("severity")
        check = key.removeprefix("check:")
        for m in moments:
            trial = highlights["trials"][m["trial"]]
            folder = trial["input_id"].split("/")[0]
            name = alias.get(folder, folder)
            label.setdefault(name, folder)
            f = per_trial.setdefault(name, {}).setdefault(
                check,
                {
                    "check": check,
                    "title": row.get("title") or check,
                    "priority": priority,
                    "moments": [],
                },
            )
            if len(f["moments"]) < MAX_MOMENTS:
                f["moments"].append(
                    {
                        "step": m["step"],
                        "channel": m["channel"],
                        "tool": m.get("tool"),
                        "said": _clip(m.get("said") or "", "end"),
                        "ran": [
                            _clip(m["ran"][0], "end"),
                            m["ran"][1],
                            _clip(m["ran"][2], "start"),
                        ],
                        "got": _clip(m.get("got") or "", "start"),
                    }
                )

    rewards = {}
    for t in highlights["trials"]:
        folder = t["input_id"].split("/")[0]
        rewards[alias.get(folder, folder)] = t.get("reward")
    cards = []
    for name, found in per_trial.items():
        if name not in where:
            continue  # a replaced original: kept as evidence, not reported
        checks = set(found)
        findings = [
            f for c, f in found.items() if not (c in restated and checks & set(_rules_over(c)))
        ]
        # The finding our decision cites comes first, so the card leads with its evidence.
        reason = (decisions.get(name) or {}).get("reason") or ""
        findings.sort(
            key=lambda f: (
                f"({f['check']}" not in reason,
                PRIORITY.get(f["priority"], 9),
                f["title"],
            )
        )
        rewarded = (rewards.get(name) or 0) > 0
        top = PRIORITY.get(findings[0]["priority"], 9) if findings else 9
        if name not in decisions and not (rewarded and top <= PRIORITY["high"]):
            continue
        task, attempt, path = where[name]
        cards.append(
            {
                "trial": label.get(name, name),
                "task": task,
                "attempt": attempt,
                "reward": rewards.get(name),
                **decisions.get(name, {"decision": None, "reason": None}),
                "folder": f"https://huggingface.co/buckets/{bucket_repo}/tree/{path}"
                if path
                else None,
                "trace": (  # the Hub's trace viewer for the trajectory
                    f"https://huggingface.co/buckets/{bucket_repo}/tree/{path}/agent/trajectory.json"
                    if path
                    else None
                ),
                "findings": findings[:MAX_FINDINGS],
            }
        )
    order = {"disqualified": 0, "cleared": 1, "auto-cleared": 2, None: 3}
    cards.sort(key=lambda c: (order[c["decision"]], c["task"], c["attempt"]))
    return {
        "format": "fast-agent-moments/1",
        "run": run_id,
        "scanner_version": highlights["scanner_version"],
        "note": "Masked excerpts (best effort) from atif-scan's highlight export: said / ran / got at each finding.",
        "cards": cards,
    }


_RULES: dict[str, list[str]] = {}


def _rules_over(check: str) -> list[str]:
    """Rule ids that restate this check."""
    if not _RULES:
        for rule, src in _restates().items():
            _RULES.setdefault(src, []).append(rule)
    return _RULES.get(check, [])


def check_clean(doc: Json) -> None:
    text = json.dumps(doc, ensure_ascii=False)
    if hit := TOKEN.search(text):
        raise SystemExit(
            f"refusing to write: token-shaped text ({len(hit.group())} chars) in an excerpt"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run", required=True)
    parser.add_argument(
        "--highlights", type=Path, required=True, help="data.js from atif-scan --highlights"
    )
    args = parser.parse_args()

    manifest = json.loads((HERE / "manifest.json").read_text())
    spec = next(r for r in manifest["runs"] if r["id"] == args.run)
    repo = (spec.get("bucket") or {}).get("repo", "evalstate/published-benchmarks")
    doc = build(args.run, load_highlights(args.highlights), repo)
    check_clean(doc)
    out = HERE / "moments" / f"{args.run}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    n = sum(len(f["moments"]) for c in doc["cards"] for f in c["findings"])
    print(
        f"{args.run}: {len(doc['cards'])} cards, {n} moments -> {out.relative_to(HERE.parent)}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
