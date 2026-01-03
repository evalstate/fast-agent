#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from statistics import median


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    if pct <= 0:
        return min(values)
    if pct >= 100:
        return max(values)
    index = int(round((pct / 100.0) * (len(values) - 1)))
    index = max(0, min(index, len(values) - 1))
    return values[index]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize streaming render timings from JSONL metrics."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=".fast-agent/streaming-demo-*.jsonl",
        help="Path or glob to JSONL metrics files.",
    )
    parser.add_argument(
        "--scenario",
        default="",
        help="Optional scenario filter (e.g. random-mix).",
    )
    parser.add_argument(
        "--phase",
        default="",
        help="Optional phase filter (pre_scroll or scrolling).",
    )
    parser.add_argument(
        "--field",
        default="render_ms",
        help="Numeric field to summarize (default: render_ms).",
    )
    parser.add_argument(
        "--overall",
        action="store_true",
        help="Print total elapsed time summaries from scenario summary events.",
    )
    args = parser.parse_args()

    paths: list[Path] = []
    if any(ch in args.path for ch in "*?[]"):
        paths = [Path(p) for p in glob.glob(args.path)]
    else:
        path = Path(args.path)
        if path.is_dir():
            paths = list(path.glob("*.jsonl"))
        else:
            paths = [path]

    if not paths:
        print(f"metrics file(s) not found: {args.path}")
        return 1

    samples: list[float] = []
    count = 0
    summaries: list[dict[str, object]] = []
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if args.scenario and payload.get("scenario") != args.scenario:
                continue
            if args.phase and payload.get("phase") != args.phase:
                continue
            if payload.get("event") == "summary":
                summaries.append({"path": str(path), **payload})
            value = payload.get(args.field)
            if isinstance(value, (int, float)):
                samples.append(float(value))
                count += 1

    if args.overall and summaries:
        print("overall:")
        for summary in summaries:
            if args.scenario and summary.get("scenario") != args.scenario:
                continue
            elapsed = summary.get("total_elapsed_ms")
            chunks = summary.get("total_chunks")
            chars = summary.get("total_chars")
            if not isinstance(elapsed, (int, float)):
                elapsed = 0.0
            elapsed_sec = elapsed / 1000.0 if elapsed else 0.0
            chunks_per_sec = (chunks / elapsed_sec) if elapsed_sec and chunks else 0.0
            chars_per_sec = (chars / elapsed_sec) if elapsed_sec and chars else 0.0
            print(
                f"{summary.get('path')} scenario={summary.get('scenario')} "
                f"elapsed_ms={elapsed:.2f} chunks={chunks} chars={chars} "
                f"chunks_per_sec={chunks_per_sec:.2f} chars_per_sec={chars_per_sec:.2f}"
            )

    if not samples:
        print(f"no {args.field} samples found")
        return 0

    samples.sort()
    unit = " ms" if args.field.endswith("_ms") else ""
    print(f"field: {args.field}")
    print(f"samples: {count}")
    print(f"min: {samples[0]:.2f}{unit}")
    print(f"p50: {median(samples):.2f}{unit}")
    print(f"p90: {_percentile(samples, 90):.2f}{unit}")
    print(f"p95: {_percentile(samples, 95):.2f}{unit}")
    print(f"p99: {_percentile(samples, 99):.2f}{unit}")
    print(f"max: {samples[-1]:.2f}{unit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
