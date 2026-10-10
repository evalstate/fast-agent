# /// script
# requires-python = ">=3.12"
# dependencies = ["pillow"]
# ///
"""Calibrate the TB4 subset against the Terminal-Bench 4.0 leaderboard and draw the charts.

Reads the leaderboard trials cached by fetch_leaderboard.py, plus subset.json, tasks.json and
selection.json (what the subset was chosen from). Writes:

    calibration.json                                  headline figures quoted on the docs page
    docs/docs/assets/benchmarks/tb4-subset/*.webp     random-picks, per-entry and cost charts

    uv run docs/benchmark_data/tb4/calibration.py [--no-charts]

Rows published after the selection's freeze date are out-of-sample and reported separately.
The charts are rendered with headless Chromium (CHROMIUM overrides the binary) and load
their fonts from Google Fonts.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics as st
import subprocess
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from html import escape
from pathlib import Path
from typing import Any

from PIL import Image

HERE = Path(__file__).resolve().parent
CACHE = HERE / ".cache"
ASSETS = HERE.parents[1] / "docs" / "assets" / "benchmarks" / "tb4-subset"
N_RANDOM = 20_000
SEED = 4
REFUSAL = "AgentSafetyRefusalError"


# ─────────────────────────────────────────────────────────────── data
@dataclass
class TaskResult:
    passes: int
    trials: int
    cost: float  # sum of recorded trial costs
    missing_costs: int


@dataclass
class Row:
    id: str
    agent: str
    model: str
    effort: str
    listed: str
    in_selection: bool
    full_score: float
    full_cost: float
    trial_passes: int
    published_passes: int
    trial_cost: float
    missing_costs: int
    tasks: dict[str, TaskResult]
    sub_score: float = 0.0
    sub_cost: float = 0.0
    est_cost: float = 0.0
    naive_cost: float = 0.0

    @property
    def label(self) -> str:
        return f"{self.model} · {self.effort}"

    @property
    def score_ok(self) -> bool:
        """Trial records reproduce the published score."""
        return self.trial_passes == self.published_passes

    @property
    def cost_ok(self) -> bool:
        """Every trial has a cost and they sum to the published total."""
        return self.missing_costs == 0 and abs(self.trial_cost - self.full_cost) < 0.5


@dataclass
class Data:
    rows: list[Row]
    tasks: list[str]
    subset: list[str]
    eligible: list[str]
    excluded: dict[str, list[str]]
    late_refusals: list[str]
    other_revision: set[str]
    frozen: str
    fetched: str
    refusals: list[tuple[str, str]] = field(default_factory=list)


def task_name(trial: dict[str, Any]) -> str:
    return trial["task_name"].split("/")[-1]


def load() -> Data:
    selection = json.loads((HERE / "selection.json").read_text())
    subset: list[str] = json.loads((HERE / "subset.json").read_text())
    tasks = sorted(json.loads((HERE / "tasks.json").read_text()))
    selection_ids = set(selection["rows"])
    rows: list[Row] = []
    refusals: list[tuple[str, str]] = []
    for path in sorted((CACHE / "rows").glob("*.json")):
        export = json.loads(path.read_text())
        row, trials = export["row"], export["trials"]
        meta, metrics = row["metadata"], row["metrics"]
        by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for trial in trials:
            by_task[task_name(trial)].append(trial)
        refusals += [(row["id"], task_name(t)) for t in trials if t["error_type"] == REFUSAL]
        rows.append(
            Row(
                id=row["id"],
                agent=meta["agent_display"]["label"],
                model=meta["model_display"]["label"],
                effort=meta["reasoning_effort"],
                listed=row["created_at"][:10],
                in_selection=row["id"] in selection_ids,
                full_score=metrics["accuracy"],
                full_cost=metrics["total_cost_usd"],
                trial_passes=sum(t["reward"] == 1 for t in trials),
                published_passes=metrics["successes"],
                trial_cost=sum(t["cost_usd"] or 0 for t in trials),
                missing_costs=sum(t["cost_usd"] is None for t in trials),
                tasks={
                    name: TaskResult(
                        passes=sum(t["reward"] == 1 for t in ts),
                        trials=len(ts),
                        cost=sum(t["cost_usd"] or 0 for t in ts),
                        missing_costs=sum(t["cost_usd"] is None for t in ts),
                    )
                    for name, ts in by_task.items()
                },
            )
        )
    for row in rows:
        if sorted(row.tasks) != tasks:
            raise ValueError(f"row {row.id} doesn't cover the 66 tasks in tasks.json")

    # Rule 3 (no safety refusals) as it stood at the freeze; later refusals are reported only.
    excluded = {t: list(reasons) for t, reasons in selection["excluded"].items()}
    refused_then = {t for rid, t in refusals if rid in selection_ids}
    for t in sorted(refused_then):
        excluded.setdefault(t, []).append("safety refusal")
    late = sorted({t for rid, t in refusals if rid not in selection_ids} - refused_then)
    eligible = [t for t in tasks if t not in excluded]
    if not set(subset) <= set(eligible):
        raise ValueError(f"subset tasks outside the eligible pool: {set(subset) - set(eligible)}")

    digests = json.loads((CACHE / "digests.json").read_text())
    pinned = selection["digests"]
    other = {rid for rid, t, d in digests if d != pinned[t]}
    board = json.loads((CACHE / "leaderboard.json").read_text())
    return Data(
        rows=rows,
        tasks=tasks,
        subset=subset,
        eligible=eligible,
        excluded=excluded,
        late_refusals=late,
        other_revision=other,
        frozen=selection["frozen"],
        fetched=board["fetched"],
        refusals=refusals,
    )


# ─────────────────────────────────────────────────────────────── statistics
def pearson(a: list[float], b: list[float]) -> float:
    ma, mb = st.mean(a), st.mean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b, strict=True))
    den = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
    return num / den


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            out[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return out


def spearman(a: list[float], b: list[float]) -> float:
    return pearson(ranks(a), ranks(b))


def sub_score(row: Row, tasks: list[str]) -> float:
    passes = sum(row.tasks[t].passes for t in tasks)
    return 100 * passes / sum(row.tasks[t].trials for t in tasks)


def sub_cost(row: Row, tasks: list[str]) -> float:
    return sum(row.tasks[t].cost for t in tasks)


def score_stats(tasks: list[str], rows: list[Row]) -> dict[str, float]:
    sub = [sub_score(r, tasks) for r in rows]
    full = [r.full_score for r in rows]
    gaps = [s - f for s, f in zip(sub, full, strict=True)]
    return {
        "rows": len(rows),
        "mean_gap": st.mean(abs(g) for g in gaps),
        "signed_gap": st.mean(gaps),
        "within5": sum(abs(g) <= 5 for g in gaps),
        "within10": sum(abs(g) <= 10 for g in gaps),
        "spearman": spearman(sub, full),
    }


def cost_share(tasks: list[str], rows: list[Row]) -> float:
    return st.median(sub_cost(r, tasks) / r.full_cost for r in rows)


def analyse(data: Data) -> dict[str, Any]:
    subset, tasks = data.subset, data.tasks
    rest = [t for t in tasks if t not in subset]
    scored = [r for r in data.rows if r.score_ok]
    costed = [r for r in data.rows if r.cost_ok]
    cohorts = {
        "selection": [r for r in scored if r.in_selection],
        "new": [r for r in scored if not r.in_selection],
    }

    # Cost: multiplier from other models' rows (every effort of the target model held out).
    ratio = {r.id: r.full_cost / sub_cost(r, subset) for r in costed}
    errors: list[float] = []
    naive: list[float] = []
    for r in data.rows:
        r.sub_score = sub_score(r, subset)
        r.sub_cost = sub_cost(r, subset)
        r.naive_cost = r.sub_cost * len(tasks) / len(subset)
    for r in costed:
        k = st.median(ratio[o.id] for o in costed if o.model != r.model)
        r.est_cost = k * r.sub_cost
        errors.append(abs(r.est_cost - r.full_cost) / r.full_cost)
        naive.append(r.naive_cost / r.full_cost - 1)
    worst = max(costed, key=lambda r: abs(r.est_cost / r.full_cost - 1))

    # Random 19-task picks from the eligible pool, scored on the same rows.
    rng = random.Random(SEED)
    draws: dict[str, list[float]] = {"all": [], "share": []}
    cohort_draws = {c: {"gap": [], "spearman": []} for c in cohorts}
    for _ in range(N_RANDOM):
        pick = rng.sample(data.eligible, len(subset))
        draws["all"].append(score_stats(pick, scored)["mean_gap"])
        draws["share"].append(cost_share(pick, costed))
        for name, rows in cohorts.items():
            s = score_stats(pick, rows)
            cohort_draws[name]["gap"].append(s["mean_gap"])
            cohort_draws[name]["spearman"].append(s["spearman"])

    score = score_stats(subset, scored)
    share = cost_share(subset, costed)
    by_cohort = {}
    for name, rows in cohorts.items():
        s = score_stats(subset, rows)
        d = cohort_draws[name]
        by_cohort[name] = {
            **s,
            "random_median_gap": st.median(d["gap"]),
            "random_closer": sum(g < s["mean_gap"] for g in d["gap"]) / N_RANDOM,
            "random_median_spearman": st.median(d["spearman"]),
            "random_rank_better": sum(v > s["spearman"] for v in d["spearman"]) / N_RANDOM,
        }
    held_sub = [sub_score(r, subset) for r in scored]
    held_rest = [sub_score(r, rest) for r in scored]
    rel_cost = {
        t: st.mean(r.tasks[t].cost / (r.full_cost / len(tasks)) for r in costed) for t in tasks
    }
    excluded_tasks = [t for t in tasks if t not in data.eligible]
    by_full = sorted(scored, key=lambda r: r.full_score)
    third = len(by_full) // 3
    return {
        "leaderboard_fetched": data.fetched,
        "frozen": data.frozen,
        "rows": len(data.rows),
        "scored_rows": len(scored),
        "scored_models": len({r.model for r in scored}),
        "unscored_rows": [r.label for r in data.rows if not r.score_ok],
        "costed_rows": len(costed),
        "tasks": len(tasks),
        "subset_tasks": len(subset),
        "eligible_tasks": len(data.eligible),
        "excluded_tasks": data.excluded,
        "late_refusals_in_subset": sorted(set(data.late_refusals) & set(subset)),
        "score": score,
        "score_by_cohort": by_cohort,
        "compression": {
            "weakest_third": st.mean(r.sub_score - r.full_score for r in by_full[:third]),
            "strongest_third": st.mean(r.sub_score - r.full_score for r in by_full[-third:]),
        },
        "held_out": {
            "mean_gap": st.mean(abs(a - b) for a, b in zip(held_sub, held_rest, strict=True)),
            "spearman": spearman(held_sub, held_rest),
        },
        "random": {
            "picks": N_RANDOM,
            "median_gap": st.median(draws["all"]),
            "median_cost_share": st.median(draws["share"]),
            "closer": sum(g < score["mean_gap"] for g in draws["all"]) / N_RANDOM,
            "cheaper_and_closer": sum(
                s <= share and g <= score["mean_gap"]
                for s, g in zip(draws["share"], draws["all"], strict=True)
            ),
        },
        "cost": {
            "share": share,
            "median_ratio": st.median(ratio.values()),
            "min_ratio": min(ratio.values()),
            "max_ratio": max(ratio.values()),
            "mean_error": st.mean(errors),
            "worst": {"row": worst.label, "error": worst.est_cost / worst.full_cost - 1},
            "naive_error": st.mean(naive),
            "subset_task_cost": share * len(tasks) / len(subset),
            "eligible_task_cost": st.mean(rel_cost[t] for t in data.eligible),
            "excluded_task_cost": st.mean(rel_cost[t] for t in excluded_tasks),
        },
        # Chart inputs; not written to calibration.json.
        "_draws": draws,
        "_cohort_draws": cohort_draws,
        "_scored": scored,
        "_costed": costed,
        "_pass_rate": {
            t: st.mean(r.tasks[t].passes / r.tasks[t].trials for r in scored) for t in subset
        },
    }


# ─────────────────────────────────────────────────────────────── drawing
# fast-agent "Forward" palette, unbranded: ivory ground, petrol ink, teal/orange accents.
BG, PAPER, DEEP = "#FFF7E8", "#FFFCF4", "#F4EAD5"
FG, FG1, FG2 = "#082C34", "#082C34", "#4D6566"
LINE = GRID = "#DCDBCF"
TEAL, ORANGE, PETROL2 = "#277C80", "#F45125", "#11414B"
BAND = "#E3E8DB"
READ = "'Figtree',system-ui,sans-serif"
VOICE = "'Fraunces',Georgia,serif"
CODE = "'DM Mono',ui-monospace,monospace"
FONTS = (
    "https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght,SOFT,WONK@9..144,100..900,0..100,0..1"
    "&family=Figtree:wght@400;500;600;700;800;900&family=DM+Mono:wght@400;500&display=swap"
)


def text(
    x: float,
    y: float,
    s: object,
    size: float = 16,
    fill: str = FG,
    weight: int = 400,
    anchor: str = "start",
    extra: str = "",
    family: str = READ,
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="{family}" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}" {extra}>{escape(str(s))}</text>'
    )


def voice(x: float, y: float, s: str, size: float = 44) -> str:
    """Headlines: Fraunces 900, SOFT 100, tracking -0.035em."""
    style = f"letter-spacing=\"{-0.035 * size:.2f}\" style=\"font-variation-settings:'SOFT' 100,'WONK' 0\""
    return text(x, y, s, size, FG, 900, "start", style, VOICE)


def mono(x: float, y: float, s: str, size: float = 14.5) -> str:
    return text(x, y, s, size, FG, 500, "start", "", CODE)


def kicker(x: float, y: float, s: str) -> str:
    return text(x, y, s.upper(), 13, TEAL, 800, "start", 'letter-spacing="1.82"')


def small_caps(x: float, y: float, s: str, anchor: str = "start") -> str:
    return text(x, y, s, 11.5, FG2, 800, anchor, 'letter-spacing="1.6"')


def line(
    x1: float, y1: float, x2: float, y2: float, c: str = LINE, w: float = 1, extra: str = ""
) -> str:
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{c}" stroke-width="{w}" {extra}/>'


def dot(x: float, y: float, r: float, fill: str = FG, stroke: str = BG, sw: float = 2) -> str:
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'


def rect(x: float, y: float, w: float, h: float, fill: str, rx: float = 3, extra: str = "") -> str:
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}" {extra}/>'


def card(x: float, y: float, w: float, h: float) -> str:
    return rect(x, y, w, h, PAPER, 14, f'stroke="{FG}" stroke-width="2"')


def section(x: float, y: float, title: str, sub: str) -> str:
    return voice(x, y, title, 26) + text(x, y + 27, sub, 15, FG2, 500)


def footer(width: float, y0: float, lines: list[str]) -> str:
    out = line(60, y0, width - 60, y0, FG, 2)
    for i, s in enumerate(lines):
        out += text(60, y0 + 30 + i * 20, s, 12.5, FG2)
    return out


def day(iso: str) -> str:
    return date.fromisoformat(iso).strftime("%-d %b %Y")


def render(body: str, width: int, height: int, name: str) -> None:
    """SVG → HTML → 2x PNG screenshot → WebP in the docs assets."""
    html = (
        f'<!doctype html><html><head><meta charset="utf-8"><link rel="stylesheet" href="{FONTS}">'
        f"<style>html,body{{margin:0;width:{width}px;height:{height}px;overflow:hidden;background:{BG}}}"
        "text{font-variant-numeric:tabular-nums}</style></head><body>"
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}"><rect width="{width}" height="{height}" fill="{BG}"/>'
        f"{body}</svg></body></html>"
    )
    with tempfile.TemporaryDirectory() as tmp:
        page, png = Path(tmp) / f"{name}.html", Path(tmp) / f"{name}.png"
        page.write_text(html)
        subprocess.run(
            [
                os.environ.get("CHROMIUM", "chromium"),
                "--headless=new",
                "--disable-gpu",
                "--hide-scrollbars",
                f"--window-size={width},{height}",
                "--virtual-time-budget=6000",
                "--force-device-scale-factor=2",
                f"--screenshot={png}",
                page.as_uri(),
            ],
            capture_output=True,
            check=True,
            timeout=120,
        )
        ASSETS.mkdir(parents=True, exist_ok=True)
        out = ASSETS / f"{name}.webp"
        Image.open(png).save(out, "WEBP", quality=88, method=6)
    print("wrote", out.relative_to(HERE.parents[2]))


# ─────────────────────────────────────────────────────────────── charts
def chart_random(a: dict[str, Any]) -> None:
    width, height = 1600, 1030
    n, picks, eligible = a["subset_tasks"], a["random"]["picks"], a["eligible_tasks"]
    share, gap = a["cost"]["share"], a["score"]["mean_gap"]
    cohorts = a["score_by_cohort"]
    b = kicker(60, 60, "Terminal-Bench 4.0 · task selection check")
    b += voice(60, 106, f"Why these {n} tasks, and not any {n}?", 46)
    b += text(
        60,
        142,
        f"We compared the subset with {picks:,} random {n}-task picks from the {eligible} tasks "
        "that meet the same constraints (single container, no GPU, no refusals).",
        19,
        FG1,
    )

    # Left: cost share against score gap, one grey dot per random pick.
    x0, y0, pw, ph = 140, 300, 620, 560
    b += section(
        60,
        228,
        "Cheaper and closer than almost any random pick",
        f"Each grey dot is one random pick, scored on all {a['scored_rows']} leaderboard entries",
    )
    xlo, xhi, ylo, yhi = 0.10, 0.42, 1.5, 12.5

    def px(v: float) -> float:
        return x0 + pw * (v - xlo) / (xhi - xlo)

    def py(v: float) -> float:
        return y0 + ph - ph * (v - ylo) / (yhi - ylo)

    for v in (0.1, 0.2, 0.3, 0.4):
        b += line(px(v), y0, px(v), y0 + ph, GRID)
        b += text(px(v), y0 + ph + 26, f"{v * 100:.0f}%", 14, FG2, 500, "middle")
    for v in range(2, 13, 2):
        b += line(x0, py(v), x0 + pw, py(v), GRID) + text(
            x0 - 12, py(v) + 5, v, 14, FG2, 500, "end"
        )
    b += rect(x0, py(gap), px(share) - x0, y0 + ph - py(gap), BAND, 0)
    cloud = [
        (s, g)
        for s, g in zip(a["_draws"]["share"], a["_draws"]["all"], strict=True)
        if xlo <= s <= xhi and ylo <= g <= yhi
    ]
    for s, g in random.Random(1).sample(cloud, min(6000, len(cloud))):
        b += f'<circle cx="{px(s):.1f}" cy="{py(g):.1f}" r="2.1" fill="#B9B6A8" fill-opacity="0.55"/>'
    b += line(x0, y0 + ph, x0 + pw, y0 + ph, "#b3ada2", 1.5) + line(
        x0, y0, x0, y0 + ph, "#b3ada2", 1.5
    )
    dash = 'stroke-dasharray="5 4"'
    b += line(px(share), py(gap), px(share), y0 + ph, TEAL, 1.5, dash)
    b += line(x0, py(gap), px(share), py(gap), TEAL, 1.5, dash)
    b += dot(px(share), py(gap), 11, ORANGE, BG, 3)
    b += rect(px(share) + 14, py(gap) - 50, 268, 52, BG, 8, 'fill-opacity="0.92"')
    b += text(px(share) + 22, py(gap) - 30, "our subset", 17, FG, 800)
    b += text(
        px(share) + 22,
        py(gap) - 10,
        f"{share * 100:.0f}% of the cost, {gap:.1f}-point gap",
        15,
        FG1,
        600,
    )
    b += text(x0 + 10, y0 + ph - 34, "cheaper and closer:", 13, TEAL, 700)
    b += text(
        x0 + 10, y0 + ph - 15, f"{a['random']['cheaper_and_closer']} of {picks:,}", 13, TEAL, 700
    )
    typical = a["random"]["median_cost_share"]
    b += text(
        px(typical),
        y0 + 24,
        f"typical random pick: {typical * 100:.0f}% of the cost, {a['random']['median_gap']:.1f}-point gap",
        14,
        FG2,
        600,
        "middle",
    )
    b += text(x0 + pw / 2, y0 + ph + 60, "Cost as a share of a full run", 17, FG, 700, "middle")
    b += text(
        x0 - 52,
        y0 + ph / 2,
        "Average gap to the full score (points)",
        17,
        FG,
        700,
        "middle",
        f'transform="rotate(-90 {x0 - 52} {y0 + ph / 2})"',
    )

    # Right: distribution of random picks' gaps, for the original and the later entries.
    rx = 860
    hw = width - 60 - rx
    b += section(
        rx,
        228,
        "It held up on entries it had never seen",
        "Average gap for every random pick, against our subset (orange)",
    )
    hlo, hhi, bin_w = 0.0, 13.0, 0.5

    def hx(v: float) -> float:
        return rx + hw * (v - hlo) / (hhi - hlo)

    frozen = day(a["frozen"])
    panels = [
        (
            "selection",
            f"The {cohorts['selection']['rows']} entries on the leaderboard when we chose it ({frozen})",
        ),
        ("new", f"The {cohorts['new']['rows']} entries published after we froze it"),
    ]
    hh, gap_y = 200, 92
    for i, (name, title) in enumerate(panels):
        top = 300 + i * (hh + gap_y)
        base = top + hh
        bins = int((hhi - hlo) / bin_w)
        counts = [0] * bins
        for v in a["_cohort_draws"][name]["gap"]:
            counts[min(bins - 1, int((v - hlo) / bin_w))] += 1
        ours = cohorts[name]["mean_gap"]
        b += text(rx, top - 14, title, 16, FG, 700)
        for k, count in enumerate(counts):
            bar = (hh - 50) * count / max(counts)
            colour = "#8FBAB5" if hlo + (k + 1) * bin_w <= ours else "#D6D2C4"
            b += rect(
                hx(hlo + k * bin_w) + 1, base - bar, hw * bin_w / (hhi - hlo) - 2, bar, colour, 2
            )
        b += line(rx, base, rx + hw, base, "#b3ada2", 1.5)
        b += line(hx(ours), top + 4, hx(ours), base, ORANGE, 3)
        closer = 1 - cohorts[name]["random_closer"]
        b += text(hx(ours) - 12, top + 70, f"our subset: {ours:.1f} points", 15, FG, 800, "end")
        b += text(hx(ours) - 12, top + 90, f"closer than {closer * 100:.0f}%", 14, FG1, 600, "end")
        b += text(hx(ours) - 12, top + 108, "of random picks", 14, FG1, 600, "end")
        median = cohorts[name]["random_median_gap"]
        b += line(hx(median), top + 24, hx(median), base, FG2, 1.5, 'stroke-dasharray="4 4"')
        b += text(hx(median) + 8, top + 22, f"median random pick: {median:.1f}", 13, FG2, 600)
        for v in range(0, 14, 2):
            b += text(hx(v), base + 20, v, 13, FG2, 500, "middle")
    b += text(
        rx + hw / 2,
        300 + 2 * hh + gap_y + 52,
        "Average gap to the full score (points)",
        15,
        FG,
        700,
        "middle",
    )

    new = [r for r in a["_scored"] if not r.in_selection]
    sel = cohorts["selection"]
    b += footer(
        width,
        height - 90,
        [
            (
                f"Source: public Terminal-Bench 4.0 leaderboard, every trial of {a['scored_rows']} entries "
                f"(fetched {day(a['leaderboard_fetched'])}; subset frozen {frozen}). Cost share is the median "
                f"over the {a['costed_rows']} entries with complete trial costs."
            ),
            (
                f"Random picks draw {n} of the {eligible} eligible tasks. On rank order alone, random picks do "
                f"about as well (Spearman {sel['random_median_spearman']:.2f} vs our {sel['spearman']:.2f} on the "
                f"original {sel['rows']}); the subset was chosen for score and cost, not ranking."
            ),
            (
                f"{len(new)} new entries is a small test: they cover {len({r.model for r in new})} models from "
                f"{len({r.agent for r in new})} agents. A single {n}-task run still carries roughly ±10 points of noise."
            ),
        ],
    )
    render(b, width, height, "random")


EFFORTS = ["max", "xhigh", "high", "medium", "low", "none"]


def heat(passes: int) -> str:
    """Passes out of 5 → flat step colour: ivory-deep, then teal tints to petrol-2."""
    return [DEEP, "#CFE2DB", "#98C3BD", "#5A9C9C", TEAL, PETROL2][passes]


def chart_models(a: dict[str, Any], data: Data) -> None:
    scored: list[Row] = a["_scored"]
    n = a["subset_tasks"]
    tasks = sorted(data.subset, key=lambda t: a["_pass_rate"][t])
    families: dict[str, list[Row]] = defaultdict(list)
    for r in scored:
        families[r.model].append(r)
    order = sorted(families, key=lambda m: -max(r.full_score for r in families[m]))
    for m in order:
        families[m].sort(key=lambda r: EFFORTS.index(r.effort))
    width = 1600
    height = 1610 + (len(scored) - 26) * 27 + (len(order) - 15) * 10

    def peers(r: Row) -> list[Row]:
        return [o for o in scored if o.model != r.model and abs(o.full_score - r.full_score) <= 8]

    b = kicker(60, 60, "Terminal-Bench 4.0 · task selection check")
    b += voice(60, 106, f"How each leaderboard entry does on the {n} tasks", 46)
    b += text(
        60,
        142,
        "Passes out of 5 attempts, for every public entry on every subset task. "
        "Existing leaderboard trials, not a new harness run.",
        19,
        FG1,
    )
    lab_r, gx0, cw, ch, gap = 290, 310, 40, 27, 3
    top = 400
    for j, t in enumerate(tasks):
        cx = gx0 + j * cw + cw / 2
        b += text(
            cx + 4,
            top - 44,
            t,
            12.5,
            FG1,
            500,
            "start",
            f'transform="rotate(-52 {cx + 4:.1f} {top - 44})"',
            CODE,
        )
        b += rect(gx0 + j * cw + 3, top - 30, cw - 6, 6, DEEP, 2)
        b += rect(gx0 + j * cw + 3, top - 30, max((cw - 6) * a["_pass_rate"][t], 2), 6, FG2, 2)
    b += text(gx0 - 12, top - 23, "leaderboard pass rate", 11.5, FG2, 600, "end")
    b += text(gx0, top - 8, "hard →", 12, FG2, 600) + text(
        gx0 + n * cw, top - 8, "→ easy", 12, FG2, 600, "end"
    )
    fx, sx, gxx, dx0, dx1 = 1150, 1230, 1300, 1330, 1540

    def dx(v: float) -> float:
        return dx0 + (dx1 - dx0) * v / 70

    b += small_caps(fx, top - 8, "FULL", "end") + small_caps(sx, top - 8, "SUBSET", "end")
    b += small_caps(gxx, top - 8, "GAP", "end")
    b += dot(dx0 + 6, top - 12, 5.5, BG, FG, 2) + text(dx0 + 16, top - 8, "full", 12, FG2, 600)
    b += dot(dx0 + 62, top - 12, 6, TEAL, BG, 1) + text(dx0 + 72, top - 8, "subset", 12, TEAL, 700)
    y = top
    for fi, m in enumerate(order):
        if fi:
            y += 10
        for r in families[m]:
            cy = y + ch / 2
            g = r.sub_score - r.full_score
            far, new = abs(g) > 5, not r.in_selection
            dagger = " †" if r.id in data.other_revision else ""
            b += text(
                lab_r,
                cy + 5,
                f"{r.label}{dagger}",
                14,
                TEAL if new else FG,
                800 if far or new else 500,
                "end",
            )
            near = peers(r)
            for j, t in enumerate(tasks):
                p = r.tasks[t].passes
                x = gx0 + j * cw
                b += rect(x + gap / 2, y + gap / 2, cw - gap, ch - gap, heat(p), 4)
                if p:
                    b += text(x + cw / 2, cy + 4.5, p, 12.5, BG if p >= 3 else FG, 700, "middle")
                if near:
                    d = p / 5 - st.mean(o.tasks[t].passes / 5 for o in near)
                    if abs(d) >= 0.6:
                        style = (
                            'stroke-width="2.5"'
                            if d > 0
                            else 'stroke-width="2" stroke-dasharray="3.5 2.5"'
                        )
                        b += rect(
                            x + gap / 2 + 1,
                            y + gap / 2 + 1,
                            cw - gap - 2,
                            ch - gap - 2,
                            "none",
                            4,
                            f'stroke="{ORANGE}" {style}',
                        )
            b += text(fx, cy + 5, f"{r.full_score:.0f}%", 14, FG1, 600, "end")
            b += text(sx, cy + 5, f"{r.sub_score:.0f}%", 14, TEAL, 700, "end")
            b += text(gxx, cy + 5, f"{g:+.1f}", 14, FG if far else FG2, 800 if far else 500, "end")
            b += line(dx0, cy, dx1, cy, "#efece5", 1)
            x1, x2 = dx(r.full_score), dx(r.sub_score)
            b += line(min(x1, x2), cy, max(x1, x2), cy, ORANGE if far else "#A9CBC7", 2.5)
            b += dot(x1, cy, 5.5, BG, FG, 2) + dot(x2, cy, 6, TEAL, BG, 1)
            y += ch
    for v in (0, 20, 40, 60):
        b += text(dx(v), y + 18, f"{v}%", 11.5, FG2, 500, "middle")

    ly = y + 46
    b += text(60, ly, "Passes of 5:", 13.5, FG1, 700)
    for p in range(6):
        b += rect(150 + p * 34, ly - 16, 30, 22, heat(p), 4)
        b += text(165 + p * 34, ly, p, 12, BG if p >= 3 else FG, 700, "middle")
    b += rect(380, ly - 16, 30, 22, DEEP, 4, f'stroke="{ORANGE}" stroke-width="2.5"')
    b += text(
        418,
        ly,
        "much better than entries with a similar full score (≥3 more passes of 5)",
        13,
        FG1,
        500,
    )
    b += rect(
        920,
        ly - 16,
        30,
        22,
        DEEP,
        4,
        f'stroke="{ORANGE}" stroke-width="2" stroke-dasharray="3.5 2.5"',
    )
    b += text(958, ly, "much worse", 13, FG1, 500)
    b += text(1060, ly, "† ran a different revision of the subset tasks", 13, FG1, 500)
    b += text(1060, ly + 20, f"teal names: published after {day(a['frozen'])}", 13, TEAL, 700)

    def family_passes(model: str, task: str) -> str:
        rows = families[model]
        return f"{sum(r.tasks[task].passes for r in rows)}/{5 * len(rows)}"

    def entry(model: str, effort: str) -> Row:
        return next(r for r in scored if r.model == model and r.effort == effort)

    luna, low, medium = (
        entry("GPT-5.6 Luna", "max"),
        entry("Fable 5.1", "low"),
        entry("Fable 5.1", "medium"),
    )
    notes = [
        (
            "Model families have signature tasks",
            [
                f"GPT-6 Astra passes wal-recovery-ordering {family_passes('GPT-6 Astra', 'wal-recovery-ordering')}",
                f"times but gsea-proteomics {family_passes('GPT-6 Astra', 'gsea-proteomics')}; Fable 5.1 is",
                (
                    f"the mirror image ({family_passes('Fable 5.1', 'gsea-proteomics')} and "
                    f"{family_passes('Fable 5.1', 'wal-recovery-ordering')}). A mix of"
                ),
                "both kinds keeps any one family from being",
                "favoured — but see † on Astra.",
            ],
        ),
        (
            "Big gaps come from a few tasks",
            [
                f"GPT-5.6 Luna's +{luna.sub_score - luna.full_score:.0f} is mostly two tasks it aces",
                f"(atrx-vep-crispr {luna.tasks['atrx-vep-crispr'].passes}/5, photonic-waveguide-routing",
                f"{luna.tasks['photonic-waveguide-routing'].passes}/5) where similar entries rarely pass.",
                f"With {n} tasks, one or two surprises move",
                "a score by 5–10 points.",
            ],
        ),
        (
            "Big effort steps show; small ones don’t",
            [
                f"Fable 5.1 low → medium: +{medium.full_score - low.full_score:.0f} pts on the full benchmark,",
                f"+{medium.sub_score - low.sub_score:.0f} on the subset. Differences under ~4 points",
                f"(e.g. high vs xhigh) can reverse on {n} tasks.",
            ],
        ),
    ]
    ny, nh = ly + 54, 184
    nw = (width - 120 - 2 * 24) / 3
    for i, (head, lines) in enumerate(notes):
        x = 60 + i * (nw + 24)
        b += card(x, ny, nw, nh) + voice(x + 22, ny + 38, head, 19.5)
        for k, s in enumerate(lines):
            b += text(x + 22, ny + 66 + k * 23, s, 14.5, FG1, 500)
    b += line(60, height - 70, width - 60, height - 70)
    b += text(
        60,
        height - 42,
        f"Source: public Terminal-Bench 4.0 leaderboard trials, fetched {day(a['leaderboard_fetched'])} "
        f"({a['scored_rows']} entries, 5 attempts per task; {', '.join(a['unscored_rows'])} excluded). "
        "Similar entries = other models within ±8 points on the full benchmark. Rows grouped by model.",
        12.5,
        FG2,
    )
    render(b, width, height, "models")


def chart_cost(a: dict[str, Any]) -> None:
    width, height = 1600, 1080
    cost = a["cost"]
    k = cost["median_ratio"]
    n, ntasks = a["subset_tasks"], a["tasks"]
    trials = 5 * n
    naive = ntasks / n
    costed: list[Row] = a["_costed"]
    b = kicker(60, 60, "Terminal-Bench 4.0 · task selection")
    b += voice(60, 106, f"What will a full run cost? About {k:.1f}× the subset", 46)
    b += text(
        60,
        142,
        f"Median full ÷ subset cost is {k:.2f}× across the {a['costed_rows']} leaderboard entries with "
        f"complete cost records (every entry between {cost['min_ratio']:.1f}× and {cost['max_ratio']:.1f}×).",
        17,
        FG1,
    )
    b += text(
        60,
        168,
        f"Fitted without the target model, estimates are off by {cost['mean_error'] * 100:.1f}% on average. "
        f"Trial-count scaling ({5 * ntasks} ÷ {trials} = {naive:.2f}×) undershoots every entry, "
        f"by {abs(cost['naive_error']) * 100:.0f}%.",
        17,
        FG1,
    )

    # Left: estimate against the published cost, per entry.
    y0, x0 = 228, 60
    b += section(
        x0,
        y0,
        "Estimated vs actual full-run cost",
        f"Subset cost × {k:.1f}, where the multiplier comes from the other models",
    )
    ly = y0 + 62
    b += dot(x0 + 8, ly - 5, 7, FG, BG, 1.5) + text(
        x0 + 22, ly, "actual (leaderboard)", 14, FG1, 600
    )
    b += dot(x0 + 200, ly - 5, 7, TEAL, BG, 1.5) + text(
        x0 + 214, ly, f"estimate (× {k:.1f})", 14, TEAL, 700
    )
    b += dot(x0 + 380, ly - 5, 5, BG, FG2, 1.8) + text(
        x0 + 392, ly, f"trial-count scaling (× {naive:.1f})", 14, FG2, 600
    )
    lab_r, ax0, ax1 = 290, 310, 740
    hi = 1000 * math.ceil(
        max(max(r.full_cost, r.est_cost, r.naive_cost) for r in costed) / 1000 + 0.3
    )

    def cx(v: float) -> float:
        return ax0 + (ax1 - ax0) * v / hi

    rows = sorted(costed, key=lambda r: -r.full_cost)
    rh = 33
    top = ly + 24
    bottom = top + len(rows) * rh
    for v in range(0, int(hi) + 1, 2000):
        b += line(cx(v), top - 4, cx(v), bottom, GRID)
        b += text(cx(v), bottom + 20, "$0" if v == 0 else f"${v // 1000}k", 13, FG2, 500, "middle")
    for i, r in enumerate(rows):
        cy = top + i * rh + rh / 2
        new = not r.in_selection
        b += text(lab_r, cy + 5, r.label, 14, TEAL if new else FG1, 700 if new else 500, "end")
        xa, xe, xn = cx(r.full_cost), cx(r.est_cost), cx(r.naive_cost)
        b += line(xn, cy, xa, cy, DEEP, 2) + line(min(xa, xe), cy, max(xa, xe), cy, "#A9CBC7", 3)
        b += (
            dot(xn, cy, 5, BG, FG2, 1.8)
            + dot(xa, cy, 7, FG, BG, 1.5)
            + dot(xe, cy, 7, TEAL, BG, 1.5)
        )
    b += text(
        (ax0 + ax1) / 2,
        bottom + 46,
        "Full-run cost (reported by the leaderboard)",
        15,
        FG,
        700,
        "middle",
    )

    # Right: why the multiplier is larger than the trial ratio.
    rx = 840
    b += section(
        rx, y0, f"Why {k:.1f}× and not {naive:.1f}×?", "The subset avoids the expensive tasks"
    )
    bars = [
        (f"All {ntasks} tasks", 1.0, FG),
        ("GPU, multi-container and refusal tasks (excluded)", cost["excluded_task_cost"], FG2),
        (
            f"Tasks meeting the constraints ({a['eligible_tasks']})",
            cost["eligible_task_cost"],
            "#8FBAB5",
        ),
        (f"This subset ({n})", cost["subset_task_cost"], TEAL),
    ]
    bw, by = 560, y0 + 80
    b += text(
        rx,
        by - 8,
        "Average cost of a task, relative to an average Terminal-Bench 4.0 task",
        14,
        FG2,
        600,
    )
    for i, (label, v, colour) in enumerate(bars):
        y = by + 14 + i * 74
        b += text(rx, y + 16, label, 16, FG, 700)
        b += rect(rx, y + 28, bw * v / 1.5, 24, colour, 6)
        b += text(
            rx + bw * v / 1.5 + 12, y + 47, f"{v:.2f}×", 17, TEAL if colour == TEAL else FG, 800
        )
    b += line(
        rx + bw / 1.5, by + 30, rx + bw / 1.5, by + 14 + 4 * 74, FG2, 1.2, 'stroke-dasharray="4 4"'
    )
    ey = by + 14 + 4 * 74 + 30
    per_trial = cost["subset_task_cost"]
    b += card(rx, ey, width - 60 - rx, 150) + voice(rx + 24, ey + 40, "The arithmetic", 21)
    b += text(rx + 24, ey + 72, f"{5 * ntasks} ÷ {trials} trials = {naive:.2f}×", 17, FG1, 600)
    b += text(
        rx + 24,
        ey + 102,
        f"…but each subset trial costs {per_trial:.2f}× an average trial",
        17,
        FG1,
        600,
    )
    b += text(
        rx + 24,
        ey + 132,
        f"{naive:.2f} ÷ {per_trial:.2f} ≈ {naive / per_trial:.1f}× — the multiplier",
        17,
        TEAL,
        800,
    )

    b += footer(
        width,
        height - 100,
        [
            (
                f"Source: public Terminal-Bench 4.0 leaderboard trial costs (fetched {day(a['leaderboard_fetched'])}). "
                f"{a['costed_rows']} of {a['rows']} entries used: all 330 trial costs present and summing to the published total."
            ),
            (
                "Estimate for each entry uses the median full ÷ subset ratio of the other models (all efforts of the entry's "
                "own model left out). Reported leaderboard costs, not invoices; sandbox compute excluded."
            ),
            (
                "The multiplier reflects these agents and models; a harness that spends very differently on cheap vs "
                "expensive tasks will shift it. Teal names: entries published after the subset was fixed."
            ),
        ],
    )
    render(b, width, height, "cost")


# ─────────────────────────────────────────────────────────────── main
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-charts", action="store_true", help="write calibration.json only")
    args = parser.parse_args()
    data = load()
    a = analyse(data)
    summary = {k: v for k, v in a.items() if not k.startswith("_")}
    (HERE / "calibration.json").write_text(json.dumps(summary, indent=1) + "\n")
    s, new = a["score"], a["score_by_cohort"]["new"]
    print(
        f"{a['scored_rows']} rows: gap {s['mean_gap']:.2f}, ±5 {s['within5']}, spearman {s['spearman']:.3f}; "
        f"after freeze: {new['rows']} rows, gap {new['mean_gap']:.2f}, ±5 {new['within5']}; "
        f"cost share {a['cost']['share'] * 100:.1f}%, ×{a['cost']['median_ratio']:.2f}"
    )
    if a["late_refusals_in_subset"]:
        print("refusals on subset tasks since the freeze:", ", ".join(a["late_refusals_in_subset"]))
    if not args.no_charts:
        chart_random(a)
        chart_models(a, data)
        chart_cost(a)


if __name__ == "__main__":
    main()
