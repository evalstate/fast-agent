"""Build-time render of the benchmarks ledger (prototype).

A port of the ledger in docs/javascripts/benchmarks.js (scoreboard, task strips,
cost dots, legend) to static HTML + SVG, from the same data that
generate_benchmark_data.build() produces. The markup reuses the existing
`fb-*` classes, so benchmarks.css styles it unchanged.

What stays in the browser (benchmarks-ledger.js, progressive enhancement):
family chips, tier toggles and sort order. Task tooltips are native SVG
<title>s, so the ledger reads fine with no JavaScript at all.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from html import escape
from typing import Any

type Entry = dict[str, Any]

# Cell codes, bottom-to-top stacking order inside a task column.
CELL_ORDER = {"1": 0, "x": 1, "0": 2, "t": 3, "e": 4, "-": 5}
CELL_LABEL = {
    "1": "pass",
    "x": "pass, disqualified (leaderboard judge or our review)",
    "0": "fail",
    "t": "agent timeout",
    "e": "error",
    "-": "missing",
}
TIER_LABEL = {"ours": "Our run", "leaderboard": "Leaderboard", "claim": "Vendor claim"}
STRIP_WIDTH = 560
COST_WIDTH = 200


# ── Formatting (round down, never up) ────────────────────────────────────
def _floor1(x: float) -> float:
    return math.floor(x * 10 + 1e-9) / 10


def pct(x: float) -> str:
    return f"{_floor1(x):.1f}%"


def money(x: float) -> str:
    if x >= 1000:
        return f"${round(x):,}"
    return f"${x:.0f}" if x >= 100 else f"${x:.2f}"


def cost_text(entry: Entry) -> str:
    text = money(entry["cost"]["total"])
    if entry["cost"].get("lowerBound"):
        return "≥" + text
    return "~" + text if entry["cost"].get("estimate") else text


def plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def title(entry: Entry) -> str:
    return f"{entry['model']} · {entry['effort']}"


def harness_line(entry: Entry) -> str:
    if entry["tier"] == "claim":
        return f"{entry['source']} · {entry['title']}"
    version = entry.get("harnessVersion")
    return f"{entry['harness']} {version}" if version else entry["harness"]


def family_name(family: dict[str, str]) -> str:
    name, maker = family["name"], family["maker"]
    return name if name.startswith(maker) else f"{maker} {name}"


# ── Data model ────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Bench:
    """One benchmark's slice of the data, as the JS `model()` builds it."""

    data: dict[str, Any]
    bench: dict[str, Any]
    runs: list[Entry]
    claims: list[Entry]
    order: list[int]  # task indices, easiest first

    @classmethod
    def of(cls, data: dict[str, Any], bench_id: str) -> Bench:
        bench = next(b for b in data["benchmarks"] if bench_id in (b["id"], b["slug"]))
        tasks, difficulty = bench["tasks"], bench["difficulty"]
        order = sorted(range(len(tasks)), key=lambda i: (-difficulty[i], tasks[i]))
        return cls(
            data=data,
            bench=bench,
            runs=[r for r in data["runs"] if r["benchmark"] == bench["id"]],
            claims=[c for c in data["claims"] if c["benchmark"] == bench["id"]],
            order=order,
        )

    @property
    def attempts(self) -> int:
        return self.bench["attempts"]

    @property
    def tasks(self) -> list[str]:
        return self.bench["tasks"]

    @property
    def families(self) -> dict[str, dict[str, str]]:
        return {f["id"]: f for f in self.data["families"]}


# ── atif-scan codes: c/h/m/l priority, upper case = fallback model, ? = none ──
def is_flagged(code: str) -> bool:
    return code != "" and code in "chCH"


def is_fallback(code: str) -> bool:
    return code != "" and code in "CHML"


def attempts_of(run: Entry, task: int, attempts: int) -> list[tuple[str, str]]:
    """A task's (cell, scan) codes in stacking order: passes first, flagged on top."""
    start = task * attempts
    cells = run["cells"][start : start + attempts]
    scan = run["scan"]["cells"][start : start + attempts] if run.get("scan") else ""
    pairs = [(c, scan[i] if i < len(scan) else "") for i, c in enumerate(cells)]
    return sorted(pairs, key=lambda p: (CELL_ORDER[p[0]], is_flagged(p[1]), is_fallback(p[1])))


def scan_lines(run: Entry, task: int, attempts: int) -> list[str]:
    if run["tier"] == "claim" or run.get("sample"):
        return []
    if not run.get("scan"):
        return ["atif-scan: this run hasn't been scanned"]
    pairs = attempts_of(run, task, attempts)
    flagged = [c for c, s in pairs if is_flagged(s)]
    passed = sum(c in "1x" for c in flagged)
    fallback = sum(is_fallback(s) for _, s in pairs)
    medium = sum(s.lower() == "m" for _, s in pairs)
    missing = sum(s == "?" for _, s in pairs)
    lines = []
    if flagged:
        lines.append(
            f"atif-scan: {plural(len(flagged), 'attempt')} with a high or critical finding"
            f" ({passed} passed)"
        )
    if fallback:
        lines.append(f"{plural(fallback, 'attempt')} ran on another model (fallback)")
    if medium:
        lines.append(f"atif-scan: {plural(medium, 'attempt')} with a medium finding")
    if missing:
        lines.append(f"atif-scan: no result for {plural(missing, 'attempt')}")
    return lines or ["atif-scan: nothing above low priority"]


def _task_tip(b: Bench, run: Entry, task: int) -> str:
    start = task * b.attempts
    cells = run["cells"][start : start + b.attempts]
    lines = [b.tasks[task], f"{cells.count('1')} of {b.attempts} attempts passed"]
    if other := [CELL_LABEL[c] for c in cells if c not in "10"]:
        lines.append(", ".join(other))
    lines += scan_lines(run, task, b.attempts)
    lines.append(f"All runs: {round(b.bench['difficulty'][task] * 100)}% of attempts pass")
    return "\n".join(lines)


# ── SVG pieces ────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Geometry:
    cell_w: int
    cell_h: int
    gap: int
    col_gap: int

    @property
    def pitch(self) -> int:
        return self.cell_w + self.col_gap

    @classmethod
    def for_columns(cls, cols: int) -> Geometry:
        """Square cells for wide benchmarks; wide bricks for small ones (TB4 subset)."""
        if cols > 40:
            return cls(5, 5, 1, 1)
        pitch = min(29, STRIP_WIDTH // cols)
        return cls(pitch - 3, 10, 2, 3)


def _shape(w: int, h: int) -> str:
    """One cell drawn from its top-left corner: square on small cells, rounded on bricks."""
    if w <= 6:
        return f"h{w}v{h}h-{w}z"
    r, arc = 1.5, "a1.5 1.5 0 0 1"
    iw, ih = w - 2 * r, h - 2 * r
    return f"m{r:g} 0h{iw:g}{arc} {r:g} {r:g}v{ih:g}{arc} -{r:g} {r:g}h-{iw:g}{arc} -{r:g} -{r:g}v-{ih:g}{arc} {r:g} -{r:g}z"


class _RelativePath:
    """Path data where each shape starts with a move relative to the previous shape,
    so repeated geometry compresses to almost nothing."""

    def __init__(self) -> None:
        self.parts: list[str] = []
        self.at: tuple[float, float] | None = None

    def add(self, x: float, y: float, shape: str) -> None:
        if self.at is None:
            self.parts.append(f"M{x:g} {y:g}{shape}")
        else:
            self.parts.append(f"m{x - self.at[0]:g} {y - self.at[1]:g}{shape}")
        # A closed subpath returns to its start; a rounded shape starts at its first "m".
        self.at = (x + (1.5 if shape.startswith("m1.5") else 0), y)

    def __str__(self) -> str:
        return "".join(self.parts)


def task_strip(b: Bench, run: Entry) -> str:
    """One path per cell code plus one each for slashes and flags, then a hit area per
    task column carrying its tooltip. Grouping like with like keeps the SVG small."""
    geo = Geometry.for_columns(len(b.order))
    flag_room = 5
    width = len(b.order) * geo.pitch - geo.col_gap
    height = flag_room + b.attempts * (geo.cell_h + geo.gap) - geo.gap
    cells: dict[str, _RelativePath] = {}
    slashes: list[str] = []
    flags: list[str] = []
    hits: list[str] = []
    w, h = geo.cell_w, geo.cell_h
    shape = _shape(w, h)
    for col, task in enumerate(b.order):
        x = col * geo.pitch
        pairs = attempts_of(run, task, b.attempts)
        for row, (cell, scan) in enumerate(pairs):
            y = flag_room + (b.attempts - 1 - row) * (h + geo.gap)
            cells.setdefault("m" if cell == "-" else cell, _RelativePath()).add(x, y, shape)
            if is_fallback(scan):
                slashes.append(f"M{x + 0.5:g} {y + h - 0.5:g}L{x + w - 0.5:g} {y + 0.5:g}")
            if is_flagged(scan) and w >= 8:
                k = min(w, h) * 0.55
                flags.append(f"M{x + w - k:g} {y}h{k:g}v{k:g}z")
        if any(is_flagged(s) for _, s in pairs):
            flags.append(f"M{x + w / 2 - 2.5:g} 0h5l-2.5 3.5z")
        hits.append(
            f'<rect x="{x}" width="{geo.pitch}" height="{height}">'
            f"<title>{escape(_task_tip(b, run, task))}</title></rect>"
        )
    paths = "".join(f'<path d="{d}" class="fb-c fb-c--{code}"/>' for code, d in cells.items())
    if slashes:
        paths += f'<path d="{"".join(slashes)}" class="fb-fallback"/>'
    if flags:
        paths += f'<path d="{"".join(flags)}" class="fb-flag"/>'
    label = f"{title(run)} on {run['harness']}: {run['passes']} of {run['slots']} trials passed"
    return (
        f'<svg class="fb-strip fb-strip--{run["tier"]}" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" role="img" aria-label="{escape(label)}">'
        f'{paths}<g class="fb-hit">{"".join(hits)}</g></svg>'
    )


def claim_strip(b: Bench, claim: Entry) -> str:
    geo = Geometry.for_columns(len(b.order))
    width = len(b.order) * geo.pitch - geo.col_gap
    height = 5 + b.attempts * (geo.cell_h + geo.gap) - geo.gap
    fill = (width - 2) * claim["score"] / 100
    label = f"{claim['source']} reports {claim['score']}%; no trial data published"
    return (
        f'<svg class="fb-strip fb-strip--claim" viewBox="0 0 {width} {height}" width="{width}" '
        f'height="{height}" role="img" aria-label="{escape(label)}">'
        f'<rect x="1" y="6" width="{width - 2}" height="{height - 7}" rx="2" class="fb-claim-frame"/>'
        f'<rect x="1" y="6" width="{fill:g}" height="{height - 7}" rx="2" class="fb-claim-fill"/>'
        f'<text x="8" y="{6 + (height - 6) / 2 + 3.5:g}" class="fb-claim-label">'
        "No trial data published · score only</text></svg>"
    )


def task_header(b: Bench) -> str:
    """Rotated task names above the strip, for benchmarks narrow enough to label."""
    geo = Geometry.for_columns(len(b.order))
    width, height = len(b.order) * geo.pitch - geo.col_gap, 118
    names = "".join(
        f'<text x="0" y="0" transform="translate({col * geo.pitch + geo.cell_w / 2 + 3:g} {height - 4}) '
        f'rotate(-58)" class="fb-taskhead__t">{escape(b.tasks[task])}</text>'
        for col, task in enumerate(b.order)
    )
    return (
        f'<svg class="fb-taskhead" viewBox="0 0 {width} {height}" width="{width}" '
        f'height="{height}" aria-hidden="true">{names}</svg>'
    )


@dataclass(frozen=True)
class CostDomain:
    """Log cost domain padded to whole 1-2-5 steps around the entries' costs."""

    min: float
    max: float
    ticks: list[float]

    @classmethod
    def of(cls, entries: list[Entry]) -> CostDomain:
        steps = [f * 10.0**k for k in range(-1, 5) for f in (1, 2, 5)]
        costs = [e["cost"]["total"] for e in entries]
        lo = max((v for v in steps if v <= min(costs) * 0.85), default=steps[0])
        hi = min((v for v in steps if v >= max(costs) * 1.15), default=steps[-1])
        ticks = [v for v in steps if lo < v < hi]
        while len(ticks) > 4:
            ticks = ticks[1::2]
        return cls(lo, hi, ticks)

    def x(self, cost: float) -> float:
        c = max(self.min, min(self.max, cost))
        return math.log(c / self.min) / math.log(self.max / self.min) * COST_WIDTH


def cost_dot(entry: Entry, dom: CostDomain) -> str:
    h = 28
    parts = [
        f'<line x1="{dom.x(t):.2f}" x2="{dom.x(t):.2f}" y1="2" y2="{h - 2}" class="fb-grid"/>'
        for t in dom.ticks
    ]
    parts.append(f'<line x1="0" x2="{COST_WIDTH}" y1="{h / 2:g}" y2="{h / 2:g}" class="fb-track"/>')
    x = dom.x(entry["cost"]["total"])
    cost = entry["cost"]
    if entry["tier"] == "ours" and cost.get("pricing") and cost.get("recorded") != cost["total"]:
        ox = dom.x(cost["recorded"])
        parts.append(
            f'<line x1="{ox:.2f}" x2="{x:.2f}" y1="{h / 2:g}" y2="{h / 2:g}" class="fb-reprice"/>'
        )
        parts.append(f'<circle cx="{ox:.2f}" cy="{h / 2:g}" r="4" class="fb-dot fb-dot--was"/>')
    parts.append(
        f'<circle cx="{x:.2f}" cy="{h / 2:g}" r="6" class="fb-dot fb-dot--{entry["tier"]}"/>'
    )
    return (
        f'<svg class="fb-costdot" viewBox="0 0 {COST_WIDTH} {h}" width="100%" height="{h}" '
        f'preserveAspectRatio="none" aria-hidden="true">{"".join(parts)}</svg>'
    )


def cost_axis(dom: CostDomain) -> str:
    labels = "".join(
        f'<text x="{dom.x(t):.2f}" y="12" text-anchor="middle">'
        f"{f'${t / 1000:g}k' if t >= 1000 else f'${t:g}'}</text>"
        for t in dom.ticks
    )
    return (
        f'<svg class="fb-costaxis" viewBox="0 0 {COST_WIDTH} 16" width="100%" height="16" '
        f'preserveAspectRatio="none" aria-hidden="true">{labels}</svg>'
    )


# ── Rows and badges ───────────────────────────────────────────────────────
def family_mark(b: Bench, family_id: str, root: str, size: str = "") -> str:
    f = b.families[family_id]
    cls = f"fb-mark fb-mark--{family_id}" + (f" fb-mark--{size}" if size else "")
    src = f"{root}assets/forward/assets/providers/{f['mark']}"
    return f'<span class="{cls}"><img src="{src}" alt="{escape(f["maker"])}" width="20" height="20"></span>'


def status_badge(entry: Entry) -> str:
    if not entry.get("status"):
        return ""
    cls = (
        "fb-status fb-status--sample"
        if entry.get("sample")
        else "fb-status fb-status--timeout"
        if entry.get("timeout") == "6h"
        else "fb-status"
    )
    return f'<span class="{cls}">{escape(entry["status"])}</span>'


def row(b: Bench, entry: Entry, dom: CostDomain, root: str, scanned: bool) -> str:
    tier = entry["tier"]
    meta = f'<span class="fb-tier fb-tier--{tier}">{TIER_LABEL[tier]}</span>'
    meta += f'<span class="fb-harness">{escape(harness_line(entry))}</span>{status_badge(entry)}'
    if level := (entry.get("scan") or {}).get("level"):
        meta += f'<span class="fb-scanlevel fb-scanlevel--{level["name"]}">atif-scan {escape(level["label"])}</span>'
    if entry.get("review"):
        meta += '<span class="fb-scanlevel fb-scanlevel--reviewed">reviewed</span>'
    if scanned and tier != "claim" and not entry.get("sample") and not entry.get("scan"):
        meta += '<span class="fb-noscan">not scanned</span>'
    strip = claim_strip(b, entry) if tier == "claim" else task_strip(b, entry)
    detail = (
        "reported"
        if tier == "claim"
        else f"{entry['passes']}/{entry['slots']} · ±{entry['se']:.1f}"
    )
    score = f'<span class="fb-score__v">{pct(entry["score"])}</span><span class="fb-score__n">{detail}</span>'
    if review := entry.get("review"):
        label = f"recorded {pct(review['recordedScore'])}"
        score += f'<span class="fb-score__full">{escape(label)}</span>'
    if b.bench.get("full"):
        full = entry.get("full")
        label = f"full run {pct(full['publishedScore'])}" if full else "subset only"
        score += f'<span class="fb-score__full">{label}</span>'
    # data-* drive the filters and sort in benchmarks-ledger.js.
    attrs = (
        f'class="fb-row fb-row--{tier}" role="row" data-family="{entry["family"]}" data-tier="{tier}" '
        f'data-score="{entry["score"]}" data-cost="{entry["cost"]["total"]}"'
        + (' data-long=""' if entry.get("timeout") == "6h" else "")
    )
    tag, href = (
        ("div", "") if tier == "claim" else ("a", f' href="{root}benchmarks/run/?id={entry["id"]}"')
    )
    return (
        f"<{tag} {attrs}{href}>"
        f'<span class="fb-who">{family_mark(b, entry["family"], root)}<span class="fb-names">'
        f'<span class="fb-model">{escape(title(entry))}</span><span class="fb-meta">{meta}</span></span></span>'
        f'<span class="fb-cell-strip">{strip}</span>'
        f'<span class="fb-score">{score}</span>'
        f'<span class="fb-cost">{cost_dot(entry, dom)}<span class="fb-cost__v">{cost_text(entry)}</span></span>'
        f"</{tag}>"
    )


def legend(runs: list[Entry], with_claim: bool) -> str:
    tiers = {r["tier"] for r in runs}
    cells = "".join(r["cells"] for r in runs)
    scans = [r["scan"]["cells"] for r in runs if r.get("scan")]
    scanned = bool(scans)
    unscanned = any(not r.get("sample") and not r.get("scan") for r in runs)
    flagged = any(set(s) & set("chCH") for s in scans)
    fallback = any(set(s) & set("CHML") for s in scans)
    items = [
        ("ours" in tiers, "ours", "our pass"),
        ("leaderboard" in tiers, "leaderboard", "comparator pass"),
        (True, "fail", "fail"),
        ("t" in cells, "t", "timeout"),
        ("e" in cells, "e", "error"),
        ("x" in cells, "x", "judge DQ"),
        (flagged, "flag", "atif-scan: high or critical finding"),
        (flagged, "flagcol", "task has a flagged attempt"),
        (fallback, "fallback", "ran on another model"),
        (unscanned and scanned, "noscan", "not scanned"),
        (unscanned and not scanned, "noscan", "no atif-scan review yet"),
        (with_claim, "claim", "vendor claim, no trials"),
    ]
    spans = "".join(
        f'<span class="fb-legend__i"><i class="fb-sw fb-sw--{sw}"></i><span>{text}</span></span>'
        for show, sw, text in items
        if show
    )
    return f'<div class="fb-legend">{spans}</div>'


def _controls(b: Bench, entries: list[Entry], root: str) -> str:
    chips = [("all", "All models", len(entries), "")]
    for f in b.data["families"]:
        if n := sum(e["family"] == f["id"] for e in entries):
            chips.append((f["id"], family_name(f), n, family_mark(b, f["id"], root, "sm")))
    chip_html = "".join(
        f'<button type="button" class="fb-chip" data-family="{fid}" aria-pressed="{str(fid == "all").lower()}">'
        f'{mark}<span>{escape(name)}</span><span class="fb-chip__n">{n}</span></button>'
        for fid, name, n, mark in chips
    )
    toggles = [("leaderboard", "Leaderboard runs", True)]
    toggles.append(("claims", "Vendor claims", bool(b.claims)))
    toggles.append(("long", "Six-hour runs", any(r.get("timeout") == "6h" for r in b.runs)))
    toggle_html = "".join(
        f'<label class="fb-toggle"><input type="checkbox" data-toggle="{key}" checked><span>{label}</span></label>'
        for key, label, show in toggles
        if show
    )
    sorts = "".join(
        f'<button type="button" data-sort="{key}" aria-pressed="{str(key == "family").lower()}">{label}</button>'
        for key, label in (("family", "By family"), ("score", "By score"), ("cost", "By cost"))
    )
    # Hidden until benchmarks-ledger.js wires them up; the table reads fine without.
    return (
        '<div class="fb-controls" hidden>'
        f'<div class="fb-chips" role="group" aria-label="Model family">{chip_html}</div>'
        f'<div class="fb-toggles">{toggle_html}'
        f'<div class="fb-seg" role="group" aria-label="Order">{sorts}</div></div></div>'
    )


def ledger(data: dict[str, Any], bench_id: str, root: str) -> str:
    """The ledger for one benchmark. `root` is the site root relative to the including page."""
    b = Bench.of(data, bench_id)
    entries = sorted(b.runs + b.claims, key=lambda e: (e["tier"] != "ours", -e["score"]))
    dom = CostDomain.of(entries)
    scanned = any(r.get("scan") for r in b.runs)

    tasks_head = (
        f'<span class="fb-h fb-h--tasks{" fb-h--labelled" if len(b.order) <= 40 else ""}">'
        f"{task_header(b) if len(b.order) <= 40 else ''}"
        f"<span>{escape(b.bench['taskLabel'])}</span>"
        '<span class="fb-h__sub">easiest → hardest across all runs</span></span>'
    )
    head = (
        '<div class="fb-ledger__head" role="row"><span class="fb-h">Model · harness</span>'
        f'{tasks_head}<span class="fb-h fb-h--num">Score</span>'
        f'<span class="fb-h fb-h--cost"><span>Run cost</span>{cost_axis(dom)}</span></div>'
    )
    body = [head]
    for f in b.data["families"]:
        group = [e for e in entries if e["family"] == f["id"]]
        if not group:
            continue
        ours = sum(e["tier"] == "ours" for e in group)
        body.append(
            f'<div class="fb-group" data-family="{f["id"]}">{family_mark(b, f["id"], root, "sm")}'
            f'<span class="fb-group__name">{escape(family_name(f))}</span>'
            f'<span class="fb-group__n">{f"{ours} of our runs" if ours else "comparators only"}</span></div>'
        )
        body += [row(b, e, dom, root, scanned) for e in group]
    body.append('<p class="fb-empty" hidden>No results match these filters.</p>')

    return (
        f'<div class="fb"><div class="fb-board" data-static-ledger>{_controls(b, entries, root)}'
        f'<div class="fb-ledger" role="table" aria-label="{escape(b.bench["name"])} results">'
        f"{''.join(body)}</div>{legend(b.runs, bool(b.claims))}</div></div>"
    )
