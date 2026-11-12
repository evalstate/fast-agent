# /// script
# requires-python = ">=3.12"
# dependencies = ["pillow"]
# ///
"""Draw the compaction chart: where fast-agent compacts against score and cost on the TB4 subset.

Figures are the run summaries for the 95-trial subset runs listed in RUNS (score = rewards ÷ 95,
± one standard error clustered by task; cost = fast-agent's recorded cost at list rates).
Writes docs/docs/assets/benchmarks/tb4-subset/compaction.webp.

    uv run docs/benchmark_data/tb4/compaction.py
"""

from __future__ import annotations

from dataclasses import dataclass

from calibration import BG, FG, FG1, FG2, GRID, TEAL, footer, kicker, line, render, text, voice


@dataclass(frozen=True)
class Run:
    run_id: str
    label: str
    score: float  # % of 95 trials
    se: float  # points, clustered by task
    cost: float  # recorded USD over 95 trials


@dataclass(frozen=True)
class Pair:
    model: str
    late: Run
    early: Run


PAIRS = [
    Pair(
        "GLM-5.3-Flash · high · Z.ai",
        Run("tb4-glm53flash-high-hf-zai-noptc-001", "No compaction (1M window)", 35.8, 8.5, 26.15),
        Run("tb4-glm53flash-high-hf-zai-compact200k-001", "Compact at 200k", 37.9, 10.0, 20.06),
    ),
    Pair(
        "Grok 4.7 · xhigh · xAI",
        Run("tb4-grok47-xhigh-xai-oauth-sse-001", "Compact at 425k (default)", 40.0, 8.4, 721.30),
        Run("tb4-grok47-xhigh-xai-oauth-sse-002", "Compact at 190k", 42.1, 9.3, 604.99),
    ),
]
TRIALS = 95


def chart() -> None:
    width, height = 1600, 820
    b = kicker(60, 60, "Terminal-Bench 4.0 subset · context compaction")
    b += voice(60, 106, "Compacting earlier cut cost with no measurable change in score", 44)
    b += text(
        60,
        142,
        "Each pair is the same model and effort run twice on the 19-task subset (95 trials), "
        "changing only the point where fast-agent compacts the context.",
        17,
        FG1,
    )

    label_r = 330
    left = (380, 760)  # score axis
    right = (900, 1480)  # cost bars
    top = 260
    row_h, gap = 52, 40

    b += voice(left[0], 216, "Score", 26) + text(left[0], 243, "± one standard error", 15, FG2, 500)
    b += voice(right[0], 216, "Cost per trial", 26)
    b += text(
        right[0],
        243,
        "Recorded cost at list prices; each model on its own scale from $0",
        15,
        FG2,
        500,
    )

    hi = 60.0

    def sx(v: float) -> float:
        return left[0] + (left[1] - left[0]) * v / hi

    y = top
    rows: list[tuple[float, Run, bool]] = []
    for pair in PAIRS:
        b += text(60, y + 22, pair.model, 18, FG, 800)
        y += 40
        for run, early in ((pair.late, False), (pair.early, True)):
            rows.append((y + row_h / 2, run, early))
            y += row_h
        y += gap
    bottom = y - gap + 8

    for v in range(0, int(hi) + 1, 10):
        b += line(sx(v), top, sx(v), bottom, GRID)
        b += text(sx(v), bottom + 22, f"{v}%", 13, FG2, 500, "middle")

    for pair in PAIRS:
        scale = (right[1] - right[0]) / max(pair.late.cost, pair.early.cost) * 0.78
        for cy, run, early in rows:
            if run not in (pair.late, pair.early):
                continue
            colour = TEAL if early else FG
            b += text(label_r, cy + 5, run.label, 15, colour, 700 if early else 500, "end")
            # score: whisker and dot
            b += line(sx(run.score - run.se), cy, sx(run.score + run.se), cy, colour, 2.5)
            for v in (run.score - run.se, run.score + run.se):
                b += line(sx(v), cy - 7, sx(v), cy + 7, colour, 2.5)
            b += (
                f'<circle cx="{sx(run.score):.1f}" cy="{cy:.1f}" r="8" fill="{colour}" '
                f'stroke="{BG}" stroke-width="2"/>'
            )
            b += text(sx(run.score + run.se) + 14, cy + 5, f"{run.score:.1f}%", 15, colour, 800)
            # cost bar
            per_trial = run.cost / TRIALS
            w = run.cost * scale
            b += f'<rect x="{right[0]}" y="{cy - 14:.1f}" width="{w:.1f}" height="28" rx="6" fill="{colour}"/>'
            value = f"${per_trial:.2f}"
            b += text(right[0] + w + 12, cy + 6, value, 17, colour, 800)
            if early:
                change = (run.cost - pair.late.cost) / pair.late.cost * 100
                b += text(
                    right[0] + w + 12 + 9.6 * len(value) + 14,
                    cy + 6,
                    f"{change:.0f}%".replace("-", "−"),
                    17,
                    colour,
                    800,
                )

    b += text(
        (left[0] + left[1]) / 2, bottom + 50, "Share of 95 trials passed", 15, FG, 700, "middle"
    )

    b += footer(
        width,
        height - 120,
        [
            (
                "Score is verifier reward ÷ 95. With 19 tasks, differences under about 10 points are within "
                "noise; both score gaps here are about 2 points."
            ),
            (
                "Cost is fast-agent's recorded cost at the provider's list prices, not billing. The "
                "later-compaction runs have more trials with incomplete usage records (11 vs 3, 17 vs 1), so "
                "their bars are, if anything, short."
            ),
            (
                "Not shown: GLM at 256k (35.8%, $0.35 per trial, or $0.22 without one trial that re-ran a "
                "script 2,238 times). Grok runs also differ in fast-agent version (0.10.44 vs 0.10.46)."
            ),
        ],
    )
    render(b, width, height, "compaction")


if __name__ == "__main__":
    chart()
