---
title: Terminal-Bench 2.1 methodology
description: "How fast-agent runs and comparators on Terminal-Bench 2.1 are selected, scored and costed."
---

# Terminal-Bench 2.1 methodology

[Terminal-Bench 2.1](https://hub.harborframework.com/datasets/terminal-bench/terminal-bench-2-1/6?tab=leaderboard&leaderboard=main)
has 89 tasks. Every result on the [benchmarks page](../../) is a complete run of
all 89 tasks with five attempts each: 445 trials, all of them public on the
Harbor Hub.

## Scoring

- A run's score is passed attempts ÷ 445. A missing or errored attempt counts as a
  failure; nothing is dropped from the denominator.
- **±** is one standard error of the mean, clustered by task: the spread of the 89
  per-task pass rates ÷ √89. That's about ±3 points for a single run, so differences
  of a point or two are noise.
- Tasks are ordered by their pass rate across every run on the page, easiest first,
  so each column in the ledger is the same task for every row.

## Our runs

- Each run lists its Harbor jobs on its run page. Where an attempt failed for an
  infrastructure reason (for example a sandbox setup timeout) and was re-run, the run
  page names the replaced trial and the replacement. The replaced trial's cost is still
  counted.
- We submitted the standard-timeout runs to the Terminal-Bench 2.1 leaderboard. The
  leaderboard closed community submissions before reviewing them, so they are marked
  **Provisional**.
- [Six-hour runs](../../six-hour/) allow a 21,600-second agent timeout per trial. They
  aren't standard leaderboard configurations and shouldn't be compared like-for-like
  with standard rows; they carry a **6h timeout** badge.

## Comparators

- Leaderboard rows use the trials the official leaderboard row lists, resolved to the
  public Harbor jobs, with the row's published score and cost.
- Where the leaderboard judge disqualified a trial, it counts as a failure (shown in
  orange). The public trial IDs don't identify which attempt was disqualified, so the
  mark sits on the first passing attempt of that task.
- One comparator (Codex · GPT-5.6 Terra max) resolves to public trials that don't match
  the submitted set exactly. Its pass count matches; its run page says so.

## Vendor claims

Figures from vendor announcements (for example OpenAI's GPT-5.6 launch chart) are shown
as published, as dashed rows with no trials behind them. We don't know the harness,
attempts or run selection behind them, so they can't be compared task by task.

## Costs

- Our costs are Harbor's recorded token costs at the configured rates, not billed
  spend. **~** marks an estimate: repriced runs, or runs where some trials have no
  recorded cost.
- GPT-5.6 Sol costs use current prices: OpenAI cut Sol token prices by 20% on 24 August
  2026. The run page shows the originally recorded total.
- Leaderboard rows show their published totals. Some of those back-fill missing costs
  at list rates; the run page gives the recorded figure alongside.

## atif-scan review

Our runs and the leaderboard comparators (all but Codex · GPT-5.6 Terra max) are scanned with
[atif-scan](https://github.com/evalstate/atif-scan), which checks every trajectory for
benchmark lookups, test-file access, evidence gaps and model fallback.

- An orange corner marks an attempt with a high or critical finding; in the ledger, a
  marker above a column means the task has one.
- A slash marks an attempt that ran on a different model from the one named (for
  example a server-side fallback).
- Findings are review priorities, not verdicts. Each run page shows the score if every
  flagged pass were counted as a failure.
