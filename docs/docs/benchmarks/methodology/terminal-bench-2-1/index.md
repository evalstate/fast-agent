---
title: Terminal-Bench 2.1 methodology
description: "How fast-agent runs and comparators on Terminal-Bench 2.1 are selected, scored and costed."
---

# Terminal-Bench 2.1 methodology

[Terminal-Bench 2.1](https://hub.harborframework.com/datasets/terminal-bench/terminal-bench-2-1/6?tab=leaderboard&leaderboard=main)
has 89 tasks, repeated 5 times for a total of 445 trials.

## Scoring

- A run's score is passed attempts ÷ 445. A missing or errored attempt counts as a
  failure; nothing is dropped from the denominator.
- **±** is one standard error of the mean, clustered by task: the spread of the 89
  per-task pass rates ÷ √89. That's about ±3 points for a single run, so differences
  of a point or two are noise.
- Tasks are ordered by their pass rate across every run on the page, easiest first,
  so each column in the ledger is the same task for every row.

## Our runs

- Each run lists its source data (Harbor Hub jobs or a Hugging Face bucket) on its run
  page. Where an attempt failed for an infrastructure reason (for example a sandbox
  setup timeout) and was re-run, the run page names the replaced trial and the
  replacement. The replaced trial's cost is still counted.
- Recent runs use a [six-hour](../../six-hour/) (21,600-second) agent timeout. They
  aren't standard leaderboard configurations and shouldn't be compared like-for-like
  with standard rows; they carry a **6h timeout** badge.
- The QEMU tasks' upstream image expired on 7 September 2026. The six-hour runs use a
  fixed QEMU image on a snapshot of the benchmark, so they are a modified evaluation,
  not unmodified Terminal-Bench 2.1. The fix is forward-compatible with `tb-legacy`,
  which the Terminal-Bench team has set as the policy for these tasks.
- Our earlier standard-timeout runs were submitted to the leaderboard, which closed
  community submissions before reviewing them; they are marked **Provisional**.

## Comparators

- Leaderboard rows use the trials the official leaderboard row lists, resolved to the
  public Harbor jobs, with the row's published score and cost.
- Where the leaderboard judge disqualified a trial, it counts as a failure (shown in
  orange). The public trial IDs don't identify which attempt was disqualified, so the
  mark sits on the first passing attempt of that task.

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
[atif-scan](https://github.com/huggingface/atif-scan), which checks every trajectory for
benchmark lookups, test-file access, evidence gaps and model fallback.

- An orange corner marks an attempt with a high or critical finding; in the ledger, a
  marker above a column means the task has one.
- A slash marks an attempt that ran on a different model from the one named (for
  example a server-side fallback).
- Findings are review priorities, not verdicts. Each run page shows the score if every
  flagged pass were counted as a failure.
