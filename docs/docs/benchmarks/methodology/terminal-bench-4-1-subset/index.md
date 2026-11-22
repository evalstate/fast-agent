---
title: Terminal-Bench 4.1 subset
description: "Verifier-only rescoring of fast-agent's original 19-task Terminal-Bench 4.0 runs on the 4.1 task packages."
---

# Terminal-Bench 4.1 subset

[4.1 results](../../?b=tb4.1) · [Archived 4.0 results](../../?b=tb4) ·
[Official Terminal-Bench 4.1 leaderboard](https://hub.harborframework.com/datasets/terminal-bench/terminal-bench/leaderboards/4-1-0)

The 4.1 view uses the same 19 tasks and five slots per task as the
[original 4.0 subset](../terminal-bench-4-subset/index.md). We ran the pinned 4.1 verifiers
on the recorded outputs of the published agent attempts. We did **not** rerun the
agents, change their outputs, or choose a different set of attempts. The original
trace bundles, model costs, replacement lineage and 4.0 scores remain available.

## What changed

The task instructions and expected artifact lists did not change for the subset.
The verifier-only rescores use the 4.1 task packages. The historical rescore
batch used Hugging Face Jobs and Docker for its follow-ups. We determined
that `wal-recovery-ordering` could not be graded reliably in those environments
and used Modal instead; the final 4.1 scores do not fall back to 4.0 WAL verdicts.
The newer rescore batch runs all 19 tasks' verifiers on Modal.

The denominator remains 95. If a source trial scored zero and produced no
collected artifact to regrade, it still contributes zero; we do not present
that as a new verifier verdict. Integrity decisions apply to the original
agent execution. If a disqualified attempt passes the new verifier, it remains
disqualified.

## What the page shows

First-party run pages link to the original execution and to the separate
[4.1 verifier evidence bundles](https://huggingface.co/buckets/evalstate/published-benchmarks/tree/tb4/19-task-v1-tb41).
Trace scan summaries and excerpts describe the original execution and use the
original verifier labels; the score and task cells use 4.1 verdicts. Model costs
and tokens are from the original runs and do not include verifier infrastructure.

External comparisons use the same comparator entries as the 4.0 view, matched
to their 4.1 Harbor rows. Their subset scores use the recorded 4.1 verifier
outcomes for those 19 tasks, and run pages still show the published full-suite
4.1 score. The 4.0 methodology and calibration figures document the original
task selection, not a new selection exercise.
