---
title: Terminal-Bench 4 subset methodology
description: "Why fast-agent runs a 19-task subset of Terminal-Bench 4.0, how the tasks were chosen, and how well the subset tracks the full benchmark."
---

# Terminal-Bench 4 subset methodology

We run 19 of [Terminal-Bench 4.0](https://hub.harborframework.com/datasets/terminal-bench/terminal-bench/leaderboards/4-0-0)'s
66 tasks, five attempts each. A full run is 330 trials and costs thousands of
dollars; the subset is 95 trials at about 15% of the cost. This page explains which
tasks we picked, why, and how far a subset score can stand in for a full one.

## Which tasks, and why

The subset has to run where we run: [HF Jobs](https://huggingface.co/docs/hub/jobs),
one container per trial, no GPU. Three rules take the 66 tasks down to an eligible
pool of 48:

| Rule | Tasks excluded |
| --- | --- |
| **Single container**: no docker-compose sidecar | ctr-optimization, cumulative-layout-shift, freight-dispatch-shift, heat-pump-warranty, intrastat-meldung, kv-live-surgery, legacy-utility-triage, live-database-cutover, medical-claims-processing, nextjs-performance, payments-pipeline-fix |
| **No GPU** | fp8-rmsnorm-gemm, jax-speedrun-gpu, math-eval-grader |
| **No safety refusals**: no leaderboard trial ended in a model refusal | batched-eval-parity, interleaved-vigenere, kv-live-surgery, shadow-relay, uefi-bootkit |

From that pool we chose 18 tasks that, together, track the full leaderboard closely at
low cost, then added hof-topology-interpenetration. Four other candidates
(distributed-dedup, ks-solver-cpp, vf2-speedup-networkx, embedding-drift-monitor) were
rejected because their verifiers have open defect reports.

The 19 tasks: atrx-vep-crispr, cad-model, cargo-flight-dispatch, fin-saccr-rwa,
gsea-proteomics, hof-topology-interpenetration, html-js-filter, mvcc-lsm-compaction,
photonic-waveguide-routing, production-planning, react-lead-form,
satb-audio-transcription, session-window-debug, sound-change-cascade,
telecom-entity-resolution, vba-userform-port, vllm-deepseek-streaming,
vpp-loss-divergence and wal-recovery-ordering.

## How representative is it?

We scored every Terminal-Bench 4.0 leaderboard row on the 19 tasks alone and compared
it with the row's published full score. 26 of the 27 rows qualify; Opus 5 · max is left
out because its trial records (173 passes) don't reproduce its published score (171).

<div data-fa-bench="calibration" data-bench="tb4"><p class="fb-loading">Loading chart…</p></div>

| | 19-task subset | Random 19 from the eligible 48 (20,000 draws) |
| --- | --- | --- |
| Average gap to the full score | **3.25 points** | median 5.16; only 6% of draws are closer |
| Within ±5 / ±10 points | **22 / 26 · 25 / 26** rows | |
| Rank correlation with the full score (Spearman) | **0.93** | median 0.94; 65% of draws rank better |
| Against the other 47 tasks only (held out) | **4.56 points**, Spearman 0.90 | |
| Cost of a subset run, as a share of a full run | **15%** (wall time 16%) | median 24%; under 1% of draws are as cheap |

The subset is closer to the full score than almost any random pick of the same size,
and much cheaper, but random picks rank runs slightly better: it was chosen for score
and cost, not ranking. The held-out row is the stricter test, because the 19 tasks are
part of the full score they're compared with.

### Estimating full-run cost

On the 15 rows with complete cost records, a full run costs a median **6.4×** the subset
(5.3× to 8.4×). Using that multiplier to estimate a row's full-run cost is off by 9.7%
on average (worst 23.9%). Scaling by task count (66 ÷ 19) underestimates every row, by
about half: the subset's tasks are cheaper than average.

## Caveats

- **Chosen with this leaderboard in view.** The agreement above flatters the subset
  relative to a model it has never seen.
- **Rows share models.** The 26 rows cover 15 models, so they aren't independent samples.
- **Wide error bars.** With 19 tasks, a single run's standard error is about ±10 points.
  Treat gaps under ten points as unresolved.
- **The range is slightly compressed.** On the weakest third of rows the subset reads
  2.3 points high on average; on the strongest third, 0.6 points low.
- **Not every task separates models.** cargo-flight-dispatch hasn't been passed by any
  leaderboard row.
- **Task revisions.** Six rows (GPT-6 Astra × 5 and Gemini 3.8 Flash) ran an earlier
  revision of some subset tasks than the other 21.

## Costs on the page

Leaderboard rows show the sum of their recorded trial costs on the 19 tasks; where a
task has a trial without a recorded cost, the figure is marked **~**. Our subset runs
show Harbor's recorded token cost. Each comparator's run page also shows its full
66-task run and published cost.
