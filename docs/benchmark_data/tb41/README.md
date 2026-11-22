# Terminal-Bench 4.1 verifier rescores

The 19-task selection is unchanged. `tb4/` and its published runs are the original
4.0 evidence; `tb41/` is a separate benchmark in `catalog.json`.

- `rescores/<source-site-run-id>.json` is the validated verifier evidence index.
  It binds the original release and pinned 4.1 definition by SHA-256. Its `trials`
  map keys are the original UUIDs; each outcome names the winning verifier
  execution, native evidence paths and digests. A missing collected artifact on
  an original zero stays zero but is not represented as a verifier verdict.
- `manifest.json` and `runs/` are the matched external comparator entries from
  the official 4.1 Harbor leaderboard. To refresh a previously selected row, use
  `uv run --no-project python docs/benchmark_data/fetch_runs.py --manifest
  docs/benchmark_data/tb41/manifest.json --run <4.1-site-run-id>`.
- The generator keeps costs, usage, original jobs and scan evidence from `tb4/`
  for the first-party trials. It recomputes the 4.1 chart and score from the
  verifier outcomes, carrying integrity decisions forward on the same traces
  and rejecting newly passing flagged trials without a decision.

Validate native results before staging the public overlay:

```sh
uv run --no-project python docs/benchmark_data/tb4/rescore_handoff.py \
  --bench-run ../bench-run --out /tmp/tb41-index \
  ../bench-run/state/rescore-handoff/*.summary.json
```

The public bundle under `tb4/19-task-v1-tb41/<source-site-run-id>/` includes the
versioned index, native selected verifier results and logs, receipts and task
pins. Audit staged files for credentials; sync and validate every public digest
before including the evidence URL in `rescores/`. Never overwrite the original
4.0 prefix or copy private output artifacts into the public overlay. Generate
the updated assets with `uv run --no-project python docs/generate_benchmark_data.py`.
