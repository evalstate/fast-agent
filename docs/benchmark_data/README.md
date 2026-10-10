# Benchmark data

`catalog.json` lists the benchmarks, model families, per-run curation, vendor claims,
comparisons and, optionally, synthesised `samples` for layout review. `generate_benchmark_data.py`
turns everything here into `docs/docs/javascripts/benchmark-data.js`.

## Terminal-Bench 4.0 subset (`tb4/`)

`tb4/tasks.json` is the full 66-task list and `tb4/subset.json` the 19 subset tasks.
`tb4/selection.json` records what the subset was chosen from: the freeze date, the
leaderboard rows published by then, the pinned task digests and the tasks excluded by
their packages (GPU, docker-compose sidecars). Rows published later are out-of-sample.

Refreshing from the leaderboard (needs the `harbor` CLI):

```bash
uv run --no-project python docs/benchmark_data/tb4/fetch_leaderboard.py   # trials -> tb4/.cache/ (git-ignored)
uv run --no-project python docs/benchmark_data/tb4/import_leaderboard.py  # -> tb4/runs/<id>.json
uv run docs/benchmark_data/tb4/calibration.py   # -> tb4/calibration.json + docs/docs/assets/benchmarks/tb4-subset/*.webp
uv run --no-project python docs/generate_benchmark_data.py
```

`import_leaderboard.py` keeps all 66 tasks per row; the generator cuts them to the subset
and keeps the full run for the run pages. Subset cost is the sum of the recorded trial
costs on the 19 tasks. `calibration.py` needs Chromium for the charts; the figures quoted
on the methodology page come from `calibration.json`. A new model needs a
`familyByModel` entry in the catalog. A new benchmark with a
subset (TB5: 20–21 tasks) needs the same three files and a `benchmarks` entry with
`subset`.

TB4 is marked `comingSoon` until our own subset runs land: its tab shows the methodology
and no results, and no ledger is generated. The leaderboard rows still feed the subset
calibration chart on the methodology page (rows listed after the bench's `frozen` date
are drawn in teal). Remove `comingSoon` once real runs exist.

# Terminal-Bench 2.1 benchmark data

Per-trial data for the benchmarks page. `manifest.json` lists every run and how its
trials are selected; `fetch_runs.py` turns that into `runs/<run_id>.json` and
`tasks.json` (the 89 canonical TB2.1 task names, sorted).

## Adding results

1. Add the run to `manifest.json` (jobs, excluded trials, published figures) and run
   `fetch_runs.py --run <id> --scan`.
2. Add it to `catalog.json` under `runs` with its `family` (add a family with a
   provider mark from `docs/docs/assets/forward/assets/providers/` if it's new).
   Vendor or marketing figures without trials go in `claims`; curated comparisons
   (`pairs`, `h2h`, `set`) go in `comparisons`.
3. Rebuild the page data: `uv run --no-project python docs/generate_benchmark_data.py`
   (writes `docs/docs/javascripts/benchmark-data.js`; don't edit that file).

## Reproduce

Requires an authenticated `harbor` CLI, `gh` (for leaderboard submission files) and,
for `--scan`, a checkout of atif-scan at `~/source/atif-scan` (`--atif-scan-dir`).
The script is standard-library only:

```bash
uv run --no-project python docs/benchmark_data/fetch_runs.py          # all runs
uv run --no-project python docs/benchmark_data/fetch_runs.py --scan   # + atif-scan summaries
uv run --no-project python docs/benchmark_data/fetch_runs.py --run luna-max-6h --refresh
```

Raw CLI responses are cached in `/tmp/bench/cache` (`--cache-dir`); `--refresh`
re-fetches. Scans are cached per full command and atif-scan source (commit plus a digest
of local changes), so a changed scanner, flag or input path rescans. Without `--scan`,
an existing `scan` block in `runs/<id>.json` is kept. First scans of uncached jobs sync
traces to `~/.cache/atif-scan` and take minutes.

`--image-model MODEL` passes the same flag to atif-scan: images that leave a check
unknown are sent to MODEL (via fast-agent) for transcription. It is the only option that
makes model calls; answers are cached privately in `~/.cache/atif-scan/images/`. The
published scans used `--image-model 'codexresponses.gpt-6-luna?reasoning=medium'`.

Bucket runs read a local mirror of the bucket (`--bucket-root`, default atif-scan's
`~/.cache/atif-scan/hf/buckets`); they need no Hub access. Sync one with
`uv run atif-scan hf://buckets/<repo>/<path>/<run>/ --sync`, or download it.

## How trials are selected

- **fast-agent runs** (`jobs` in the manifest): every trial of the listed Hub jobs,
  minus `exclude_trials` (infrastructure failures replaced by a replacement job). Excluded
  trials don't count toward score or coverage, but their recorded cost is included.
- **Leaderboard comparators** (`leaderboard_row`): the trial associations of the
  official TB2.1 leaderboard row (`harbor hub leaderboard row trial list`), resolved to
  the public "scrubbed" Hub jobs. Published score/cost/date come from the row.
  `disqualified_trials` come from the merged submission file in
  `harbor-framework/terminal-bench-2-1`.
- **Bucket runs** (`bucket` in the manifest): harbor-hf runs in a Hugging Face bucket,
  `{"repo", "path" (default "runs"), "runs": [main, replacement...]}`. Each later run's
  `run.json` `operator_selection` names the trials of the main run it replaces; those
  originals are excluded (and kept as evidence). Published runs live in
  `evalstate/published-benchmarks` under `<benchmark>/<revision>/<run id>/`, with a
  `manifest.json` of file digests and the payload policy; `source` records where they
  were copied from. A `pricing` block computes cost from tokens when the run recorded
  none (`cost.computed`, a lower bound when any trial's usage is incomplete).
- **Publish-side scan policy** (`scan_rules`, default `atif-rules.json`): atif-scan rules
  applied to every scan, raising our own review priorities without changing atif-scan's
  detectors. `review.task_recalled` (high) marks a trial that named Terminal-Bench tasks
  unprompted, its own or three others (`tb21.recall.task_catalog`); `review.benchmark_named` (medium) one that
  named Terminal-Bench (`recall.benchmark_unprompted`). The file's sha256 is in the scan
  cache key and in `scan.scanner.rules_sha256`.
- **Our review** (`review`): the publication decision on every pass atif-scan flags high
  or critical. `disqualified` lists exact trials (id or folder name) with a reason; they
  show as `x` and leave the score. `cleared` lists flagged passes that were kept, with the
  reason (e.g. a detector false positive), so every flag has a recorded decision.
  `recorded_passes` keeps the score before review; `coverage` notes evidence gaps.
  `clear_when_only` maps rule ids to a reason: a flagged pass whose only high/critical
  findings are those rules is cleared with that reason (`auto`), e.g. task recognition is
  a high observation, not a disqualification. A scan with any flagged pass left without a
  decision stops `fetch_runs.py`, so a reviewed run never shows an undecided flag.
- A score counts 445 slots (89 tasks × 5). Missing or errored trials score 0.

- **Multi-cohort releases**: a bench-run release may hold several cohorts (the GPT-6 Luna
  effort sweep `tb21-luna-flex-r1`). `bucket.cohort` names the run's cohort; the run's
  folder holds that cohort's jobs and a copy of the full release manifest.

## Safety stops and the safety allowance

`safety_cells` lists, per task, the attempt indices (aligned with `tasks`) of unrewarded
trials Harbor ended with `AgentSafetyRefusalError` or `AgentSafetyStopError`; they stay
`e` in `tasks`. The page marks them as safety stops. `safety-allowance.json` defines the optional *Safety allowance* scenario: a refused trial
counts as a pass when its task is easy for the run's reference model, i.e. the reference
passed at least `minPasses` (4) of its 5 attempts. Anthropic runs use Claude Haiku 5.5
high (our `haiku55-high-6h` run); every other run uses our GPT-6 Luna high run (both counted
after review). The generator emits `safety.refused` / `safety.reference` / `safety.allowance` per run and
`safetyAllowance` per benchmark; the page applies it only when switched on
(`?safety=allow`) and labels it a scenario. A run with safety errors but no
`safety_cells` stops the generator.

## Review details (`moments/`)

`fetch_runs.py --scan --moments` reruns each run's scan with atif-scan's `--highlights`
(same inputs and options) and `moments.py` turns the export into `moments/<run>.json`:
one card per pass with a high or critical finding and per review decision, with up to
three findings and two masked excerpts each (said / ran / got at the matched step), led by
the finding the decision cites. Cards link to the trial folder and the Hub's trace viewer
in the public bucket (bucket runs only). `fetch_runs.py` records each task's trials in
cell order (`trials`) for this. Excerpts are trace text: atif-scan masks secret shapes
(best effort) and `moments.py` refuses token-shaped text; check the files against known
credentials before committing. The generator copies them to `docs/benchmarks/moments/`;
a run page loads its file when opened.

## Review policy

Every pass atif-scan flags high or critical gets a recorded decision (`review` in the
manifest). The rules, as applied across the reviewed runs:

- **Disqualify** when the agent obtained, or deliberately sought, material from the
  benchmark's repository or mirrors (task README, task files, tests, reference solution,
  expected answer), even if the material was only descriptive or the search came back
  empty; or the task's upstream fix (fix-ocaml-gc).
- **Disqualify** a deliverable that isn't what the task asks for, or that games the
  check: pytorch-model-cli's `cli_tool` delivered as a Python script, or an ELF stub that
  only runs Python (the TB2.1 maintainers ruled the same); path-tracing-reverse code that
  hard-codes pixels.
- **Clear** recall alone: naming Terminal-Bench or its tasks unprompted, with no lookup
  (`clear_when_only` for `review.task_recalled`).
- **Clear** a network workaround used only for a download the task allows (POV-Ray
  sources), and a shortcut only considered in reasoning and never delivered.
- **Clear** detector false positives, with the reason.

## Cell codes (`tasks`)

One character per trial, ordered by `started_at`:
`1` rewarded · `0` failed without an error · `t` AgentTimeoutError (unrewarded) ·
`e` other error (unrewarded) · `x` rewarded but disqualified (leaderboard judge or our review) ·
`-` missing slot. A timed-out trial that was still rewarded shows as `1`.

## Scan block

`scan` summarises `atif-scan --brief --format json` over the run's full jobs; `scan.version`
and `scan.scanner` (commit, local changes, flags) record what produced it. On 2026-10-08
every scanned run (ours and the Claude Code / Terminus 2 comparators) was rescanned with
atif-scan 0.7.0 (`e973303`), and again on 2026-10-09 with 0.7.1 (`60844ae`), and `--image-model 'codexresponses.gpt-6-luna?reasoning=medium'`.
`codex-terra-max` has no scan (see below).
`scan.cells` holds one code per selected trial, in the same order as `tasks`, from the
full scan's per-trial results: `c`/`h`/`m`/`l` is the trial's highest unexcused priority
(critical, high, medium, or low/info/none), upper case means the trial's steps used a
model other than the run's (fallback), and `?` means no scan result. The page marks
attempts from these codes, so its counts match `review.high_or_critical_rewarded` and
`review.other_model_trials`. Findings are review priorities, not verdicts. For Hub runs
with excluded trials, the other scan totals still include them (e.g. 446 trials). Bucket
runs report the review, evidence and trial counts over the reported trials (`scan.scope`);
finding tallies still include scanned replaced originals.

## Caveats and runs not fully reconciled

All 14 runs match the published pass count with 445 filled slots. One run is still
marked `"reconciled": false`:

- `codex-terra-max`: the leaderboard row's trial associations (re-set 2026-08-12)
  resolve to public trials mixing Codex 0.144.0, 0.144.1 and unknown versions. Their error
  mix (15 timeouts, 12 non-zero exits, 1 setup and 1 verifier timeout) differs from PR
  #115's submitted set (0.144.1, 12 AgentTimeoutError). The submitted source job
  (dd8bc272) isn't public. The pass count matches (350 rewarded, 349 after one DQ), but
  the cells aren't the submitted trials.

Cost does not match the published figure for these runs:

- `cc-fable5-xhigh`, `terminus2-fable5-high`: the published totals ($552.67, $438.64)
  back-fill null Fable `cost_usd` at list rates on cost-fixed jobs that aren't public
  (PR #79). Recorded Hub cost is $366.37 (412 of 445 costed) and $70.20 (only the 79
  Opus-fallback trials costed). Recorded as-is, mixing pricing bases.
- `codex-terra-max`: the trials above record $406.51 (434 costed) against the published
  $421.15.
- `fa-deepseek-v4-flash-max`: Hub records $8.571 (434 costed); PR #189 states
  $8.73520228. The 11 uncosted trials also have no Hub token counts.

Disqualified trials (`cc-fable5-xhigh` ×1, `cc-sonnet5-high` ×3, `codex-terra-max` ×1)
are known only by their leaderboard-clone trial id and task, and the clone ids don't map
to the public trial ids. So `x` marks the first rewarded cell of that task, not
necessarily the exact attempt. The pass count matches the published score.

`codex-terra-max` has no scan because its public job (77fc16b9) mixes the Terra and
Luna Codex runs.

The fast-agent standard-timeout PRs (#170, #174, #189, #212, #221, #160) were closed
unmerged because community submissions were closed. Their scores are the submitters'
figures, not leaderboard entries. Source jobs come from each PR's submission file
(for #212/#221/#160, from the original submitter PRs #211/#220/#159).
