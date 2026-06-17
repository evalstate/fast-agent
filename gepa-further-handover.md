# GEPA further handover

Date: 2026-06-11

Branch: `dev/0.7.18`

Current pushed HEAD:

```text
f354485d Fix CI formatting and optional GEPA import
```

Working tree note:

```text
?? deploy/
```

`deploy/` is still untracked and was intentionally left alone because it did not
look like GEPA demo data.

## What landed on this branch

### 1. Important max-token behavior fix

Commit:

```text
da21211b Omit unset max token defaults
```

Purpose:

- stop inventing small default `maxTokens` caps for unknown/blank models;
- allow unset model limits to remain `None`;
- omit OpenAI/Responses max-token request fields when unset;
- avoid importing llama.cpp `/slots` request/previous-request state as a
  persistent overlay max-output-token default.

Validation at the time:

```text
uv run scripts/lint.py --fix
uv run scripts/typecheck.py
uv run pytest tests/unit -q
```

Result:

```text
5460 passed
```

Follow-up generated docs now show:

```text
RequestParams.maxTokens: int | None = None
```

### 2. Row-wise GEPA adapter

Commit:

```text
39df3d8e Add GEPA row-wise batch adapter
```

Added to:

```text
src/fast_agent/integrations/gepa.py
```

New public pieces:

- `FastAgentRowWiseBatchAdapter`
- `RowWiseEvaluationRun`
- `RowWiseScore`
- `RowWiseBatchScorer`
- `ReflectiveDatasetBuilder`

Existing pieces retained:

- `FastAgentReflectionLM`
- `FastAgentBatchEvaluator`

Intent:

- `FastAgentBatchEvaluator` remains the aggregate `candidate -> full batch ->
  one score + side_info` path for GEPA `optimize_anything()`.
- `FastAgentRowWiseBatchAdapter` is the lower-level GEPA adapter-protocol path
  for `gepa.api.optimize()`, where GEPA supplies minibatches and expects
  per-row scores/trajectories/objective scores.

Artifact layout for row-wise runs:

```text
row-wise-evals/eval-00001/
  input.jsonl
  results.jsonl
  batch-summary.json
  telemetry.jsonl
  row-wise-score.json
```

Implementation details:

- Writes GEPA minibatches to JSONL.
- Runs `BatchRunner`.
- Applies candidate variables.
- Aligns outputs back to inputs by `id_field` when provided.
- Returns real `gepa.core.adapter.EvaluationBatch` if GEPA is installed.
- Falls back to a local structural `_FallbackEvaluationBatch` when GEPA is not
  installed.
- Optional GEPA import is dynamic via `importlib.import_module()` so CI
  typecheck does not require the `[gepa]` extra.

Relevant unit test:

```text
tests/unit/fast_agent/integrations/test_gepa.py
```

### 3. GEPA docs and Extension Reference

Commits:

```text
f63bd718 Document GEPA row-wise batch adapter
5a62751b Add GEPA extension reference docs
```

Updated:

```text
docs/docs/guides/gepa.md
```

The guide now distinguishes three evidence/API shapes:

1. **Aggregate batch evals**
   - full JSONL batch;
   - one aggregate candidate score;
   - `FastAgentBatchEvaluator`;
   - GEPA `optimize_anything()`.

2. **Row-wise batch evals**
   - fast-agent batch execution over GEPA minibatches;
   - per-row scores/trajectories/objective scores;
   - `FastAgentRowWiseBatchAdapter`;
   - GEPA `gepa.api.optimize()`.

3. **Artifact-focused evals**
   - generated files/reports/screenshots/logs/test output;
   - `EvalRun`/`CandidateRun` or custom project runners.

Added:

```text
docs/docs/ref/extension_reference.md
docs/docs/_generated/extension_reference.md
docs/docs/assets/social/ref/extension_reference.png
```

Navigation updated:

```text
Reference
  Class Reference
  Extension Reference
  Open Telemetry
```

Generator updated:

```text
docs/generate_reference_docs.py
```

The generator now produces GEPA adapter signatures for the Extension Reference.
It also normalizes some Python-version-specific signature output:

- `pathlib._local.Path` -> `pathlib.Path`
- `"\'BatchBackend\'"` -> `BatchBackend`

Docs build validation:

```text
uv run scripts/docs.py build
```

Result:

```text
No issues found
Built site in docs/site
```

### 4. `[gepa]` optional extra

Commit:

```text
5a62751b Add GEPA extension reference docs
```

Added to `pyproject.toml`:

```toml
[project.optional-dependencies]
gepa = [
    "gepa @ git+https://github.com/evalstate/gepa.git@feat/trackio",
    "trackio>=0.26.0",
]

[tool.hatch.metadata]
allow-direct-references = true
```

Reason:

- Trackio support is important for fast-agent GEPA workflows.
- The currently useful GEPA Trackio integration is on the
  `evalstate/gepa` branch `feat/trackio`.
- Hatch needs `allow-direct-references = true` for this VCS optional dependency.

Lockfile:

```text
uv.lock
```

was updated and resolves:

```text
gepa 0.1.1 from git+https://github.com/evalstate/gepa.git@e3b12d3...
trackio 0.27.0
```

Local verification:

```text
uv sync --extra gepa --frozen --no-dev
```

passed.

Important release caveat:

- `uv build` passes locally.
- `twine check dist/*` passes after cleaning `dist/`.
- But published wheel metadata includes:

```text
Requires-Dist: gepa @ git+https://github.com/evalstate/gepa.git@feat/trackio ; extra == 'gepa'
```

PyPI/Warehouse may reject direct VCS dependencies at upload time. If PyPI upload
fails, likely fix is to switch the published extra to a normal release spec once
GEPA Trackio support is released, or document the branch install separately.

## CI issue found and fixed

Commit:

```text
f354485d Fix CI formatting and optional GEPA import
```

Problems reproduced locally:

### Format job

CI runs:

```text
uv run scripts/format.py
```

Before the fix it reformatted files. It also attempted to format files in the
`skills-repo` submodule.

Fix:

```toml
[tool.ruff]
extend-exclude = [
    "docs/docs/assets/gepa-monty-runs/**/candidate-*/*.monty.py",
    "skills-repo/**",
]
```

After fix:

```text
uv run scripts/format.py
1328 files left unchanged
```

### Typecheck job

CI installs:

```text
uv sync --locked --group dev
```

It does not install `fast-agent-mcp[gepa]`, so a direct import of
`gepa.core.adapter` caused:

```text
error[unresolved-import]: Cannot resolve imported module `gepa.core.adapter`
```

Fix:

- changed `FastAgentRowWiseBatchAdapter`’s `EvaluationBatch` construction to use
  `importlib.import_module("gepa.core.adapter")`;
- kept `_FallbackEvaluationBatch` for environments without GEPA.

## Validation after CI fixes

Commands run after the final fix:

```bash
uv sync --locked --group dev
uv run scripts/format.py
uv run scripts/lint.py
uv run scripts/typecheck.py
uv run scripts/check_internal_resources.py
uv run pytest tests/unit/fast_agent/integrations/test_gepa.py tests/unit/fast_agent/utils/test_async_utils.py -q
uv run pytest tests/unit -q
env -u ENVIRONMENT_DIR -u FAST_AGENT_RUNTIME_ENVIRONMENT bash scripts/test_package_install.sh
```

Results:

```text
format: clean
lint: passed
typecheck: passed
internal resources: passed
focused tests: 6 passed
unit tests: 5453 passed, 1 skipped
package install smoke: passed when local env overrides were unset
```

Local package smoke note:

- This shell had:

```text
ENVIRONMENT_DIR=/home/ssmith/source/fast-agent-pr/.fast-agent
FAST_AGENT_RUNTIME_ENVIRONMENT=/home/ssmith/source/fast-agent-pr/.fast-agent
```

- With those set, `scripts/test_package_install.sh` failed in the plugin smoke
  because temp-project plugin installs resolved against the repo environment.
- With them unset, the package smoke passed. CI should not have those variables.

## Related OpenClaw / GEPA context

Project inspected:

```text
~/temp/gepa-batch-openclaw/
```

Important local prototype:

```text
scripts/openclaw-vanilla-f1-gepa.py
```

It contains:

- `--gepa-mode batch|row-wise`
- `--score-mode f1|row-aware`
- an `OpenClawRowWiseAdapter` prototype
- row-aware composite:

```text
0.50 * topic_micro_f1
+ 0.20 * row_exact_accuracy
+ 0.30 * avg_row_jaccard
```

Decision made:

- fast-agent should own the generic row-wise batch adapter mechanics;
- OpenClaw-specific scoring, policy hygiene, topic hints, static ASI, and
  Trackio table schemas stay project/evaluator-specific.

## Follow-ups / watch list

1. **CI**
   - Branch has been pushed after the CI fixes.
   - Watch GitHub Actions for `format`, `lint/typecheck`, and `package-test`.

2. **PyPI publish**
   - The direct VCS dependency in `[project.optional-dependencies].gepa` may
     block PyPI upload.
   - If that happens, either:
     - release GEPA Trackio support and switch to `gepa>=...`; or
     - remove direct dependency from published metadata and document a separate
       `uv add "gepa @ git+..."` command.

3. **Trackio ergonomics**
   - We have not yet added pure helper functions for metric flattening or safe
     Trackio logging.
   - Candidate future helpers:

```python
gepa_numeric_metrics(side_info, score_prefix="candidate/", detail_prefix="candidate/detail/")
safe_trackio_log(payload)
```

4. **Package build hygiene**
   - `uv build` currently copies `__pycache__` files from `examples/` into
     `src/fast_agent/resources/examples`.
   - This did not fail CI locally, but it is noisy and probably worth cleaning
     in a separate focused change.

5. **Untracked `deploy/`**
   - Still present locally.
   - Not touched as part of GEPA changes.
