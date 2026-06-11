---
title: GEPA Optimization
description: Use fast-agent to build GEPA evaluators, Actionable Side Information, and optimization loops.
social:
  title: GEPA Optimization
  tagline: Optimize prompts, tool descriptions, and agent behavior with feedback-rich eval loops.
  description: Use fast-agent to build GEPA evaluators, Actionable Side Information, and optimization loops.
  alt: fast-agent social card - GEPA Optimization
---

# GEPA Optimization

[GEPA](https://github.com/gepa-ai/gepa) is a text optimizer for prompts,
instructions, tool descriptions, code, and other textual system parameters. It
works best when an evaluator returns both a score and enough diagnostic feedback
for a reflection model to explain what went wrong and propose a better
candidate. That diagnostic payload is GEPA's **Actionable Side Information**
(ASI).

**fast-agent** is a good fit for GEPA loops because it gives you the runtime
pieces that normally take the longest to build:

- batch and structured runs over repeatable datasets;
- AgentCards that package prompts, tools, request parameters, and model choices;
- AgentCard variables for candidate-scoped prompt and policy text;
- `BatchRunner`, `EvalRun`, and GEPA-compatible adapters for repeatable
  candidate directories, artifacts, reflection calls, and result contracts;
- local Python `function_tools` for cheap tool-call simulations;
- `tool_result_mode: passthrough` for routing evals where the tool call itself
  is the result, so the model does not spend another turn summarizing fake tool
  output;
- provider aliases, prompt caching, telemetry, and result JSONL you can score
  outside the model.

The core pattern is:

1. Put the text GEPA should mutate in a prompt file, AgentCard, tool docstring,
   schema description, or another file.
2. Evaluate each candidate with fast-agent.
3. Score the output deterministically where possible.
4. Return a scalar score plus ASI that names the failures, evidence, and useful
   edits.
5. Let GEPA propose another candidate and repeat.

!!! tip "Try the GEPA fast-agent skill"

    The `gepa-fast-agent` skill in the fast-agent skills registry distills this
    guide into a compact implementation checklist. Use it when designing or
    reviewing GEPA evaluators, scorer ASI, frontier metrics, Trackio monitoring,
    or batch/artifact optimizer layouts.

    ```bash
    fast-agent skills add gepa-fast-agent
    ```

## Pick the evidence shape: rows or artifacts

GEPA loops need a candidate, a score, and enough evidence for reflection.
Practically, fast-agent GEPA loops split by the evidence they produce and by the
GEPA API shape they use.

**Aggregate batch evals** run the same worker over many independent rows, then
score the output JSONL as one candidate result. Use this shape for
classification, extraction, labeling, routing, grading, and other dataset-style
tasks where the main signal is corpus-level. This is the
`candidate -> (score, side_info)` shape used by
`FastAgentBatchEvaluator` and GEPA's `optimize_anything()`.

```text
candidate instructions/card variables
  -> fast-agent batch run over input rows
  -> results.jsonl + summary/telemetry
  -> deterministic scorer
  -> score + ASI
```

**Row-wise batch evals** still use fast-agent batch execution, but expose each
input row as a GEPA optimization instance. Use this when GEPA should sample
minibatches, compare per-row scores, and reflect over row-level trajectories.
This is the adapter protocol shape used by `FastAgentRowWiseBatchAdapter` and
GEPA's `gepa.api.optimize()`.

```text
candidate instructions/card variables + GEPA minibatch rows
  -> fast-agent batch run over that minibatch
  -> aligned per-row outputs
  -> row scorer
  -> per-row scores + trajectories + objective scores
```

**Artifact-focused evals** ask the task model to create files or other
inspectable outputs, then run external checks. Use this shape for HTML pages,
code generation, docs, charts, screenshots, reports, or tool workflows where
the important result is an artifact rather than one JSON row.

```text
candidate skill/card/prompt files
  -> fast-agent go or a project eval runner
  -> generated artifacts + reports/screenshots/logs
  -> deterministic checker, optional VLM/manual review hooks
  -> score + ASI
```

The evidence hygiene is the same in every mode even when the GEPA API shape is
different. Keep candidate materialization isolated, write every
prompt/result/report to a candidate or evaluation directory, and make the
scorer produce compact evidence that the reflection model can act on. Batch
evals usually center on `results.jsonl`; row-wise evals center on minibatch
JSONL plus per-row trajectories; artifact evals center on generated files,
checker reports, screenshots, logs, or test output.

## Start with a batch evaluator

For data labeling, extraction, routing, grading, and classification tasks, use
`fast-agent batch run` as the evaluator body.

```bash
fast-agent batch run \
  --input eval/input.jsonl \
  --output runs/candidate-001/results.jsonl \
  --agent-card .fast-agent/agent-cards/classifier.md \
  --agent classifier \
  --var-file policy=runs/candidate-001/policy.md \
  --template eval/template.md \
  --json-schema eval/output.schema.json \
  --model "responses.gpt-5.4-mini?service_tier=flex" \
  --parallel 8 \
  --include-input \
  --telemetry-output runs/candidate-001/telemetry.jsonl \
  --summary-output runs/candidate-001/summary.json
```

Use the batch output as GEPA evidence. A checker can compare every row against
gold labels, parse failures, or business rules, then write ASI like this:

```json
{
  "scores": {
    "gepa_score": 0.72,
    "exact_match": 0.68,
    "valid_json": 0.96
  },
  "raw_metrics": {
    "latency_seconds": 14.2,
    "failure_count": 8
  },
  "failures": [
    {
      "id": "row-17",
      "type": "confused_label",
      "expected": "billing",
      "actual": "account_access",
      "evidence": "The row asks about an invoice PDF, not sign-in."
    }
  ],
  "actionable_feedback": [
    "Add an explicit billing-vs-account boundary rule near the label list.",
    "Keep the JSON fields unchanged; failures are semantic, not schema-related."
  ]
}
```

The ASI should be compact but specific. Include row IDs, expected and actual
behavior, excerpts, checker messages, and the candidate artifact path. Avoid
returning only a score; GEPA needs the reason for the score to make useful
edits.

All values in `side_info["scores"]` must be **higher is better** because GEPA
uses those keys for Pareto/frontier tracking. Do not place raw loss, latency,
cost, timeout seconds, failure counts, token counts, or policy length in
`scores` unless you first transform them into maximize-style values such as
`latency_score`, `cost_score`, `policy_length_compliance`, or
`failure_free_rate`. Keep raw lower-is-better diagnostics elsewhere, for
example under `raw_metrics`, `details`, or `summary`.

## Run structured labeling first

GEPA loops often need labels or validation data before optimization starts.
`fast-agent batch run` can produce structured labels from a stronger model, then
you can audit or override them before using them as ground truth.

```bash
fast-agent batch run \
  --input raw-examples.jsonl \
  --output labels/strong-model-labels.jsonl \
  --instruction labels/labeler.md \
  --template labels/template.md \
  --json-schema labels/label.schema.json \
  --model "responses.gpt-5.5"
```

A practical workflow is:

- generate structured labels with a strong model;
- review a sample and write explicit overrides for ambiguous rows;
- freeze the label file before starting GEPA;
- score all candidates with the same frozen labels.

This keeps the optimizer from chasing a moving target.

## Optimize tool descriptions

Tool-routing evals are one of the fastest ways to use GEPA with fast-agent. You
can expose production-shaped tools as local Python stubs, ask the model to make
one tool call, and score the call name and arguments without performing any real
network work.

```python title="eval/tools.py"
def hub_repo_details(repo_id: str, files: list[str] | None = None) -> dict[str, object]:
    """Get metadata and selected file contents for one Hugging Face repository.

    Args:
        repo_id: Repository id such as `org/name`.
        files: Optional file paths to read from the repository.
    """
    return {"repo_id": repo_id, "files": files or []}
```

```markdown title="eval/tool-routing-agent.md"
---
type: agent
name: tool_router
model: "$system.default"
request_params:
  tool_result_mode: passthrough
  max_iterations: 1
  parallel_tool_calls: false
function_tools:
  - ./tools.py:hub_repo_details
---

Call the most appropriate tool for the user's request.
```

Then run the card over your routing dataset:

```bash
fast-agent batch run \
  --agent-card eval/tool-routing-agent.md \
  --agent tool_router \
  --input eval/routing-input.jsonl \
  --output runs/candidate-001/results.jsonl \
  --model "responses.gpt-5.4-mini" \
  --parallel 8
```

For this style of eval, GEPA can mutate the card body, the tool docstrings, the
argument descriptions, or all of them. Keep the scoring outside the model: parse
the batch result JSONL, compare the observed tool call to expected alternatives,
and return ASI that explains wrong routes, missing arguments, over-broad
descriptions, or tool-local hygiene problems.

## Build artifact evaluators

For artifact tasks, make the evaluator create an isolated candidate directory,
run fast-agent or a project runner, then score the generated files.

```text
runs/gepa/candidate-001/
  candidate.json
  variables.json
  prompts/
  artifacts/
  reports/
  logs/
  score.json
```

The scorer can combine deterministic checks with optional review hooks:

- parse generated HTML, Markdown, code, CSV, JSON, images, or reports;
- run linters, unit tests, screenshot checks, render checks, or schema
  validators;
- capture stdout/stderr, return code, timeout status, and report paths;
- add VLM or manual-review findings as ASI when visual quality matters;
- keep the mutable candidate files separate from fixed fixtures and source
  data.

Artifact evals often benefit from subprocess isolation. A failed candidate
should usually produce side-info and artifacts, not corrupt the working tree or
abort the optimization without evidence.

## Make Trackio the default dashboard

GEPA runs can be long-lived. [Trackio](https://github.com/gradio-app/trackio) is recommended to provide a live view of GEPA optimize metrics, and other metadata you want to track. Use one Trackio run for both GEPA's optimizer metrics and your evaluator-specific candidate metrics:

- GEPA logs frontier, candidate, proposal, validation, objective, and summary
  data under the `gepa/` prefix.
- Your scorer can log task-specific candidate metrics under `candidate/` and
  `candidate/detail/`.
- Keep `side_info["scores"]` frontier-safe: every value should be numeric and
  higher-is-better. Put raw lower-is-better diagnostics in
  `side_info["score_details"]` or `side_info["raw_metrics"]`.

The useful side-info shape is:

```python
side_info = {
    "scores": {
        "gepa_score": score,
        "valid_output_rate": valid_output_rate,
        "failure_free_rate": failure_free_rate,
    },
    "score_details": {
        "failure_count": failure_count,
        "latency_seconds": latency_seconds,
    },
    "failures": failures[:20],
    "actionable_feedback": summarize_failures(failures),
}
```

`fast_agent.integrations.gepa.gepa_numeric_metrics()` turns that into Trackio
metrics such as `candidate/gepa_score` and
`candidate/detail/failure_count`; `safe_trackio_log()` makes that logging
best-effort so a dashboard outage does not fail an evaluator.

## Call fast-agent from GEPA

Install GEPA and Trackio support with the `gepa` optional dependency:

```bash
uv add "fast-agent-mcp[gepa]"
```

PyPI packages cannot declare direct Git dependencies in extras, so the published
extra depends on the latest PyPI GEPA release plus Trackio. Trackio-specific GEPA
helpers require a GEPA release with Trackio support; until that support is
available on PyPI, install the integration branch in your application
environment:

```bash
uv add "gepa @ git+https://github.com/evalstate/gepa.git@feat/trackio"
```

See the [Extension Reference](../ref/extension_reference/#gepa) for the exact
dependency set and adapter class signatures.

For row-oriented evaluators, use the public batch adapter instead of
hand-rolling candidate directories, subprocess calls, JSONL parsing, summary
paths, and telemetry paths.

```python title="gepa-run.py"
from __future__ import annotations

from pathlib import Path
from typing import Any

import trackio
from gepa.optimize_anything import GEPAConfig, EngineConfig, ReflectionConfig, optimize_anything
from fast_agent.batch import BatchRunResult
from fast_agent.eval import CandidateRun
from fast_agent.integrations.gepa import (
    FastAgentBatchEvaluator,
    FastAgentReflectionLM,
    gepa_numeric_metrics,
    gepa_trackio_init_kwargs,
    make_gepa_tracking_config,
    safe_trackio_log,
)


ROOT = Path(__file__).resolve().parent


def score_candidate(
    result: BatchRunResult,
    candidate: dict[str, str],
    candidate_run: CandidateRun,
) -> tuple[float, dict[str, Any]]:
    score, failures = score_rows(result.rows)
    side_info = {
        "scores": {
            "gepa_score": score,
            "valid_output_rate": sum(1 for row in result.rows if row.get("ok")) / max(len(result.rows), 1),
        },
        "score_details": {
            "failure_count": len(failures),
        },
        "candidate_dir": str(candidate_run.path),
        "summary": result.summary,
        "failures": failures[:20],
        "actionable_feedback": summarize_failures(failures),
    }
    safe_trackio_log(
        {
            "gepa/iteration": (candidate_run.index or 1) - 1,
            "candidate/local_idx": candidate_run.index or 0,
            **gepa_numeric_metrics(side_info),
        }
    )
    return score, side_info


evaluator = FastAgentBatchEvaluator(
    env_dir=".fast-agent",
    agent_card=".fast-agent/agent-cards/classifier.md",
    agent="classifier",
    candidate_variables={"policy": "policy"},
    input="eval/input.jsonl",
    template_source="eval/template.md",
    schema="eval/output.schema.json",
    model="responses.gpt-5.4-mini",
    parallel=8,
    scorer=score_candidate,
    run_dir=ROOT / "runs",
    backend="process",
)

reflection_lm = FastAgentReflectionLM(
    env_dir=".fast-agent",
    model="responses.gpt-5.5?reasoning=high",
    audit_dir=ROOT / "runs" / "reflection",
)

trackio.init(
    **gepa_trackio_init_kwargs(
        project="fast-agent-gepa",
        name="classifier-policy",
        config={
            "mode": "aggregate",
            "agent": "classifier",
            "run_dir": str(ROOT / "runs"),
        },
    )
)

result = optimize_anything(
    seed_candidate={"policy": Path("seed/policy.md").read_text(encoding="utf-8")},
    evaluator=evaluator,
    objective="Improve the policy so the batch worker returns accurate structured labels.",
    config=GEPAConfig(
        engine=EngineConfig(max_metric_calls=24, cache_evaluation=True),
        reflection=ReflectionConfig(reflection_lm=reflection_lm),
        tracking=make_gepa_tracking_config(
            name="classifier-policy",
            attach_existing=True,
        ),
    ),
)

Path("runs/best-policy.md").write_text(result.best_candidate["policy"], encoding="utf-8")
```

Replace `score_rows()` and `summarize_failures()` with your deterministic
checker. The checker is the most important part of the loop: it should encode
the product decision you actually care about and explain failures in language a
reflection model can act on.

### Use row-wise GEPA when each row is an optimization instance

`FastAgentBatchEvaluator` treats the whole JSONL batch as one GEPA metric call:
`candidate -> full batch -> one aggregate score + side_info`. That is usually
the simplest and most stable mode.

GEPA also supports a lower-level adapter protocol where it samples minibatches
from a trainset and expects **one score per row**. Use
`FastAgentRowWiseBatchAdapter` for that mode. fast-agent still owns the JSONL
minibatch file, `BatchRunner` call, result alignment, summaries, telemetry, and
artifact directories; your code supplies row scoring and row-level trajectories.

```python title="row-wise-gepa.py"
from __future__ import annotations

from pathlib import Path
from typing import Any

from gepa.api import optimize

from fast_agent.integrations.gepa import (
    FastAgentReflectionLM,
    FastAgentRowWiseBatchAdapter,
    RowWiseEvaluationRun,
    RowWiseScore,
    gepa_api_trackio_kwargs,
)


def score_row(
    output_row: dict[str, Any],
    input_row: dict[str, Any],
    candidate: dict[str, str],
    evaluation: RowWiseEvaluationRun,
) -> RowWiseScore:
    result = output_row.get("result") if isinstance(output_row.get("result"), dict) else {}
    expected = input_row["expected_label"]
    actual = result.get("label")
    score = 1.0 if actual == expected else 0.0
    return RowWiseScore(
        score=score,
        trajectory={
            "scores": {"gepa_score": score, "row_exact": score},
            "expected": expected,
            "actual": actual,
            "feedback": (
                "Preserve this row behavior."
                if score == 1.0
                else f"Expected {expected!r}, got {actual!r}."
            ),
        },
        objective_scores={"gepa_score": score, "row_exact": score},
    )


adapter = FastAgentRowWiseBatchAdapter(
    env_dir=".fast-agent",
    agent_card=".fast-agent/agent-cards/classifier.md",
    agent="classifier",
    candidate_variables={"policy": "policy"},
    template_source="eval/template.md",
    schema="eval/output.schema.json",
    model="responses.gpt-5.4-mini",
    parallel=8,
    row_scorer=score_row,
    run_dir=Path("runs/row-wise-gepa"),
    id_field="id",
    backend="process",
)

rows = load_jsonl("eval/train.jsonl")

result = optimize(
    seed_candidate={"policy": Path("seed/policy.md").read_text(encoding="utf-8")},
    trainset=rows,
    valset=rows,
    adapter=adapter,
    reflection_lm=FastAgentReflectionLM(
        env_dir=".fast-agent",
        model="responses.gpt-5.5?reasoning=high",
        audit_dir="runs/row-wise-gepa/reflection",
    ),
    reflection_minibatch_size=3,
    max_metric_calls=400,
    cache_evaluation=True,
    frontier_type="hybrid",
    **gepa_api_trackio_kwargs(
        project="fast-agent-gepa",
        name="classifier-row-wise",
        config={
            "mode": "row-wise",
            "agent": "classifier",
        },
    ),
)
```

See the [Extension Reference](../ref/extension_reference/#gepa-integration-adapters)
for the generated signatures of the GEPA adapter classes.

Use row-wise mode when the row is the natural optimization instance and GEPA
should reflect over row-level successes/failures. Keep aggregate metrics such as
micro-F1, exact-match rate, or average Jaccard in your scorer or Trackio logs as
needed, but remember that GEPA's adapter protocol accepts per-row scores and
sums/minibatches them during candidate selection. If your signal is mostly a
single corpus-level metric, prefer `FastAgentBatchEvaluator`.

For artifact evaluators, use `EvalRun` and `CandidateRun` directly:

```python title="artifact-evaluator.py"
from fast_agent.eval import EvalRun

run = EvalRun("runs/gepa-artifacts")


def evaluate(candidate: dict[str, str]) -> tuple[float, dict[str, Any]]:
    candidate_run = run.candidate()
    candidate_run.materialize_candidate(candidate)
    materialize_skill_tree(candidate, candidate_run.path / "skills")
    command = candidate_run.run_command(
        ["uv", "run", "python", "scripts/run_eval.py", "--candidate", str(candidate_run.path)],
        timeout_seconds=900,
        log_prefix="eval",
    )
    score, side_info = score_reports(candidate_run.path / "reports", command)
    candidate_run.write_score(score, side_info)
    return score, side_info
```

Use the reflection adapter when you want GEPA reflection calls to use the same
fast-agent model aliases and configuration as the rest of your project. The
adapter writes prompt, request, response, timing, stdout/stderr, errors, and
usage when available, under the configured audit directory.

## Candidate hygiene

GEPA will optimize exactly what you score. Add guardrails to the evaluator when
you need production-safe outputs:

- reject invalid JSON schemas, invalid AgentCards, or Python files that do not
  import;
- penalize tool descriptions that hide global routing policy inside one
  unrelated tool;
- enforce token, latency, or cost budgets as secondary scores;
- keep train and validation splits separate;
- write every candidate, score, ASI payload, stdout, stderr, and result JSONL to
  disk so a winning candidate can be audited before adoption.

## Card packs

The default fast-agent card-pack registry includes
[`gepa-demo`](https://github.com/fast-agent-ai/card-packs/tree/main/packs/gepa-demo),
a small runnable starter pack with one batch evaluator demo and one artifact
evaluator demo.

Install and open the helper agent:

```bash
fast-agent go --pack gepa-demo
```

Then smoke-test the two evaluator shapes without external model calls:

```bash
uv run .fast-agent/scripts/gepa-run.py --evaluate-only
uv run .fast-agent/scripts/gepa-artifact-run.py --evaluate-only
```

If you create or ship a GEPA card pack, keep its AgentCards and scripts aligned
with this guide:

- declare mutable AgentCard text with `variables`;
- call `fast-agent batch run` with `--var`, `--var-file`, or `--vars-json`;
- use `--json-schema`, `--telemetry-output`, and `--summary-output`;
- use `FastAgentBatchEvaluator` or `EvalRun`/`CandidateRun` rather than
  duplicating candidate directory and subprocess boilerplate;
- audit every `side_info["scores"]` key as higher-is-better.
