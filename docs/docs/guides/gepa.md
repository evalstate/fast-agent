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

GEPA only requires an evaluator with one shape: `candidate -> (score,
side_info)`. Philosophically, batch and artifact evals are the same kind of
optimization loop. Practically, fast-agent GEPA loops almost always split by the
evidence they produce.

**Batch-focused evals** run the same worker over many independent rows, then
score the output JSONL. Use this shape for classification, extraction,
labeling, routing, grading, and other dataset-style tasks.

```text
candidate instructions/card variables
  -> fast-agent batch run over input rows
  -> results.jsonl + summary/telemetry
  -> deterministic scorer
  -> score + ASI
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

The evaluator contract is the same in both cases. The operational details are
different: batch evals usually center on `results.jsonl`; artifact evals usually
center on generated files, checker reports, screenshots, logs, or test output.
Keep candidate materialization isolated, write every prompt/result/report to a
candidate directory, and make the scorer produce compact evidence that the
reflection model can act on.

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

!!! tip "Monitor long runs with Trackio"

    GEPA runs can be long-lived. If you want a live view of candidate scores,
    frontier metrics, alerts, or run metadata, initialize
    [Trackio](https://github.com/huggingface/trackio) in your evaluator script
    and log after each candidate evaluation. Keep raw lower-is-better values out
    of `side_info["scores"]`, but log them to Trackio normally for monitoring.

## Call fast-agent from GEPA

For row-oriented evaluators, use the public batch adapter instead of
hand-rolling candidate directories, subprocess calls, JSONL parsing, summary
paths, and telemetry paths.

```python title="gepa-run.py"
from __future__ import annotations

from pathlib import Path
from typing import Any

from gepa.optimize_anything import GEPAConfig, EngineConfig, ReflectionConfig, optimize_anything
from fast_agent.batch import BatchRunResult
from fast_agent.eval import CandidateRun
from fast_agent.integrations.gepa import FastAgentBatchEvaluator, FastAgentReflectionLM


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
        "candidate_dir": str(candidate_run.path),
        "summary": result.summary,
        "failures": failures[:20],
        "actionable_feedback": summarize_failures(failures),
    }
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

result = optimize_anything(
    seed_candidate={"policy": Path("seed/policy.md").read_text(encoding="utf-8")},
    evaluator=evaluator,
    objective="Improve the policy so the batch worker returns accurate structured labels.",
    config=GEPAConfig(
        engine=EngineConfig(max_metric_calls=24, cache_evaluation=True),
        reflection=ReflectionConfig(reflection_lm=reflection_lm),
    ),
)

Path("runs/best-policy.md").write_text(result.best_candidate["policy"], encoding="utf-8")
```

Replace `score_rows()` and `summarize_failures()` with your deterministic
checker. The checker is the most important part of the loop: it should encode
the product decision you actually care about and explain failures in language a
reflection model can act on.

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
