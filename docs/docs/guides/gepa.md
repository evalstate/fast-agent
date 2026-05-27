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

## Start with a batch evaluator

For data labeling, extraction, routing, grading, and classification tasks, use
`fast-agent batch run` as the evaluator body.

```bash
fast-agent batch run \
  --input eval/input.jsonl \
  --output runs/candidate-001/results.jsonl \
  --instruction runs/candidate-001/instructions.md \
  --template eval/template.md \
  --schema eval/output.schema.json \
  --model "responses.gpt-5.4-mini?service_tier=flex" \
  --parallel 8 \
  --include-input \
  --telemetry runs/candidate-001/telemetry.jsonl \
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
  --schema labels/label.schema.json \
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

## Call fast-agent from GEPA

The smallest integration is a GEPA evaluator function that materializes a
candidate, runs fast-agent, scores the results, and returns `(score, side_info)`.

```python title="gepa-run.py"
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from gepa.optimize_anything import GEPAConfig, EngineConfig, ReflectionConfig, optimize_anything


ROOT = Path(__file__).resolve().parent


def evaluate(candidate: dict[str, str]) -> tuple[float, dict[str, Any]]:
    run_dir = ROOT / "runs" / f"candidate-{len(list((ROOT / 'runs').glob('candidate-*'))) + 1:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)

    instruction_path = run_dir / "instructions.md"
    instruction_path.write_text(candidate["instructions"], encoding="utf-8")
    output_path = run_dir / "results.jsonl"

    proc = subprocess.run(
        [
            "fast-agent",
            "batch",
            "run",
            "--input",
            "eval/input.jsonl",
            "--output",
            str(output_path),
            "--instruction",
            str(instruction_path),
            "--template",
            "eval/template.md",
            "--schema",
            "eval/output.schema.json",
            "--model",
            "responses.gpt-5.4-mini",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    score, failures = score_jsonl(output_path)
    side_info = {
        "scores": {"gepa_score": score},
        "candidate_dir": str(run_dir),
        "stdout_tail": proc.stdout[-2000:],
        "stderr_tail": proc.stderr[-2000:],
        "failures": failures[:20],
        "actionable_feedback": summarize_failures(failures),
    }
    (run_dir / "score.json").write_text(json.dumps(side_info, indent=2), encoding="utf-8")
    return score, side_info


result = optimize_anything(
    seed_candidate={"instructions": Path("seed/instructions.md").read_text(encoding="utf-8")},
    evaluator=evaluate,
    objective="Improve the instruction so the batch worker returns accurate structured labels.",
    config=GEPAConfig(
        engine=EngineConfig(max_metric_calls=24, cache_evaluation=True),
        reflection=ReflectionConfig(reflection_lm="openai/gpt-5"),
    ),
)

Path("runs/best-instructions.md").write_text(result.best_candidate["instructions"], encoding="utf-8")
```

Replace `score_jsonl()` and `summarize_failures()` with your deterministic
checker. The checker is the most important part of the loop: it should encode
the product decision you actually care about and explain failures in language a
reflection model can act on.

## Use fast-agent for reflection

If you want GEPA's reflection calls to use the same fast-agent model aliases and
configuration as the rest of your project, wrap one-shot `fast-agent go` as a
callable reflection LM.

```python title="fast_agent_reflection.py"
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any


class FastAgentReflectionLM:
    def __init__(self, model: str, run_dir: Path) -> None:
        self.model = model
        self.run_dir = run_dir
        self.count = 0
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def __call__(self, prompt: str | list[dict[str, Any]]) -> str:
        self.count += 1
        call_dir = self.run_dir / f"call-{self.count:03d}"
        call_dir.mkdir(parents=True, exist_ok=True)
        prompt_path = call_dir / "prompt.md"
        prompt_path.write_text(prompt if isinstance(prompt, str) else repr(prompt), encoding="utf-8")

        proc = subprocess.run(
            [
                "fast-agent",
                "go",
                "--prompt-file",
                str(prompt_path),
                "--model",
                self.model,
                "--quiet",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        (call_dir / "stdout.txt").write_text(proc.stdout, encoding="utf-8")
        (call_dir / "stderr.txt").write_text(proc.stderr, encoding="utf-8")
        if proc.returncode:
            raise RuntimeError(proc.stderr[-2000:])
        return proc.stdout
```

Pass an instance where GEPA accepts a language-model callable, or use a model
string when GEPA's built-in provider integration is enough.

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

## Quick demo pack

Install the starter pack:

```bash
fast-agent go --pack gepa-demo
```

The pack installs a small AgentCard, a toy dataset, a deterministic checker, and
`scripts/gepa-run.py`. After installation, run:

```bash
uv run .fast-agent/scripts/gepa-run.py --evaluate-only
```

That command smoke-tests the seed candidate without requiring a long GEPA run.
Remove `--evaluate-only` when you are ready to install GEPA and run an
optimization loop.
