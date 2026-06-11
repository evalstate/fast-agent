<!--
  GENERATED FILE — DO NOT EDIT.
  Source: generate_reference_docs.py
-->

## GEPA integration adapters

Import GEPA helpers from `fast_agent.integrations.gepa`. These adapters keep
fast-agent responsible for batch execution, artifact paths, and reflection LM
calls while leaving scoring policy in your evaluator code.

### `FastAgentReflectionLM`

Synchronous language-model callable for GEPA reflection calls backed by
`fast-agent go`. It writes prompt, request, response, timing, stdout/stderr,
error, and usage artifacts under `audit_dir`.

```python
FastAgentReflectionLM(
    *,
    env_dir: str | Path | None = None,
    model: str | None = None,
    audit_dir: str | Path,
    agent: str | None = None,
    timeout_seconds: float | None = None,
    command_runner: CommandRunner | None = None
)
```
```python
reflection_lm.__call__(prompt: str | list[dict[str, Any]]) -> str
```
### `FastAgentBatchEvaluator`

Aggregate GEPA evaluator for `gepa.optimize_anything.optimize_anything`: one
candidate runs one full fast-agent batch and returns one `(score, side_info)`
pair. Use this when the primary metric is corpus-level.

```python
FastAgentBatchEvaluator(
    *,
    env_dir: str | Path | None = None,
    agent_card: str | Path,
    agent: str | None = None,
    candidate_variables: Mapping[str, str],
    input: str | Path,
    template: str | None = None,
    template_source: str | Path | None = None,
    schema: str | Path | None = None,
    model: str | None = None,
    parallel: int | None = None,
    scorer: BatchScorer,
    run_dir: str | Path,
    backend: BatchBackend = 'harness',
    include_input: bool = True
)
```
```python
evaluator.__call__(candidate: Mapping[str, str]) -> tuple[float, Any]
```
### `FastAgentRowWiseBatchAdapter`

Lower-level GEPA adapter protocol implementation for `gepa.api.optimize`: GEPA
supplies minibatches of input rows, fast-agent runs each minibatch through
`BatchRunner`, and your `row_scorer` returns one score/trajectory per row.

```python
FastAgentRowWiseBatchAdapter(
    *,
    env_dir: str | Path | None = None,
    agent_card: str | Path,
    agent: str | None = None,
    candidate_variables: Mapping[str, str],
    template: str | None = None,
    template_source: str | Path | None = None,
    schema: str | Path | None = None,
    model: str | None = None,
    parallel: int | None = None,
    row_scorer: RowWiseBatchScorer,
    run_dir: str | Path,
    backend: BatchBackend = 'harness',
    include_input: bool = True,
    id_field: str | None = None,
    reflective_dataset_builder: ReflectiveDatasetBuilder | None = None,
    batch_runner_factory: BatchRunnerFactory | None = None
)
```
```python
adapter.evaluate(batch: list[JsonRow], candidate: dict[str, str], capture_traces: bool = False) -> Any
```
```python
adapter.make_reflective_dataset(candidate: dict[str, str], eval_batch: Any, components_to_update: list[str]) -> Mapping[str, Sequence[Mapping[str, Any]]]
```
### `RowWiseScore`

`row_scorer` may return `RowWiseScore`, a bare float, `(score, trajectory)`,
or `(score, trajectory, objective_scores)`. `objective_scores` values should
be higher-is-better because GEPA uses them for frontier tracking.

```python
RowWiseScore(
    score: float,
    trajectory: Any = None,
    objective_scores: Mapping[str, float] | None = None
) -> None
```
### `RowWiseEvaluationRun`

Metadata passed to each `row_scorer` call for the current minibatch evaluation.

```python
RowWiseEvaluationRun(index: int, path: Path, input_path: Path, result: BatchRunResult) -> None
```
