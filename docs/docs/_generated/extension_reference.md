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
### `FastAgentSingleTaskAdapter`

`optimize_anything()`-friendly evaluator for one fast-agent task at a
time. It runs a batch of one, records the same candidate artifacts,
`results.jsonl`, summary, and telemetry files as larger batch evals, and
exposes evaluation metrics for `FastAgentGEPATrackioCallback`. Use this
when you want the simple `candidate -> score + side_info` API without
building a JSONL dataset first.

```python
FastAgentSingleTaskAdapter(
    *,
    env_dir: str | Path | None = None,
    agent_card: str | Path | None = None,
    agent: str | None = None,
    model: str | None = None,
    input_builder: SingleTaskInputBuilder,
    scorer: SingleTaskScorer,
    run_dir: str | Path,
    template: str | None = '{{prompt}}',
    template_source: str | Path | None = None,
    schema: str | Path | None = None,
    backend: BatchBackend = 'harness',
    include_input: bool = True,
    batch_runner_factory: BatchRunnerFactory | None = None
) -> None
```
```python
FastAgentSingleTaskAdapter.prompt(
    *,
    model: str | None = None,
    scorer: SingleTaskScorer,
    run_dir: str | Path,
    candidate_key: str = 'prompt',
    env_dir: str | Path | None = None,
    template: str = '{{prompt}}',
    backend: BatchBackend = 'harness',
    batch_runner_factory: BatchRunnerFactory | None = None
) -> FastAgentSingleTaskAdapter
```
```python
adapter.__call__(candidate: Mapping[str, str], example: Any | None = None) -> tuple[float, Any]
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
adapter.evaluate(
    batch: list[JsonRow],
    candidate: dict[str, str],
    capture_traces: bool = False
) -> Any
```
```python
adapter.make_reflective_dataset(
    candidate: dict[str, str],
    eval_batch: Any,
    components_to_update: list[str]
) -> Mapping[str, Sequence[Mapping[str, Any]]]
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
RowWiseEvaluationRun(
    index: int,
    path: Path,
    input_path: Path,
    result: BatchRunResult
) -> None
```
### `SingleTaskEvaluationRun`

Metadata passed to a single-task scorer, including the candidate directory,
one-row input file, `BatchRunResult`, and `CandidateRun`.

```python
SingleTaskEvaluationRun(
    index: int,
    path: Path,
    input_path: Path,
    result: BatchRunResult,
    candidate_run: CandidateRun
) -> None
```
### Trackio helpers

Trackio helpers provide fast-agent defaults for GEPA dashboards. Use
`gepa_trackio_init_kwargs()` when your script initializes Trackio, use
`gepa_api_trackio_kwargs()` with `gepa.api.optimize()`, and use
`make_gepa_trackio_dashboard()` or `make_gepa_tracking_config()` with
`optimize_anything()`.

```python
gepa_trackio_init_kwargs(
    *,
    project: str = 'fast-agent-gepa',
    name: str | None = None,
    group: str | None = None,
    config: Mapping[str, Any] | None = None,
    embed: bool = False,
    auto_log_gpu: bool = False,
    **overrides: Any
) -> dict[str, Any]
```
```python
gepa_api_trackio_kwargs(
    *,
    project: str = 'fast-agent-gepa',
    name: str | None = None,
    group: str | None = None,
    config: Mapping[str, Any] | None = None,
    step_metric: str = 'gepa/iteration',
    key_prefix: str = 'gepa/',
    attach_existing: bool = False,
    **trackio_init_overrides: Any
) -> dict[str, Any]
```
```python
make_gepa_tracking_config(
    *,
    project: str = 'fast-agent-gepa',
    name: str | None = None,
    group: str | None = None,
    config: Mapping[str, Any] | None = None,
    step_metric: str = 'gepa/iteration',
    key_prefix: str = 'gepa/',
    attach_existing: bool = False,
    **trackio_init_overrides: Any
) -> Any
```
```python
make_gepa_trackio_dashboard(
    *,
    project: str = 'fast-agent-gepa',
    name: str | None = None,
    group: str | None = None,
    config: Mapping[str, Any] | None = None,
    reflection_lm: FastAgentReflectionLM | None = None,
    eval_adapter: Any | None = None,
    step_metric: str = 'gepa/iteration',
    key_prefix: str = 'gepa/',
    attach_existing: bool = False,
    include_gepa_context: bool = False,
    **trackio_init_overrides: Any
) -> GEPATrackioDashboard
```
### Evaluator metric helpers

`gepa_numeric_metrics()` flattens `side_info['scores']`,
`side_info['score_details']`, and `side_info['raw_metrics']` into
Trackio-friendly numeric metrics. `safe_trackio_log()` logs them without
making Trackio a hard dependency of evaluator code.

```python
gepa_numeric_metrics(
    side_info: Mapping[str, Any],
    *,
    score_prefix: str = 'candidate/',
    detail_prefix: str = 'candidate/detail/',
    raw_prefix: str = 'candidate/raw/'
) -> dict[str, NumericMetric]
```
```python
safe_trackio_log(payload: Mapping[str, Any], *, step: int | None = None) -> bool
```
