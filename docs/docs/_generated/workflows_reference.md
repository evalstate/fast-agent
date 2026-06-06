<!--
  GENERATED FILE — DO NOT EDIT.
  Source: generate_reference_docs.py
-->

## Workflow Decorators (Generated)

These signatures are generated from the installed `fast_agent` package to prevent drift.

### `chain`

```python
fast.chain(
    name: str,
    *,
    sequence: list[str],
    instruction: str | pathlib.Path | pydantic.networks.AnyUrl | None = None,
    cumulative: bool = False,
    default: bool = False
) -> Callable[
    [Callable[~P, Coroutine[Any, Any, +R]]],
    Callable[~P, Coroutine[Any, Any, +R]],
]
```
### `parallel`

```python
fast.parallel(
    name: str,
    *,
    fan_out: list[str],
    fan_in: str | None = None,
    instruction: str | pathlib.Path | pydantic.networks.AnyUrl | None = None,
    include_request: bool = True,
    default: bool = False
) -> Callable[
    [Callable[~P, Coroutine[Any, Any, +R]]],
    Callable[~P, Coroutine[Any, Any, +R]],
]
```
### `evaluator_optimizer`

```python
fast.evaluator_optimizer(
    name: str,
    *,
    generator: str,
    evaluator: str,
    instruction: str | pathlib.Path | pydantic.networks.AnyUrl | None = None,
    min_rating: str = 'GOOD',
    max_refinements: int = 3,
    refinement_instruction: str | None = None,
    default: bool = False
) -> Callable[
    [Callable[~P, Coroutine[Any, Any, +R]]],
    Callable[~P, Coroutine[Any, Any, +R]],
]
```
### `router`

```python
fast.router(
    name: str,
    *,
    agents: list[str],
    instruction: str | pathlib.Path | pydantic.networks.AnyUrl | None = None,
    servers: list[str] | None = None,
    tools: dict[str, list[str]] | None = None,
    resources: dict[str, list[str]] | None = None,
    prompts: dict[str, list[str]] | None = None,
    model: str | None = None,
    use_history: bool = False,
    request_params: fast_agent.llm.request_params.RequestParams | None = None,
    human_input: bool = False,
    default: bool = False,
    elicitation_handler: mcp.client.session.ElicitationFnT | None = None,
    api_key: str | None = None
) -> Callable[
    [Callable[~P, Coroutine[Any, Any, +R]]],
    Callable[~P, Coroutine[Any, Any, +R]],
]
```
### `orchestrator`

```python
fast.orchestrator(
    name: str,
    *,
    agents: list[str],
    instruction: str | pathlib.Path | pydantic.networks.AnyUrl = '...',
    model: str | None = None,
    request_params: fast_agent.llm.request_params.RequestParams | None = None,
    use_history: bool = False,
    human_input: bool = False,
    plan_type: Literal['full', 'iterative'] = 'full',
    plan_iterations: int = 5,
    default: bool = False,
    api_key: str | None = None
) -> Callable[
    [Callable[~P, Coroutine[Any, Any, +R]]],
    Callable[~P, Coroutine[Any, Any, +R]],
]
```
### `iterative_planner`

```python
fast.iterative_planner(
    name: str,
    *,
    agents: list[str],
    instruction: str | pathlib.Path | pydantic.networks.AnyUrl = '...',
    model: str | None = None,
    request_params: fast_agent.llm.request_params.RequestParams | None = None,
    plan_iterations: int = -1,
    default: bool = False,
    api_key: str | None = None
) -> Callable[
    [Callable[~P, Coroutine[Any, Any, +R]]],
    Callable[~P, Coroutine[Any, Any, +R]],
]
```
### `maker`

```python
fast.maker(
    name: str,
    *,
    worker: str,
    k: int = 3,
    max_samples: int = 50,
    match_strategy: str = 'exact',
    red_flag_max_length: int | None = None,
    instruction: str | pathlib.Path | pydantic.networks.AnyUrl | None = None,
    default: bool = False
) -> Callable[
    [Callable[~P, Coroutine[Any, Any, +R]]],
    Callable[~P, Coroutine[Any, Any, +R]],
]
```
