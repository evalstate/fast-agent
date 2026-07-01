<!--
  GENERATED FILE — DO NOT EDIT.
  Source: generate_reference_docs.py
-->

## AgentHarness Class

`AgentHarness` is an async context manager returned by `FastAgent.harness()`.

```python
from fast_agent import AgentHarness


async with fast.harness() as harness:
    typed: AgentHarness = harness
```
### Properties

| Property | Type | Description |
|----------|------|-------------|
| `sessions` | `HarnessSessions` | Session manager for the running harness |
| `environment` | `ShellEnvironment` | Shell environment used by the running harness; may also implement `SessionFilesystem` for model-facing file tools |

### Methods

#### `session()`

```python
await harness.session(session_id: str | None = None, *, agent_name: str | None = None) -> HarnessSession
```
Returns an existing session or creates one by delegating to
`harness.sessions.get_or_create()`.

#### `shell()`

```python
await harness.shell(
    command: str,
    *,
    cwd: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    timeout: float | None = None
) -> ShellExecutionResult
```
Runs a shell command through the harness shell environment and returns a
`ShellExecutionResult` with `stdout`, `stderr`, and `exit_code`. This is
programmatic shell access: it does not create a session and does not add
the command or output to chat history.

## HarnessSessions Class

Manager for harness sessions. When `session_history` is enabled, creating a
session also creates or loads `environment_dir/sessions/<session_id>/`.

```python
session = await harness.sessions.get("demo")
session = await harness.sessions.create("demo")
session = await harness.sessions.get_or_create("demo")
await harness.sessions.delete("demo")
```
Session ID rules:

- `None` means `"default"`;
- strings are stripped;
- empty strings raise `ValueError`;
- IDs must be 1-128 characters;
- IDs must start and end with a letter or digit;
- IDs may contain only letters, digits, dashes, and underscores.

Generated from `HARNESS_SESSION_ID_PATTERN`: `^[A-Za-z0-9](?:[A-Za-z0-9_-]{0,126}[A-Za-z0-9])?$`.

| Method | Signature | Behavior |
|--------|-----------|----------|
| `get` | `get(session_id: str \| None = None) -> HarnessSession` | Return an existing session. Raises if missing. |
| `create` | `create(session_id: str \| None = None, *, agent_name: str \| None = None) -> HarnessSession` | Create a session. Raises if it already exists. |
| `get_or_create` | `get_or_create(session_id: str \| None = None, *, agent_name: str \| None = None) -> HarnessSession` | Return an existing session or create it. |
| `delete` | `delete(session_id: str \| None = None) -> None` | Delete a session if present; missing sessions are ignored. |

## HarnessSession Class

A stable session backed by one owned `AgentInstance`.

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `id` | `str` | Normalized session ID |
| `default_agent_name` | `str \| None` | Session default agent used when calls omit `agent_name` |

### Methods

```python
await session.send(
    message: MessageParam,
    *,
    agent_name: str | None = None,
    request_params: RequestParams | None = None
) -> str

await session.generate(
    messages: MessageParam,
    *,
    agent_name: str | None = None,
    request_params: RequestParams | None = None
) -> PromptMessageExtended

await session.structured(
    messages: MessageParam,
    model: type[ModelT],
    *,
    agent_name: str | None = None,
    request_params: RequestParams | None = None
) -> tuple[ModelT | None, PromptMessageExtended]

await session.structured_schema(
    messages: MessageParam,
    schema: dict[str, Any],
    *,
    agent_name: str | None = None,
    request_params: RequestParams | None = None
) -> tuple[Any | None, PromptMessageExtended]

await session.clear(*, agent_name: str | None = None, clear_prompts: bool = False) -> None

await session.shell(
    command: str,
    *,
    cwd: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    timeout: float | None = None
) -> ShellExecutionResult

await session.delete() -> None
```
Calls resolve their target agent in this order:

1. explicit per-call `agent_name`;
2. the session `default_agent_name`;
3. the app default agent.

`clear()` clears only the resolved target agent, not every agent in the session.

`shell()` returns `ShellExecutionResult` with `stdout`, `stderr`, and
`exit_code`. It does not add the command or output to chat history, but it
is serialized with other operations on the same `HarnessSession`.

### Session lifecycle and concurrency

- the same session ID returns the same `HarnessSession` object and the same owned
  `AgentInstance`;
- different session IDs receive isolated `AgentInstance` objects;
- when `session_history` is enabled, session IDs map to persisted
  `environment_dir/sessions/<session_id>/` directories and existing histories
  are hydrated on creation;
- deleting a session disposes its instance;
- deleting a session removes its persisted session folder when persistence is enabled;
- exiting the harness context disposes all remaining session instances;
- deleted `HarnessSession` objects are closed.

Concurrent operations on the same `HarnessSession` are rejected:

```text
RuntimeError: Session 'support-123' is already running generate; start another session for parallel conversation branches.
```
Deleting an active session is also rejected:

```text
RuntimeError: Session 'support-123' is running generate; wait before deleting it.
```
