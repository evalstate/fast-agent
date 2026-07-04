<!--
  GENERATED FILE — DO NOT EDIT.
  Source: generate_reference_docs.py
-->

#### `harness()`

```python
fast.harness(
    *,
    model: str | None = None,
    environment: "'EnvironmentSelection'" = None
) -> AgentHarness
```
Creates a headless `AgentHarness` for typed, session-oriented Python usage.
The harness uses the same initialization path as `run()` but does not enter the
TUI, CLI message/prompt-file modes, MCP server mode, or ACP server mode.
On startup, it loads AgentCards from the active fast-agent home's `agent-cards/`
directory when that directory exists and contains cards.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `str \| None` | `None` | Optional global model override, similar to the CLI `--model` override |
| `environment` | `ShellEnvironment \| None` | `None` | Optional shell environment override for `harness.shell(...)` and `session.shell(...)`; environments that also implement `EnvironmentFilesystem` back model-facing file tools |
