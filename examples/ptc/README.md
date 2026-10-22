# Programmatic tool calling (PTC) demo

`--ptc` replaces an agent's tools with a single `execute_python` tool. The model writes Python
that runs in a [pydantic-monty](https://github.com/pydantic/monty) sandbox. Each tool is an
async function in that sandbox, and only printed output and the final expression value come
back into context.

```bash
uv pip install 'fast-agent-mcp[ptc]'   # already included in the dev environment
cd examples/ptc

# Direct tool calling: every order list passes through the model's context
uv run fast-agent go --servers orders \
  -m "Which EMEA customer has the highest total revenue from delivered orders? Include the runner-up."

# Programmatic tool calling: one script that fans out concurrent calls and aggregates in code
uv run fast-agent go --ptc --servers orders \
  -m "Which EMEA customer has the highest total revenue from delivered orders? Include the runner-up."
```

PTC combines with the other runtimes. For example, `--ptc --shell` exposes the shell tool as
`await shell(...)`. Configure limits and directly exposed tools under `ptc:` in
`fast-agent.yaml`.

The expected answer is Stark (C006) with $9,703.40, and the runner-up is Aperture (C012) with $6,385.75.
