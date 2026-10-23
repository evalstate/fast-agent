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

## Evals

`evals.py` runs a handful of quick end-to-end checks through the real CLI. Each one checks the
answer and *how* the model got there (from the saved history):

| Case | Checks |
| --- | --- |
| `orders` | aggregation in one script (`execute_python` used) |
| `tool-image` | an MCP tool image reaches the model when the script returns its handle |
| `tool-image-select` | several images fetched; exactly one returned and shown |
| `resource-image` | `get_resource` on an image resource (`swatch://west`) |
| `attach-media` | `attach_media` from a script stages `assets/swatch-attach.png` |
| `subagents` | two subagents run concurrently from one script |
| `slash-direct` | `slash_command` stays a direct tool rather than a sandbox function |

The swatch colours exist only in the image pixels, so the media cases pass only if the image
actually reached the model.

```bash
uv run evals.py                 # all cases, default model from fast-agent.yaml
uv run evals.py --model sonnet  # another model
uv run evals.py tool-image      # one case
uv run evals.py --list          # print each case's command to run and watch it by hand
```

The cases use `agent-cards/ptc-eval.md`, which enables the `orders` and `media`
(`media_server.py`) servers, shell, subagents and harness tools.
