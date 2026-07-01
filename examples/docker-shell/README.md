# Docker shell environment example

This example runs `harness.shell(...)` inside a disposable Docker container
instead of the host shell.

It is intentionally a manual smoke test rather than part of the automated test
suite, so the project does not require Docker for normal development or CI.

## Run

From the repository root:

```bash
uv run python examples/docker-shell/docker_shell_harness.py
```

Use a different image if needed:

```bash
uv run python examples/docker-shell/docker_shell_harness.py --image ubuntu:24.04
```

The script:

1. creates a temporary host workspace,
2. mounts it into a disposable Docker container at `/workspace`,
3. runs shell commands through `fast.harness(environment=...)`, and
4. verifies a file written in the container appears on the host.

The container is removed when the harness exits.

## Interactive agent

To run the normal interactive fast-agent prompt with the model-facing `execute`
tool backed by Docker instead of your host shell:

```bash
uv run python examples/docker-shell/interactive.py
```

The interactive example creates a shell-enabled agent and starts:

```python
async with fast.run(environment=environment) as agent_app:
    await agent_app.interactive()
```

When the model calls `execute`, commands run in the Docker container mounted at
`/workspace`.

The interactive example uses `DockerMountedSessionEnvironment`, so model-facing
filesystem helper tools such as `read_text_file`, `write_text_file`, and
`apply_patch` target the same `/workspace` bind mount used by Docker shell
execution.

By default the example mounts a temporary host directory. To mount a specific
workspace:

```bash
uv run python examples/docker-shell/interactive.py --workspace .
```
