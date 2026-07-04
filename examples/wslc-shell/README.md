# WSLC shell environment example

This example runs fast-agent's model-facing `execute` and filesystem helper
tools through `wslc`, the Docker-compatible Windows Subsystem for Linux
container CLI available in recent Windows builds.

It mirrors the Docker shell example, but passes `container_cli="wslc"` to
`DockerMountedEnvironment`. The host workspace is mounted into the
container at `/workspace`, so files written by the agent are visible on the
Windows host.

## Run

From the repository root:

```bash
uv run python examples/wslc-shell/interactive.py
```

By default the example creates a temporary host directory and mounts it
read-write at `/workspace`.

To demo against a real directory:

```bash
uv run python examples/wslc-shell/interactive.py --workspace .
```

Use a different image or shell if needed:

```bash
uv run python examples/wslc-shell/interactive.py \
  --image python:3.12 \
  --shell bash \
  --workspace examples/wslc-shell/workspace
```

The script writes `README.txt` into the mounted host workspace before starting
the agent. Ask the agent to run commands such as:

```text
List the files in /workspace, append a line to README.txt, then show the file.
```

When the model calls `execute`, commands run via:

```text
wslc exec -w /workspace <container> bash -lc "<command>"
```

The container is removed when the interactive session exits.
