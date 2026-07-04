# Named execution environments

This example shows the config-driven environment flow:

1. define reusable environments in `.fast-agent/fast-agent.yaml`;
2. select one by name with `environment="..."` or `--environment`;
3. use `harness.local` and transfer helpers when you need to move files between
   the host workspace and the active environment.

Run the local examples from this directory:

```bash
uv run python named_harness.py
uv run python artifacts_roundtrip.py
```

Run the interactive prompt in a named environment:

```bash
uv run python interactive.py --environment local
```

If Docker is available, try:

```bash
uv run python interactive.py --environment ubuntu
```

If `wslc` is available instead of Docker, use the configured `wslc` environment:

```bash
fast-agent go -E wslc
uv run python interactive.py --environment wslc
```

`container_cli` must name an executable that can be found on `PATH`; shell
aliases and functions are not visible to fast-agent's subprocess launcher. If
your `wslc` command is an alias, create a small wrapper script instead and point
the config at it:

```bash
mkdir -p ~/bin
cat > ~/bin/wslc <<'EOF'
#!/usr/bin/env bash
exec /path/to/real/wslc "$@"
EOF
chmod +x ~/bin/wslc
```

Then either put `~/bin` on `PATH` before starting fast-agent, or set
`container_cli: /home/you/bin/wslc` in `.fast-agent/fast-agent.yaml`.

`artifacts_roundtrip.py` copies `inputs/` into the active environment, runs a
shell command there, and copies generated files back to `outputs/`.

The `hf-gpu` entry is a template for Hugging Face Sandbox usage and requires the
provider credentials normally needed by that adapter.

The older `examples/docker-shell/` and `examples/huggingface-sandbox/`
directories construct environment instances directly in code. This directory is
the equivalent name-based setup.
