# Hugging Face sandbox environment example

This example runs the model-facing `execute` and filesystem helper tools inside
a Hugging Face Sandbox. It is a manual smoke test, not part of the automated
test suite, because it creates remote Hugging Face Jobs resources.

For the config-driven equivalent using `environment="hf-gpu"`, see
[`examples/environments/`](../environments/).

## Run

From the repository root:

```bash
uv run python examples/huggingface-sandbox/interactive.py
```

To mount a Hugging Face Storage Bucket read-write:

```bash
uv run python examples/huggingface-sandbox/interactive.py \
  --bucket username/my-bucket:/workspace:rw
```

To mount a bucket read-only:

```bash
uv run python examples/huggingface-sandbox/interactive.py \
  --bucket username/my-dataset:/data:ro
```

The sandbox is terminated when the interactive session exits.
