# Hugging Face Space MCP server from AgentCards

This example deploys a custom FastMCP server backed by fast-agent Harness API,
Hugging Face OAuth, and AgentCards.

It exposes two MCP tools:

- `research(topic, depth)` — structured tool backed by the `researcher` card.
- `chat(message)` — message-style smoke tool backed by the same card.

The Space requests the `inference-api` OAuth scope and uses the caller's token
for Hugging Face Inference Providers.

## Run locally

```bash
export FAST_AGENT_SERVE_OAUTH=huggingface
python app.py
```

For local non-OAuth smoke tests, set `HF_TOKEN` or run `hf auth login` and
remove/disable `FAST_AGENT_SERVE_OAUTH`.

## Deploy to Spaces

Use Docker SDK with the README metadata in this directory. For a released
package, the Dockerfile installs from PyPI. For unreleased local changes, copy a
wheel into the Space and adjust the install line to use that wheel.
