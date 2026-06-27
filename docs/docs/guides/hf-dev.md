---
title: Code with Hugging Face
description: An introduction to using `fast-agent`
social:
  title: Code with Hugging Face
  tagline: Build coding workflows with fast-agent and Hugging Face developer tools.
  description: Build coding workflows with fast-agent and Hugging Face developer tools.
  alt: fast-agent social card — Code with Hugging Face
---


# Code with HF Inference Providers

Use the latest open weight models via Hugging Face Inference Providers:

```bash
uvx fast-agent-mcp@latest --pack hf-dev
```

This starts **fast-agent** pre-configured with a coding agent and filesystem search sub-agent.

By default you will be prompted for the coding model to use, with `gpt-oss-120b` used for search.

To change these defaults use:

```bash
uvx fast-agent-mcp@latest model setup
```

The agent has a minimal system prompt and tools for accessing the shell, filesystem, and **fast-agent** services. The system prompt includes `AGENTS.md` if present. Customize the agent by modifying `.fast-agent/agent-cards/dev.md`.

Use `/skills` to discover, add, remove, and update skills. Use `/connect` to connect to MCP servers.

Use the `compaction-strategies` skill to set up your preferred compaction scheme, if any.

## Installation

**fast-agent** requires Python 3.12 or newer. Install with:

```bash
uv tool install -U fast-agent-mcp
```

Or a specific version of Python:

```bash
uv tool install --python 3.12 -U fast-agent-mcp
```

This installs the `fast-agent` executable.

## llama.cpp

`fast-agent` has support for [llama.cpp](https://llama.app). Read the 
[llama.cpp provider guide](../models/providers/llamacpp.md) to get
started.

## Export agent traces to Hugging Face URLs

Persisted sessions can be exported as Codex-style JSONL traces and uploaded directly
to Hugging Face Hub storage.

From inside `fast-agent`:

```text
/session export latest --hf-url hf://buckets/your-name/fast-agent-traces/
/session export latest --hf-url hf://datasets/your-name/fast-agent-traces/trace.jsonl
```

From the shell:

```bash
fast-agent export latest --hf-url hf://buckets/your-name/fast-agent-traces/
fast-agent export latest --output trace.jsonl --hf-url hf://datasets/your-name/fast-agent-traces/trace.jsonl
```

Notes:

- Traces come from persisted sessions, so `--noenv` runs are not exportable.
- Set `git_aware: true` in `fast-agent.yaml` to include git provenance in
  exported trace metadata when the session runs inside a git repository.
- If you omit `--output`, fast-agent writes a default file named
  `{session_id}__{agent_name}__codex.jsonl` in the current directory before upload.
- `--hf-url` supports `hf://buckets/...` and `hf://datasets/...`. If it ends with
  `/`, fast-agent appends the local filename.
- `--hf-dataset` remains available for compatibility and constructs an equivalent
  `hf://datasets/...` upload URL.
- Uploads require `huggingface_hub`. Dataset repos are created automatically if
  they do not already exist.

For the full CLI reference, see [fast-agent export](../ref/export_command/).
