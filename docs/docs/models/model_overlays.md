---
title: Model Overlays
description: Working with local model overlays in fast-agent
---

# Model Overlays

**Model overlays** let you create local, named model entries that behave like built-in model presets, but carry extra runtime configuration with them.

They are useful when you want a short token such as `qwen-local` or `sonnet-lab` to stand for:

- a specific provider and model
- a custom `base_url`
- authentication rules for that endpoint
- default request settings such as `temperature`, `top_p`, or `max_tokens`
- local metadata used by the model picker and status displays

In practice, overlays are the easiest way to work with:

- local OpenAI-compatible servers
- self-hosted `llama.cpp` endpoints
- alternate gateways for the same wire model
- multiple differently-tuned variants of the same underlying model

## What an overlay does

When you select an overlay token, fast-agent resolves it before model creation.

That means the overlay can supply:

- the **provider** used to dispatch requests
- the **wire model name** sent to the backend
- **connection settings** such as `base_url` and authentication
- **request defaults** that are applied unless you override them explicitly
- **metadata** such as context window and output token limits

For example, this overlay:

```yaml
name: qwen-local
provider: openresponses
model: unsloth/Qwen3.5-9B-GGUF
connection:
  base_url: http://localhost:8080/v1
  auth: none
defaults:
  temperature: 0.8
  top_p: 0.95
  max_tokens: 2048
metadata:
  context_window: 75264
  max_output_tokens: 2048
picker:
  label: Qwen local
  description: Local llama.cpp import
  current: true
```

lets you run:

```bash
fast-agent go --model qwen-local
```

even though the underlying model string sent at runtime is:

```text
openresponses.unsloth/Qwen3.5-9B-GGUF?temperature=0.8&top_p=0.95
```

## Where overlays live

Model overlays are stored in the active environment directory:

- `ENV_DIR/model-overlays/*.yaml` — overlay manifests
- `ENV_DIR/model-overlays.secrets.yaml` — optional companion secrets

With the default environment directory, that usually means:

```text
.fast-agent/model-overlays/
.fast-agent/model-overlays.secrets.yaml
```

If you run with `--env <path>` or configure `environment_dir`, overlays are loaded from that environment instead.

## Overlay manifest format

An overlay manifest is a YAML document with these top-level sections:

```yaml
name: qwen-local
provider: openresponses
model: unsloth/Qwen3.5-9B-GGUF

connection:
  base_url: http://localhost:8080/v1
  auth: none
  # api_key_env: LLAMA_CPP_TOKEN
  # secret_ref: llama-lab
  # default_headers:
  #   X-My-Header: value

defaults:
  reasoning: off
  temperature: 0.8
  top_p: 0.95
  top_k: 40
  min_p: 0.05
  max_tokens: 2048
  transport: sse
  service_tier: fast
  web_search: false
  web_fetch: false

metadata:
  context_window: 75264
  max_output_tokens: 2048
  tokenizes:
    - text/plain
    - image/jpeg
    - image/png
  default_temperature: 0.8
  fast: true

picker:
  label: Qwen local
  description: Imported from llama.cpp
  current: true
  featured: false
```

### Required fields

- `name`: the token you use at the CLI or in config, for example `qwen-local`
- `provider`: the fast-agent provider to use
- `model`: the backend model name sent on the wire

### Connection settings

Use `connection` when the overlay needs endpoint-specific transport details:

- `base_url`: custom API base URL
- `auth`: one of `none`, `env`, or `secret_ref`
- `api_key_env`: environment variable name to read when `auth: env`
- `secret_ref`: companion secret entry name when `auth: secret_ref`
- `default_headers`: optional headers to send on each request

### Request defaults

Use `defaults` for model-string-style runtime defaults that should travel with the overlay.

These values behave like query parameters on the resolved model string and are applied unless an explicit run overrides them.

Common examples:

- `temperature`
- `top_p`
- `top_k`
- `min_p`
- `max_tokens`
- `reasoning`
- `transport`
- `service_tier`
- `web_search`
- `web_fetch`

### Metadata

`metadata` is used by fast-agent for local model understanding and UI display.

This is especially helpful for self-hosted models that are not part of the built-in catalog.

Common fields:

- `context_window`
- `max_output_tokens`
- `tokenizes`
- `fast`

## Authentication options

There are three supported auth modes.

### No auth

```yaml
connection:
  base_url: http://localhost:8080/v1
  auth: none
```

Use this for local servers with no API key requirement.

### Environment variable auth

```yaml
connection:
  base_url: https://gateway.example/v1
  auth: env
  api_key_env: LAB_MODEL_TOKEN
```

fast-agent reads the API key from `LAB_MODEL_TOKEN` at runtime.

### Secret reference auth

Overlay manifest:

```yaml
connection:
  base_url: https://gateway.example/v1
  auth: secret_ref
  secret_ref: lab-qwen
```

Companion secrets file:

```yaml
lab-qwen:
  api_key: your-secret-token
```

You can also store default headers in the companion secret entry if needed.

## Using overlays

Once an overlay exists, you can use it anywhere you would normally supply a model string:

- `fast-agent go --model qwen-local`
- `default_model: "qwen-local"`
- agent card `model: qwen-local`
- model references such as `$system.local`

Example:

```yaml
default_model: "$system.fast"

model_references:
  system:
    fast: "qwen-local"
    plan: "claude-sonnet-4-5"
```

## Overlays and precedence

Overlay names behave like local runtime presets.

If an overlay name collides with a built-in preset or another preset token, the overlay wins for that environment. `fast-agent check` reports this as informational output so the override is visible.

Overlays are environment-local, so different environments can define different overlay sets without changing project config.

## Creating overlays from llama.cpp

The easiest way to create a local overlay is the `fast-agent model llamacpp` command.

It queries a llama.cpp-compatible server, discovers models from the models endpoint, reads runtime defaults from the props endpoint, and writes an overlay into the active environment. The generated overlay uses the `openresponses` provider, the normalized `/v1` base URL, the selected auth mode, and the discovered request defaults and metadata.

### Discover available models

```bash
fast-agent model llamacpp list --url http://localhost:8080 --json
```

This queries the server's model listing and prints the discovered catalog.

### Import a model as an overlay

```bash
fast-agent model llamacpp import \
  --url http://localhost:8080 \
  unsloth/Qwen3.5-9B-GGUF \
  --name qwen-local
```

fast-agent will:

1. discover models from the server
1. interrogate the selected model for runtime defaults
1. generate an overlay manifest
1. write it to `model-overlays/<name>.yaml`

### Dry-run and print the generated YAML

```bash
fast-agent model llamacpp preview \
  --url http://localhost:8080/v1 \
  meta-llama/Llama-3.2-3B-Instruct \
  --name llama-local
```

### Import with environment-based auth

```bash
fast-agent model llamacpp import \
  --url https://lab.example \
  unsloth/Qwen3.5-9B-GGUF \
  --name qwen-lab \
  --auth env \
  --api-key-env LLAMA_CPP_TOKEN
```

## Model setup and doctor flows

fast-agent also includes helper flows for model references:

```bash
fast-agent model setup
fast-agent model doctor
```

- `model setup` helps create or update namespaced model references such as `$system.fast`
- `model doctor` inspects model onboarding readiness and reference resolution

These commands work well with overlays, because a reference can point to either a built-in model/preset or a local overlay token.

## Example: local overlay + model reference

```yaml
default_model: "$system.fast"

model_references:
  system:
    fast: "qwen-local"
```

Then run:

```bash
fast-agent go
```

This gives you a stable project-facing token (`$system.fast`) while keeping the actual endpoint wiring in the environment-local overlay.

## Troubleshooting

### The overlay is not found

Check that:

- the overlay file is in the active environment directory
- the file has a `.yaml` or `.yml` extension
- the overlay `name` matches the token you are using exactly

### The overlay requires an API key

If `auth: env`, make sure the configured environment variable is set.

If `auth: secret_ref`, make sure the referenced entry exists in `model-overlays.secrets.yaml` and includes `api_key`.

### The model picker does not show my overlay

Make sure the overlay file loads cleanly and includes valid YAML. Invalid overlay manifests are skipped with a warning.

### I want different endpoints for the same model

That is a good fit for overlays. You can create multiple overlays that point at the same wire model but use different `base_url`, auth, and defaults.

For example:

- `qwen-local`
- `qwen-remote`
- `qwen-fast`

Each can resolve to the same backend model name while carrying distinct runtime settings.
