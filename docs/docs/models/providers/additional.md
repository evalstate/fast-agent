---
title: Additional Providers
social:
  title: Additional Providers
  tagline: Configure Groq, Aliyun, OpenRouter, OrcaRouter, TensorZero, and generic endpoints.
  description: Configure Groq, Aliyun, OpenRouter, OrcaRouter, TensorZero, and generic endpoints.
  alt: fast-agent social card — Additional model providers
---

# Additional Providers

These providers are useful when you want a specific hosted model, a router, or
an OpenAI-compatible endpoint without needing a dedicated first-class provider
guide. For Grok models, see the dedicated [xAI / Grok](xai.md) guide. For native
GLM models, see [Z.ai](zai.md). For Hugging Face Inference Providers, see the
dedicated [Hugging Face](huggingface.md) guide. For the native DeepSeek
Responses route, see [DeepSeek](deepseek.md).

Most entries use the same small configuration shape:

```yaml
<provider>:
  api_key: "${PROVIDER_API_KEY}"
  # base_url: "https://api.example.com/v1" # optional override
  # default_model: "model-name"             # optional
  # default_headers:                        # optional
  #   X-Custom-Header: "value"
```

Run `fast-agent check` after adding credentials to confirm they are visible to fast-agent.

## Quick reference

| Provider | Config key | API key environment variable | Default endpoint | Model string examples |
| --- | --- | --- | --- | --- |
| Groq | `groq` | `GROQ_API_KEY` | `https://api.groq.com/openai/v1` | `groq.openai/gpt-oss-120b` |
| Aliyun | `aliyun` | `ALIYUN_API_KEY` | `https://dashscope-intl.aliyuncs.com/compatible-mode/v1` | `qwen-turbo`, `aliyun.qwen3-max` |
| OpenRouter | `openrouter` | `OPENROUTER_API_KEY` | `https://openrouter.ai/api/v1` | `openrouter.google/gemini-2.5-pro-exp-03-25:free` |
| OrcaRouter | `orcarouter` | `ORCAROUTER_API_KEY` | `https://api.orcarouter.ai/v1` | `orcarouter.openai/gpt-4o-mini` |
| Open Responses | `openresponses` | `OPENRESPONSES_API_KEY` | Your Open Responses endpoint | `openresponses.openai/gpt-oss-120b:groq` |
| Generic OpenAI-compatible | `generic` | `GENERIC_API_KEY` | `http://localhost:11434/v1` for Ollama-style local use | `generic.llama3.2:latest` |
| TensorZero | `tensorzero` | None; configure provider credentials in the TensorZero Gateway | `http://localhost:3000` | `tensorzero.test_chat` |

!!! note "Capabilities vary by provider and model"

    Structured outputs, tool calling, reasoning controls, multimodal input, and provider-managed web tools are all model-dependent. Use the [Models Reference](../models_reference/) for fast-agent's known capability metadata.

## OpenAI-compatible hosted providers

Use these when the provider exposes an OpenAI-compatible API but has its own credentials, model catalog, or small behavior differences.

### Groq

```yaml
groq:
  api_key: "${GROQ_API_KEY}"
```

```bash
fast-agent --model groq.openai/gpt-oss-120b
```

Groq is optimized for fast hosted inference. It uses OpenAI-compatible request handling in fast-agent.
The shortcut `gpt-oss` currently resolves through the Hugging Face provider; use the explicit
`groq.` prefix when you want Groq.

--8<-- "_generated/model_aliases_groq.md"

### Aliyun

```yaml
aliyun:
  api_key: "${ALIYUN_API_KEY}"
```

```bash
fast-agent --model qwen-turbo
fast-agent --model aliyun.qwen3-max
```

Aliyun uses the DashScope compatible-mode endpoint by default. Override `base_url` only when you need a different Aliyun region, gateway, or compatible endpoint.

--8<-- "_generated/model_aliases_aliyun.md"

### OpenRouter

```yaml
openrouter:
  api_key: "${OPENROUTER_API_KEY}"
```

```bash
fast-agent --model openrouter.google/gemini-2.5-pro-exp-03-25:free
```

OpenRouter routes requests to many upstream providers. Model names and capabilities are controlled by OpenRouter and the selected upstream model.

### OrcaRouter

```yaml
orcarouter:
  api_key: "${ORCAROUTER_API_KEY}"
```

```bash
fast-agent --model orcarouter.openai/gpt-4o-mini
```

[OrcaRouter](https://www.orcarouter.ai) is an OpenAI-compatible model routing gateway. Model names are namespaced by upstream provider, for example `orcarouter.anthropic/claude-sonnet-4-6` or `orcarouter.google/gemini-2.5-flash`. The `orcarouter.auto` router picks the best upstream per request.

!!! note "Tool calling and the auto router"

    The `orcarouter.auto` router may select an upstream that does not support tool calling. When fast-agent runs with tools (the default), prefer an explicit namespaced model such as `orcarouter.openai/gpt-4o-mini`.

## Open Responses endpoints

Open Responses is an open standard for interoperable LLM interfaces. Use the `openresponses` provider for compatible endpoints:

```yaml
openresponses:
  api_key: "${OPENRESPONSES_API_KEY}"
  base_url: "https://api.example.com"
  reasoning: "medium" # minimal, low, medium, high
```

```bash
fast-agent --model openresponses.openai/gpt-oss-120b:groq
```

For a one-off endpoint test, keep the `openresponses` route explicit and
override only its destination:

```bash
OPENRESPONSES_API_KEY=... fast-agent go \
  --model openresponses.deepseek-v4-flash \
  --base-url https://responses.example/v1
```

Provider-managed MCP is not supported by `openresponses`. Use the OpenAI `responses` provider when you need `management: provider`.

## TensorZero

[TensorZero](https://tensorzero.com/) is an open-source framework for production LLM applications. It combines an LLM gateway, observability, optimization, evaluations, and experimentation.

Use TensorZero when you want fast-agent to call task-specific TensorZero functions while the gateway owns model selection, fallbacks, retries, prompt templates, observability, and provider credentials.

The fastest way to start is the bundled quickstart:

```bash
fast-agent quickstart tensorzero
```

That creates a dockerized example with a TensorZero Gateway, a custom MCP server, MiniIO-backed multimodal support, and a ready-to-run fast-agent example.

Configure the gateway endpoint if you are not using the default `http://localhost:3000`:

```yaml
tensorzero:
  base_url: "http://localhost:3000"
```

Call a TensorZero function with the `tensorzero.` model prefix:

```bash
uv run agent.py --model=tensorzero.test_chat
```

Provider credentials should normally be configured in the TensorZero Gateway, not in fast-agent.

## Generic OpenAI-compatible endpoints

Use `generic` for local or self-hosted OpenAI-compatible APIs, including Ollama-style endpoints.

```yaml
generic:
  base_url: "http://localhost:11434/v1"
  api_key: "ollama"
```

```bash
fast-agent --model generic.llama3.2:latest
```

Use `--base-url` for a one-off Chat Completions endpoint without changing the
configured generic provider:

```bash
GENERIC_API_KEY=... fast-agent go \
  --model generic.deepseek-v4-flash \
  --base-url https://gateway.example/v1
```

`generic` continues to use Chat Completions; `--base-url` changes the host, not
the protocol.

For reusable local names, defaults, metadata, and authentication behavior, prefer [Model Overlays](../model_overlays/).
