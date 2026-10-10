---
title: Additional Providers
social:
  title: Additional Providers
  tagline: Configure hosted providers, routers, and generic OpenAI-compatible endpoints.
  description: Configure Groq, Aliyun, OpenRouter, Neuralwatt, DeepInfra, SiliconFlow, OrcaRouter, TensorZero, and generic endpoints.
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
| Open Responses | `openresponses` | `OPENRESPONSES_API_KEY` | Your Open Responses endpoint | `openresponses.openai/gpt-oss-120b:groq` |
| Generic OpenAI-compatible | `generic` | `GENERIC_API_KEY` or an explicitly configured key variable | Set `base_url` for hosted services; defaults to `http://localhost:11434/v1` for local use | `generic.Qwen/Qwen3-32B`, `generic.llama3.2:latest` |
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
  --model openresponses.deepseek-flash \
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

Use `generic` for **hosted, local, or self-hosted OpenAI-compatible Chat
Completions APIs**. You do not need a dedicated fast-agent provider or a code
change to connect another compatible service. Configure its API base URL and
credentials, then select `generic.<model-id>`.

### Hosted provider setup

The following services use the same `generic` configuration. The environment
variable names below are suggestions: bind the chosen variable explicitly in
`generic.api_key`, as in the example that follows.

| Service | API base URL | Suggested key variable | Official setup and model discovery |
| --- | --- | --- | --- |
| Neuralwatt | `https://api.neuralwatt.com/v1` | `NEURALWATT_API_KEY` | [Quickstart](https://docs.neuralwatt.com/quickstart), [models](https://docs.neuralwatt.com/api/models) |
| DeepInfra | `https://api.deepinfra.com/v1/openai` | `DEEPINFRA_API_KEY` | [Quickstart](https://docs.deepinfra.com/quickstart), [models](https://docs.deepinfra.com/models) |
| SiliconFlow | `https://api.siliconflow.com/v1` | `SILICONFLOW_API_KEY` | [Quickstart](https://docs.siliconflow.com/en/userguide/quickstart), [models](https://docs.siliconflow.com/en/api-reference/models/get-model-list) |
| OrcaRouter | `https://api.orcarouter.ai/v1` | `ORCAROUTER_API_KEY` | [OpenAI SDK setup](https://docs.orcarouter.ai/compatibility/openai-sdk), [models](https://docs.orcarouter.ai/getting-started/models) |

Use the full base URL shown above, without appending `/chat/completions`.
DeepInfra requires the `/v1/openai` suffix. For a SiliconFlow China account, use
the [China endpoint](https://docs.siliconflow.cn/docs/userguide/quickstart),
`https://api.siliconflow.cn/v1`, with the corresponding account's key and model
catalog.

For example, to use SiliconFlow, export your key in the shell where you run
fast-agent:

```bash
export SILICONFLOW_API_KEY="your-api-key"
```

Create or update `fast-agent.yaml` in your working directory:

```yaml
default_model: "generic.Qwen/Qwen3-32B"

generic:
  base_url: "https://api.siliconflow.com/v1"
  api_key: "${SILICONFLOW_API_KEY}"
```

Then check the configuration and send a short message:

```bash
fast-agent check
fast-agent go --message "Reply with hello."
```

`fast-agent check` inspects configuration and credential visibility; the message
command verifies an actual API request to the service.

To use another service, replace the base URL, key variable, and model ID with
values from its official documentation. Copy the model ID exactly, including
case, organization prefixes, slashes, dots, or version suffixes. Only prepend
`generic.` for fast-agent; that prefix is not sent to the service. A model does
not have to appear in fast-agent's built-in catalog. Provider model availability
can change, so check the provider's current model list if an example is unavailable.

These services do not have built-in `deepinfra.`, `siliconflow.`, `neuralwatt.`,
or `orcarouter.` prefixes or config sections. Keep the `generic` route even when
the hosted model is from OpenAI, Anthropic, or DeepSeek. The route selects the
API protocol, not the model's author.

### One-off endpoint override

For a single run, override the endpoint with `--base-url`:

```bash
GENERIC_API_KEY="your-api-key" fast-agent go \
  --model generic.Qwen/Qwen3-32B \
  --base-url https://api.siliconflow.com/v1 \
  --message "Reply with hello."
```

This environment-only example assumes `generic.api_key` is not already set in
your configuration. A configured key takes precedence over `GENERIC_API_KEY`.
`--base-url` changes the destination for this run; it does not select a different
API key or change Chat Completions into the Responses API.

### Multiple hosted providers

The `generic` config block supplies one shared endpoint and key. Use
[Model Overlays](../model_overlays.md) to give each model its own endpoint and
credentials when working with multiple services.

For example, save this as `.fast-agent/model-overlays/siliconflow-qwen.yaml`
(or under `model-overlays/` in your configured fast-agent home):

```yaml
name: siliconflow-qwen
provider: generic
model: Qwen/Qwen3-32B
connection:
  base_url: https://api.siliconflow.com/v1
  auth: env
  api_key_env: SILICONFLOW_API_KEY
```

With `SILICONFLOW_API_KEY` exported, run:

```bash
fast-agent go --model siliconflow-qwen
```

Create a separate overlay for each additional service, changing `name`, `model`,
`base_url`, and `api_key_env`. In an overlay, `model` is the provider's exact model
ID **without** the `generic.` prefix; `provider: generic` selects the route.
Overlay authentication and endpoints apply to that model without replacing the
shared `generic` settings.

### Troubleshooting hosted endpoints

| Symptom | What to check |
| --- | --- |
| Authentication error (`401` or `403`) | Export the configured key variable in the current shell and use the endpoint for that account/region. Check for an older `generic.api_key` overriding `GENERIC_API_KEY`. |
| Endpoint or model not found (`404`) | Use the full API base URL, without `/chat/completions`, and copy the exact model ID from the provider's catalog. |
| Unknown fast-agent provider | Use `generic.<model-id>` or an overlay name, rather than inventing a provider prefix. |
| Unsupported parameter, tool, or output format | Check the selected provider and model's capabilities. OpenAI compatibility does not imply support for every OpenAI feature. |

Start with a plain text request before adding MCP tools, structured outputs,
reasoning controls, or multimodal input. Those capabilities depend on both the
service and the selected model; an overlay does not add support to the backend.

### Local endpoints

For Ollama-style local endpoints:

```yaml
generic:
  base_url: "http://localhost:11434/v1"
  api_key: "ollama"
```

```bash
fast-agent --model generic.llama3.2:latest
```

For reusable local names, defaults, metadata, and authentication behavior, prefer [Model Overlays](../model_overlays.md).
