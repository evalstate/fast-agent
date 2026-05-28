---
title: Providers and Models
social:
  title: Providers and Models
  tagline: Pick providers, model aliases, and first-class features quickly.
  description: Pick providers, model aliases, and first-class features quickly.
  alt: fast-agent social card — Getting Started with Models
---

**`fast-agent`** has native support for **OpenAI Responses** and **Chat Completions**, **Anthropic Messages**, **Google GenAI** and **Amazon Bedrock** APIs. 

OpenAI Codex users can use their subscription with **`fast-agent`**, using their existing installation or logging in with `fast-agent auth codexplan`. 

Chat Completions models are also available via **Microsoft Azure**, and supported Anthropic models are available on **Google Vertex**.

Local models with [**llama.cpp**](https://llama.app/) are directly supported, with automatic configuration and connection with the Responses API.

## Selecting a Model

#### Model Picker and Defaults

In interactive mode, with no model specified or default configured, **`fast-agent`** shows a model selector on startup, highlighting available models.

<div
  class="fa-terminal-demo"
  data-fa-asciinema-cast="../../assets/models/model-picker.cast"
  data-fa-asciinema-cols="96"
  data-fa-asciinema-rows="21"
  data-fa-asciinema-poster="npt:0:03"
  data-fa-asciinema-speed="1"
  data-fa-asciinema-idle-time-limit="1.3"
  data-fa-asciinema-fit="width"
  data-fa-asciinema-autoplay="true"
>
  <div class="fa-terminal-theme-switch" aria-label="Terminal theme">
    <button type="button" data-fa-terminal-theme="auto">Auto</button>
    <button type="button" data-fa-terminal-theme="light">Light</button>
    <button type="button" data-fa-terminal-theme="dark">Dark</button>
  </div>
  <div data-fa-asciinema-target></div>
</div>

<!--
Cast asset:
- Source: docs/docs/assets/models/model-picker.cast
- Regenerate: uv run scripts/docs.py cast-build model-picker
- Replay locally: asciinema play docs/docs/assets/models/model-picker.cast
-->

### Using Presets

The quickest way to get started is to use the convenience presets for popular models, for example:

```bash
fast-agent --model opus      # Use the most recent opus model
fast-agent --model codexplan # Use the latest supported Codex Subscription Model
```

Use `fast-agent model presets` to see the current shortcuts.

### Model Strings and Configuration

Models in **fast-agent** are specified with a model string:

```text
provider.model_name[?reasoning=high&...]
```

The query string allows configuration of provider, model, and sampling parameters.

Custom models and configurations can be defined using [Model Overlays](model_overlays.md).

- [Providers and Models](llm_providers/) lists provider configuration and authentication details.
- [Models Reference](models_reference/) lists generated model capabilities such as structured
  outputs, reasoning, verbosity, and supported input modalities.

## Provider families

Start with the native providers for common use, or use additional providers for hosted OpenAI-compatible APIs, routers, and local endpoints.

| Provider family | Start with | Main features |
| --- | --- | --- |
| OpenAI Responses | `gpt55`, `gpt54`, `gpt52`, `gpt-5-mini`, `codex` | GPT-5 class models, reasoning, text verbosity, structured outputs, `web_search`, SSE/WebSocket transports, service tiers, connectors |
| Anthropic | `sonnet`, `opus`, `opus48`, `opus47`, `haiku` | Claude 4.x, prompt caching, adaptive reasoning/effort, structured outputs, `web_search`, `web_fetch`, long context, task budget where supported |
| Google | `gemini`, `gemini3`, `gemini3.1`, `gemini3flash` | Gemini native API, structured outputs, thinking controls, text/image/PDF/audio/video input, YouTube links through media attachments |
| xAI / Grok | `grok`, `grok4`, `grok-4.3` | Grok models, reasoning controls, `web_search`, `x_search`, SSE/WebSocket transports |
| Hugging Face | `kimi`, `kimi26instant`, `deepseek-hf`, `glm`, `minimax` | Hugging Face Inference Providers routing, curated aliases, and HF MCP authentication |
| Additional providers | `deepseek`, `qwen-turbo`, `gpt-oss` | Groq, DeepSeek, Aliyun, OpenRouter, Open Responses, TensorZero, and generic OpenAI-compatible endpoints |

### OpenAI Responses

Use the `responses` provider for GPT-5 class OpenAI models.

```bash
fast-agent --model "responses.gpt-5.5?reasoning=medium"
fast-agent --model "responses.gpt-5.5?web_search=on"
fast-agent --model "responses.gpt-5.5?verbosity=high&transport=ws"
fast-agent --model "responses.gpt-5.5?service_tier=fast"
```

Useful query parameters:

- `reasoning=none|minimal|low|medium|high|xhigh` depending on model
- `verbosity=low|medium|high`
- `web_search=on|off`
- `transport=sse|ws|auto`
- `service_tier=fast|flex` where supported

Use the `openai` provider for Chat Completions-style models such as `openai.gpt-4.1`.

### Anthropic

Anthropic support includes Claude-specific reasoning, caching, web tools, and structured-output
selection.

```bash
fast-agent --model sonnet
fast-agent --model "sonnet?reasoning=4096"
fast-agent --model "opus?reasoning=auto"
fast-agent --model "opus?reasoning=xhigh"
fast-agent --model "opus?web_search=on&web_fetch=on"
fast-agent --model "opus?task_budget=128k"
```

Useful query parameters and config:

- `reasoning=auto|low|medium|high|max|off` on adaptive-thinking models
- `reasoning=xhigh` on Opus models that advertise it, such as `opus`, `opus48`, and `opus47`
- `reasoning=<tokens>` on older budget-thinking models, for example `reasoning=4096`
- `web_search=on|off`
- `web_fetch=on|off`
- `task_budget=20k|128k|off` where supported
- `anthropic.cache_mode: auto|prompt|off`
- `anthropic.cache_ttl: 5m|1h`

`opus` currently resolves to `claude-opus-4-8`; use `opus47` or `opus46` when you need to pin an
older Opus generation. Claude Opus 4.7+ uses adaptive reasoning rather than fixed thinking budgets:
`reasoning=auto` lets the model choose, effort levels tune depth and token spend, and `task_budget`
sets a model-visible budget for a whole agentic loop. `task_budget` is separate from `max_tokens`,
which remains the enforced per-response ceiling.

Structured outputs default to JSON schema on models that support Anthropic's structured-output
feature. Older models fall back to the legacy `tool_use` flow.

### Google

Use the native Google provider for Gemini models.

```bash
fast-agent --model gemini
fast-agent --model "gemini3?reasoning=auto"
fast-agent --model "google.gemini-3.1-pro-preview?reasoning=high"
```

Google models support structured outputs and multimodal inputs. Current fast-agent model metadata
advertises text, image, PDF, audio, and video tokenization for Gemini models. YouTube links can be
attached as media links when using a model that supports video input.

Useful query parameters:

- `reasoning=auto|minimal|low|medium|high|off`
- `structured=json`
- sampling controls such as `temperature`, `top_p`, and `top_k` where applicable

### xAI

Use the [xAI provider](providers/xai/) for Grok models.

```bash
fast-agent --model grok
fast-agent --model "xai.grok-4.3?web_search=on"
fast-agent --model "xai.grok-4.3?x_search=on"
```

### Hugging Face

Use the [Hugging Face provider](providers/huggingface/) for Hugging Face Inference Providers routing and curated aliases.

```bash
fast-agent --model kimi
fast-agent --model kimi26instant
fast-agent --model "hf.moonshotai/Kimi-K2.6:novita?reasoning=on"
```

### Additional providers

Use [Additional Providers](providers/additional/) for hosted OpenAI-compatible APIs, routers, and local endpoints such as Groq, DeepSeek, Aliyun, OpenRouter, Open Responses, TensorZero, and generic endpoints.

```bash
fast-agent --model deepseek
fast-agent --model qwen-turbo
fast-agent --model groq.openai/gpt-oss-120b
fast-agent --model openrouter.google/gemini-2.5-pro-exp-03-25:free
fast-agent --model generic.llama3.2:latest
```

That page keeps the long-tail reference in one place, including config keys, API key environment variables, default endpoints, and provider-specific notes.

## Model string format

Model strings follow this format:

```text
provider.model_name[?reasoning=value][&query=value...]
```

- **provider**: the LLM provider, for example `responses`, `anthropic`, `google`, `xai`,
  `hf`, `azure`, `openrouter`, `generic`, or `tensorzero`
- **model_name**: the model or deployment name
- **query parameters**: provider/model-specific overrides such as `reasoning`, `structured`,
  `context`, `transport`, `service_tier`, `temperature` (`temp` alias), `web_search`,
  `web_fetch`, `x_search`, and `task_budget`

Examples:

- `responses.gpt-5.5?reasoning=medium`
- `responses.gpt-5.5?web_search=on`
- `sonnet?reasoning=4096`
- `opus?web_search=on&web_fetch=on`
- `gemini3?reasoning=auto`
- `xai.grok-4.3?x_search=on`
- `kimi26instant`
- `hf.moonshotai/Kimi-K2.6:novita?reasoning=on`
- `azure.my-deployment`
- `generic.llama3.2:latest`
- `openrouter.google/gemini-2.5-pro-exp-03-25:free`
- `tensorzero.my_tensorzero_function`

### Precedence

Model specifications follow this precedence order, highest to lowest:

1. Explicitly set in agent decorators
1. Command-line arguments with `--model`
1. Default model in `fast-agent.yaml`
1. `FAST_AGENT_MODEL` environment variable
1. System default (`gpt-5.4-mini?reasoning=low`)

### Reasoning

You can also set reasoning directly in the model string query. This is especially useful for provider-specific reasoning modes:

- `responses.gpt-5.5?reasoning=medium`
- `sonnet?reasoning=4096` (budget tokens)
- `opus?reasoning=auto` (adaptive default)
- `opus?reasoning=xhigh&task_budget=128k` (adaptive Opus + task budget)
- `gemini3?reasoning=high`
- `xai.grok-4.3?reasoning=none`

Reasoning, Verbosity and Task Budget settings are also available from the `/model` command, or by using ++f6++ or ++f7++ keys.

### Temperature and sampling

You can set sampling temperature directly in the model string query:

- `responses.gpt-5.5?temperature=0.2`
- `openai.gpt-4.1?temp=0.7`
- `hf.moonshotai/Kimi-K2.6:novita?temperature=1.0&top_p=0.95`

If temperature is omitted, fast-agent does not send a temperature parameter.
Only explicit values (for example via `?temperature=` / `?temp=` or request
params/config) are forwarded.

### Model presets and model references

For convenience, popular models have built-in **model presets** such as `codex` or `sonnet`.
These are documented on the [LLM Providers](llm_providers/) page.

You can also create local **model overlays**. These are environment-local named model entries that
bundle endpoint settings, auth, request defaults, and local metadata under a short token such as
`qwen-local`. See [Model Overlays](model_overlays/).

You can also define your own namespaced **model references** in `fast-agent.yaml` and
reference them with exact tokens like `$system.fast`.

If a configured model reference cannot be resolved, fast-agent logs a warning and automatically falls back
to the next lower-precedence model source.

## Default configuration

You can set a default model for your application in your `fast-agent.yaml`:

```yaml
default_model: "gpt-5-mini?reasoning=low"
```

## History saving

You can save the conversation history to a file by sending a `***SAVE_HISTORY <filename>` message. This can then be reviewed, edited, loaded, or served with the `prompt-server` or replayed with the `playback` model.

!!! Note "File Format / MCP Serialization"

    If the filetype is `json`, fast-agent saves a `{"messages": [...]}` JSON container. It can contain either MCP `PromptMessage` objects (legacy) or `PromptMessageExtended` objects (preserves tool calls, channels, etc). `fast_agent.load_prompt` and `prompt-server` will load either the text or JSON format directly.

This can be helpful when developing applications to:

* Save a conversation for editing
* Set up in-context learning
* Produce realistic test scenarios to exercise edge conditions etc. with the [Playback model](internal_models/#playback)
