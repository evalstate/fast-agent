---
title: Moonshot
social:
  title: Moonshot
  tagline: Run Kimi K3 with native reasoning, streaming, tools, structured output, and vision.
  description: Configure Moonshot's native Kimi K3 Chat Completions provider in fast-agent.
---

# Moonshot

fast-agent provides a native Moonshot Chat Completions route for Kimi K3.

## Setup

Create an API key in the [Kimi platform](https://platform.kimi.ai/) and set:

```bash
export MOONSHOT_API_KEY="..."
```

Run the native alias:

```bash
fast-agent go --model kimik3 \
  --message "Explain why 37 × 41 equals 1517."
```

Equivalent model strings are:

```text
kimik3
kimi-k3
moonshot.kimi-k3
```

The existing `kimi`, `kimi27code`, `kimi26`, and other Kimi aliases continue
to use their Hugging Face Inference Providers routes. Existing
`moonshotai/Kimi-K2.*` repository identifiers also remain Hugging Face routes.

### Hugging Face Inference Providers

Kimi K3 is also available through Hugging Face Inference Providers:

```text
hf.moonshotai/Kimi-K3:fireworks-ai
hf.moonshotai/Kimi-K3:together
```

Both routes accept K3's top-level `reasoning_effort` values (`low`, `high`, and
`max`), return reasoning separately in `reasoning_content`, and support image
input. Their HF deployments do not provide usable video input.

## Configuration

The built-in endpoint and model are:

```yaml
moonshot:
  api_key: "${MOONSHOT_API_KEY}"
  base_url: "https://api.moonshot.ai/v1"
  default_model: "kimi-k3"
```

Only `api_key` is required. `base_url`, `default_model`, and `default_headers`
can be overridden for a compatible gateway.

## Model limits and parameters

Kimi K3 has a 1,048,576-token context window shared by input and output. The API
defaults `max_completion_tokens` to 131,072 when the field is omitted and
allows it up to 1,048,576, provided the prompt plus requested completion still
fits the context window.

K3 fixes its sampling configuration. Moonshot documents `top_p=0.95`, `n=1`,
`presence_penalty=0`, and `frequency_penalty=0` as non-modifiable, and does not
allow `temperature` to be modified. Native presets therefore do not add
sampling parameters. If reusable request defaults provide fixed or unsupported
sampling fields, the Moonshot adapter warns and omits them.

## Reasoning

Kimi K3 always reasons. The supported reasoning efforts are:

```text
low
high
max
```

`max` is the default. K3 does not support disabling reasoning.

```bash
fast-agent go --model "kimik3?reasoning=low" --message "Solve this problem."
fast-agent go --model "kimik3?reasoning=high" --message "Solve this problem."
fast-agent go --model "kimik3?reasoning=max" --message "Solve this problem."
```

Moonshot returns hidden reasoning separately in `reasoning_content`; fast-agent
keeps it in the reasoning channel rather than merging it into visible assistant
content.

During onboarding on July 29, 2026, two identical modular-arithmetic probes per
effort returned these reasoning-token counts:

| Effort | Runs | Median |
|---|---:|---:|
| `low` | 141, 134 | 137.5 |
| `high` | 299, 279 | 289 |
| `max` | 347, 392 | 369.5 |

This confirms that effort changes reasoning volume for that task; it is not a
fixed token budget or a guarantee for every prompt.

## Multi-turn conversations

Moonshot requires the complete assistant message from each response in later
requests. This includes `reasoning_content` and, when present, `tool_calls`.
fast-agent replays all required assistant fields in conversation history and
never copies hidden reasoning into visible `content`.

## Structured output

K3 supports strict `json_schema` responses:

```bash
fast-agent go --model kimik3 \
  --json-schema ./result.schema.json \
  --message "Return the requested result."
```

The native model profile therefore uses `structured_tool_policy:
no_tools`: structured generation is available, but regular tools are suppressed
for that structured turn.

## Image and video input

K3 supports image and video understanding. fast-agent advertises and has tested Moonshot's
documented image MIME types (`JPEG`, `PNG`, `GIF`, `WebP`, `BMP`, `HEIC`, and
`HEIF`) and video MIME types (`MP4`, `MPEG`, `MOV`, `AVI`, `FLV`, `MPG`,
`WebM`, `WMV`, and `3GPP`). SVG and PDF are not advertised as native K3 media
inputs.

## Usage and caching

Moonshot automatically caches reusable prompt prefixes. Usage can report
`cached_tokens` at the top level and in `prompt_tokens_details.cached_tokens`.
The integration relies on provider-reported usage and does not configure or
claim a fixed cache TTL.

## Official documentation

- [Kimi API overview](https://platform.kimi.ai/docs/overview.md)
- [Kimi K3](https://platform.kimi.ai/docs/guide/kimi-k3-quickstart.md)
- [Models](https://platform.kimi.ai/docs/models.md)
- [Model parameter reference](https://platform.kimi.ai/docs/api/models-overview.md)
- [Chat Completions](https://platform.kimi.ai/docs/api/chat.md)
- [Reasoning effort](https://platform.kimi.ai/docs/guide/use-reasoning-effort.md)
- [Multi-turn conversations](https://platform.kimi.ai/docs/guide/engage-in-multi-turn-conversations-using-kimi-api.md)
- [Tool calls](https://platform.kimi.ai/docs/guide/use-kimi-api-to-complete-tool-calls.md)
- [Vision input](https://platform.kimi.ai/docs/guide/use-kimi-vision-model.md)
- [OpenAPI specification](https://platform.kimi.ai/docs/openapi.json)

See [Models Reference](../models_reference/) for the generated capability row.
