---
title: Hugging Face
social:
  title: Hugging Face
  tagline: Configure Hugging Face Inference Providers, routing, and model aliases in fast-agent.
  description: Configure Hugging Face Inference Providers, routing, and model aliases in fast-agent.
  alt: fast-agent social card — Hugging Face
---

# Hugging Face

Use the `hf` provider for [Hugging Face Inference Providers](https://huggingface.co/docs/inference-providers/en/index).

```yaml
hf:
  api_key: "${HF_TOKEN}"
  # default_provider: groq # optional: groq, fireworks-ai, cerebras, etc.
```

Use `hf.<model_name>[:provider]` to specify models. If no provider suffix is supplied, Hugging Face auto-routes the request.

```bash
fast-agent --model kimi
fast-agent --model kimi26instant
fast-agent --model hf.openai/gpt-oss-120b
fast-agent --model hf.moonshotai/kimi-k2-instruct-0905:groq
fast-agent --model "hf.moonshotai/Kimi-K2.6:novita?reasoning=on"
```

Curated aliases such as `kimi`, `deepseek-hf`, `glm`, and `minimax` include provider choices and request defaults tested with fast-agent features such as structured outputs and tool use. Capability can still vary by backing provider.

## Qwen3.8 27B

Use the canonical [Qwen3.8 27B](https://huggingface.co/Qwen/Qwen3.8-27B)
model ID with the Hugging Face router:

```bash
fast-agent --model "hf.Qwen/Qwen3.8-27B?reasoning=xhigh"
fast-agent --model "hf.Qwen/Qwen3.8-27B?reasoning=low"
fast-agent --model "hf.Qwen/Qwen3.8-27B?reasoning=off"
```

The model has a native 262,144-token context window. Its optional 1M context
requires an explicitly configured YaRN deployment; dedicated endpoints should
use the limit returned by their own `/v1/models` response.

Qwen3.8 thinking is enabled by default. Fast-agent supports the documented
`low`, `medium`, and `xhigh` reasoning efforts. Although the upstream model
defaults to `xhigh`, fast-agent uses `medium`: repeated fixed-answer trials
matched `xhigh` correctness with materially lower completion usage and latency.
Use `reasoning=xhigh` when maximum deliberation matters more than throughput.
`reasoning=off` sends `chat_template_kwargs.enable_thinking: false`. Historical
reasoning is replayed as `reasoning_content`, preserving the model's default
multi-turn thinking behavior.

Qwen3.8 uses the `grok_shell` execution profile by default. It exposes `shell`
with explicit timeout, background, and working-directory fields alongside the
unified `process` tool. Set `shell_execution.tool_profile` explicitly to
override automatic model selection.

The model card recommends `temperature=1.0`, `top_p=0.95`, `top_k=20`,
`min_p=0.0`, `presence_penalty=0.0`, and `repetition_penalty=1.0` for thinking
mode. Put these values in a model overlay when a dedicated endpoint should use
them by default.

The model profile supports text, JPEG/PNG/WebP images, and MP4 video. Live
fast-agent onboarding on August 22, 2026 verified streamed reasoning, reasoning
disable, shell tool execution and continuation, JSON-object structured output,
tool-assisted structured output, local image and video attachments, and usage
accounting on a dedicated Hugging Face OpenAI Chat Completions endpoint. JSON
object mode is advertised; JSON Schema mode is not.

For tool-assisted structured output, fast-agent defers JSON-object enforcement
until after the tool result. In repeated live trials, enforcing JSON while
tools were still active caused the model to fabricate an answer instead of
calling the tool; the deferred workflow called the tool and preserved its
payload reliably. This policy performs one tool-gathering phase followed by a
schema-only final phase; workflows that require sequential dependent tool
rounds should gather those results before requesting the structured final.

## Muse Glimmer via Together

`glimmer` routes [Muse Glimmer 30B](https://huggingface.co/meta-models/Muse-Glimmer-30B)
through the Hugging Face Inference Providers router using Together:

```bash
fast-agent --model glimmer
fast-agent --model "glimmer?reasoning=xhigh"
```

The preset resolves to `hf.meta-models/Muse-Glimmer-30B:together` and applies
Meta's recommended sampling defaults: `temperature=1.0`, `top_p=0.95`, and
`top_k=64`.

Muse Glimmer supports text and image input with text output and a 131,072-token
context window. Its reasoning control is a chat-template setting rather than an
OpenAI `reasoning_effort` field. Fast-agent maps `low`, `medium`, `high`, and
`xhigh` to `chat_template_kwargs.reasoning_strength`; the default is `high`.
The model advertises a 128,000-token output capability, but fast-agent omits
`max_tokens` by default so the serving backend can account for the serialized
input. Glimmer routes reserve 32,768 output tokens when that field is omitted,
so fast-agent uses the resulting 98,304-token prompt window for context
monitoring and automatic compaction. Use `glimmer?max_tokens=...` to send an
explicit output cap; the monitored prompt window adjusts to that reservation.

Meta's released chat template and [Together's model page](https://www.together.ai/models/muse-glimmer)
describe tool calling, while Together's serverless model catalog currently marks
function calling and structured outputs as unavailable for this endpoint. Live
fast-agent testing confirms regular streamed tool calls and tool-result
continuation work; the Hugging Face adapter uses manual stream accumulation for
Together's null-valued tool-call continuation fragments.

Fast-agent does not advertise structured JSON support for `glimmer`. Prompted
JSON and tool-assisted JSON can succeed, but native Pydantic structured output
was not reliable in live testing.

## Kimi instant mode

Kimi models that support instant mode can disable reasoning with the `instant` query parameter:

```bash
fast-agent --model "hf.moonshotai/Kimi-K2.5?instant=on"  # thinking disabled
fast-agent --model "hf.moonshotai/Kimi-K2.5?instant=off" # thinking enabled
```

## Gemma thinking mode

`gemma4` routes Gemma 4 31B through Hugging Face Inference Providers on Cerebras:

```bash
fast-agent --model gemma4
fast-agent --model "hf.google/gemma-4-31B-it:cerebras?temperature=1.0&top_p=0.95"
```

Gemma 4 reasoning is disabled by default on Cerebras. Enable it with `reasoning_effort`
values through fast-agent's `reasoning` query parameter:

```bash
fast-agent --model "gemma4?reasoning=medium" # sends reasoning_effort=medium
fast-agent --model "gemma4?reasoning=none"   # sends reasoning_effort=none
```

## Hugging Face MCP authentication

`HF_TOKEN` is automatically applied when connecting to Hugging Face MCP servers:

- `hf.co` / `huggingface.co` uses `Authorization: Bearer {HF_TOKEN}`
- `*.hf.space` uses both `Authorization: Bearer {HF_TOKEN}` and
  `X-HF-Authorization: Bearer {HF_TOKEN}`

## Model aliases

--8<-- "_generated/model_aliases_hf.md"
