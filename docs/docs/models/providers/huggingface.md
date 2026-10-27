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

Curated aliases such as `kimi`, `deepseek-hf`, `glm`, and `minimax` include provider choices and request defaults. Capabilities and supported parameters vary by backing provider; see the route-specific notes below.

## GLM-5.3 and GLM-5.3-Flash

`glm` and `glm53` select `hf.zai-org/GLM-5.3:zai-org`; `glm53flash` selects
`hf.zai-org/GLM-5.3-Flash:zai-org`. The HF picker offers both models through
Together, Novita, and Z.ai, replacing its GLM-5.2 entries:

| Picker alias | Route |
| --- | --- |
| `GLM 5.3 (together)` | `hf.zai-org/GLM-5.3:together` |
| `GLM 5.3 (novita)` | `hf.zai-org/GLM-5.3:novita` |
| `GLM 5.3 (zai-org)` | `hf.zai-org/GLM-5.3:zai-org` |
| `GLM 5.3 Flash (together)` | `hf.zai-org/GLM-5.3-Flash:together` |
| `GLM 5.3 Flash (novita)` | `hf.zai-org/GLM-5.3-Flash:novita` |
| `GLM 5.3 Flash (zai-org)` | `hf.zai-org/GLM-5.3-Flash:zai-org` |

The explicit `glm52` and `GLM 5.2 (zai-org)`, `GLM 5.2 (fireworks-ai)`, and
`GLM 5.2 (deepinfra)` aliases remain available for compatibility. Native
[Z.ai](zai.md) aliases and its GLM-5.2 provider default are unchanged.

```bash
fast-agent --model glm
fast-agent --model "glm53flash?reasoning=low"
fast-agent --model "hf.zai-org/GLM-5.3:together?reasoning=high"
fast-agent --model "hf.zai-org/GLM-5.3-Flash:novita?reasoning=max"
```

Both fast-agent model profiles keep reasoning enabled and expose `low`, `high`, and
`max` effort levels, defaulting to `max`; disabled thinking is not supported.
The existing effort normalizer maps `off`/`none` back to `max`, not disabled
thinking. Each route sends top-level `reasoning_effort`, with backend-specific preserved
thinking controls:

| Backend | Additional request fields | Evidence |
| --- | --- | --- |
| Together | `chat_template_kwargs.clear_thinking: false` | [Together's GLM-5.3 guide](https://docs.together.ai/docs/glm-5.3-quickstart) documents effort and preserved thinking. |
| Novita | No additional thinking fields | [Novita's model page](https://novita.ai/models/model-detail/zai-org-glm-5.3) lists `low`/`high`/`max`; its [chat reference](https://novita.ai/docs/api-reference/model-apis-llm-create-chat-completion) does not fully specify the wire contract. Top-level effort was accepted live without `separate_reasoning`. |
| Z.ai (`zai-org`) | `thinking: {type: enabled, clear_thinking: false}` | [Z.ai's GLM-5.3 guide](https://docs.z.ai/guides/llm/glm-5.3) and [thinking guide](https://docs.z.ai/guides/capabilities/thinking-mode) document always-on and preserved thinking. |

Together additionally documents `medium`, but it is not part of this common
HF profile. Accepted effort parameters do not prove that every backend applies
distinct reasoning budgets. Fast-agent preserves streamed reasoning exactly,
including whitespace, and replays it as `reasoning_content` on later turns and
tool-result continuations. Offline stream simulations exercise this separately
from live request-acceptance checks.

Structured output uses JSON object mode, schema instructions, and fast-agent
validation—not strict OpenAI `json_schema` mode. The smoke checks below do not
establish support for every OpenAI-compatible request parameter.

[GLM-5.3](https://huggingface.co/zai-org/GLM-5.3) is text-only.
[GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) supports text and
image input through HF. Native Z.ai PDF/MOV support is **not** advertised for
these HF routes.

### Dated evidence and limitations

Research and synthetic FastAgent harness testing on **October 27, 2026**
covered all six routes. The final live run passed **39 of 39 inference cases**:
default/low/high/max reasoning acceptance, actual function-tool execution and
continuation, schema-validated structured output, and Flash PNG recognition
on all three backends. Every successful case reported prompt/completion usage.
All six offline capability checks passed after restricting HF Flash metadata
to text/images.

Together Flash image recognition timed out at 120 seconds in the initial run,
then passed in 71 seconds in the final run; latency can vary substantially.
Reasoning content was observed across all routes, but is not necessarily
present for every trivial prompt or structured response.

Discovery was inconsistent: HF's per-model
[inference provider mappings](https://huggingface.co/docs/hub/models-inference)
listed Together as live, while the [router model listing](https://router.huggingface.co/v1/models)
omitted those Together routes. Explicit routed requests nevertheless passed.
Treat listings as discovery hints, not proof of availability or failover.

To reproduce with `HF_TOKEN` or an HF cached login, explicitly authorize
billable synthetic requests:

```bash
uv run examples/huggingface-glm53-smoke.py --live
uv run examples/huggingface-glm53-smoke.py --live --backend together --model GLM-5.3-Flash --case image
uv run examples/huggingface-glm53-smoke.py --case capabilities
```

The script isolates workspace configuration, tools, and skills. By default it
caps each inference at 1,024 output tokens and each case at 120 seconds with at
most three iterations; it does not validate full context/output limits or
native PDF/video support. `--backend`, `--model`, and `--case` can be repeated;
`--max-tokens` accepts 256, 512, 1024, or 2048. Sanitized results are written to
`/tmp/glm53-live-results.md` (override with `--report`); request acceptance alone
does not establish effort effectiveness.

## DeepSeek V4.1 Flash

```bash
fast-agent --model deepseek41-hf
fast-agent --model "hf.deepseek-ai/DeepSeek-V4.1-Flash:novita?reasoning=off"
fast-agent --model "hf.deepseek-ai/DeepSeek-V4.1-Flash:fireworks-ai"
```

`deepseek41-hf`, `deepseek-v41-hf`, and `DeepSeek V4.1 Flash (novita)` select
Novita. Native DeepSeek aliases are unchanged. Omitting the backend uses the
configured HF default provider or HF auto-routing; other backends fall back to
the same model-level request profile, not automatic retry/failover.

The HF profile inherits V4.1's vision support, 1,048,576-token context,
393,216-token output limit, and JSON Schema mode. Backend limits and capabilities
may differ. Reasoning history is replayed as `reasoning_content`.

For compatibility with the existing HF DeepSeek profile, requests send top-level
`reasoning_effort`: `max` by default, `low`/`high`/`max` when selected, and `none`
for `reasoning=off` or `reasoning=none`. No numeric effort conversion is made:
the model README's raw-weights range of 1–100 does not establish API semantics.
HF's maintained [Novita adapter](https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/inference/_providers/novita.py)
uses the [shared chat parameter passthrough](https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/inference/_providers/_common.py)
without an effort mapping. Other backends may differ in which effort values they honor.

As of September 11, 2026, HF lists Novita and Fireworks as live routes. Local
request-shaping tests cover default/max/off and backend profile fallback. Live
FastAgent harness tests on Novita passed default/max/off reasoning, a tool-call
roundtrip, and synthetic image recognition. Reasoning appeared for default/max
and was absent for off. Low/high effort and other backends remain unverified live.

For a credential-free, explicitly opt-in synthetic harness smoke script, run
`uv run examples/huggingface-novita-smoke.py --live` from the repository with
`HF_TOKEN` configured (or an HF cached login). It isolates workspace configuration
and caps each request at 2,048 output tokens; it does not test the default output limit.

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


## DeepSeek V4 Flash 0731 route limits

`hf.deepseek-ai/DeepSeek-V4-Flash-0731:<provider>` uses the model's default
reasoning profile unless a backend-specific profile applies. Baseten caps output
at 384,000 tokens; Scaleway caps output at 32,768 tokens. Fast-agent applies these
route limits to defaults and clamps larger explicit `max_tokens` values before
sending a request. Smaller explicit budgets are preserved.

Scaleway exposes a 262,144-token context window; context display and automatic
compaction use this route-specific window, not the model-wide 1M window.
Together omits the default output limit. DeepInfra defaults to 131,072 output
tokens because omission still reserves 384,000 tokens on that backend. Explicit
`max_tokens` overrides are preserved on Together and DeepInfra.

The V4.1 Flash Novita route also omits the default output limit while preserving
explicit overrides. Other routes retain the model-wide output budget.

These limits apply to raw model strings, picker aliases, and routes selected with
`hf.default_provider`; no `?max_tokens=...` workaround is needed for Baseten.
