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

Kimi K3.1 Pro is available via Together using
`hf.moonshotai/Kimi-K3.1-Pro:together` (`HF_TOKEN` required). The suffix is
`together`, not `togetherai`. This route supports images and top-level
`reasoning_effort` (`low`, `high`, `max`; default `max`), with preserved reasoning
history. It is also the default HF model and the target of `kimi`/`kimithink`.
PTC is enabled by default for K3.1 Pro; `--no-ptc` or `?ptc=off` disables it.
Its structured-output requests use Together's legacy schema-bearing envelope
to work around HF router translation; output is validated locally.
See [Moonshot](moonshot.md) for the native route and model details.

Curated aliases such as `kimi`, `deepseek-hf`, `glm`, and `qwen35` include provider choices and request defaults. Capabilities and supported parameters vary by backing provider; see the route-specific notes below.

## Hugging Face routing update — 15 November 2026

Most catalog routes now pass through OpenRouter, even with an explicit backend
suffix. Backend-specific `thinking`, chat-template reasoning controls and
`disable_reasoning` are silently ignored there; fast-agent uses top-level
`reasoning_effort` instead. Kimi K2.5/K2.6, GLM 5, GLM 5.2 on Z.ai and
Qwen 3.6 can again disable reasoning with `?reasoning=off` or `?reasoning=none`,
both sent as `reasoning_effort: "none"`.
Qwen 3.5 provisionally uses the same control as Qwen 3.6: upstream rate limits
prevented the initial
investigation, but a post-change smoke check confirmed off returns zero
reasoning tokens and text. Kimi K2.6 off returns no reasoning text, but still
reported one reasoning token in both post-change checks.

Inkling 2 Small requires reasoning on both routes: its ladder now starts at
`minimal` (default `high`), and `none` is rejected upstream. DeepSeek V4.1 Flash
10 September (`deepseek41-hf`) exposes only `low`, `high`, `max`; these are
accepted but showed no measurable effort difference. Its 1017 successor
(`deepseek-hf`) still supports `none`/off. The unprobed 0731 checkpoint is
unchanged. MiniMax M2.5 on Novita is still served directly and cannot disable
reasoning; no toggle is offered.

Muse Glimmer now sends top-level effort, retaining `low`, `medium`, `high`,
`xhigh` and default `high`. GLM 5.3/Flash remain always-on (`low`, `high`, `max`);
probes distinguished `max`, but not `low` from `high`. Kimi K3/K3.1 Pro and
GLM 5.2 DeepInfra settings are unchanged. Reasoning streaming and replay remain
unchanged. Together's GLM `clear_thinking` field is retained pending a multi-turn
check of its effect. Z.ai's `thinking` object, including `clear_thinking: false`,
was dropped on single-turn evidence; a post-change tool loop confirmed replay
still works, but did not establish the field's multi-turn effect.

For Hugging Face effort-based models that cannot disable reasoning, saved or typed
`?reasoning=none` / `?reasoning=off` now maps to the lowest exposed effort,
not the default. This adjustment is logged once per model instance; model-string
adjustments have no separate console notification. `/model reasoning` reports
the effective effort when changed interactively. Reasoning remains enabled.
This fallback is Hugging Face-only; other providers keep their existing behavior.
HF models with a disable control still disable for both `none` and `off`.

The DeepSeek 10 September and MiniMax capability changes are model-wide,
including unprobed backends and dedicated endpoints. Dedicated endpoint wire
adapters are retained, but their exposed settings are not wholly unchanged.
MiniMax has no exposed effort ladder or reasoning control; saved `reasoning=none`
or `reasoning=off` has no effect (there is no lowest effort to select).

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

The explicit `glm52`, `GLM 5.2 (zai-org)`, and `GLM 5.2 (deepinfra)` aliases
remain available. For the native provider, see
[Z.ai](zai.md).

```bash
fast-agent --model glm
fast-agent --model "glm53flash?reasoning=low"
fast-agent --model "hf.zai-org/GLM-5.3:together?reasoning=high"
fast-agent --model "hf.zai-org/GLM-5.3-Flash:novita?reasoning=max"
```

Reasoning is always on, with `low`, `high`, and `max` effort levels (default
`max`). `minimal` maps to `low` and `medium` to `high`; requests to disable
reasoning (`none`/off) map to `low` with a one-time logged warning. Each route
sends top-level `reasoning_effort`, plus backend-specific preserved-thinking
controls:

| Backend | Additional request fields |
| --- | --- |
| Together | `chat_template_kwargs.clear_thinking: false` ([guide](https://docs.together.ai/docs/glm-5.3-quickstart)) |
| Novita | None ([model page](https://novita.ai/models/model-detail/zai-org-glm-5.3)) |
| Z.ai (`zai-org`) | None |

Streamed reasoning is preserved exactly and replayed as `reasoning_content` on
later turns and tool-result continuations.

Structured output uses JSON object mode with schema instructions and
fast-agent validation, not strict `json_schema` mode.

[GLM-5.3](https://huggingface.co/zai-org/GLM-5.3) is text-only.
[GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) accepts text and
images through HF; native Z.ai PDF/MOV support is not available on HF routes.

## DeepSeek V4.1 Flash

`deepseek-hf` selects DeepSeek V4.1 Flash on Novita, using the current
[DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)
checkpoint. DeepSeek's native `deepseek-flash` API model has served this
checkpoint since October 17, 2026, so `deepseek-hf` is the same model as the
native [DeepSeek](deepseek.md) route:

```bash
fast-agent --model deepseek-hf
fast-agent --model "deepseek-hf?reasoning=high"
fast-agent --model "hf.deepseek-ai/DeepSeek-V4.1-Flash-1017:novita?reasoning=off"
```

`deepseek-hf` resolves to `hf.deepseek-ai/DeepSeek-V4.1-Flash-1017:novita`.
It supplies no sampling overrides, so the provider's sampling defaults apply.
`deepseek41-hf`, `deepseek-v41-hf` and `DeepSeek V4.1 Flash (novita)` remain on
the original September 10 checkpoint (see
[Original checkpoint](#original-september-10-checkpoint)).

The model supports text and image input (JPEG, PNG, WebP and GIF) with text
output and a 1,048,576-token context window. Fast-agent advertises a
393,216-token output ceiling, also reported by Novita's rejection of a larger
limit. It omits `max_tokens` by default so the backend default applies;
`deepseek-hf?max_tokens=...` sends an explicit limit. Contract probes exceeded
114,000 output tokens with the limit omitted, but did not establish the
backend's exact default. The HF router's request-body size limit still
applies (see [inline images](#inline-images-and-the-request-size-limit)).

Reasoning defaults to `max`, as on the native DeepSeek route. Fast-agent
exposes `none`, `low`, `high` and `max`, sent unchanged as top-level
`reasoning_effort`; `reasoning=off` sends `none`. Generic efforts are
normalized: `minimal` to `low`, `medium` to `high` and `xhigh` to `max`. No
numeric effort conversion is made. The off/none checks returned no
reasoning-channel content, although the model can still show its working in
the visible answer. Reasoning history is replayed as `reasoning_content`.

Structured output uses standard `json_schema`. With tools available, the
default `defer` policy runs a tools-only phase before requesting a schema-only
final answer without tools. Sending schema and tools together sometimes
produced a fenced JSON answer that failed validation; the two-phase default
passed the live conformance checks. `?structured_tools=always` restores the
combined request.

PDF, audio and video input are not advertised. Fast-agent removes unsupported
attachments and warns the user, but the request currently does not tell the
model that an attachment was removed. **When no tools are offered, the model
can write tool-call markup as plain text**, including DeepSeek DSML,
`<tool_call>` JSON or `<terminal>` tags. This occurred after an unsupported
PDF was removed and on file-listing prompts without tools. That text is not an
executed tool call. With tools offered, the tested calls were native and
executed correctly.

Programmatic tool calling is off by default. The PTC stage has not run for
this checkpoint; this is an inherited default, not a new comparative
performance finding.

Live fast-agent onboarding on November 15, 2026 verified streaming, every
exposed reasoning effort including `off`, shell and function tool loops with
parallel calls, multi-turn reasoning replay, PNG input and image tool results,
and schema output directly and after a tool call through
`hf.deepseek-ai/DeepSeek-V4.1-Flash-1017:novita`. Other backends are not covered
by this fast-agent conformance result.

### Original September 10 checkpoint

```bash
fast-agent --model deepseek41-hf
fast-agent --model "hf.deepseek-ai/DeepSeek-V4.1-Flash:novita?reasoning=low"
fast-agent --model "hf.deepseek-ai/DeepSeek-V4.1-Flash:fireworks-ai"
```

`deepseek41-hf`, `deepseek-v41-hf`, and `DeepSeek V4.1 Flash (novita)` select
the original `deepseek-ai/DeepSeek-V4.1-Flash` checkpoint on Novita; they are
pinned there and do not follow `deepseek-hf`. Omitting the backend uses the
configured HF default provider or HF auto-routing; other backends fall back to
the same model-level request profile, not automatic retry/failover.

The HF profile inherits V4.1's vision support, 1,048,576-token context,
393,216-token output limit, and JSON Schema mode. Backend limits and capabilities
may differ. Reasoning history is replayed as `reasoning_content`. Structured
output with tools uses the same two-phase `defer` policy as `deepseek-hf`: the
combined schema+tools request failed validation intermittently on Novita.

Requests send top-level `reasoning_effort`: `max` by default, with `low`,
`high`, and `max` available. As of 15 November 2026, this checkpoint cannot
disable reasoning: `none` and off are no longer offered. Saved or typed
`?reasoning=none` / `?reasoning=off` maps to `low`, not the default `max`,
with a one-time logged warning. The accepted efforts showed no measurable
difference in the probes. No numeric effort conversion is
made; the README's raw-weights range of 1–100 does not establish API semantics.
The 1017 checkpoint retains working `none`/off support.

Earlier September harness checks passed a tool-call roundtrip and synthetic
image recognition. Their off result predates the OpenRouter routing change
and no longer describes the current route.

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

## Gemma 4 31B

`hf.google/gemma-4-31B-it` (with or without a backend suffix) exposes reasoning
as an on/off toggle, sent as `chat_template_kwargs.enable_thinking`. Effort
levels such as `reasoning=medium` have no wire representation on these routes.

Fast-agent now sends `enable_thinking: false` explicitly by default. Earlier
versions sent no thinking control unless `reasoning=off` was selected, so the
backend's own default applied. This is a wire change: on a backend whose
template or deployment previously enabled thinking by default, requests now
disable it. Fast-agent has not verified each backend's default, so the new
default should not be read as equivalent to the provider default. Use
`?reasoning=on` to request thinking:

```bash
fast-agent --model "hf.google/gemma-4-31B-it?reasoning=on"
```

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
context window. Fast-agent sends `low`, `medium`, `high`, and `xhigh` as
top-level `reasoning_effort`; the default is `high`. The former chat-template
`reasoning_strength` field has no effect through OpenRouter.
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

## Reflection Beam 1.0

`beam` routes [Reflection Beam 1.0](https://huggingface.co/reflection/Beam-1.0-501B-A23B)
through the Hugging Face Inference Providers router, served through OpenRouter
with Nebius as the default upstream:

```bash
fast-agent --model beam
fast-agent --model "beam?reasoning=minimal"
fast-agent --model "hf.reflection/Beam-1.0-501B-A23B:together"
```

The preset resolves to `hf.reflection/Beam-1.0-501B-A23B:nebius` and does
not set sampling defaults; the provider's defaults apply. The HF picker lists
it as `Beam 1.0 (nebius)`. Together uses the same request profile through
OpenRouter; it is an alternative route, not automatic failover.

Beam supports text input and output only, with a 1,048,576-token context
window and a 131,072-token output ceiling, including reasoning. Fast-agent
omits `max_tokens` by default; both tested upstreams enforce the ceiling even
when a larger value is requested. Use `beam?max_tokens=...` for a smaller
explicit output cap. Image, PDF, audio and video attachments are not supported
by fast-agent for this model; unsupported content is removed with a warning.

Reasoning is mandatory. Fast-agent accepts `minimal`, `low`, `medium`, `high`,
`xhigh` and `max`, sends the value unchanged as top-level `reasoning_effort`,
and explicitly defaults to `max`. Neither `none` nor off is offered.
A requested `?reasoning=none` or `?reasoning=off` maps to `minimal`, with a
one-time logged warning; that warning is not currently shown on the console.

Reasoning is replayed as `reasoning_content` on every turn, including plain
assistant replies and tool-call turns. A history containing an assistant turn
without reasoning (for example, from another model) is rejected by the provider
with an explanatory error identifying the missing reasoning. Switching models
mid-conversation into Beam is not supported; start a new conversation instead.
Fast-agent does not insert placeholder reasoning or override the provider's
requirement. The shared fatal-error path still prints a traceback.

Structured output uses the standard `json_schema` response format, directly
and with tools; no legacy envelope or separate schema-only final phase is
required. Programmatic tool calling is off by default. The PTC stage has not
run for this model; this is not a comparative performance finding.

**Stop-sequence caveat:** stop sequences can match inside Beam's reasoning
and end a turn with no answer. Fast-agent sends requested stop sequences
unchanged and retries an empty completion once, but the turn can still fail.
Provider token usage is recorded; OpenRouter's reported cost is not currently
exposed as fast-agent `cost_usd`, which remains unknown rather than zero.

Live fast-agent onboarding on November 16, 2026 verified streaming, all six
reasoning efforts, shell and function tool loops with parallel calls,
multi-turn reasoning replay, unsupported-media removal, and schema output
directly and after a tool call through `beam` (Nebius) and
`hf.reflection/Beam-1.0-501B-A23B:together`. The missing-reasoning error
explanation was also verified through `beam`.

## Inkling 2 Small

`inkling` routes [Inkling 2 Small](https://huggingface.co/thinkingmachines/Inkling-2-Small)
from Thinking Machines through the Hugging Face Inference Providers router using
DeepInfra:

```bash
fast-agent --model inkling
fast-agent --model "inkling?reasoning=max"
```

The preset resolves to `hf.thinkingmachines/Inkling-2-Small:deepinfra` and does
not set sampling defaults; the provider's defaults apply. The HF picker lists it
as `Inkling 2 Small (deepinfra)`. DeepInfra is the maintainer-selected default;
Together remains available (see [Together route](#together-route)). In two
small, separate agentic benchmark probes (15 trials per mode on each backend,
run with different fast-agent builds, scheduling and times of day), DeepInfra
completed 30 of 30 trials and Together 28 of 30. These are observations from
non-matched runs, not a reliability or speed comparison between the backends.

Inkling 2 Small supports text and image input (JPEG, PNG and WebP) with text
output and a 1,048,576-token context window. PDF, audio and video attachments
are not supported by fast-agent for this model; fast-agent removes them from the
request and shows a warning. No output maximum is published. Fast-agent omits `max_tokens` by
default, so the provider default applies, and uses a nominal 128,000-token
output budget for its own accounting. Use `inkling?max_tokens=...` to send an
explicit output cap.

DeepInfra ends every response at 65,536 output tokens, reasoning included,
whatever `max_tokens` or `max_completion_tokens` requests (verified November
14, 2026), and fast-agent clamps explicit values to that. A turn that spends
the whole allowance on reasoning returns no answer. Together has not shown this
cap: it returned an 82,475-token completion in a benchmark run, though its
ceiling is not published or measured. Use `Inkling 2 Small (together)` for
work that needs longer single responses.

Reasoning is controlled by the top-level `reasoning_effort` field. Fast-agent
accepts `minimal`, `low`, `medium`, `high`, `xhigh` and `max`, sends the
value unchanged, and defaults to `high`. Reasoning is mandatory on both routes:
`none` now returns HTTP 400, rather than the pre-switch soft minimum. Neither
`none` nor off is offered. Saved or typed `?reasoning=none` / `?reasoning=off`
maps to `minimal`, not the default `high`, with a one-time logged warning.
Reasoning content is replayed unchanged in conversation history.

**15 November 2026:** `none` was removed after the routing change. This
supersedes the "soft minimum" description in the dated
[Inkling 2 Small integration report](../integration/inkling-2-small.md).

Structured output uses the standard `json_schema` response format. When tools
are available, fast-agent runs a tool phase first and then requests the
schema-only final answer without tools, because the schema can stop the model
from calling a tool. DeepInfra streams tool-call continuation fragments with a
`null` type, so fast-agent accumulates this route's streams manually.

On DeepInfra the model sometimes ends a turn with reasoning only and no answer
when no tools are declared; in a probe, a tool-free follow-up turn (a question
about a removed PDF) returned empty replies in most attempts. fast-agent
retries an empty completion once; a turn can still fail. This was not seen on
Together in the same test.

Programmatic tool calling is off by default for this model; enable it with
`inkling?ptc=on` or `--ptc`.

Live fast-agent testing on November 14, 2026 verified streaming, every
reasoning effort (including the then-supported `off`), shell and function tool
loops with parallel calls, multi-turn reasoning replay, JPEG, PNG and WebP image input, image tool
results, and schema output (direct and after a tool call) through
`hf.thinkingmachines/Inkling-2-Small:deepinfra`.

### Together route

Together also serves the model. Pin it with the `Inkling 2 Small (together)`
preset or the full model string:

```bash
fast-agent --model "Inkling 2 Small (together)"
fast-agent --model "hf.thinkingmachines/Inkling-2-Small:together?reasoning=low"
```

The context window, image input, reasoning efforts and default, omitted
`max_tokens`, reasoning replay, programmatic tool calling default and the
tool-phase-first handling of schema output are the same as on DeepInfra. What
differs:

- Together rejects the standard `json_schema` response format for this route,
  so structured-output requests use the legacy `json_object` envelope with a
  `schema` field.
- Together rejects PDF input. (DeepInfra accepts it, but fast-agent does not
  advertise PDF for this model, so PDF attachments are removed on both routes.)
- Streams use the standard OpenAI SDK stream path rather than manual
  accumulation.

Live fast-agent onboarding on November 14, 2026 verified streaming, every
reasoning effort (including the then-supported `off`), shell and function tool
loops, multi-turn reasoning replay, JPEG, PNG and WebP image input, and schema output (direct and
after a tool call) through `hf.thinkingmachines/Inkling-2-Small:together`.

### Un-pinned route

`hf.thinkingmachines/Inkling-2-Small` without a backend lets the router choose
(DeepInfra at the time of testing). fast-agent provisionally applies the
DeepInfra settings to this un-pinned route, but the router may serve it from
any backend: if it selects Together, structured (schema) output fails with an
error. Pin a backend (`:deepinfra` or `:together`) for schema output and for
predictable behaviour. Other explicitly selected backends and custom endpoints
get no Inkling-specific settings and are untested.

## Kimi instant mode

Kimi models that support instant mode can disable reasoning with the `instant` query parameter:

```bash
fast-agent --model "hf.moonshotai/Kimi-K2.5?instant=on"  # thinking disabled
fast-agent --model "hf.moonshotai/Kimi-K2.5?instant=off" # thinking enabled
```

## Hugging Face MCP authentication

`HF_TOKEN` is automatically applied when connecting to Hugging Face MCP servers:

- `hf.co` / `huggingface.co` uses `Authorization: Bearer {HF_TOKEN}`
- `*.hf.space` uses both `Authorization: Bearer {HF_TOKEN}` and
  `X-HF-Authorization: Bearer {HF_TOKEN}`

## Model aliases

--8<-- "_generated/model_aliases_hf.md"


## DeepSeek V4 Flash 0731 route limits

The earlier `deepseek-ai/DeepSeek-V4-Flash-0731` checkpoint is no longer listed
in the model picker or as an alias, but it can still be selected by model ID,
for example `hf.deepseek-ai/DeepSeek-V4-Flash-0731:baseten`. It uses the same
model profile as the 1017 [DeepSeek V4.1 Flash](#deepseek-v41-flash) checkpoint,
not the original `deepseek41-hf` checkpoint: reasoning defaults
to `max` (top-level `reasoning_effort`, `reasoning=off` sends `none`), reasoning
history is replayed as `reasoning_content`, and structured output with tools
uses the two-phase `defer` policy. Unlike V4.1, the 0731 checkpoint is text-only;
image attachments are removed before the request is sent.

Backend-specific limits still apply. Baseten caps output at 384,000 tokens;
Scaleway caps output at 32,768 tokens. Fast-agent applies these route limits to
defaults and clamps larger explicit `max_tokens` values before sending a request.
Smaller explicit budgets are preserved.

Scaleway exposes a 262,144-token context window; context display and automatic
compaction use this route-specific window, not the model-wide 1M window.
Together omits the default output limit. DeepInfra defaults to 131,072 output
tokens because omission still reserves 384,000 tokens on that backend. Explicit
`max_tokens` overrides are preserved on Together and DeepInfra.

The V4.1 Flash Novita route also omits the default output limit while preserving
explicit overrides. Other routes retain the model-wide output budget.

These limits apply to raw model strings and routes selected with
`hf.default_provider`; no `?max_tokens=...` workaround is needed for Baseten.

## Inline images and the request size limit

The router rejects request bodies larger than about 5 MB with HTTP 413. Image-heavy
turns (document scans, video frames) can reach that quickly, because every image in
the history is resent on each turn. Fast-agent fits inline images to the limit when
it builds each request; the conversation history is never changed:

1. Images whose longest side exceeds `image_max_dimension` (default 2048 px) are
   downscaled. Images larger than `image_reencode_min_bytes` (default 300 KB) are
   re-encoded as the smaller of their own format and JPEG; images with transparency
   stay PNG. The result is deterministic, so repeated turns keep a stable prompt prefix.
2. If the request is still larger than `max_request_bytes` (default 4.5 MiB), the oldest
   images are replaced with a short text placeholder until it fits.
3. If the router still answers 413, the request is resent once with the images trimmed
   to half its size.

This applies to the Hugging Face router only; requests to a custom `hf.base_url`
endpoint are sent unchanged.

```yaml
hf:
  # image_max_dimension: 2048       # null disables downscaling
  # image_reencode_min_bytes: 300000 # null disables re-encoding
  # max_request_bytes: 4718592      # null disables the budget
```
