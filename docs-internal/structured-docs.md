# Structured outputs

fast-agent can ask a model to produce a final answer that matches a JSON Schema.
The runtime chooses the strongest structured-output mechanism available for the
selected model/provider, then validates the final answer locally.

The user-facing goal is simple:

- if you request structured output without tools, fast-agent makes one structured
  model call;
- if tools are available, fast-agent uses the selected structured/tool policy to
  decide whether tools are exposed, deferred, or combined with structured
  constraints;
- the final answer is parsed and validated against the schema.

## Structured output modes

Model/provider metadata exposes the structured-output capability as:

```python
json_mode: Literal["schema", "object"] | None
```

Meaning:

| Mode | Behavior |
|---|---|
| `schema` | Use provider-native JSON Schema / structured-output constraints. |
| `object` | Use provider JSON-object mode plus prompt instructions and local validation. |
| `None` | Use provider-specific fallback or prompt/local validation, depending on provider. |

Users normally do not need to choose between these modes. The model catalog and
overlays provide metadata, and fast-agent resolves the correct path.

## Structured output with tools

Structured output and regular tools are controlled by:

```python
structured_tool_policy: Literal["auto", "always", "defer", "no_tools"]
```

| Policy | Behavior |
|---|---|
| `auto` | Use model/provider compatibility metadata and provider defaults. This is the normal default. |
| `always` | Send regular tools and structured constraints together in the same request. |
| `defer` | Use a two-phase flow: tools first, then a structured final answer without tools. |
| `no_tools` | Suppress regular tools and produce one structured answer immediately. |

### `always`

Use this when the model/provider reliably supports regular tool calls and
structured constraints in the same request.

```text
turn 1:
  regular tools: yes
  structured constraints: yes
```

If the model calls a tool, the normal tool loop continues. Subsequent calls keep
the structured request active unless the provider has special handling.

### `defer`

Use this when a model or gateway cannot reliably combine regular tools and
structured constraints in one provider request.

```text
turn 1:
  regular tools: yes
  structured constraints: no

turn 2:
  regular tools: no
  structured constraints: yes
```

`defer` now finalizes even if the model does not call a tool on the first turn.
That preserves the contract that the final answer is structured.

### `no_tools`

Use this when structured output should be produced immediately and regular tools
should not be available for this call.

```text
turn 1:
  regular tools: no
  structured constraints: yes
```

This is useful for extraction, classification, routing, or any structured task
where an agent may have tools generally, but this specific call should not use
them.

## Model string overrides

Structured-output behavior can be selected in model strings.

Provider-specific structured mode:

```text
sonnet?structured=json
sonnet?structured=tool_use
```

Structured/tools policy:

```text
sonnet?structured_tools=always
sonnet?structured_tools=defer
sonnet?structured_tools=no_tools
sonnet?structured_tools=auto
```

Combined examples:

```text
sonnet?structured=json&structured_tools=always
sonnet?structured=tool_use&structured_tools=defer
sonnet?structured=tool_use&structured_tools=no_tools
```

## Model overlays

Local model overlays can describe structured-output capabilities:

```yaml
metadata:
  json_mode: object                 # schema | object | none/null
  structured_tool_policy: defer     # auto | always | defer | no_tools
```

Use `json_mode: none` or `json_mode: null` when provider-native structured modes
are unavailable or unreliable and prompt/local validation is preferred.

Use `structured_tool_policy: defer` when a model should use tools first and then
produce a structured final answer.

## Checking behavior

The structured-tools probe verifies actual tool + structured-output behavior:

```bash
fast-agent check structured-tools --model sonnet --json
fast-agent check structured-tools --models opus,opus46,sonnet,haiku --json
```

To compare policies:

```bash
fast-agent check structured-tools --model sonnet --structured-tool-policy always --json
fast-agent check structured-tools --model sonnet --structured-tool-policy defer --json
```

The probe reports:

- resolved provider/model;
- `json_mode`;
- selected structured tool policy;
- whether the tool was called;
- whether final JSON parsed and validated;
- whether the final JSON matched the tool payload.

---

# Anthropic

Anthropic has two structured-output mechanisms in fast-agent.

## Native JSON Schema mode

Selected with:

```text
?structured=json
```

or automatically for Anthropic models whose metadata has:

```python
json_mode = "schema"
```

This path uses Anthropic's native structured-output API:

```python
output_config = {
  "format": {
    "type": "json_schema",
    "schema": ...
  }
}
```

and the Anthropic structured-output beta header.

Properties:

- strongest Anthropic structured-output path;
- compatible with reasoning/thinking;
- compatible with regular tools on current first-party Claude models;
- works well with `structured_tools=always`.

Current first-party Anthropic models use this path by default, including:

- `claude-opus-4-7`
- `claude-opus-4-6`
- `claude-sonnet-4-6`
- `claude-haiku-4-5`

Example:

```text
sonnet?structured=json&structured_tools=always
```

Empirical probe results for `haiku`, `opus46`, `opus`, and `sonnet` showed that
native JSON Schema mode can call regular tools and return valid schema-matching
JSON in the same structured tool flow.

## Legacy `tool_use` structured output

Selected with:

```text
?structured=tool_use
```

or automatically for Anthropic models/providers that do not use native JSON
Schema mode.

This path creates a synthetic Anthropic tool:

```text
return_structured_output
```

and forces the model to call it:

```python
tool_choice = {
  "type": "tool",
  "name": "return_structured_output"
}
```

The final structured JSON is read from that tool's input.

Properties:

- reliable legacy structured-output path;
- not compatible with Anthropic thinking/reasoning, so thinking is disabled for
  that request;
- does not combine with regular tools in a single request, because the structured
  output tool is forced.

For this reason, when Anthropic's effective structured mode is `tool_use`,
`structured_tool_policy=auto` resolves to:

```python
"no_tools"
```

That means:

```text
sonnet?structured=tool_use
```

produces a single structured answer using only the synthetic
`return_structured_output` tool.

## Using regular tools with Anthropic `tool_use`

If you want regular tools to run before the final structured answer, choose
`defer` explicitly:

```text
sonnet?structured=tool_use&structured_tools=defer
```

This produces:

```text
turn 1:
  regular tools: yes
  structured output tool: no

turn 2:
  regular tools: no
  structured output tool: yes, forced
```

This is the recommended way to use regular tools with Anthropic legacy
`tool_use` structured output.

## Anthropic Vertex

Anthropic Vertex currently does not advertise fast-agent's direct Anthropic
structured-output beta support in this provider path. As a result, it may fall
back to the legacy `tool_use` structured-output mechanism even for models whose
first-party Anthropic equivalents support native JSON Schema mode.

Use:

```text
anthropic-vertex.claude-sonnet-4-6?structured_tools=defer
```

when regular tools should be used before a structured final answer.

Use:

```text
anthropic-vertex.claude-sonnet-4-6?structured_tools=no_tools
```

when the request should produce structured output immediately without regular
tools.

## Anthropic recommendations

| Goal | Recommended setting |
|---|---|
| Current Claude model, structured output only | `?structured=json` or default `auto` |
| Current Claude model, regular tools and schema together | `?structured=json&structured_tools=always` |
| Anthropic legacy `tool_use`, no regular tools | `?structured=tool_use` |
| Anthropic legacy `tool_use`, tools first then structured output | `?structured=tool_use&structured_tools=defer` |
| Tool-capable agent but this call should not use tools | `?structured_tools=no_tools` |

---

# OpenAI Responses and Codex Responses

The Responses-family providers use the modern schema path in fast-agent:

```python
json_mode = "schema"
```

This includes both:

- `responses`
- `codexresponses`

For these providers, structured output is represented as provider-native schema
constraints rather than prompt-only JSON instructions or legacy forced tool
output.

## Structured output behavior

Current Responses and Codex Responses models are expected to support regular
tools and structured schema constraints in the same request.

In normal `auto` mode, these models resolve to same-request behavior:

```text
turn 1:
  regular tools: yes
  structured constraints: yes
```

If the model calls a tool, the normal tool loop continues and the final answer is
validated against the schema.

Recommended default:

```python
json_mode = "schema"
structured_tool_policy = None  # auto -> same-request provider default
```

Users normally do not need `defer` for these models.

## Responses provider

Current tested Responses models:

| Model string | Resolved model | Structured mode | `structured_tools=auto` probe |
|---|---|---|---|
| `responses.gpt-5.5` | `gpt-5.5` | schema | PASS |
| `responses.gpt-5.4` | `gpt-5.4` | schema | PASS |
| `responses.gpt-5.4-mini` | `gpt-5.4-mini` | schema | PASS |
| `responses.gpt-5.4-nano` | `gpt-5.4-nano` | schema | PASS |
| `responses.gpt-5.3-chat-latest` | `gpt-5.3-chat-latest` | schema | PASS |
| `responses.gpt-5.3-codex` | `gpt-5.3-codex` | schema | PASS |
| `responses.gpt-5.2` | `gpt-5.2` | schema | PASS |

Probe result meaning:

- the model called `get_probe_payload`;
- the final answer parsed as JSON;
- the final answer validated against the schema;
- the final JSON matched the tool payload.

Example:

```bash
fast-agent check structured-tools \
  --models responses.gpt-5.5,responses.gpt-5.4,responses.gpt-5.3-codex \
  --structured-tool-policy auto \
  --json
```

## Codex Responses provider

Codex Responses uses the same schema path, but dispatches through the Codex OAuth
provider:

```text
codexresponses.<model>
```

Current tested Codex Responses aliases:

| Alias | Resolved model | Structured mode | `structured_tools=auto` probe |
|---|---|---|---|
| `codexplan` | `gpt-5.5` | schema | PASS |
| `codexplan54` | `gpt-5.4` | schema | PASS |
| `codexplan53` | `gpt-5.3-codex` | schema | PASS |
| `codexspark` | `gpt-5.3-codex-spark` | schema | PASS |

Example:

```bash
fast-agent check structured-tools \
  --models codexplan,codexplan54,codexplan53,codexspark \
  --structured-tool-policy auto \
  --json
```

## Removed unsupported Codex aliases

The following older Codex Responses aliases/models were removed after live
testing showed they are not supported with the current ChatGPT/Codex account
path:

- `codexplan52`
- `codexplan51`
- `gpt-5.2-codex`
- `gpt-5.1-codex`

Observed provider errors:

```text
The 'gpt-5.2-codex' model is not supported when using Codex with a ChatGPT account.
The 'gpt-5.1-codex' model is not supported when using Codex with a ChatGPT account.
```

Use one of the current aliases instead:

```text
codexplan
codexplan54
codexplan53
codexspark
```

## Responses recommendations

| Goal | Recommended setting |
|---|---|
| Structured output only | default `auto` or `?structured_tools=no_tools` if tools are present but should be ignored |
| Structured output with regular tools | default `auto` |
| Force same-request schema + tools | `?structured_tools=always` |
| Tool-first/two-phase behavior | usually unnecessary; use `?structured_tools=defer` only for experimentation or a specific gateway issue |

---

# xAI Grok

xAI Grok models use the provider-native JSON Schema path in fast-agent:

```python
json_mode = "schema"
```

The current Grok models tested support regular tools and structured schema
constraints in the same request.

Recommended default:

```python
json_mode = "schema"
structured_tool_policy = None  # auto -> same-request provider default
```

## Current xAI probe results

The `--json-schema` CLI path was smoke-tested with a schema containing an
optional nullable field with `default: null`:

```bash
fast-agent go \
  --model xai.grok-4.3 \
  --message "Return answer='ok' and context=null." \
  --json-schema /tmp/fast-agent-xai-structured-smoke.schema.json \
  --quiet
```

Observed output:

```json
{"answer": "ok", "context": null}
```

Structured output with regular tools was tested with:

```bash
fast-agent check structured-tools \
  --models xai.grok-4-fast-non-reasoning,xai.grok-4-fast-reasoning,xai.grok-4-1-fast-non-reasoning,xai.grok-4-1-fast-reasoning \
  --structured-tool-policy always \
  --json
```

and individually for the 4.3 aliases.

| Model string | Resolved model | Structured mode | `structured_tools=always` probe |
|---|---|---|---|
| `xai.grok-4.3` | `grok-4.3` | schema | PASS |
| `xai.grok-4.3-latest` | `grok-4.3-latest` | schema | PASS |
| `xai.grok-4-fast-non-reasoning` | `grok-4-fast-non-reasoning` | schema | PASS |
| `xai.grok-4-fast-reasoning` | `grok-4-fast-reasoning` | schema | PASS |
| `xai.grok-4-1-fast-non-reasoning` | `grok-4-1-fast-non-reasoning` | schema | PASS |
| `xai.grok-4-1-fast-reasoning` | `grok-4-1-fast-reasoning` | schema | PASS |

Additional one-shot `--json-schema` smokes passed for:

- `xai.grok-4-fast-non-reasoning`
- `xai.grok-4-1-fast-non-reasoning`
- `xai.grok-4.3`
- `xai.grok-4.3-latest`

There is currently no `grok-4.2` / `grok-4-2` entry in the fast-agent model
catalog.

## xAI recommendations

| Goal | Recommended setting |
|---|---|
| Structured output only | default `auto` or `?structured_tools=no_tools` if tools are present but should be ignored |
| Structured output with regular tools | default `auto` |
| Force same-request schema + tools | `?structured_tools=always` |
| Tool-first/two-phase behavior | unnecessary based on current probe results |

---

# Google native Gemini

Google native uses the `google.genai` `GenerateContentConfig` structured output
surface:

```python
config.response_mime_type = "application/json"
config.response_schema = schema
```

fast-agent records current Gemini text/vision models as `json_mode="schema"`.

## Current native Gemini aliases

As of 2026-05-03, `google.genai` model listing showed these Gemini model IDs as
active for `generateContent` and relevant to the text/chat path:

| Alias | Model |
|---|---|
| `gemini` | `gemini-3.1-pro-preview` |
| `gemini3.1` | `gemini-3.1-pro-preview` |
| `gemini3.1flashlite` | `gemini-3.1-flash-lite-preview` |
| `gemini3` | `gemini-3-pro-preview` |
| `gemini3flash` | `gemini-3-flash-preview` |
| `gemini25` | `gemini-2.5-flash` |
| `gemini25pro` | `gemini-2.5-pro` |
| `gemini2` | `gemini-2.0-flash` |

Removed inactive dated/legacy Gemini 2.5 preview IDs from the native Google
catalog:

- `gemini-2.5-flash-preview`
- `gemini-2.5-pro-preview`
- `gemini-2.5-flash-preview-05-20`
- `gemini-2.5-pro-preview-05-06`
- `gemini-2.5-flash-preview-09-2025`

## Structured output with tools

Google native supports combining `response_schema` and regular function tools in
the same request for some, but not all, current Gemini models. Direct fast-agent
probes produced this matrix:

| Model | `always` | `defer` | Default policy | Notes |
|---|---:|---:|---|---|
| `gemini-3.1-pro-preview` | pass | — | `always` | Same-request schema + tools worked. |
| `gemini-3-pro-preview` | pass | — | `always` | Same-request schema + tools worked. |
| `gemini-3-flash-preview` | pass | — | `always` | Same-request schema + tools worked. |
| `gemini-3.1-flash-lite-preview` | fail | pass | `no_tools` | Repeated tool calls instead of finalizing under `always`. |
| `gemini-2.5-pro` | fail | pass | `no_tools` | API rejected function calling with `application/json` response MIME type. |
| `gemini-2.5-flash` | fail | pass | `no_tools` | API rejected function calling with `application/json` response MIME type. |
| `gemini-2.0-flash` | fail | not probed | `no_tools` | Repeated tool calls instead of finalizing under `always`. |

fast-agent allows Google native structured requests to keep tools when the
resolved structured-tools policy is `always`, but models that failed the
same-request probe are registered with `structured_tool_policy="no_tools"` so
default `auto` remains a structured single call. Use `structured_tools=defer`
when tool-informed structured output is required for those models.

`no_tools` still suppresses regular tools for a structured-only answer, and
`defer` remains available as a two-phase tools-first mode:

```text
google.gemini-3-flash-preview?structured_tools=no_tools
google.gemini-3-flash-preview?structured_tools=defer
google.gemini-3.1-flash-lite-preview?structured_tools=defer
```

## Google recommendations

| Goal | Recommended setting |
|---|---|
| Structured output only | default `auto`, or `?structured_tools=no_tools` if regular tools are configured but should be ignored |
| Structured output with regular tools | default `auto` |
| Force same-request schema + tools | `?structured_tools=always` |
| Tool-first/two-phase behavior | `?structured_tools=defer` |

---

# Hugging Face Inference Providers

Hugging Face routed models are the most variable structured-output surface in
fast-agent. The same model family can behave differently depending on the
selected inference provider (`novita`, `fireworks-ai`, `together`, `cerebras`,
and so on).

For this reason, HF support should be maintained as a measured compatibility
matrix rather than guessed from model family alone.

## Two-pass HF compatibility process

### Pass 1: choose the structured mode

For each model route, test structured output without regular tools in this
order:

1. `json_mode="schema"`
2. `json_mode="object"`
3. `json_mode=None`

Choose the strongest mode that:

- does not produce a provider error;
- returns parseable JSON;
- validates against the schema;
- matches the requested payload.

### Pass 2: choose the structured/tools policy

Using the recommended mode from pass 1:

1. test `structured_tools=always`;
2. if that fails, test `structured_tools=defer` as an explicit tools-first mode;
3. if `always` fails, use `no_tools` as the default policy so `auto` remains a
   single structured call unless the user explicitly asks for `defer`.

## Matrix probe script

The work-in-progress matrix probe lives at:

```bash
scripts/probe_structured_support_matrix.py
```

Example:

```bash
uv run scripts/probe_structured_support_matrix.py \
  --models kimi26,qwen35,glm51,minimax25,gpt-oss,gpt-oss-20b,deepseek4pro
```

The script:

- temporarily tests each model with `schema`, `object`, and prompt/local mode;
- recommends a `json_mode`;
- tests `always`;
- tests `defer` when `always` fails;
- prints a table and JSON payload.

## Initial HF probe results

These results are from quick live probes and should be treated as an initial
support matrix, not a permanent guarantee. Provider routes change frequently.

| Model | Route | Recommended structured mode | `always` | `defer` | Default policy | Tool-informed policy | Notes |
|---|---|---:|---:|---:|---|---|---|
| `hf.moonshotai/Kimi-K2-Instruct-0905:novita` | Novita | `schema` | pass | — | `always` | `always` | Older Kimi route is still live on Novita. |
| `hf.moonshotai/Kimi-K2-Thinking:novita` | Novita | `schema` | fail | pass | `no_tools` | `defer` | Older thinking route is still live; use `defer` when tools should inform output. |
| `kimi25` | `moonshotai/Kimi-K2.5:novita` | `schema` | pass | — | `always` | `always` | Schema appears viable on Novita. |
| `kimi26` | `moonshotai/Kimi-K2.6:novita` | `schema` | fail | pass | `no_tools` | `defer` | Use `defer` explicitly for tool-informed structured output. |
| `qwen35` | `Qwen/Qwen3.5-397B-A17B:novita` | `schema` | fail | pass | `no_tools` | `defer` | Schema appears viable; default suppresses tools. |
| `glm51` | `zai-org/GLM-5.1:together` | `schema` | fail | pass | `no_tools` | `defer` | Provider rejected combined response format + tools under `always`. |
| `minimax25` | `MiniMaxAI/MiniMax-M2.5:fireworks-ai` | `schema` | fail | pass | `no_tools` | `defer` | Provider rejected combined response format + tools under `always`. |
| `gpt-oss` | `openai/gpt-oss-120b:cerebras` | `schema` | fail | pass | `no_tools` | `defer` | Provider rejected `tools` with `response_format` under `always`. |
| `gpt-oss-20b` | default HF route | `schema` | fail | pass | `no_tools` | `defer` | Provider rejected JSON mode with tool/function calling under `always`. |
| `deepseek4pro` | `deepseek-ai/DeepSeek-V4-Pro:fireworks-ai` | `schema` | fail | pass | `no_tools` | `defer` | Use `defer` explicitly for tool-informed structured output. |

Additional legacy checks:

| Model | Recommended structured mode | `always` | `defer` | Recommended policy | Notes |
|---|---:|---:|---:|---|---|
| `glm5` | `schema` | pass | — | `always` | Older GLM route still works in same-request mode. |
| `minimax21` | `schema` | pass | — | `always` | Older MiniMax route still works in same-request mode. |
| `glm47` | `schema` | fail | fail | `no_tools` pending investigation | Failure involved provider rejecting replayed `reasoning_content`; may need provider-specific cleanup rather than a model capability change. |
| `deepseek31` | `schema` | fail | fail | `no_tools` pending investigation | Same caveat: failure involved provider/request replay behavior. |
| `deepseek32` | `schema` | fail | fail | `no_tools` pending investigation | Same caveat: failure involved provider/request replay behavior. |

## HF availability observations

Recent provider lookup showed:

- `moonshotai/Kimi-K2-Instruct-0905` is still live on `novita` and
  `featherless-ai`;
- `moonshotai/Kimi-K2-Thinking` is still live on `novita` and `featherless-ai`;
- `moonshotai/Kimi-K2` and `moonshotai/Kimi-K2-Thinking-0905` had no active HF
  providers;
- older GLM, MiniMax, DeepSeek, Qwen, and GPT-OSS routes are mostly still live.

This suggests HF cleanup should mostly distinguish current vs legacy aliases
rather than deleting older models just because they are no longer preferred.

## HF recommendations

| Situation | Recommendation |
|---|---|
| New/current HF route | run the two-pass matrix before changing catalog metadata |
| `always` rejects `tools` + `response_format` | set `structured_tool_policy="no_tools"` as the default; document that `defer` works when tool-informed output is required |
| schema passes without tools | prefer `json_mode="schema"` unless route-specific tool probing shows problems |
| schema fails but object passes | use `json_mode="object"` |
| native modes are unreliable | use `json_mode=None` and local validation |
| route is live but no longer preferred | keep metadata if useful, but mark selector alias non-current |
