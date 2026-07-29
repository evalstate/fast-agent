---
title: Z.ai
social:
  title: Z.ai
  tagline: Configure native GLM models, reasoning streams, tools, and structured output.
  description: Configure Z.ai GLM models through the native Chat Completions provider in fast-agent.
  alt: fast-agent social card — Z.ai provider
---

# Z.ai

Use the `zai` provider for Z.ai's native OpenAI-compatible Chat Completions API.
The initial native model is GLM-5.2.

## Configure

Set the environment variable:

```bash
export ZAI_API_KEY="your-api-key"
```

Or configure it in `fast-agent.secrets.yaml`:

```yaml
zai:
  api_key: "${ZAI_API_KEY}"
```

Optional provider settings:

```yaml
zai:
  api_key: "${ZAI_API_KEY}"
  base_url: "https://api.z.ai/api/paas/v4/"
  default_model: "glm-5.2"
  # default_headers:
  #   X-Custom-Header: value
```

## Use GLM-5.2

These model strings all select the native provider:

```bash
fast-agent --model zaiglm
fast-agent --model glm-5.2
fast-agent --model zai.glm-5.2
```

`zaiglm` is the explicit native-provider preset. The older `glm` and `glm52`
presets continue to route through Hugging Face for compatibility:

```text
zaiglm  -> zai.glm-5.2
glm     -> hf.zai-org/GLM-5.2:zai-org
glm52   -> hf.zai-org/GLM-5.2:zai-org
```

## GLM-5.2 capabilities

fast-agent configures GLM-5.2 with:

- 1,000,000 input tokens;
- up to 131,072 output tokens;
- text input and text output;
- streaming messages;
- separate `reasoning_content` streaming;
- function calling and streamed chat-completion tool calls;
- JSON object structured output.

The Z.ai Chat Completions endpoint also serves multimodal GLM models, but the
official GLM-5.2 model guide identifies GLM-5.2 itself as text-only. Image,
audio, video, and file attachment tests therefore belong to compatible vision
models such as GLM-5V, not this model profile.

## Reasoning

Reasoning defaults to `max`. Select an effort in the model string:

```bash
fast-agent --model "zaiglm?reasoning=minimal"
fast-agent --model "zaiglm?reasoning=medium"
fast-agent --model "zaiglm?reasoning=high"
fast-agent --model "zaiglm?reasoning=max"
fast-agent --model "zaiglm?reasoning=none"
```

Supported values are `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, and
`max`. `none` sends disabled thinking; enabled reasoning is returned separately
from visible assistant content in `reasoning_content`.

## Context caching and process polling

Z.ai performs automatic context caching and reports hits through
`usage.prompt_tokens_details.cached_tokens`. The service documentation does not
publish a fixed or configurable TTL.

In a controlled GLM-5.2 observation with independent entries and valid controls,
both prefixes were cached at 300 seconds idle, while both missed at 600 seconds
and at the longer tested delays. This is operational evidence, not a service
guarantee.

For the native Z.ai route, fast-agent therefore uses a 240-second default
managed-process wait. The margin encourages the next polling turn to occur
inside the shortest observed cache-hit window. It does not configure or assert
a 240- or 300-second provider TTL. Hugging Face GLM routes are unchanged.

## Tools and streaming

fast-agent uses the normal streaming Chat Completions tool loop:

1. GLM-5.2 streams a tool call with an ID, function name, and JSON argument
   fragments.
2. fast-agent executes the tool and appends a matching tool-result message.
3. GLM-5.2 streams the final assistant response.

For streamed requests containing tools, fast-agent sends Z.ai's
`tool_stream: true` extension. This enables incremental `function.arguments`
deltas instead of waiting for the complete tool call to be buffered. fast-agent
groups fragments by tool-call index and assembles the JSON arguments before
execution. The extension is omitted when no tools are present and removed from
the non-streaming fallback request.

To capture the provider request and raw chunks:

```bash
FAST_AGENT_LLM_TRACE=1 fast-agent go --no-home \
  --model "zaiglm?reasoning=medium" \
  --message "Explain why 17 × 19 equals 323."
```

Captures are written under `stream-debug/`:

- `*_request.json` contains the normalized provider request;
- `*_chunks.jsonl` contains each streamed Chat Completions chunk.

Trace files can contain prompts, tool arguments, model output, and usage data.
Review and sanitize them before sharing or committing fixtures.

## Structured output

GLM-5.2 uses Z.ai's JSON object mode plus schema instructions and fast-agent
validation:

```bash
fast-agent go --model zaiglm \
  --json-schema ./result.schema.json \
  --message "Return the requested result."
```

This provider does not advertise strict OpenAI `json_schema` semantics for
GLM-5.2.

## Official documentation

- [GLM-5.2](https://docs.z.ai/guides/llm/glm-5.2)
- [Chat Completion API](https://docs.z.ai/api-reference/llm/chat-completion)
- [Thinking Mode](https://docs.z.ai/guides/capabilities/thinking-mode)
- [Streaming Messages](https://docs.z.ai/guides/capabilities/streaming)
- [Function Calling](https://docs.z.ai/guides/capabilities/function-calling)
- [Structured Output](https://docs.z.ai/guides/capabilities/struct-output)

See [Models Reference](../models_reference/) for the generated capability row.
