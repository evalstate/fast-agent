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

### Streaming channel normalization

Z.ai streaming deltas can carry `reasoning_content`, visible `content`, and
`tool_calls`. fast-agent processes each field independently rather than treating
them as mutually exclusive:

- reasoning fragments remain in the structured reasoning channel;
- visible content fragments remain in assistant content;
- tool arguments are grouped and concatenated by tool-call index.

This preserves arrival order within each channel and supports alternating or
same-chunk reasoning, content, and tool deltas. fast-agent does not merge hidden
reasoning into visible assistant text or assume that the provider must finish
all reasoning before emitting another delta type.



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
