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
The native catalog includes GLM-5.3, GLM-5.3-Flash, and GLM-5.2.

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

The default endpoint is Z.ai's standard API. GLM Coding Plan users can select
its Chat Completions endpoint without changing providers:

```yaml
zai:
  api_key: "${ZAI_API_KEY}"
  base_url: "https://api.z.ai/api/coding/paas/v4/"
```

## Models

| Model | Native alias | Input | Context | Maximum output |
|---|---|---|---:|---:|
| GLM-5.3 | `zaiglm53` | Text | 1,000,000 | 131,072 |
| GLM-5.3-Flash | `zaiglm53flash` | Text, image, linked MOV/PDF | 1,000,000 | 131,072 |
| GLM-5.2 | `zaiglm` | Text | 1,000,000 | 131,072 |

Use either an alias, bare canonical model ID, or provider-qualified ID:

```bash
fast-agent go --model zaiglm53 --message "Review this code."
fast-agent go --model glm-5.3 --message "Review this code."
fast-agent go --model zai.glm-5.3 --message "Review this code."

fast-agent go --model zaiglm53flash --message "Describe the attached image."
fast-agent go --model glm-5.3-flash --message "Describe the attached image."
fast-agent go --model zai.glm-5.3-flash --message "Describe the attached image."
```

For compatibility, the provider default and `zaiglm` alias remain GLM-5.2.
The older `glm` and `glm52` aliases also retain their Hugging Face routes:

```text
zaiglm53       -> zai.glm-5.3
zaiglm53flash  -> zai.glm-5.3-flash
zaiglm         -> zai.glm-5.2
glm            -> hf.zai-org/GLM-5.2:zai-org
glm52          -> hf.zai-org/GLM-5.2:zai-org
```

## GLM-5.3

GLM-5.3 is text-only. It supports streaming, function calling, context
caching, separate `reasoning_content`, and JSON object structured output.

Reasoning is always enabled. Select one of the three documented efforts:

```bash
fast-agent go --model "zaiglm53?reasoning=low" --message "Solve this."
fast-agent go --model "zaiglm53?reasoning=high" --message "Solve this."
fast-agent go --model "zaiglm53?reasoning=max" --message "Solve this."
```

`max` is the default. GLM-5.3 rejects disabled thinking and unsupported effort
values such as `none` and `medium`.

### Sampling

Z.ai defaults GLM-5.3 sampling to `temperature=1.0` and `top_p=0.95`.
fast-agent leaves both fields unset unless requested, allowing the provider
defaults to apply. Z.ai recommends tuning only one sampling control at a time:

```bash
fast-agent go --model "zaiglm53?temperature=0.8" --message "Write an introduction."
fast-agent go --model "zaiglm53?top_p=0.8" --message "Write stable technical documentation."
```

The same sampling defaults and recommendation apply to GLM-5.3-Flash.

## GLM-5.3-Flash

GLM-5.3-Flash uses the same forced-reasoning contract and adds native
multimodal input. fast-agent supports the model guide's JPEG and PNG inputs as
remote URLs or Base64 data URLs, including multiple images:

```bash
fast-agent go --model zaiglm53flash
```

Then attach the images in the interactive prompt:

```text
/attach https://example.test/first.png
/attach https://example.test/second.jpg
Compare these screenshots.
```

Z.ai limits images to JPG/JPEG/PNG, less than 5 MB each, and at most
6000×6000 pixels. WebP and GIF are not advertised for this profile.

The current Chat Completions API also accepts `video_url` and `file_url`
content blocks for GLM-5.3-Flash. On August 26, 2026, fast-agent onboarding
verified remote MOV (`video/quicktime`) and PDF links through the standard
endpoint. These are therefore available as linked attachments. Embedded/Base64
video and document input, other video formats, and Office document input are
not advertised without an equally specific contract.

## Reasoning and tool continuations

Both GLM-5.3 models use preserved thinking:

```json
{
  "thinking": {
    "type": "enabled",
    "clear_thinking": false
  },
  "reasoning_effort": "max"
}
```

fast-agent keeps hidden reasoning separate from visible assistant content. For
later turns—including tool-result continuations—it returns the complete
assistant `reasoning_content` unchanged and in its original order, as required
by Z.ai.

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

For streamed requests containing tools, fast-agent sends `tool_stream: true`.
The non-streaming fallback removes only that streaming extension.

## Structured output

All three native profiles use Z.ai's JSON object mode plus schema instructions
and fast-agent validation:

```bash
fast-agent go --model zaiglm53 \
  --json-schema ./result.schema.json \
  --message "Return the requested result."
```

This is not strict OpenAI `json_schema` mode.

## GLM-5.2 compatibility

GLM-5.2 retains its existing reasoning controls: `none`, `minimal`, `low`,
`medium`, `high`, `xhigh`, and `max`. Unlike the 5.3 models, `none` disables
thinking. GLM-5.2 remains text-only through the native profile.

## Official documentation

- [GLM-5.3](https://docs.z.ai/guides/llm/glm-5.3)
- [GLM-5.3-Flash](https://docs.z.ai/guides/vlm/glm-5.3-flash)
- [GLM-5.2](https://docs.z.ai/guides/llm/glm-5.2)
- [Chat Completion API](https://docs.z.ai/api-reference/llm/chat-completion)
- [Thinking Mode](https://docs.z.ai/guides/capabilities/thinking-mode)
- [Streaming Messages](https://docs.z.ai/guides/capabilities/streaming)
- [Function Calling](https://docs.z.ai/guides/capabilities/function-calling)
- [Structured Output](https://docs.z.ai/guides/capabilities/struct-output)

See [Models Reference](../models_reference/) for the generated capability row.
