---
title: DeepSeek
social:
  title: DeepSeek
  tagline: Configure DeepSeek models, credentials, and model aliases in fast-agent.
  description: Configure DeepSeek models, credentials, and model aliases in fast-agent.
  alt: fast-agent social card — DeepSeek
---

# DeepSeek

Use the `deepseek` provider for fast-agent's native DeepSeek Responses route.
It uses a stateless Responses API over SSE and currently supports
`deepseek-v4-flash`.

## Setup

Create a key in the [DeepSeek platform](https://platform.deepseek.com/) and set:

```bash
export DEEPSEEK_API_KEY="..."
```

Then run the built-in alias:

```bash
fast-agent go --model deepseek \
  --message "Explain why 37 × 41 equals 1517."
```

The explicit model string is:

```text
deepseek.deepseek-v4-flash
```

The native provider rejects other model names. In particular,
`deepseek-chat`, `deepseek-reasoner`, and `deepseek-v4-pro` are not native
DeepSeek model strings in fast-agent.

## Configuration

Only the API key is required:

```yaml
deepseek:
  api_key: "${DEEPSEEK_API_KEY}"
```

The complete provider shape is:

```yaml
deepseek:
  api_key: "${DEEPSEEK_API_KEY}"
  base_url: "https://api.deepseek.com"
  default_model: "deepseek-v4-flash"
  reasoning: "max"
  web_search:
    enabled: false
  # default_headers:
  #   X-Custom-Header: value
```

`base_url`, `default_model`, and `default_headers` are optional. A configured
`default_model` must currently be `deepseek-v4-flash`.

Run `fast-agent check` after configuring credentials.

## Reasoning

Reasoning defaults to `max`. Select an effort in the model string:

```bash
fast-agent go --model "deepseek?reasoning=none" --message "Answer directly."
fast-agent go --model "deepseek?reasoning=low" --message "Solve this problem."
fast-agent go --model "deepseek?reasoning=high" --message "Solve this problem."
fast-agent go --model "deepseek?reasoning=max" --message "Solve this problem."
```

Supported values are `none`, `low`, `high`, and `max`. DeepSeek returns
reasoning separately from visible assistant text; fast-agent preserves it in
the reasoning channel and replays it when continuing a tool-use turn.

`max_output_tokens` includes hidden reasoning. Leave enough output headroom
when reasoning is enabled rather than treating the setting as a visible-text
budget.

## Tools, structured output, and web search

The native route supports:

- function tools;
- JSON Schema structured output;
- provider-managed web search.

Enable web search in configuration:

```yaml
deepseek:
  web_search:
    enabled: true
```

Or enable it for one model selection:

```bash
fast-agent go --model "deepseek?web_search=true"
```

DeepSeek currently accepts the web-search enablement toggle. Generic
OpenAI-style search context, domain, and location options are not forwarded by
the adapter.

When forcing a specific function with `tool_choice`, use `reasoning=none`.
Automatic tool selection supports reasoning.

## Stateless Responses behavior

DeepSeek's route differs from OpenAI's stateful Responses API:

- requests use SSE; WebSocket transport is not supported;
- server-side response storage and continuation are not used;
- service tiers are not supported;
- image, PDF, audio, video, and file inputs are not supported;
- OpenAI-only request fields such as `include`, `parallel_tool_calls`,
  `service_tier`, and `store` are omitted.

Conversation and tool continuation still work because fast-agent sends the
required history on each stateless request.

## Hugging Face routes are separate

Aliases such as `deepseek-hf`, `deepseek4-hf`, and `deepseek4pro-hf` use
[Hugging Face Inference Providers](huggingface.md), not the native DeepSeek API.
They use `HF_TOKEN`, provider-specific routing, and their own capability
metadata.

Use `deepseek` when you want the native DeepSeek Responses route. Use an `hf.`
model string or a `deepseek-*-hf` alias when you want a Hugging Face-hosted
route.

## Model aliases

--8<-- "_generated/model_aliases_deepseek.md"

## Official documentation

- [DeepSeek platform](https://platform.deepseek.com/)
- [DeepSeek API documentation](https://api-docs.deepseek.com/)

See [Models Reference](../models_reference/) for the generated capability row,
context limit, output limit, and supported input modalities.
