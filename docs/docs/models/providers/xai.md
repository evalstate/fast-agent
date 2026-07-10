---
title: xAI
social:
  title: xAI
  tagline: Configure Grok models, reasoning, web search, and X Search in fast-agent.
  description: Configure Grok models, reasoning, web search, and X Search in fast-agent.
  alt: fast-agent social card — xAI / Grok provider
---

# xAI / Grok

Use the `xai` provider for xAI Grok models. xAI supports both `web_search` and `x_search`; fast-agent sends `x_search` as xAI's provider-managed X Search tool.

## Configure

```yaml
xai:
  api_key: "${XAI_API_KEY}"
  # base_url: "https://api.x.ai/v1" # default
```

Environment variables:

- `XAI_API_KEY`: Your xAI API key
- `XAI_BASE_URL`: Override the API endpoint

## Use a model

```bash
fast-agent --model "xai.grok-4.3?reasoning=high"
fast-agent --model "xai.grok-4.3?web_search=on"
fast-agent --model "xai.grok-4.3?x_search=on"
fast-agent --model "xai.grok-4.5"
```

## Reasoning and search tools

Useful xAI query parameters:

- `reasoning=none|low|medium|high` on reasoning-capable Grok models
- `web_search=on|off` for xAI web search
- `x_search=on|off` for xAI's X Search remote tool

`web_search` and `x_search` are distinct provider-managed tools.

## Capabilities

Capabilities are model-dependent. See [Models Reference](../models_reference/) for fast-agent's known structured output, reasoning, modality, and tool metadata.

## Model aliases

--8<-- "_generated/model_aliases_xai.md"
