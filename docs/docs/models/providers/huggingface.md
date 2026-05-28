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

## Kimi instant mode

Kimi models that support instant mode can disable reasoning with the `instant` query parameter:

```bash
fast-agent --model "hf.moonshotai/Kimi-K2.5?instant=on"  # thinking disabled
fast-agent --model "hf.moonshotai/Kimi-K2.5?instant=off" # thinking enabled
```

## Hugging Face MCP authentication

`HF_TOKEN` is automatically applied when connecting to Hugging Face MCP servers:

- `hf.co` / `huggingface.co` uses `Authorization: Bearer {HF_TOKEN}`
- `*.hf.space` uses `X-HF-Authorization: Bearer {HF_TOKEN}`

## Model aliases

--8<-- "_generated/model_aliases_hf.md"
