---
title: MetaAI
description: Configure MetaAI Muse Spark models in fast-agent
---

# MetaAI

Use the `metaai` provider for MetaAI Muse Spark models through the Responses API.

## Configuration

```yaml
metaai:
  api_key: "${META_AI_API_KEY}"
  base_url: "https://api.meta.ai/v1"
```

Environment variables:

- `META_AI_API_KEY`: Your MetaAI API key
- `META_AI_BASE_URL`: Override the API endpoint

## Models

```bash
fast-agent --model "metaai.muse-spark-1.1"
```

Muse Spark 1.1 supports text, image, video, and PDF input with a 1,048,576 token context window.

### Current models

--8<-- "_generated/current_models_metaai.md"

### Model aliases

--8<-- "_generated/model_aliases_metaai.md"
