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
  web_search:
    enabled: false
    search_context_size: medium  # Optional: low | medium | high
    # user_location:  # Optional approximate locale bias
    #   type: approximate
    #   country: "GB"
    #   region: "London"
    #   city: "London"
    #   timezone: "Europe/London"
```

Environment variables:

- `META_AI_API_KEY`: Your MetaAI API key
- `META_AI_BASE_URL`: Override the API endpoint

## Models

```bash
fast-agent --model "metaai.muse-spark-1.1"
fast-agent --model "metaai.muse-spark-1.1?web_search=on"
```

Muse Spark 1.1 supports text, image, video, and PDF input with a 1,048,576 token context window.

## Search grounding

MetaAI search grounding is exposed as the same Responses-family `web_search` tool
used elsewhere in fast-agent:

- Config default: `metaai.web_search.enabled`
- Model string toggle: `?web_search=on|off`
- Interactive toggle: `/model web_search` and the F8 toolbar control

When enabled, fast-agent sends `tools=[{"type":"web_search", ...}]` and requests
`include: ["web_search_call.results"]` so raw search hits are available alongside
`url_citation` annotations on the answer.

Optional tool fields from config:

- `search_context_size`: `low` | `medium` | `high`
- `user_location`: approximate country/region/city/timezone bias

### Current models

--8<-- "_generated/current_models_metaai.md"

### Model aliases

--8<-- "_generated/model_aliases_metaai.md"
