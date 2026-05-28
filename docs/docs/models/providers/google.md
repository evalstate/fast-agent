---
title: Google
social:
  title: Google
  tagline: Configure Gemini models with the native Google provider.
  description: Configure Gemini models with the native Google provider.
  alt: fast-agent social card — Google
---

# Google

Google is natively supported in `fast-agent` using the Google GenAI libraries.

Google models have support for attaching YouTube URLs for video and transcript understanding.
Gemini models that support Grounding with Google Search can also use provider-side web search via
the standard `web_search` model-string option.

**YAML Configuration:**

```yaml
google:
  api_key: "your_google_key"
```

**Environment Variables:**

- `GOOGLE_API_KEY`: Your Google API key

## Reasoning, search, and multimodal input

Google models support model-dependent thinking controls, structured outputs, multimodal inputs, and
Grounding with Google Search.

```bash
fast-agent --model gemini
fast-agent --model "gemini3?reasoning=auto"
fast-agent --model "google.gemini-3.1-pro-preview?reasoning=high"
fast-agent --model "gemini3?web_search=on"
fast-agent --model "google.gemini-3.5-flash?web_search=on"
```

Useful query parameters:

- `reasoning=auto|minimal|low|medium|high|off` where the selected Gemini model advertises thinking
  controls
- `web_search=on|off` for Grounding with Google Search on supported Gemini models
- `structured=json` for JSON schema structured outputs
- sampling controls such as `temperature`, `top_p`, and `top_k` where applicable

When `web_search=on`, fast-agent sends Google's native `GoogleSearch` tool and formats grounding
metadata as citations in the text response when Google returns citation spans. Search can be toggled
from the `/model web_search` command or the interactive model controls on models that advertise
support.

Current fast-agent metadata marks Google Search as supported on Gemini 2.0 Flash, Gemini 2.5, and
Gemini 3 / 3.5 aliases. Check [Models Reference](../models_reference/) for the generated capability
view.

**Model Name Aliases:**

--8<-- "_generated/model_aliases_google.md"

### OpenAI Mode

You can also access Google via the OpenAI-compatible provider. Use `googleoai` in the YAML file, or `GOOGLEOAI_API_KEY` for API key access; that path uses the Google OpenAI-compatible endpoint by default.
