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
fast-agent --model "gemini37flash?reasoning=medium"
fast-agent --model "gemini3?reasoning=auto"
fast-agent --model "google.gemini-3.1-pro-preview?reasoning=high"
fast-agent --model "gemini3?web_search=on"
fast-agent --model "google.gemini-3.7-flash?web_search=on"
fast-agent --model "gemini37flash?service_tier=flex"
```

Useful query parameters:

- `reasoning=auto|minimal|low|medium|high|off` where the selected Gemini model advertises thinking
  controls
- `web_search=on|off` for Grounding with Google Search on supported Gemini models
- `structured=json` for JSON schema structured outputs
- sampling controls such as `temperature`, `top_p`, and `top_k` where applicable

Gemini 3.7 Flash is a GA coding and agent model with a 1M-token context window, 64K maximum
output, and `low`, `medium`, or `high` thinking levels (`medium` by default). It does not accept
sampling parameters, numeric thinking budgets, or `candidate_count`; fast-agent omits those fields
from native requests. Introductory pricing through December 31, 2026 is $0.75 per million input
tokens and $3.75 per million output tokens.

## Flex inference

Gemini 3.7 Flash supports Google's synchronous, best-effort Flex tier:

```bash
fast-agent --model "gemini37flash?service_tier=flex"
```

Or configure Flex as the Google default:

```yaml
google:
  api_key: "your_google_key"
  service_tier: flex
```

In the interactive prompt, press **Shift+Tab** to cycle between Standard and Flex. The existing
`/fast flex` command is a shortcut for `/model fast flex`; `/fast off` returns to Standard.
The command name predates Gemini support, so Flex remains available even though it is intentionally
slower than the default tier.

Flex targets latency-tolerant work and may take 1–15 minutes. Fast-agent uses a 900-second
stream/request timeout by default while Flex is selected; an explicit `streaming_timeout` query
still takes precedence. Capacity errors such as HTTP 503 remain retryable.

Developer API requests use Google's native `serviceTier: flex` field. Vertex Flex requires the
`global` location and uses the documented Vertex shared/Flex routing headers with API version `v1`.

| Period | Input | Output including thinking | Cached input | Cache storage |
| --- | ---: | ---: | ---: | ---: |
| Through Dec 31, 2026 | $0.375/M | $1.875/M | $0.0375/M | $0.50/M tokens/hour |
| From Jan 1, 2027 | $0.75/M | $3.75/M | $0.075/M | $1.00/M tokens/hour |

When `web_search=on`, fast-agent sends Google's native `GoogleSearch` tool and formats grounding
metadata as citations in the text response when Google returns citation spans. Search can be toggled
from the `/model web_search` command or the interactive model controls on models that advertise
support.

Current fast-agent metadata marks Google Search as supported on Gemini 2.0 Flash, Gemini 2.5, and
Gemini 3.x aliases, including Gemini 3.7 Flash. Check
[Models Reference](../models_reference/) for the generated capability view.

**Model Name Aliases:**

--8<-- "_generated/model_aliases_google.md"

### OpenAI Mode

You can also access Google via the OpenAI-compatible provider. Use `googleoai` in the YAML file, or `GOOGLEOAI_API_KEY` for API key access; that path uses the Google OpenAI-compatible endpoint by default.
