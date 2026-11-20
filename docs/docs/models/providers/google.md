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

The generic `gemini` alias selects Gemini 3.8 Flash (`gemini38`, `gemini38flash`, or
`gemini3.8flash`). Explicit older aliases, including `gemini37flash`, `gemini35flash`, and
`gemini3flash`, retain their original models.

Gemini 3.8 Flash capabilities were verified against the official [Gemini model page](https://ai.google.dev/gemini-api/docs/models/gemini-3.8-flash),
[Vertex model page](https://cloud.google.com/vertex-ai/generative-ai/docs/models/gemini/3-8-flash),
[migration guide](https://ai.google.dev/gemini-api/docs/latest-model), and
[pricing](https://ai.google.dev/gemini-api/docs/pricing#gemini-3.8-flash) on September 5, 2026.

## Gemini 4 (native API)

| Native model | Short alias | Thinking levels | Default thinking |
| --- | --- | --- | --- |
| [`google.gemini-4-neon`](https://aistudio.google.com/docs/models/gemini-4-neon) | `neon` | low, medium, high, xhigh | high |
| [`google.gemini-4-argon`](https://aistudio.google.com/docs/models/gemini-4-argon) | `argon` | low, medium, high | medium |
| [`google.gemini-4-hydrogen`](https://aistudio.google.com/docs/models/gemini-4-hydrogen) | `hydrogen` | low, medium, high, xhigh | high |
| [`google.gemini-4-carbon`](https://aistudio.google.com/docs/models/gemini-4-carbon) | `carbon` | low, medium, high | medium |

```bash
fast-agent go --model 'neon?reasoning=xhigh'
fast-agent go --model 'argon?reasoning=low'
fast-agent go --model 'google.gemini-4-hydrogen?reasoning=xhigh'
fast-agent go --model 'carbon?web_search=on'
```

All four accept text, images, audio, video and PDFs, return text, and support function calling,
JSON Schema output, Google Search grounding and Flex. They share a **1,048,576-token window
for input and output combined**. fast-agent uses a 65,536-token output limit by default;
a single non-streaming hop on Argon or Carbon is limited to 32,768. The separate
[long decode](https://aistudio.google.com/docs/long-decode?generate-content=true) capability
does not make 1M the default output budget. Neon is paid-tier only (no Free tier). Use a named [thinking level](https://aistudio.google.com/docs/thinking)
rather than sampling controls or a numeric thinking budget.

The native provider uses `generateContent` / `streamGenerateContent`, not a chat-completions
proxy or a server-stored Interactions session. fast-agent keeps the Gemini Developer API on
`v1beta` for compatibility with preview models and beta features. To opt into a different
version, set `GOOGLE_GENAI_API_VERSION` (subject to model availability on that version).
Copilot routes have [separate credentials and capabilities](copilot.md).

**Tracing:** The current OpenTelemetry Google instrumentor requires `google-genai <3`;
it skips SDK-level Google spans with SDK 3.4. fast-agent's agent and MCP spans do not
depend on that instrumentor.

## Reasoning, search, and multimodal input

Google models support model-dependent thinking controls, structured outputs, multimodal inputs, and
Grounding with Google Search.

```bash
fast-agent --model gemini
fast-agent --model "gemini38flash?reasoning=medium"
fast-agent --model "gemini3?reasoning=auto"
fast-agent --model "google.gemini-3.1-pro-preview?reasoning=high"
fast-agent --model "gemini3?web_search=on"
fast-agent --model "google.gemini-3.8-flash?web_search=on"
fast-agent --model "gemini38flash?service_tier=flex"
```

Useful query parameters:

- `reasoning=auto|minimal|low|medium|high|xhigh|off` where the selected Gemini model advertises thinking
  controls
- `web_search=on|off` for Grounding with Google Search on supported Gemini models
- `structured=json` for JSON schema structured outputs
- sampling controls such as `temperature`, `top_p`, and `top_k` where applicable

Gemini 3.8 Flash is a GA coding and agent model with a 1M-token context window, 64K maximum
output, and `low`, `medium`, or `high` thinking levels (`medium` by default). It does not accept
sampling parameters, numeric thinking budgets, or `candidate_count`; fast-agent omits those fields
from native requests. Introductory pricing through December 31, 2026 is $0.75 per million input
tokens and $3.75 per million output tokens.

## Flex inference

Gemini 3.8 Flash supports Google's synchronous, best-effort Flex tier:

```bash
fast-agent --model "gemini38flash?service_tier=flex"
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
Gemini 3.x aliases, including Gemini 3.8 Flash. Check
[Models Reference](../models_reference/) for the generated capability view.

**Model Name Aliases:**

--8<-- "_generated/model_aliases_google.md"

### OpenAI Mode

You can also access Google via the OpenAI-compatible provider. Use `googleoai` in the YAML file, or `GOOGLEOAI_API_KEY` for API key access; that path uses the Google OpenAI-compatible endpoint by default.
