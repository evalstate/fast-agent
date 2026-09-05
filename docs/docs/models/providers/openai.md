---
title: OpenAI
social:
  title: OpenAI
  tagline: Configure OpenAI Responses, Chat Completions, Codex Responses, and provider-managed tools.
  description: Configure OpenAI Responses, Chat Completions, Codex Responses, and provider-managed tools.
  alt: fast-agent social card — OpenAI
---

# OpenAI

OpenAI multimodal models support text, images, and PDF input (`application/pdf`). For PDFs, provide a local file rather than a URL.

fast-agent exposes three OpenAI-facing provider paths. Use the provider prefix when you want to
force a specific API surface:

| Provider path | API surface | Use for |
| --- | --- | --- |
| `responses` | OpenAI Responses API | OpenAI API models, provider-managed tools, connectors, service tiers, WebSockets |
| `codexresponses` | Codex Responses backend | Codex subscription models |
| `openai` | Chat Completions API | Legacy Chat Completions-compatible models and deployments |

Prefer models hosted on `responses` or `codexresponses` for OpenAI API work unless a legacy model is specifically needed. 

Encrypted reasoning blocks are used to maintain model intelligence between tool calls and turns, with user-facing reasoning summaries made available.

## Feature availability by OpenAI provider

| Feature | `responses` | `codexresponses` | `openai` Chat Completions |
| --- | --- | --- | --- |
| Auth | `OPENAI_API_KEY` | `fast-agent auth provider login codex` or `CODEX_API_KEY` | `OPENAI_API_KEY` |
| API surface | OpenAI Responses API | Codex Responses backend | Chat Completions API |
| Curated aliases | `gpt55`, `gpt54`, `gpt52`, `codex`, `chatgpt` | `codexplan`, `codexplan54`, `codexspark` | `openai.gpt-4.1`, `openai.gpt-4o` |
| Structured outputs | Yes, JSON schema where model supports it | Yes, JSON schema where model supports it | Yes, model-dependent Chat Completions structured outputs |
| Reasoning controls | Yes, model-dependent `reasoning` effort | Yes for Codex planning models; Spark does not expose effort controls | Limited/model-dependent; prefer `responses` for GPT-5-class reasoning |
| Text verbosity | Yes, where advertised | Yes, where advertised | No |
| `web_search` | Yes | Yes | No |
| Provider-managed remote MCP | Yes | No | No |
| OpenAI hosted connectors | Yes | No | No |
| WebSocket transport | Yes, with SSE fallback | Yes, with SSE fallback where supported | No |
| `service_tier` | `fast` / `flex` where the model supports it | `fast` only; no `flex` | No |
| Configurable output limit | `max_tokens` | No | `max_tokens` |

## Model availability

The tables below show current recommended aliases for each OpenAI-facing provider path.
For the complete generated capability reference, see [Models Reference](../models_reference/).

### Responses models

Use `responses` for OpenAI Responses API models and for using the `flex` service tier or Remote MCP/Connectors.

#### WebSocket Support

Responses-compatible models use WebSockets as the default transport, with continuation support so repeated turns can avoid resending unchanged input items. fast-agent sends `store=false` on Responses requests. Read [compatibility with ZDR policies](https://developers.openai.com/api/docs/guides/websocket-mode#how-continuation-works) for more details.

WebSocket mode can be disabled by using `transport=sse` in the model string. 

WebSockets are kept alive for up to 55 minutes, and have a robust retry and SSE
fallback mechanisms for error recovery. 

#### Encrypted Reasoning

Reasoning summaries are displayed, with encrypted blocks stored locally for session resumption.

!!! Note

    Encrypted reasoning blocks are not transferable between API keys or credentials.


Current Responses models:

--8<-- "_generated/current_models_responses.md"

Examples:

- `responses.gpt-5.5?reasoning=medium`
- `responses.gpt-5.5?web_search=on`
- `responses.gpt-5.4?service_tier=flex`

### Codex Responses models

Use `codexresponses` for Codex subscription-backed models. Authenticate with
`fast-agent auth provider login codex` or provide `CODEX_API_KEY`.

The `codexresponses` provider is similar to `responses`, with these main differences:

- The `flex` service tier is **not supported**.
- Remote MCP and Connectors are **not supported**.
- Output token limits are **not supported**; explicit `max_tokens` settings are rejected.
- The supported model list includes `gpt-5.3-codex-spark`.
- Billing is via the Codex subscription.

Current Codex Responses models:

--8<-- "_generated/current_models_codexresponses.md"

Examples:

- `astra`
- `codexresponses.gpt-6-astra?reasoning=max`
- `codexplan`
- `codexresponses.gpt-5.5?reasoning=high`
- `codexresponses.gpt-5.3-codex-spark?web_search=on`

### Legacy Chat Completions models

!!! note "Legacy Models" 

    Use `openai` when you specifically need the legacy Chat Completions-compatible path. Prefer
    the explicit `openai.` prefix so the selected API surface is obvious.

Current legacy Chat Completions models:

--8<-- "_generated/current_models_openai.md"

Examples:

- `openai.gpt-4.1`
- `openai.gpt-4o`
- `openai.my-custom-deployment`

## Configuration

**YAML Configuration:**

```yaml
openai:
  api_key: "your_openai_key" # Default
  base_url: "https://api.openai.com/v1" # Default, only include if required
```

**Environment Variables:**

- `OPENAI_API_KEY`: Your OpenAI API key
- `OPENAI_BASE_URL`: Override the API endpoint

## Responses (OpenAI Responses API)

Use the `responses` provider for OpenAI Responses API models (for example `gpt-5`, `o3`, `o4-mini`).

```yaml
responses:
  api_key: "your_openai_key"
  base_url: "https://api.openai.com/v1" # Optional override
  reasoning: "medium" # Optional default
  text_verbosity: "medium" # Optional default for supporting models
  transport: "sse" # sse | websocket | auto
  web_search:
    enabled: false
    tool_type: web_search # web_search | web_search_preview
    # search_context_size: medium # low | medium | high
    # allowed_domains: ["openai.com", "docs.openai.com"]
    # external_web_access: false # only applies to tool_type=web_search
    # user_location:
    #   type: approximate
    #   city: "Minneapolis"
    #   region: "Minnesota"
    #   country: "US"
    #   timezone: "America/Chicago"
```

Per-run override via model string is also supported:

- `responses.gpt-5-mini?web_search=on`
- `responses.gpt-5-mini?web_search=off`
- `responses.gpt-5.3-codex?transport=ws`


**Provider-managed remote MCP and connectors:**

The OpenAI `responses` provider supports provider-managed remote MCP servers and
OpenAI hosted connectors declared with `management: provider` under
`mcp.servers` or card `mcp_connect` entries.

- Remote MCP servers must be remote `http`/`sse` URLs.
- Connector entries use `connector_id` instead of `url`. See OpenAI's
  [hosted connector documentation](https://developers.openai.com/api/docs/guides/tools-connectors-mcp?quickstart-panels=connector#available-connectors)
  for current connector behavior and authorization requirements.
- Set exactly one of `url` or `connector_id`.
- Use `access_token` for bearer auth / connector authorization.
- `defer_loading: true` enables server-side lazy tool loading.
- Not supported by `codexresponses`, Codex OAuth aliases, `openresponses`, or
  generic `openai` chat-completions models.

Connector IDs are validated against the installed OpenAI SDK. At the time this page was generated,
the accepted IDs are:

- `connector_dropbox`
- `connector_gmail`
- `connector_googlecalendar`
- `connector_googledrive`
- `connector_microsoftteams`
- `connector_outlookcalendar`
- `connector_outlookemail`
- `connector_sharepoint`

Example connector entry:

```yaml
mcp:
  servers:
    dropbox:
      management: provider
      connector_id: connector_dropbox
      access_token: "${DROPBOX_CONNECTOR_TOKEN}"
      description: "Dropbox connector"
```

See [Configuration Reference](../../ref/config_file/#mcp-server-configuration)
for the MCP server schema and
[Agent Cards](../../agents/defining/agent_cards/#runtime-mcp-targets-mcp_connect)
for card-scoped runtime targets.


## Codex (OAuth Responses)

**`fast-agent`** supports using your OpenAI Codex subscription. Run `fast-agent auth provider login codex`
once, then use a Codex OAuth model alias such as `astra` (GPT-6-Astra), `codexplan` (GPT-6-Astra, medium reasoning),
`codexplan54` (GPT-5.4 planning), `codexplan53` (GPT-5.3 Codex planning), or
`codexspark` (GPT-5.3 Codex Spark).

**Quick Start:**

```bash
# Start OAuth login (stores tokens in your OS keyring)
fast-agent auth provider login codex

# Use GPT-6-Astra through the Codex subscription
fast-agent --model astra

# Pin a previous planning model via OAuth
fast-agent --model codexplan54
```

**Provider Configuration:**

```yaml
codexresponses:
  # Optional: override defaults
  base_url: "https://chatgpt.com/backend-api/codex"
  text_verbosity: "medium"  # low | medium | high
  web_search:
    enabled: false
  default_headers:
    X-Custom-Header: "value"
```

**Environment Variables:**

- `CODEX_API_KEY`: Optional. Provide a Codex OAuth access token directly.

**Notes:**

- Tokens are stored in your OS keyring, with a secure file fallback, via `fast-agent auth provider login codex`.
- `astra` maps to `codexresponses.gpt-6-astra?reasoning=medium`.
- `gpt-6-astra` maps to the API-key-backed `responses.gpt-6-astra?reasoning=medium` route; availability still depends on the OpenAI API account's model access.
- `codexplan` maps to `codexresponses.gpt-6-astra?reasoning=medium`.
- `codexplan54` maps to `codexresponses.gpt-5.4?reasoning=high`.
- `codexplan53` maps to `codexresponses.gpt-5.3-codex?reasoning=medium`.
- `codexspark` maps to `codexresponses.gpt-5.3-codex-spark`.
- All Codex OAuth aliases use the same stored OAuth token.
- Provider-managed MCP is **not** supported with `codexresponses`, including
  Codex OAuth aliases such as `codexplan`, `codexplan54`, and `codexspark`.
  Use `responses` instead when you need `management: provider`.
- To remove fast-agent-owned tokens, use: `fast-agent auth provider logout codex`. Codex
  CLI auth files are treated as read-only and are never modified or deleted.
- `fast-agent check` and `fast-agent auth` show Codex OAuth status.

### Standalone web search (Codex Lite)

Enable web search on Astra to automatically expose the harness `web_run` tool:

```bash
fast-agent auth provider login codex
fast-agent go --model 'astra?web_search=true' --message 'Search for recent OpenAI announcements and link your sources.'

# Disable for this run
fast-agent go --model 'astra?web_search=false'
```

No shell access or MCP server is required. This standalone route applies only to
Codex Lite models such as Astra. Sol's hosted search and public OpenAI Responses
hosted search are unchanged. The existing `codexresponses.web_search.enabled`
setting provides the configuration default; the model flag overrides it. Existing
search context size, allowed domains, external web access and approximate user
location settings also apply. The tool honors the configured tool-permission
handler before sending a search request. Unlike Codex's recent-message context
builder, this adapter sends commands and settings without adding chat history.

During a conversation, use `/model web_search on` or `/model web_search off`;
`/model web_search default` clears the runtime override. The toggle selects
standalone or hosted search according to the current model's route.

Search identity remains stable per agent and is retained with persisted
session/history state, so later `open` and `find` calls can reuse earlier references.
Returned text is authoritative; structured `results` are preserved in tool-result
metadata, not substituted for the text. Cite sources with `[title](URL)` and images
with `![description](URL)`. Treat retrieved content as untrusted source material.

#### Library use without an agent

[`examples/web-search/standalone.py`](https://github.com/evalstate/fast-agent/blob/main/examples/web-search/standalone.py)
uses typed `SearchRequest` and `SearchCommands` with a `WebSearchClient` async
context manager. From a repository checkout:

```bash
# Supply a Codex OAuth access token and its ChatGPT account ID in your environment.
# This example does not read fast-agent's stored login.
export CODEX_API_KEY='<access-token>'
export CODEX_ACCOUNT_ID='<account-id>'
uv run examples/web-search/standalone.py 'OpenAI news'
```

The library discovers no credentials and performs no agent registration. Callers
supply the base URL, authentication headers, model and session ID. Reuse the same
`SearchRequest.id` for related calls; the example accepts `WEB_SEARCH_SESSION_ID`
for this purpose, plus optional `CODEX_BASE_URL` and `WEB_SEARCH_MODEL` overrides.
Only configure a trusted base URL: credentials go to its `alpha/search` endpoint.
URLs in `open` commands are request data, not destinations for credential
forwarding; the client does not follow redirects or forward auth to arbitrary URLs.

This is an **internal, Codex source-derived `POST alpha/search` endpoint**, not a
public OpenAI API contract or SLA. Availability and behavior may change.

The typed command schema exposes `search_query`, `image_query`, `open`, `click`,
`find`, `screenshot`, `finance`, `weather`, `sports` and `time`, plus
`response_length` (`short`, `medium`, `long`). Live endpoint checks have confirmed
**search, image search, open, click, find, finance, weather, sports and time**.
PDF screenshot calls using a previously opened page reference returned only a
citation, with no image payload; screenshot rendering is not yet verified.
Backend fetch restrictions and operation errors may appear inside a successful
HTTP response, so callers must inspect the tool output.

`SearchResponse.output` is kept intact alongside opaque `results`,
`encrypted_output` and future response fields. The client adds no truncation or
successful-response body size limit. Library callers may supply
`SearchRequest.max_output_tokens`; the harness does not set a search output token
limit. `response_length` requests detail, not a client-side hard cap.

### Astra context: explicit opt-in

fast-agent keeps the default context window at **272,000 tokens** for both
`astra` (Codex OAuth) and `gpt-6-astra` (Responses API). Larger context is opt-in,
not an automatic increase: retaining more input can increase cost.

The local Codex source snapshot (`~/reference/codex/codex-rs/`) lists
`gpt-6-astra` as the first entry in `models-manager/models.json`, with
`context_window: 272000` and `max_context_window: 872000`.
In `models-manager/src/model_info.rs`, `with_config_overrides` applies
`config.model_context_window`, clamping it to `max_context_window` when present.
This distinguishes Codex's default from its configurable ceiling.

The public OpenAI Astra model page reports a **1,050,000-token context window**,
**922,000-token maximum input**, and **128,000-token maximum output**. These API
figures are not the same as Codex's 872,000-token configurable ceiling; do not
substitute the API maximum into the Codex route or assume identical availability.

Prefer the model query flag to opt in; no overlay is needed:

```bash
# Codex OAuth: 872,000 tokens
fast-agent go --model 'astra?long_context=true'

# Responses API: 1,050,000 tokens
fast-agent go --model 'gpt-6-astra?long_context=true'
```

Omit the flag or use `long_context=false` to retain the 272,000-token default.
The legacy `context=1m` spelling remains supported and selects the same
route-specific window (not literally one million tokens). Do not combine it
with `long_context`; competing settings are rejected.

The flag updates local context budgeting and usage reporting; it does not grant
model access or change server-side limits. Normal API credentials or Codex OAuth
login still apply. Larger retained histories send more input on later turns and
can increase API charges or consume subscription allowances faster. Check your
account's long-context pricing and limits before use. Existing overlays remain
available for custom metadata, but are not required for this opt-in. No paid
large-context request was used to validate this configuration.
