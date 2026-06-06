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
| Auth | `OPENAI_API_KEY` | `fast-agent auth codexplan` or `CODEX_API_KEY` | `OPENAI_API_KEY` |
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

## Model availability

### Responses models

Use `responses` for OpenAI Responses API models and for using the `flex` service tier or Remote MCP/Connectors.

#### WebSocket Support

Responses-compatible models use WebSockets as the default transport, with continuation support so repeated turns can avoid resending unchanged input items. fast-agent sends `store=false` on Responses requests. Read [compatibility with ZDR policies](https://developers.openai.com/api/docs/guides/websocket-mode#how-continuation-works) for more details.

WebSocket mode can be disabled by using `transport=sse` in the model string. 

#### Encrypted Reasoning

Reasoning summaries are displayed, with encrypted blocks stored locally for session resumption.

!!! Note

    Encrypted reasoning blocks are not transferable between API keys or credentials.


Responses models also have short aliases.

| Model string or alias | Resolves to / equivalent | Notes |
| --- | --- | --- |
| `gpt55` | `responses.gpt-5.5` | Current GPT-5.5 shortcut |
| `gpt54` | `responses.gpt-5.4` | GPT-5.4 shortcut |
| `gpt54-mini` | `responses.gpt-5.4-mini` | Smaller GPT-5.4 shortcut |
| `gpt54-nano` | `responses.gpt-5.4-nano` | Smallest GPT-5.4 shortcut |
| `gpt52` | `responses.gpt-5.2` | GPT-5.2 shortcut |
| `gpt51` | `responses.gpt-5.1` | GPT-5.1 shortcut |
| `gpt-5`, `gpt-5-mini`, `gpt-5-nano` | `responses.<model>` | GPT-5 family defaults to Responses |
| `o3`, `o3-mini`, `o4-mini`, `o1` | `responses.<model>` | OpenAI reasoning models default to Responses |
| `chatgpt`, `chat-latest` | `responses.chat-latest` | ChatGPT-latest shortcut |
| `codex` | `responses.gpt-5.3-codex` | OpenAI API Codex model, not Codex subscription auth |
| `responses.<model>` | exact Responses model name | Explicit form for any supported Responses model |

Examples:

- `responses.gpt-5.5?reasoning=medium`
- `responses.gpt-5.5?web_search=on`
- `responses.gpt-5.4?service_tier=flex`

### Codex Responses models

Use `codexresponses` for Codex subscription-backed models. Authenticate with
`fast-agent auth codexplan` or provide `CODEX_API_KEY`.

The `codexresponses` provider is similar to `responses`, with these main differences:

- The `flex` service tier is **not supported**.
- Remote MCP and Connectors are **not supported**.
- The supported model list includes `gpt-5.3-codex-spark`.
- Billing is via the Codex subscription.

| Model string or alias | Resolves to / equivalent | Notes |
| --- | --- | --- |
| `codexplan` | `codexresponses.gpt-5.5?reasoning=medium` | Default Codex planning alias |
| `codexplan54` | `codexresponses.gpt-5.4?reasoning=high` | Pinned GPT-5.4 planning alias |
| `codexplan53` | `codexresponses.gpt-5.3-codex?reasoning=medium` | Pinned GPT-5.3 Codex planning alias |
| `codexspark` | `codexresponses.gpt-5.3-codex-spark` | Spark model; no reasoning effort controls |
| `codexresponses.<model>` | exact Codex Responses model name | Explicit form for supported Codex Responses models |

Examples:

- `codexplan`
- `codexresponses.gpt-5.5?reasoning=high`
- `codexresponses.gpt-5.3-codex-spark?web_search=on`

### Legacy Chat Completions models

!!! note "Legacy Models" 

    Use `openai` when you specifically need the legacy Chat Completions-compatible path. Prefer
    the explicit `openai.` prefix so the selected API surface is obvious.

| Model string | API model | Notes |
| --- | --- | --- |
| `openai.gpt-4.1` | `gpt-4.1` | Legacy Chat Completions-compatible GPT-4.1 |
| `openai.gpt-4.1-mini` | `gpt-4.1-mini` | Smaller GPT-4.1 |
| `openai.gpt-4.1-nano` | `gpt-4.1-nano` | Smallest GPT-4.1 |
| `openai.gpt-4o` | `gpt-4o` | GPT-4o Chat Completions path |
| `openai.gpt-4o-mini` | `gpt-4o-mini` | Smaller GPT-4o |
| `openai.<model_name>` | exact Chat Completions model name | Explicit form for custom deployments or newly released Chat Completions models |

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

**`fast-agent`** supports using your OpenAI Codex subscription. Run `fast-agent auth codexplan`
once, then use a Codex OAuth model alias such as `codexplan` (GPT-5.5 planning),
`codexplan54` (GPT-5.4 planning), `codexplan53` (GPT-5.3 Codex planning), or
`codexspark` (GPT-5.3 Codex Spark).

**Quick Start:**

```bash
# Start OAuth login (stores tokens in your OS keyring)
fast-agent auth codexplan

# Use the Codex planning model
fast-agent --model codexplan

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

- Tokens are stored in your OS keyring via `fast-agent auth codexplan`.
- `codexplan` maps to `codexresponses.gpt-5.5?reasoning=medium`.
- `codexplan54` maps to `codexresponses.gpt-5.4?reasoning=high`.
- `codexplan53` maps to `codexresponses.gpt-5.3-codex?reasoning=medium`.
- `codexspark` maps to `codexresponses.gpt-5.3-codex-spark`.
- All Codex OAuth aliases use the same stored OAuth token.
- Provider-managed MCP is **not** supported with `codexresponses`, including
  Codex OAuth aliases such as `codexplan`, `codexplan54`, and `codexspark`.
  Use `responses` instead when you need `management: provider`.
- To remove tokens, use: `fast-agent auth codex-clear`.
- `fast-agent check` and `fast-agent auth` show Codex OAuth status.
