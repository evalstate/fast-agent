# OpenAI

## Configuration

## Responses API

#### WebSocket Support

Responses compatible models use WebSockets as the default transport, with each turn only sending new input itmes for efficiency. fast-agent uses stateless mode to retain compatibility with Zero Data Retention policies (`store=false`). Read [compatibility with ZDR policies](https://developers.openai.com/api/docs/guides/websocket-mode#how-continuation-works) for more details. 

WebSocket mode can be disabled by using `transport=sse` in the model string. 

#### Encrypted Reasoning

Reasoning summaries are displayed, with encrypted blocks being stored locally for session resumption. Note that  

!!! Note

    Encrypted reasoning blocks are not transferable between API keys or credentials.

#### Remote MCP and Connectors

#### Web Search



## Codex Responses

The `codexresponses` provider is similar to `responses`, with these main differences:

- The `flex` service tier is **not supported**.
- Remote MCP and Connectors are **not supported**.
- The supported model list includes `gpt-5.3-codexspark`.
- Billing is via the Codex Subscription

## Chat Completions

!!! note "Legacy Models" 

A number of models remain available via the legacy Chat Completions API. 



For now, see [Providers and Models](../llm_providers/) and [Models Reference](../models_reference/).



## OpenAI

**fast-agent** supports OpenAI `gpt-5` series, `gpt-4.1` series, `o1-preview`, `o1` and `o3-mini` models. Arbitrary model names are supported with `openai.<model_name>`. Supported modalities are model-dependent, check the [OpenAI Models Page](https://platform.openai.com/docs/models) for the latest information.

OpenAI multimodal models support text, images, and PDF input (`application/pdf`). For PDFs, provide a local file/blob rather than a URL.

For reasoning models, you can specify `low`, `medium`, or `high` effort as follows:

```bash
fast-agent --model o3-mini.medium
fast-agent --model gpt-5.high
```

`gpt-5` also supports a `minimal` reasoning effort.

Structured outputs use the OpenAI API Structured Outputs feature.

**YAML Configuration:**

```yaml
openai:
  api_key: "your_openai_key" # Default
  base_url: "https://api.openai.com/v1" # Default, only include if required
```

**Environment Variables:**

- `OPENAI_API_KEY`: Your OpenAI API key
- `OPENAI_BASE_URL`: Override the API endpoint

**Model Name Aliases:**

--8<-- "_generated/model_aliases_openai.md"

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

Websocket transport is available for all models used through the `responses` provider. When
websocket transport is active, follow-up turns may be sent incrementally for efficiency.

**Provider-managed remote MCP and connectors:**

The OpenAI `responses` provider supports provider-managed remote MCP servers and
OpenAI hosted connectors declared with `management: provider` under
`mcp.servers` or card `mcp_connect` entries.

- Remote MCP servers must be remote `http`/`sse` URLs.
- Connector entries use `connector_id` instead of `url`.
- Set exactly one of `url` or `connector_id`.
- Use `access_token` for bearer auth / connector authorization.
- `defer_loading: true` enables server-side lazy tool loading.
- Not supported by `codexresponses`, Codex OAuth aliases, `openresponses`, or
  generic `openai` chat-completions models.

See [Configuration Reference](../ref/config_file/#mcp-server-configuration)
for the MCP server schema and
[AgentCards and ToolCards](../ref/agent_cards/#runtime-mcp-targets-mcp_connect)
for card-scoped runtime targets.


## Codex (OAuth Responses)

**`fast-agent`** supports using your OpenAI Codex subscription. Run `fast-agent auth codexplan`
once, then use a Codex OAuth model alias such as `codexplan` (GPT-5.3 Codex) or
`codexplan52` (GPT-5.2 Codex).

**Quick Start:**

```bash
# Start OAuth login (stores tokens in your OS keyring)
fast-agent auth codexplan

# Use the Codex planning model
fast-agent --model codexplan

# Use the GPT-5.2 Codex planning model via OAuth
fast-agent --model codexplan52
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
- `codexplan` maps to `codexresponses.gpt-5.3-codex` and `codexplan52` maps to
  `codexresponses.gpt-5.2-codex`; both use the same stored OAuth token.
- Provider-managed MCP is **not** supported with `codexresponses`, including
  Codex OAuth aliases such as `codexplan`, `codexplan52`, and `codexspark`.
  Use `responses` instead when you need `management: provider`.
- To remove tokens, use: `fast-agent auth codex-clear`.
- `fast-agent check` and `fast-agent auth` show Codex OAuth status.

**Model Name Aliases:**

--8<-- "_generated/model_aliases_codexresponses.md"