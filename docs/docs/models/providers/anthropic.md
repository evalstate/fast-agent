# Anthropic


## Anthropic

Anthropic models support Text, Vision and PDF content.

**YAML Configuration:**

```yaml
anthropic:
  api_key: "your_anthropic_key" # Required
  base_url: "https://api.anthropic.com/v1" # Default, only include if required
  cache_mode: "auto" # Options: off, prompt, auto (default: auto)
  cache_ttl: "5m" # Options: 5m, 1h (default: 5m)
  web_search:
    enabled: false
    # max_uses: 3
    # allowed_domains: ["example.com", "*.docs.example.com"]
    # blocked_domains: ["social.example"]  # mutually exclusive with allowed_domains
    # user_location:
    #   type: approximate
    #   city: "London"
    #   country: "UK"
  web_fetch:
    enabled: false
    citations_enabled: false
    # max_uses: 3
    # max_content_tokens: 4096
    # allowed_domains: ["example.com"]
    # blocked_domains: ["tracking.example"]  # mutually exclusive with allowed_domains
```

**Environment Variables:**

- `ANTHROPIC_API_KEY`: Your Anthropic API key
- `ANTHROPIC_BASE_URL`: Override the API endpoint

**Caching Options:**

The `cache_mode` setting controls how prompt caching is applied:

- `off`: No caching, even if global `prompt_caching` is enabled
- `prompt`: Caches tools, system prompt, and template content
- `auto`: Same as `prompt` (default)

The `cache_ttl` setting controls how long cached content persists:

- `5m`: Standard 5-minute cache (default)
- `1h`: Extended 1-hour cache (additional cost)

**Reasoning + Structured Outputs:**

`claude-opus-4-6` uses adaptive thinking by default. Use effort levels (`low`, `medium`, `high`,
`max`) or `auto` with `anthropic.reasoning`:

```yaml
anthropic:
  reasoning: "high"
```

Adaptive models default to `auto` (provider‑chosen) and do not accept explicit budgets.

Anthropic models using budget-based thinking default to **reasoning on** with a **1024 token budget**.
Use `anthropic.reasoning` to set a budget, map from effort aliases, or disable reasoning entirely:

```yaml
anthropic:
  reasoning: 16000 # Reasoning budget tokens (minimum: 1024)
```

- Disable reasoning with `reasoning: "0"`, `reasoning: "off"`, or `reasoning: false`.
- Budget models also accept `low`/`medium`/`high`/`max` to map to preset budgets.
- The reasoning budget must be less than `max_tokens`. If you set a budget that meets/exceeds
  `max_tokens`, fast-agent raises `max_tokens` so the budget fits.

You can also set reasoning per run using the model string:

- `sonnet?reasoning=4096`
- `anthropic.claude-4-5-sonnet-latest?reasoning=4096`
- `claude-opus-4-6?reasoning=auto`

**Structured output selection (Anthropic JSON schema vs tool_use):**

- Models that support the `structured-outputs-2025-11-13` feature default to JSON schema output
  (`structured_output_mode: json`). This mode **is compatible with reasoning**.
- Older models default to the legacy `tool_use` structured output flow. `tool_use` **is not compatible
  with reasoning** — fast-agent disables reasoning when tool-forced structured output is selected.

You can override the structured output mode explicitly:

```yaml
anthropic:
  structured_output_mode: auto # auto (default), json, or tool_use
```

Deprecated: `thinking_enabled` and `thinking_budget_tokens` are ignored. Use `reasoning`.

**Built-in Anthropic web tools (`web_search` + `web_fetch`):**

fast-agent can enable Anthropic server-side web tools directly (these are not MCP tool calls):

- `anthropic.web_search.enabled: true`
- `anthropic.web_fetch.enabled: true`

Optional controls:

- `max_uses`
- `allowed_domains` / `blocked_domains` (mutually exclusive)
- `web_search.user_location` (approximate city/region/country/timezone)
- `web_fetch.max_content_tokens`
- `web_fetch.citations_enabled`

You can override per run in the model string:

- `claude-opus-4-6?web_search=on&web_fetch=on`
- `sonnet?web_search=off`

Supported values are `on`/`off` (also accepts `true`/`false`, `1`/`0`).

Version policy is model-aware:

- Claude 4.6 models use `web_search_20260209` and `web_fetch_20260209`
  (with required beta header `code-execution-web-tools-2026-02-09`).
- Other supported Anthropic models use legacy versions
  (`web_search_20250305`, `web_fetch_20250910`).

**Provider-managed remote MCP:**

The direct `anthropic` provider supports provider-managed remote MCP servers
declared with `management: provider` under `mcp.servers` or card `mcp_connect`
entries.

- Supported on `anthropic`
- Not supported on `anthropic-vertex`
- Server must be a remote `http`/`sse` URL
- Use `access_token` for bearer auth if required

See [Configuration Reference](../ref/config_file/#mcp-server-configuration)
for the MCP server schema and
[AgentCards and ToolCards](../ref/agent_cards/#runtime-mcp-targets-mcp_connect)
for card-scoped runtime targets.


**Model Name Aliases:**

--8<-- "_generated/model_aliases_anthropic.md"
