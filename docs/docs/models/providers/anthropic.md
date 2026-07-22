---
title: Anthropic Provider
social:
  title: Anthropic Provider
  tagline: Configure Claude models, prompt caching, reasoning, structured outputs, and Anthropic web tools.
  description: Configure Claude models, prompt caching, reasoning, structured outputs, and Anthropic web tools.
  alt: fast-agent social card — Anthropic
---

Anthropic models support Text, Vision and PDF content. Caching is enabled by default, and Remote MCP is supported.

**YAML Configuration:**

```yaml
anthropic:
  api_key: "your_anthropic_key" # Optional if ANTHROPIC_API_KEY or Anthropic SDK credentials are available
  base_url: "https://api.anthropic.com/v1" # Default, only include if required
  cache_mode: "auto" # Options: off, prompt, auto (default: auto)
  cache_ttl: "5m" # Options: 5m, 1h (default: 5m)
  cache_diagnostics: false # First-party Anthropic cache-miss diagnosis (debug only)
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
- `ANTHROPIC_AUTH_TOKEN`: Bearer token auth supported by the Anthropic SDK
- `ANTHROPIC_PROFILE` / `ANTHROPIC_CONFIG_DIR`: Select Anthropic SDK profile credentials
- `ANTHROPIC_BASE_URL`: Override the API endpoint

**Authentication precedence:**

**fast-agent** first uses an explicit `anthropic.api_key`, then `ANTHROPIC_API_KEY`. If neither is
set, it constructs the Anthropic SDK client without an API key so the SDK can use its own credential
chain, including `ANTHROPIC_AUTH_TOKEN`, profile credentials, and workload identity federation
environment variables. `fast-agent check` reports these as Anthropic SDK credentials when the SDK
finds an auth source.

**Caching Options:**

The `cache_mode` setting controls how prompt caching is applied:

- `off`: No caching, even if global `prompt_caching` is enabled
- `prompt`: Caches tools, system prompt, and template content
- `auto`: Also advances cache checkpoints through recent conversation turns (default)

The `cache_ttl` setting controls how long cached content persists:

- `5m`: Standard 5-minute cache (default)
- `1h`: Extended 1-hour cache (additional cost)

The TTL is a user policy choice. A 5-minute cache has a lower write premium and
usually pays off after one future read. A 1-hour cache survives longer gaps but
has a higher write premium and generally needs two future reads to pay off.
Anthropic silently skips cache writes when the marked prefix is below the
model's minimum cacheable size; check the cache creation/read usage fields
before treating an early request as a cache miss.

Set `cache_diagnostics: true` only while debugging against the first-party
Anthropic API. It enables Anthropic's cache-diagnosis beta and records the
provider's cache-miss reason in the `fast-agent-provider-diagnostics` response
channel. It is disabled by default and is not sent by Anthropic-on-Vertex.

**Reasoning + Structured Outputs:**

Claude reasoning support depends on the model family:

| Model family | fast-agent aliases | Reasoning mode | Effort values | Task budget |
| --- | --- | --- | --- | --- |
| Claude Opus 4.8 | `opus`, `opus48` | adaptive | `auto`, `low`, `medium`, `high`, `xhigh`, `max`, `off` | supported |
| Claude Opus 4.7 | `opus47` | adaptive | `auto`, `low`, `medium`, `high`, `xhigh`, `max`, `off` | supported |
| Claude Opus 4.6 | `opus46` | adaptive | `auto`, `low`, `medium`, `high`, `max`, `off` | not supported |
| Claude Sonnet 5 | `sonnet`, `sonnet5` | adaptive | `auto`, `low`, `medium`, `high`, `xhigh`, `max`, `off` | not supported |
| Claude Sonnet 4.6 | `sonnet46` | adaptive | `auto`, `low`, `medium`, `high`, `max`, `off` | not supported |
| Older Claude 4.x / Haiku | `haiku`, pinned older IDs | token budget | `1024+` token budgets, or preset aliases | not supported |

Adaptive models use `thinking: {"type": "adaptive"}` under the hood. Use effort levels
(`low`, `medium`, `high`, `xhigh` where supported, `max`) or `auto` with `anthropic.reasoning`:

```yaml
anthropic:
  reasoning: "high"
```

Adaptive models default to `auto` (provider-chosen). Do not configure fixed thinking budgets for
these models; use effort levels instead.

`task_budget` is available for Claude Opus 4.7+ in **fast-agent**. It gives the model a visible token
budget for a full agentic loop, so the model can self-moderate. It is different from `max_tokens`,
which is still the enforced ceiling for one response:

```yaml
anthropic:
  reasoning: "xhigh"
  task_budget: 128k
```

Anthropic models using budget-based thinking default to **reasoning on** with a **1024 token budget**.
Use `anthropic.reasoning` to set a budget, map from effort aliases, or disable reasoning entirely:

```yaml
anthropic:
  reasoning: 16000 # Reasoning budget tokens (minimum: 1024)
```

- Disable reasoning with `reasoning: "0"`, `reasoning: "off"`, or `reasoning: false`.
- Budget models also accept `low`/`medium`/`high`/`max` to map to preset budgets.
- The reasoning budget must be less than `max_tokens`. If you set a budget that meets/exceeds
  `max_tokens`, **fast-agent** raises `max_tokens` so the budget fits.

You can also set reasoning per run using the model string:

- `sonnet?reasoning=4096`
- `opus?reasoning=xhigh&task_budget=128k`
- `opus47?reasoning=auto&task_budget=64k`
- `claude-opus-4-6?reasoning=auto`

**Structured output selection (Anthropic JSON schema vs tool_use):**

- Models that support the `structured-outputs-2025-11-13` feature default to JSON schema output
  (`structured_output_mode: json`). This mode **is compatible with reasoning**.
- Older models default to the legacy `tool_use` structured output flow. `tool_use` **is not compatible
  with reasoning** — **fast-agent** disables reasoning when tool-forced structured output is selected.
- Anthropic on Vertex does not support modern structured outputs in **fast-agent**; choose
  `structured_output_mode: tool_use` / `?structured=tool_use` there.

You can override the structured output mode explicitly:

```yaml
anthropic:
  structured_output_mode: auto # auto (default), json, or tool_use
```

Deprecated: `thinking_enabled` and `thinking_budget_tokens` are ignored. Use `reasoning`.

**Built-in Anthropic web tools (`web_search` + `web_fetch`):**

**fast-agent** can enable Anthropic server-side web tools directly (these are not MCP tool calls):

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

See [Configuration Reference](../../ref/config_file/#mcp-server-configuration)
for the MCP server schema and
[Agent Cards](../../agents/defining/agent_cards/#runtime-mcp-targets-mcp_connect)
for card-scoped runtime targets.


**Model Name Aliases:**

--8<-- "_generated/model_aliases_anthropic.md"
