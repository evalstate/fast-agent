---
title: Configuration Reference
description: Complete reference for fast-agent configuration settings
social:
  title: Configuration Reference
  tagline: Understand every setting in fast-agent configuration files.
  description: Understand every setting in fast-agent configuration files.
  alt: fast-agent social card — Configuration Reference
---


# Configuration Reference

**fast-agent** can be configured through the `fast-agent.yaml` file. Place it in the active fast-agent home (by default `./.fast-agent`) or pass an explicit path with `--config-path` (`-c`). For sensitive information, you can use `fast-agent.secrets.yaml` with the same structure - values from both files will be merged, with secrets taking precedence.

Configuration can also be provided through environment variables, with the naming pattern `SECTION__SUBSECTION__PROPERTY` (note the double underscores).

## Configuration File Location

fast-agent loads configuration from the active fast-agent home first, then from the current working directory if no home config exists. The home is selected by `--home`, `--workspace <path>/.fast-agent`, `FAST_AGENT_HOME`, or the default `./.fast-agent`. You can also specify a configuration file path or URI with the `--config-path` (`-c`) command-line argument.

## General Settings

```yaml
# Default model for all agents
default_model: "gpt-5-mini?reasoning=low"  # Format: provider.model_name with optional query params

# Optional namespaced model references (configured under model_references and
# used via exact tokens like $system.fast)
model_references:
  system:
    fast: "gpt-5-mini?reasoning=low"
    plan: "claude-sonnet-4-6"

# Whether to automatically enable Sampling. Model selection precedence is Agent > Default.
auto_sampling: true

# Number of times to retry transient LLM API errors (falls back to FAST_AGENT_RETRIES env)
llm_retries: 2

# Execution engine (only asyncio is currently supported)
execution_engine: "asyncio"

# Base directory for fast-agent runtime data
home: ".fast-agent"

# Session history storage (on/off)
session_history: true

# Session history rolling window (number of recent sessions to keep)
session_history_window: 20

# Persist git repository provenance in session snapshots and trace exports
git_aware: false
```

History compaction defaults are generated from `fast_agent.config.CompactionSettings`:

--8<-- "_generated/compaction_config_snippet.md"

Relative `compaction.prompt` file paths are resolved from the loaded config file's directory. If
`FAST_AGENT_HOME` points at a home config, the path is relative to that home config file; it is not
resolved from the process current working directory.

`llm_retries` defaults to `2` and is the preferred way to control retry attempts. If unset in
config, the `FAST_AGENT_RETRIES` environment variable is used as a fallback.

## Namespaced Model References

Use `model_references` to create exact-token model references such as `$system.fast` and reuse them in
`default_model`, `--model`, environment overrides, and agent card `model` fields.

```yaml
default_model: "$system.fast"

model_references:
  system:
    fast: "gpt-5-mini?reasoning=low"
    plan: "claude-sonnet-4-6"
```

Notes:

- Model reference tokens must match this form exactly: `$<namespace>.<key>` (for example `$system.fast`).
- Model references can point to other model references (recursive expansion is supported with cycle detection).
- If a model reference cannot be resolved, fast-agent logs a warning and falls back to the next
  lower-precedence model source (explicit model → CLI → config → env → hardcoded default).
  This warning is emitted through the normal logger/event pipeline and may be surfaced in UIs.
- If a selected model is not a model reference token (doesn't start with `$`), normal validation behavior
  applies.

## Model Overlays

Model overlays are home-local named model entries stored under the active home.

Files are loaded from:

- `model-overlays/*.yaml`
- `model-overlays.secrets.yaml`

Use overlays when you want a short local token such as `qwen-local` to carry:

- a provider
- a wire model name
- a custom `base_url`
- auth settings
- request defaults
- local metadata for picker/display use

Example overlay manifest:

```yaml
name: qwen-local
provider: openresponses
model: unsloth/Qwen3.5-9B-GGUF
connection:
  base_url: http://localhost:8080/v1
  auth: none
defaults:
  temperature: 0.8
  top_p: 0.95
  max_tokens: 2048
metadata:
  context_window: 75264
  max_output_tokens: 2048
picker:
  label: Qwen local
  description: Imported from llama.cpp
  current: true
```

Once present, the overlay name can be used anywhere a model string is accepted:

```yaml
default_model: "qwen-local"
```

or:

```yaml
model_references:
  system:
    fast: "qwen-local"
```

For a complete guide, see [Model Overlays](../models/model_overlays/).

--8<-- "_generated/execution_environments_reference.md"

## Runtime Environment Variables

- `FAST_AGENT_DISABLE_UV_LOOP=1`: Disable uvloop even if installed (non-Windows). By default, uvloop is used when available.
`session_history` controls whether fast-agent persists session metadata and history files in the environment sessions folder (default `.fast-agent/sessions`). `session_history_window` limits how many recent sessions are kept; older sessions are pruned when new sessions are created. The same window is used for session resume completions and ordinal selection (e.g. `/session resume 1`).

`git_aware` adds best-effort git provenance to persisted sessions and exported traces. When enabled and the session working directory is inside a git repository, fast-agent records the repository root, commit, capture time, branch, dirty state, GitHub `owner/repo` when available, and a sanitized `origin` remote URL. The first captured state is kept as `started`; later saves update `current`.

`home` sets the base folder for local fast-agent data such as skills, sessions, and permission history. You can also override this per run with `fast-agent --home <path>`, or choose a workspace with `fast-agent --workspace <path>` and let the home default to `<workspace>/.fast-agent`. Use `--no-home` for ephemeral runs that intentionally skip home-based side effects.

### History Compaction

When a conversation grows large, fast-agent can compact older turns into a single checkpoint summary, freeing context while preserving the work done so far. The summary is produced by the agent's own model and inserted into history as a clearly-marked message (shown as `compacted` in `/history`); the most recent turns are kept verbatim.

--8<-- "_generated/compaction_settings_reference.md"

You can also compact on demand:

- `/compact` &mdash; compact now, showing the before/after context usage.
- `/compact <instructions>` &mdash; steer the summary (for example, `/compact focus on the database migration`).
- `/compact preview` &mdash; show what would be kept and dropped, without calling the model.
- `/compact prompt` &mdash; print the active summarization prompt.

The pre-compaction history is archived to a `compacted_*.json` file in the session directory so the original conversation is never lost.

## Model Providers

### Anthropic

```yaml
anthropic:
  api_key: "your_anthropic_key"  # Optional; can also use ANTHROPIC_API_KEY or Anthropic SDK credentials
  base_url: "https://api.anthropic.com/v1"  # Optional, only include to override
  reasoning: auto  # Adaptive models: auto/low/medium/high/max. Budget models: integer tokens or off.
  structured_output_mode: auto  # auto (default), json, or tool_use
  web_search:
    enabled: false
    max_uses: 3  # Optional, must be > 0
    allowed_domains: ["example.com"]  # Optional; mutually exclusive with blocked_domains
    # blocked_domains: ["tracking.example"]
    user_location:  # Optional
      type: approximate
      city: "London"
      country: "UK"
      region: "England"
      timezone: "Europe/London"
  web_fetch:
    enabled: false
    citations_enabled: false
    max_uses: 3  # Optional, must be > 0
    max_content_tokens: 4096  # Optional, must be > 0
    allowed_domains: ["example.com"]  # Optional; mutually exclusive with blocked_domains
    # blocked_domains: ["ads.example"]
```

Anthropic authentication uses this precedence:

1. `anthropic.api_key` in fast-agent config/secrets.
2. `ANTHROPIC_API_KEY`.
3. Anthropic SDK credential discovery, including `ANTHROPIC_AUTH_TOKEN`, Anthropic profiles
   (`ANTHROPIC_PROFILE` / `ANTHROPIC_CONFIG_DIR` / active profile), and workload identity
   federation environment variables.

This means `api_key` is no longer required when Anthropic SDK credentials are available.

Anthropic models fall into three groups:

- **No reasoning support**: `reasoning` is ignored with a warning.
- **Budget-based thinking** (older models): defaults to a 1024 token budget. Set `reasoning` to a
  budget integer or disable with `"0"`/`off`/`false`. You can also pass `low`/`medium`/`high`/`max`,
  which map to preset budgets.
- **Adaptive thinking** (e.g. `claude-opus-4-6`, `claude-opus-4-7`, `claude-opus-4-8`): defaults
  to `auto` (provider-chosen). Use effort levels (`low`/`medium`/`high`/`max`, plus `xhigh` where
  advertised) to set `output_config.effort`. Fixed thinking budgets are deprecated for these models;
  Opus 4.7+ additionally supports `task_budget` for model-visible agent-loop budgets.

For budget models, the reasoning budget must be lower than `max_tokens` (fast-agent raises
`max_tokens` if needed).

Structured outputs default to JSON schema for newer models that support the
`structured-outputs-2025-11-13` feature and are compatible with reasoning. Older models fall back to
`tool_use` structured output, which is **not compatible** with reasoning (fast-agent disables
reasoning for tool-forced structured outputs). Override with `structured_output_mode: json` or
`structured_output_mode: tool_use` as needed.

Legacy `thinking_enabled` and `thinking_budget_tokens` settings are deprecated and ignored.

Anthropic built-in web tools can also be toggled per run in the model string:

- `claude-opus-4-6?web_search=on&web_fetch=on`
- `sonnet?web_search=off`

Allowed values: `on`/`off` (also accepts `true`/`false`, `1`/`0`).

### OpenAI

```yaml
openai:
  api_key: "your_openai_key"  # Can also use OPENAI_API_KEY env var
  base_url: "https://api.openai.com/v1"  # Optional, only include to override
  reasoning: "medium"  # Optional unified reasoning setting where supported
```

For OpenAI Responses API models, use the `responses` section below. The
Responses-family `web_search` block is also supported for `openresponses`,
`codexresponses`, and `metaai` provider sections.

Responses-family providers can also be toggled per run in the model string:

- `responses.gpt-5?web_search=on`
- `openresponses.openai/gpt-oss-120b:groq?web_search=on`
- `codexresponses.gpt-5.3-codex?web_search=off`
- `metaai.muse-spark-1.1?web_search=on`

Allowed values: `on`/`off` (also accepts `true`/`false`, `1`/`0`).

### Responses (OpenAI Responses API)

```yaml
responses:
  api_key: "your_openai_key"  # Can also use OPENAI_API_KEY env var
  base_url: "https://api.openai.com/v1"  # Optional, only include to override
  reasoning: "medium"  # Optional default reasoning setting
  text_verbosity: "medium"  # Optional: low | medium | high
  transport: "sse"  # sse | websocket | auto
  web_search:
    enabled: false
    tool_type: web_search  # web_search | web_search_preview
    search_context_size: medium  # Optional: low | medium | high
    allowed_domains: ["openai.com"]  # Optional, max 100 domains
    external_web_access: false  # Optional, only for tool_type=web_search
    user_location:  # Optional
      type: approximate
      city: "Minneapolis"
      region: "Minnesota"
      country: "US"
      timezone: "America/Chicago"
```

`web_search` can be toggled per run in the model string:

- `responses.gpt-5-mini?web_search=on`
- `responses.gpt-5-mini?web_search=off`
- `responses.gpt-5.3-codex?transport=ws`

Websocket transport is available for all models used through the `responses` provider. When
websocket transport is active, follow-up turns may be sent incrementally for efficiency.

### Azure OpenAI

```yaml
# Option 1: Using resource_name and api_key (standard method)
azure:
  api_key: "your_azure_openai_key"  # Required unless using DefaultAzureCredential
  resource_name: "your-resource-name"  # Resource name in Azure
  azure_deployment: "deployment-name"  # Required - deployment name from Azure
  api_version: "2024-10-21"  # Optional API version
  default_headers:
    Ocp-Apim-Subscription-Key: "${AZURE_OPENAI_API_KEY}"
  # Do NOT include base_url if you use resource_name

# Option 2: Using base_url and api_key (custom endpoints or sovereign clouds)
# azure:
#   api_key: "your_azure_openai_key"
#   base_url: "https://your-endpoint.openai.azure.com/"
#   azure_deployment: "deployment-name"
#   api_version: "2024-10-21"
#   default_headers:
#     Ocp-Apim-Subscription-Key: "${AZURE_OPENAI_API_KEY}"
#   # Do NOT include resource_name if you use base_url

# Option 3: Using DefaultAzureCredential (for managed identity, Azure CLI, etc.)
# azure:
#   use_default_azure_credential: true
#   base_url: "https://your-endpoint.openai.azure.com/"
#   azure_deployment: "deployment-name"
#   api_version: "2024-10-21"
#   default_headers:
#     Ocp-Apim-Subscription-Key: "${AZURE_OPENAI_API_KEY}"
#   # Do NOT include api_key or resource_name in this mode
```

Important configuration notes:
- Use either `resource_name` or `base_url`, not both.
- When using `DefaultAzureCredential`, do NOT include `api_key` or `resource_name` (the `azure-identity` package must be installed).
- When using `base_url`, do NOT include `resource_name`.
- When using `resource_name`, do NOT include `base_url`.
- `default_headers` can be used with any option to pass API management headers.
- The model string format is `azure.deployment-name`

### DeepSeek

```yaml
deepseek:
  api_key: "your_deepseek_key"  # Can also use DEEPSEEK_API_KEY env var
  base_url: "https://api.deepseek.com"  # Optional, only include to override
```

### Google

```yaml
google:
  api_key: "your_google_key"  # Can also use GOOGLE_API_KEY env var
  base_url: "https://generativelanguage.googleapis.com/v1beta/openai"  # Optional
```

### xAI (Grok)

```yaml
xai:
  api_key: "your_xai_key"  # Can also use XAI_API_KEY env var
  base_url: "https://api.x.ai/v1"  # Optional, defaults to this value
```

### MetaAI

```yaml
metaai:
  api_key: "${META_AI_API_KEY}"
  base_url: "https://api.meta.ai/v1"  # Optional, defaults to this value
  default_model: "muse-spark-1.1"
  web_search:
    enabled: false
    search_context_size: medium  # Optional: low | medium | high
    user_location:  # Optional approximate locale bias
      type: approximate
      country: "GB"
```

MetaAI search grounding can also be toggled per run with
`metaai.muse-spark-1.1?web_search=on`. See the
[MetaAI provider guide](../models/providers/metaai/) for supported media,
interactive toggles, and search-result behavior.

### Groq

```yaml
groq:
  api_key: "your_groq_key"  # Can also use GROQ_API_KEY env var
  base_url: "https://api.groq.com/openai/v1"  # Optional, defaults to this value
```

### Aliyun (Qwen via OpenAI-compatible API)

```yaml
aliyun:
  api_key: "your_aliyun_key"  # Provide via secrets/env as appropriate
  base_url: "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"  # Optional, defaults to this value
```

### Hugging Face (Inference Providers)

```yaml
hf:
  api_key: "${HF_TOKEN}"  # Can also use HF_TOKEN env var
  base_url: "https://router.huggingface.co/v1"  # Optional
  default_provider: null  # Optional: groq, fireworks-ai, cerebras, etc.
```

### Generic (Ollama, etc.)

```yaml
generic:
  api_key: "ollama"  # Default for Ollama, change as needed
  base_url: "http://localhost:11434/v1"  # Default for Ollama
```

### OpenRouter

```yaml
openrouter:
  api_key: "your_openrouter_key"  # Can also use OPENROUTER_API_KEY env var
  base_url: "https://openrouter.ai/api/v1"  # Optional, only include to override
```

### TensorZero

```yaml
tensorzero:
  base_url: "http://localhost:3000"  # Optional, only include to override
```

See the [TensorZero Quick Start](https://tensorzero.com/docs/quickstart) and the [TensorZero Gateway Deployment Guide](https://www.tensorzero.com/docs/gateway/deployment/) for more information on how to deploy the TensorZero Gateway.

### AWS Bedrock

```yaml
bedrock:
  region: "us-east-1"  # Required - AWS region where Bedrock is available
  profile: "default"   # Optional - AWS profile to use (defaults to "default")
```

AWS Bedrock uses standard AWS authentication through the boto3 credential provider chain. You can configure credentials using:

- **AWS CLI**: Run `aws configure` to set up credentials (AWS SSO recommended for local development)
- **Environment variables**: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN` (for temporary credentials)
- **IAM roles**: Use IAM roles when running on EC2 or other AWS services
- **AWS profiles**: Use named profiles with the `profile` setting or `AWS_PROFILE` environment variable

Additional environment variables:
- `AWS_REGION` or `AWS_DEFAULT_REGION`: Override the region setting
- `AWS_PROFILE`: Override the profile setting

The model string format is `bedrock.model-id` (e.g., `bedrock.amazon.nova-lite-v1:0`)

## MCP Server Configuration

MCP Servers are defined under the `mcp.servers` section:

`mcp.servers` contains preconfigured aliases. AgentCards may additionally declare
runtime targets via `mcp_connect`; those are resolved at startup and do not need
to be prelisted here.

You can define servers in canonical form (`transport`/`url`/`command`) or with
the shorthand `target` field:

```yaml
mcp:
  servers:
    githubcopilot:
      target: "https://api.githubcopilot.com/mcp/"
      load_on_start: false
      auth:
        oauth: true

    filesystem:
      target: "@modelcontextprotocol/server-filesystem /workspace"
      load_on_start: false
```

You can also provide target-first entries as a list under `mcp.targets`. This
matches AgentCard-style runtime target declarations while still normalizing to
canonical named servers:

```yaml
mcp:
  targets:
    - target: "https://demo.hf.space"
    - target: "@modelcontextprotocol/server-filesystem /workspace"
      name: "filesystem"
      load_on_start: false
```

`mcp.targets` entries normalize into `mcp.servers` aliases. If both
`mcp.targets` and `mcp.servers` define the same name, the explicit
`mcp.servers.<name>` entry wins.

When `target` is used, explicit fields override target-derived defaults.
For example, `transport`, `url`, `headers`, and `auth` on the server entry take
precedence over derived values.

`target` must be a pure target string. Do not embed fast-agent CLI flags
(`--auth`, `--oauth`, `--timeout`, etc.) inside `target`; use structured fields
like `headers` and `auth` instead.

### Provider-managed remote MCP

Use `management: provider` when you want the model provider to execute a remote
MCP server directly instead of having fast-agent connect to it as a local MCP
client.

```yaml
mcp:
  servers:
    stripe:
      management: provider
      transport: "http"
      url: "https://mcp.stripe.com"
      access_token: "${STRIPE_TOKEN}"
      description: "Stripe official MCP"
      defer_loading: true  # OpenAI Responses hint; ignored by Anthropic
```

Provider-managed remote MCP is supported only for:

- `anthropic`
- `responses`

It is not available for `codexresponses`, Codex OAuth aliases, `openresponses`,
`openai`, `google`, `anthropic-vertex`, or other providers.

Rules for provider-managed remote MCP servers:

- Must be URL-based remote servers (`http` or `sse`)
- `url` is required unless using an OpenAI Responses connector
- Use `access_token` for bearer auth when needed
- `command`, `args`, `env`, `cwd`, `headers`, `auth`, and `roots` are not supported
- `defer_loading` is only used by the OpenAI `responses` provider

When a provider-managed server is attached to an agent/card:

- `tools.<server_name>` must use exact tool names only
- wildcard tool filters are not supported
- `prompts.<server_name>` and `resources.<server_name>` filters are not supported

```yaml
mcp:
  servers:
    stripe:
      management: provider
      transport: "http"
      url: "https://mcp.stripe.com"
      access_token: "${STRIPE_TOKEN}"

agents:
  billing:
    model: "responses.gpt-5-mini"
    servers: ["stripe"]
    tools:
      stripe: ["customers_create", "customers_retrieve"]
```

### OpenAI Responses connectors

The OpenAI `responses` provider can also manage OpenAI hosted connectors. Use
`connector_id` instead of `url`:

```yaml
mcp:
  servers:
    dropbox:
      management: provider
      connector_id: connector_dropbox
      access_token: "${DROPBOX_OAUTH_ACCESS_TOKEN}"
      description: "Dropbox connector"
      defer_loading: true
```

Connector rules:

- Supported only by the OpenAI `responses` provider
- Set exactly one of `url` or `connector_id`
- `connector_id` must be one of the connector IDs supported by the installed OpenAI SDK
- `access_token` is required
- Omit `transport`, `url`, `command`, `args`, `env`, `cwd`, `headers`, `auth`, and `roots`
- `defer_loading: true` enables server-side lazy tool loading

```yaml
mcp:
  servers:
    # Example stdio server
    server_name:
      transport: "stdio"  # "stdio", "sse", or "http"
      command: "npx"  # Command to execute
      args: ["@package/server-name"]  # Command arguments as array
      read_timeout_seconds: 60  # Optional timeout in seconds
      # HTTP transport only:
      # http_timeout_seconds: 30        # Overall HTTP timeout (seconds). If unset, uses MCP SDK default (30s).
      # http_read_timeout_seconds: 300  # Per-read timeout for streaming (seconds). If unset, uses MCP SDK default (300s).
      ping_interval_seconds: 30  # Optional ping interval; <=0 disables (default: 30)
      max_missed_pings: 3  # Optional; consecutive missed pings before marking failed (default: 3)
      env:  # Optional environment variables
        ENV_VAR1: "value1"
        ENV_VAR2: "value2"
      sampling:  # Optional sampling settings
        model: "gpt-5-mini"  # Model to use for sampling requests

    # Example Stremable HTTP server
    streamable_http__server:
      transport: "http"
      url: "http://localhost:8000/mcp"
      read_transport_sse_timeout_seconds: 300  # Timeout for HTTP/SSE connections
      http_timeout_seconds: 300  # Overall HTTP timeout (StreamableHTTP)
      http_read_timeout_seconds: 300  # Read timeout (StreamableHTTP)
      headers:  # Optional HTTP headers
        Authorization: "Bearer token"
      auth:  # Optional authentication
        oauth: true
      include_instructions: true  # Whether to include instructions in {{serverInstructions}}

    # Example SSE server
    sse_server:
      transport: "sse"
      url: "http://localhost:8000/sse"
      read_transport_sse_timeout_seconds: 300  # Timeout for SSE connections
      headers:  # Optional HTTP headers
        Authorization: "Bearer token"
      auth:  # Optional authentication
        oauth: true


    # Server with roots
    file_server:
      transport: "stdio"
      command: "command"
      args: ["arguments"]
      roots:  # Root directories accessible to this server
        - uri: "file:///path/to/dir"  # Must start with file://
          name: "Optional Name"  # Optional display name for the root
          server_uri_alias: "file:///server/path"  # Optional, for consistent paths
```

Ping settings are optional and configured per server. `ping_interval_seconds` defaults to 30 seconds (<=0 disables), and `max_missed_pings` defaults to 3.

## Skills Configuration

Configure skill directories and marketplace registries:

```yaml
skills:
  # Override default skill directories
  directories:
    - ".fast-agent/skills"
    - "~/my-custom-skills"

  # Available skill registries (marketplaces)
  marketplace_urls:
    - "https://github.com/fast-agent-ai/skills"
    - "https://github.com/huggingface/skills"
    - "https://github.com/anthropics/skills"
```

| Setting | Description | Default |
|---------|-------------|---------|
| `directories` | List of directories to search for `SKILL.md` files | `.fast-agent/skills`, `.agents/skills`, `.claude/skills` |
| `marketplace_urls` | List of skill registries for `/skills add` | fast-agent, Hugging Face, and Anthropic registries |

See [Agent Skills](../guides/skills/) for more information on using skills.

## Command Plugin Configuration

Enable installed command plugins and configure plugin registries:

```yaml
plugins:
  enabled:
    - agent-finder
    - edit-assistant
  marketplace_urls:
    - "https://github.com/fast-agent-ai/card-packs"
  config:
    agent-finder:
      page_size: 10
      prompt_when_multiple: true
```

| Setting | Description | Default |
|---------|-------------|---------|
| `enabled` | Plugin names to load from the active fast-agent home's `plugins/` directory | `[]` |
| `marketplace_url` | Single plugin registry for `fast-agent plugins add` | fast-agent card-packs registry |
| `marketplace_urls` | Ordered plugin registries | fast-agent card-packs registry |
| `config` | Namespaced plugin-specific configuration, keyed by plugin name | `{}` |

Plugin registries configure direct plugin operations such as
`fast-agent plugins add` and `fast-agent plugins update`. Required plugins for a
card pack are resolved from the marketplace that supplied that pack, not
necessarily from these plugin registry settings. If you publish a custom card
pack that declares `plugins.required`, include matching `command_plugins`
entries in the same marketplace file as the pack.

Plugin-specific settings belong under `plugins.config.<plugin-name>`. The shape
inside each plugin's namespace is owned by that plugin.

Global plugins are layered separately from the active fast-agent home. When
`FAST_AGENT_HOME` is set, plugin names enabled in
`$FAST_AGENT_HOME/fast-agent.yaml` are merged with the active project config,
including when `--home <dir>` selects a different active fast-agent home. If
`FAST_AGENT_HOME` is not set, `~/.fast-agent/fast-agent.yaml` is used as the
global plugin layer when it exists. Only the global file's `plugins` block is
layered in: global plugins load from the global `plugins/` directory, and
project-enabled plugins load from the active fast-agent home's `plugins/`
directory. Project plugin commands override global commands with the same name,
and inline `commands:` entries override both.
See [Command Plugins](../agents/plugins/) for install, update, and card-pack
usage.

## OpenTelemetry Settings

```yaml
otel:
  enabled: false  # Enable or disable OpenTelemetry
  service_name: "fast-agent"  # Service name for tracing
  otlp_endpoint: "http://localhost:4318/v1/traces"  # OTLP endpoint for tracing
  console_debug: false  # Log spans to console
  sample_rate: 1.0  # Sample rate (0.0-1.0)
```

## Logging Settings

```yaml
logger:
  type: "file"  # "none", "console", "file", or "http"
  level: "warning"  # "debug", "info", "warning", or "error"
  progress_display: true  # Enable/disable progress display
  path: "fast-agent-log.jsonl"  # Explicit path to log file (for "file" type)
  batch_size: 100  # Events to accumulate before processing
  flush_interval: 2.0  # Flush interval in seconds
  max_queue_size: 2048  # Maximum queue size for events
  
  # HTTP logger settings
  http_endpoint: "https://logging.example.com"  # Endpoint for HTTP logger
  http_headers:  # Headers for HTTP logger
    Authorization: "Bearer token"
  http_timeout: 5.0  # Timeout for HTTP logger requests
  
  # Console display options
  show_chat: true  # Show chat messages on console
  show_tools: true  # Show MCP Server tool calls on console
  truncate_tools: true  # Truncate long tool calls in display
  enable_markup: true # Disable if outputs conflict with rich library markup
  enable_prompt_marks: true # Emit OSC 133 prompt marks in supported terminals
  streaming: "markdown"  # "markdown", "plain", or "none"
```

When `logger.path` is omitted, file logging writes to
`<fast-agent-home>/fast-agent-log.jsonl`. Under `--no-home`, it falls back to
`<current-working-directory>/fast-agent-log.jsonl`. Explicit relative paths continue to resolve
from the process current working directory.

## MCP UI Settings

```yaml
mcp_ui_mode: "enabled"  # "disabled", "enabled", or "auto"
mcp_ui_output_dir: ".fast-agent/ui"  # Output directory for generated HTML files
```

## MCP Timeline Settings

```yaml
mcp_timeline:
  steps: 20
  step_seconds: 30  # seconds per bucket (also supports strings like "30s", "2m")
```

## Skills Settings

```yaml
skills:
  directory: null  # Override the default skills directory
```

## Shell Execution Settings

```yaml
shell_execution:
  tool_profile: minimal_process  # Bash + Process (default); native retains legacy tools
  timeout_seconds: 90
  warning_interval_seconds: 30
  interactive_use_pty: true  # Use PTY for interactive prompt shell commands
  output_byte_limit: 8192  # Set null for model-specific automatic sizing
  retain_truncated_output: true
  retained_output_max_bytes: 2097152  # Per shell process
  retained_output_temp_directory: null  # Optional parent directory
  process_poll_max_wait_seconds: 250  # Accepted range: 1–600
  managed_process_poll_history_folding: auto  # auto | on | off
```

`tool_profile` controls the model-facing contract only. The default
`minimal_process` profile exposes `Bash(command, run_in_background?)` and
`Process(process_id, action)`. The `native` profile remains available as a
compatibility escape hatch for the legacy `execute`, `poll_process`, and
`terminate_process` schemas; both profiles use the same managed-process runtime.

When `retain_truncated_output` is enabled, fast-agent creates a private
session-scoped directory and lazily writes a `0600` file only when a shell
result exceeds its model-facing preview. The truncation notice includes the
temporary path so the model can inspect selected ranges or search the complete
output. Each process is limited by `retained_output_max_bytes`; retained files
are removed when the shell runtime closes.

`process_poll_max_wait_seconds` caps a single managed-process wait. The default
stays below Anthropic's five-minute prompt-cache TTL and applies even when a
model or model overlay declares a longer default poll wait.

`managed_process_poll_history_folding` controls whether repetitive quiet
managed-process polling exchanges are collapsed before the next model call:

- `auto` enables folding only for models whose metadata marks the behavior as
  validated.
- `on` forces folding for every model.
- `off` preserves every poll exchange in model history.

## LLM Retries

```yaml
llm_retries: 2
```

## Example Full Configuration

```yaml
default_model: "gpt-5-mini?reasoning=low"

# Model provider settings
anthropic:
  api_key: API_KEY

responses:
  api_key: API_KEY
  reasoning: "high"

# MCP servers
mcp:
  servers:
    fetch:
      transport: "stdio"
      command: "uvx"
      args: ["mcp-server-fetch"]
      
    prompts:
      transport: "stdio"
      command: "prompt-server"
      args: ["prompts/myprompt.txt"]
      
    filesys:
      transport: "stdio"
      command: "uvx"
      args: ["mcp-server-filesystem"]
      roots:
        - uri: "file://./data"
          name: "Data Directory"

# Logging configuration
logger:
  type: "file"
  level: "info"
  path: "logs/fast-agent-log.jsonl"
```

## Environment Variables

All configuration options can be set via environment variables using a nested delimiter:

```
ANTHROPIC__API_KEY=your_key
OPENAI__API_KEY=your_key
LOGGER__LEVEL=debug
```

Environment variables take precedence over values in the configuration files. For nested arrays or complex structures, use the YAML configuration file.

The `fast-agent.yaml` file supports referencing environment variables inline using the `${ENV_VAR}` syntax. When the configuration is loaded, any value specified as `${ENV_VAR}` will be automatically replaced with the value of the corresponding environment variable. This allows you to securely inject sensitive or environment-specific values into your configuration files without hardcoding them.

For example:

```yaml
openai:
  api_key: "${OPENAI_API_KEY}"
```

In this example, the `api_key` value will be set to the value of the `OPENAI_API_KEY` environment variable at runtime.
