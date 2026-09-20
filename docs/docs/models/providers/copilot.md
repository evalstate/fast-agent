---
title: GitHub Copilot
description: Use Copilot models with fast-agent
social:
  title: GitHub Copilot
  tagline: Use Anthropic and OpenAI models through native GitHub Copilot authentication with Messages or Responses wire formats.
  description: Use Anthropic and OpenAI models through native GitHub Copilot authentication with Messages or Responses wire formats.
  alt: fast-agent social card — GitHub Copilot
---

# GitHub Copilot

The `copilot` provider uses native GitHub authentication and provides
direct access to Anthropic and OpenAI models via Messages or
Responses wire formats respectively. The Responses API supports websockets.

The native provider sends `Copilot-Integration-Id: copilot-sdk` by default on
Messages, Responses HTTP/SSE and WebSocket requests. 

## Setup

```bash
uv pip install fast-agent-mcp
fast-agent auth provider login copilot
fast-agent --model copilot.claude-sonnet-5
```

Alternatively, you can use `fast-agent go` and login direct from the model
selection screen.

NB: OAuth credentials are **not model entitlement**: your account and
application still need access to the selected model. Routing uses the local
curated model registry, not the remote `/models` catalog. The fast-agent public
OAuth client ID is
`Ov23li9BBH5sVoKopuI6`; no client secret is required.

Credentials use shared provider-auth storage: the OS keyring when writable,
otherwise `~/.fast-agent/auth.json`. Set `FAST_AGENT_AUTH_FILE` to use an explicit
portable provider credential file. Manage the stored Copilot credential with:

```bash
fast-agent auth provider show copilot
fast-agent auth provider token copilot
fast-agent auth provider export copilot ./copilot.auth.json
fast-agent auth provider logout copilot
```

`show` reports credential status. `token` prints a secret and `export` writes a
secret-bearing file: keep both out of chat, logs and version control.

### Headless authentication

Supply either an exported provider file via `FAST_AGENT_AUTH_FILE` or an
appropriately authorized `COPILOT_GITHUB_TOKEN` from your secret manager.
`COPILOT_GITHUB_TOKEN` takes precedence over stored credentials; `GH_TOKEN` and
`GITHUB_TOKEN` are ignored. A rejected environment credential never starts
interactive login or silently falls back to stored credentials. Noninteractive
inference does not launch login.


## Configuration

Optional defaults in `fast-agent.yaml`, generated from `CopilotSettings`:

--8<-- "_generated/copilot_config_snippet.md"

- `base_url`: HTTPS origin, default `https://api.githubcopilot.com`. No path,
  query, credentials or fragment allowed.
- `integration_id`: configurable routing identity, default `copilot-sdk`.
  Validated as a safe HTTP header value and applied consistently across native
  transports. Configure in YAML or with `COPILOT__INTEGRATION_ID` for a custom ID;
  incidental request-header overrides cannot replace it. A valid value does not imply GitHub registration
  or model access.
- `runtime_timeout_seconds`: positive timeout for runtime operations.
- `cache_mode`: Claude Messages caching policy: `auto` caches the stable prompt
  and advances through recent conversation turns; `prompt` caches tools, system
  instructions and prompt templates; `off` disables automatic markers.
- `cache_ttl`: optional Claude Messages TTL override. `null` preserves the model
  default; explicit values are `5m` and `1h`. One-hour caching has not been live
  verified through Copilot. These cache settings do not affect GPT Responses.

When migrating older configuration, rename `direct_base_url` to `base_url` and
remove `backend`, `use_environment_token` and `cli_path`. These obsolete settings
are rejected, not silently ignored. Remove `COPILOT__BACKEND` from runner
configuration too.

## Exact model names

Use these provider-qualified names; there are no Copilot aliases. Unknown IDs
are rejected rather than routed to another provider. `copilot.claude-opus5` is
not canonical and has no alias; use `copilot.claude-opus-5`.

| Model | Wire API |
| --- | --- |
| `copilot.claude-haiku-4.5` | Messages |
| `copilot.claude-sonnet-5` | Messages |
| `copilot.claude-opus-5` | Messages |
| `copilot.claude-fable-5` | Messages |
| `copilot.claude-fable-5.1` | Messages |
| `copilot.gpt-5.6-luna` | Responses |
| `copilot.gpt-5.6-terra` | Responses |
| `copilot.gpt-5.6-sol` | Responses |
| `copilot.gpt-6-astra` | Responses |

Provider identity stays Copilot regardless of wire API; Anthropic/OpenAI API
keys are not used for these models. The status bar prefixes Copilot model labels
with `(cp)`; this does not change the model name used in configuration.

## Model parameters

Copilot-routed models inherit the base model's local metadata parameters, so
`copilot.claude-opus-5` gets the same `model_specific` prompt text, edit-tool
contract, poll settings, structured-output policy and tokenizer support as
`anthropic.claude-opus-5`. Only route-specific fields differ:

- `default_provider`, `response_transports`, `response_websocket_providers`:
  set from the curated Copilot model spec, not a remote catalog. Gateway
  acceptance is checked only by actual inference.
- `response_service_tiers`: cleared; Copilot rejects service-tier selection.
- `codex_responses_lite`: off; this is a Codex-only request contract.
- `long_context_window`: cleared; extended-context tiers are an upstream
  billing feature the gateway does not expose.
- `anthropic_web_search_version`, `anthropic_web_fetch_version`,
  `anthropic_required_betas`: cleared; the gateway exposes no Anthropic
  server-side web tools or beta headers.

## Prompt caching

Claude Messages uses the normal Anthropic automatic cache planner **by default**:
markers cover the tools/system prefix and advance through recent conversation
turns while retaining the previous cache boundary. No manual `cache_control`
metadata is needed. The TTL is inherited from the resolved model unless
`copilot.cache_ttl` overrides it; missing TTL metadata falls back to the standard
Anthropic settings default, not your Anthropic account configuration.

Use `copilot.cache_mode` to change this policy. Direct `anthropic.cache_mode`,
`anthropic.cache_ttl` and diagnostic settings do not affect Copilot. Explicit
request cache controls are still forwarded even when automatic planning is off.

## Transport and compatibility

The four GPT models default to **WebSockets**, using static curated capability
metadata; no remote catalog is required. Select HTTP/SSE explicitly if needed:

```bash
fast-agent --model 'copilot.gpt-6-astra?transport=sse'
```

The default (and explicit `?transport=websocket`) has **no SSE fallback**.
Explicit `?transport=auto` tries WebSocket first, with SSE fallback on early errors.
Ordinary OpenAI/Codex Responses defaults to `auto`.
Claude models use SSE only. The native runtime does not query `/models`.
