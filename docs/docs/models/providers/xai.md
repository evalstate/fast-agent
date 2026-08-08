---
title: xAI
social:
  title: xAI
  tagline: Configure Grok models, reasoning, web search, and X Search in fast-agent.
  description: Configure Grok models, reasoning, web search, and X Search in fast-agent.
  alt: fast-agent social card — xAI / Grok provider
---

# xAI / Grok

Use the `xai` provider for xAI Grok models. xAI supports both `web_search` and `x_search`; fast-agent sends `x_search` as xAI's provider-managed X Search tool.

## Sign in with a Grok/X subscription

```bash
fast-agent auth provider login xai
```

The device login opens an xAI verification URL and displays a code. Provider
credentials use the OS keyring when it is writable and otherwise fall back to
`~/.fast-agent/auth.json`. Access tokens refresh automatically before expiry.

The model selector also offers this login when an xAI model is selected without
a configured credential.

Useful credential commands:

```bash
fast-agent auth provider show xai
fast-agent auth provider token xai
fast-agent auth provider export xai ./xai.auth.json
fast-agent auth provider logout xai
```

An exported file contains only the selected provider and includes its refresh
token. Set `FAST_AGENT_AUTH_FILE` to use that portable file. This is the
recommended form for long-running Harbor jobs because refreshed credentials are
written back to the staged file.

## Configure

```yaml
xai:
  api_key: "${XAI_API_KEY}"
  # base_url: "https://api.x.ai/v1" # default
```

Environment variables:

- `XAI_API_KEY`: Your xAI API key
- `XAI_BASE_URL`: Override the API endpoint
- `FAST_AGENT_AUTH_FILE`: Explicit portable provider credential file

An explicit `xai.api_key` or `XAI_API_KEY` takes precedence over stored OAuth.

## Use a model

```bash
fast-agent --model "xai.grok-4.3?reasoning=high"
fast-agent --model "xai.grok-4.3?web_search=on"
fast-agent --model "xai.grok-4.3?x_search=on"
fast-agent --model "xai.grok-4.5"
```

## Reasoning and search tools

Useful xAI query parameters:

- `reasoning=low|medium|high` on reasoning-capable Grok models
- `web_search=on|off` for xAI web search
- `x_search=on|off` for xAI's X Search remote tool

`web_search` and `x_search` are distinct provider-managed tools.

Grok 4.5 with `reasoning=high` defaults to a 180-second idle timeout between
stream events. Other model and reasoning combinations retain the global
120-second default. Set `streaming_timeout=<seconds>` to override the default,
or `streaming_timeout=none` to disable stream-idle enforcement.

## Managed process polling

Grok models default to a 240-second managed-process wait when `process(action="wait")`
omits `wait_sec`. This is local fast-agent runtime policy, not an xAI request parameter.
Override it for a model selection with `poll_period=<seconds>`:

```bash
fast-agent --model "xai.grok-4.5?poll_period=420"
```

The value must be an integer from 10 through 3600 and cannot exceed
`shell_execution.process_poll_max_wait_seconds`. For a persistent per-model
default, use an overlay's `metadata.process_poll_default_wait_seconds`.

## Capabilities

Capabilities are model-dependent. See [Models Reference](../models_reference/) for fast-agent's known structured output, reasoning, modality, and tool metadata.

## Model aliases

--8<-- "_generated/model_aliases_xai.md"
