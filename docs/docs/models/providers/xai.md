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
  # reasoning_summary: concise # experimental; Grok 4.5/4.6
  # stream_tool_calls: true # experimental; Grok 4.5/4.6
  # image_upload_mode: inline # disable temporary image uploads; default: public_url
  # image_upload_ttl_seconds: 86400 # 1 hour to 30 days
```

Environment variables:

- `XAI_API_KEY`: Your xAI API key
- `XAI_BASE_URL`: Override the API endpoint
- `FAST_AGENT_AUTH_FILE`: Explicit portable provider credential file

An explicit `xai.api_key` or `XAI_API_KEY` takes precedence over stored OAuth.

## Reuse images across turns

xAI's Responses API replays conversation context across Grok turns. By default Images are uploaded to temporary URLs to keep transfer size short (references rather than base64 encoding).

!!! info "Temporary public image URLs"

    xAI image understanding accepts public URLs rather than uploaded file IDs.
    fast-agent therefore creates an opaque public xAI CDN URL for each uploaded
    image. Anyone possessing that URL can access it until it expires. Treat
    debug logs and request traces containing these URLs as sensitive until
    expiry.

The file and URL expire together after `image_upload_ttl_seconds`, which
defaults to 1 day (86,400) seconds and accepts values from 3,600 seconds (one hour)
through 2,592,000 seconds (30 days). Canonical conversation history retains the
original image data, so persisted sessions remain portable and expired images
can be uploaded again.

JPEG and PNG images up to xAI's 20 MiB image-understanding limit are uploaded;
remote URLs and other media remain unchanged. If upload is unavailable,
fast-agent falls back to the original inline image. To disable uploads:

```yaml
xai:
  image_upload_mode: inline
```

## Use a model

```bash
fast-agent --model "xai.grok-4.6?reasoning=xhigh"
fast-agent --model "xai.grok-4.6?web_search=on"
fast-agent --model "xai.grok-4.6?x_search=on"
fast-agent --model "xai.grok-4.5"
```

## Reasoning and search tools

Useful xAI query parameters:

- `reasoning=low|medium|high|xhigh` on Grok 4.6
- `reasoning=low|medium|high` on Grok 4.3 and 4.5
- `web_search=on|off` for xAI web search
- `x_search=on|off` for xAI's X Search remote tool

`web_search` and `x_search` are distinct provider-managed tools.

Grok 4.5 and 4.6 also support two opt-in experimental Responses settings:


Grok 4.5 with `reasoning=high`, and Grok 4.6 with `reasoning=high` or
`reasoning=xhigh`, default to a 300-second idle timeout between stream events.
Other model and reasoning combinations retain the global 120-second default.
Set `streaming_timeout=<seconds>` to override the default, or
`streaming_timeout=none` to disable stream-idle enforcement.

fast-agent creates an opaque `prompt_cache_key` for each xAI conversation and
sends it on every Responses API request. The key remains stable across turns so


## Managed process polling

Grok models default to a 240-second managed-process wait when `process(action="wait")`
omits `wait_sec`. This is local fast-agent runtime policy, not an xAI request parameter.
Override it for a model selection with `poll_period=<seconds>`:

```bash
fast-agent --model "xai.grok-4.6?poll_period=420"
```

The value must be an integer from 10 through 3600 and cannot exceed
`shell_execution.process_poll_max_wait_seconds`. For a persistent per-model
default, use an overlay's `metadata.process_poll_default_wait_seconds`.

## Capabilities

Capabilities are model-dependent. See [Models Reference](../models_reference/) for fast-agent's known structured output, reasoning, modality, and tool metadata.

## Model aliases

--8<-- "_generated/model_aliases_xai.md"
