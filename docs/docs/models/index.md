---
title: Model Features and History Saving
---

# Model Features and History Saving

Models in **fast-agent** are specified with a model string that takes the format
`provider.model_name[.reasoning_effort][?query=value&...]`.

### Precedence

Model specifications in fast-agent follow this precedence order (highest to lowest):

1. Explicitly set in agent decorators
1. Command line arguments with `--model` flag
1. Default model in `fastagent.config.yaml`
1. `FAST_AGENT_MODEL` environment variable
1. System default (`gpt-5-mini?reasoning=low`)

### Format

Model strings follow this format: `provider.model_name[?reasoning=value][&query=value...]`

- **provider**: The LLM provider (e.g., `anthropic`, `openai`, `azure`, `deepseek`, `generic`,`openrouter`, `tensorzero`)
- **model_name**: The specific model to use in API calls (for Azure, this is your deployment name)
- **query parameters** (optional): provider/model-specific runtime overrides such as
  `reasoning`, `structured`, `context`, `transport`, `temperature` (`temp` alias),
  `web_search`, and `web_fetch`.
  (`web_fetch` is Anthropic-only; `web_search` is available for Anthropic and
  Responses-family providers).

Examples:

- `anthropic.claude-4-5-sonnet-latest`
- `openai.gpt-5.2`
- `openai.o3-mini?reasoning=high`
- `sonnet?reasoning=4096`
- `openai.o3-mini?reasoning=high`
- `gpt-5?temperature=0.2`
- `claude-opus-4-6?web_search=on&web_fetch=on`
- `openai.gpt-5?web_search=on`
- `azure.my-deployment`
- `generic.llama3.2:latest`
- `openrouter.google/gemini-2.5-pro-exp-03-25:free`
- `tensorzero.my_tensorzero_function`

#### Reasoning Effort

For models that support it (e.g. `o1`, and `gpt-5` etc), you can specify a reasoning effort with
the `reasoning` query parameter: **`high`**, **`medium`** or **`low`** - for example
`openai.o3-mini?reasoning=high`. **`medium`** is the default if not specified.

Anthropic models use either adaptive thinking (effort levels + `auto`) or budget-based thinking
(integer token budgets). Budget models also accept `low`/`medium`/`high`/`max` to map to preset
budgets. Adaptive models default to `auto` and do not accept explicit budgets.

You can also set reasoning directly in the model string query. This is especially useful for
budget-style reasoning models:

- `sonnet?reasoning=4096` (budget tokens)
- `claude-opus-4-6?reasoning=auto` (adaptive default)
- `openai.o3-mini?reasoning=high`

`gpt-5` class models additionally support a `minimal` reasoning effort.

#### Temperature

You can set sampling temperature directly in the model string query:

- `gpt-5?temperature=0.2`
- `openai.gpt-4.1?temp=0.7`

If temperature is omitted, fast-agent does not send a temperature parameter.
Only explicit values (for example via `?temperature=` / `?temp=` or request
params/config) are forwarded.

#### Model presets and model references

For convenience, popular models have built-in **model presets** such as `codex` or `sonnet`.
These are documented on the [LLM Providers](llm_providers.md) page.

You can also create local **model overlays**. These are environment-local named model entries that
bundle endpoint settings, auth, request defaults, and local metadata under a short token such as
`qwen-local`. See [Model Overlays](model_overlays.md).

You can also define your own namespaced **model references** in `fastagent.config.yaml` and
reference them with exact tokens like `$system.fast`.

If a configured model reference cannot be resolved, fast-agent logs a warning and automatically falls back
to the next lower-precedence model source.

### Default Configuration

You can set a default model for your application in your `fastagent.config.yaml`:

```yaml
default_model: "openai.gpt-4o" # Default model for all agents
```

### History Saving

You can save the conversation history to a file by sending a `***SAVE_HISTORY <filename>` message. This can then be reviewed, edited, loaded, or served with the `prompt-server` or replayed with the `playback` model.

!!! Note "File Format / MCP Serialization"

    If the filetype is `json`, fast-agent saves a `{"messages": [...]}` JSON container. It can contain either MCP `PromptMessage` objects (legacy) or `PromptMessageExtended` objects (preserves tool calls, channels, etc). `fast_agent.load_prompt` and `prompt-server` will load either the text or JSON format directly.

This can be helpful when developing applications to:

* Save a conversation for editing
* Set up in-context learning
* Produce realistic test scenarios to exercise edge conditions etc. with the [Playback model](internal_models.md#playback)
