# Anthropic structured outputs

## Summary
Anthropic's current public docs describe structured outputs as generally available on
the Claude API for supported models. In fast-agent, Anthropic structured output support
still has two execution modes:

- **JSON outputs** via `output_config.format` (default on supported models)
- **Legacy `tool_use` fallback** via a synthetic strict tool (used for older models or
  explicit `structured=tool_use` overrides)

Only JSON mode can coexist cleanly with normal tool calling and Anthropic reasoning.
The legacy `tool_use` path is kept as a compatibility fallback and changes request
semantics by replacing normal tools with a synthetic structured-output tool.

## Current SDK / transport notes
The repo is currently pinned to `anthropic[vertex]==0.97.0`.

Fast-agent's provider implementation still uses the SDK beta namespace and beta stream
types internally:

- `AsyncAnthropic(...).beta.messages.create(...)`
- `AsyncAnthropic(...).beta.messages.stream(...)`

This is now slightly behind Anthropic's public documentation terminology, which documents
JSON structured output using `output_config.format` and strict tool use as first-class
structured-output features.

## Structured output modes
Structured calls (`agent.chat.structured(...)`) can operate in two modes:

| Mode | Description | Anthropic API usage |
| --- | --- | --- |
| `json` (default) | JSON outputs with schema validation | `output_config.format={"type": "json_schema", "schema": ...}` |
| `tool_use` | Legacy fallback using a synthetic strict tool | `tools=[{..., strict: true}]` + `tool_choice` |

### Default selection
The Anthropic provider selects **JSON outputs** by default when the model database
marks the model with `json_mode="schema"`. Older / legacy Anthropic entries retain
`json_mode=None`, which causes automatic fallback to `tool_use`.

### Override (`structured=` query parameter)
You can override the mode in a model string query parameter:

- `claude-sonnet-4-5?structured=json`
- `claude-sonnet-4-5?structured=tool_use`

This is parsed in `ModelFactory.parse_model_string(...)` and passed to the provider
in the same style as `reasoning=`.

## Model database updates
Anthropic models are annotated for structured output support via `json_mode`:

- `json_mode="schema"`: use Anthropic JSON structured output by default
- `json_mode=None`: use legacy `tool_use` fallback by default

Current fast-agent Anthropic entries that default to JSON mode:

- `claude-opus-4-1`
- `claude-opus-4-5`
- `claude-opus-4-6`
- `claude-opus-4-7`
- `claude-sonnet-4-5`
- `claude-sonnet-4-5-20250929`
- `claude-sonnet-4-6`
- `claude-haiku-4-5`
- `claude-haiku-4-5-20251001`

## Request construction
### JSON outputs
- Build schema from the Pydantic model.
- Use `anthropic.transform_schema()` for unsupported JSON Schema constraints.
- Pass through `output_config.format={"type": "json_schema", "schema": ...}`.
- Include beta flag: `betas=["structured-outputs-2025-11-13"]`.
- Preserve normal tools when they are supplied alongside structured output.

### Legacy `tool_use` fallback
- Define a synthetic tool named `return_structured_output`.
- Set `strict: true` on the tool definition.
- Apply `tool_choice` to force tool usage.
- Include beta flag: `betas=["structured-outputs-2025-11-13"]`.
- Suppress normal tools for that structured turn.

## Streaming and telemetry wiring
The beta streaming events map directly to existing stream hooks:

| Beta event/block | Usage | Existing hook |
| --- | --- | --- |
| `BetaRawContentBlockStartEvent` | tool use starts | `_notify_tool_stream_listeners("start", ...)` |
| `BetaRawContentBlockDeltaEvent` + `BetaInputJSONDelta` | streaming tool JSON | `_notify_tool_stream_listeners("delta", ...)` |
| `BetaRawContentBlockStopEvent` | tool use ends | `_notify_tool_stream_listeners("stop", ...)` |
| `BetaTextDelta` | streaming text | `_notify_stream_listeners(StreamChunk(...))` |
| `BetaThinkingDelta` | streaming reasoning | `_notify_stream_listeners(..., is_reasoning=True)` |
| `BetaRawMessageDeltaEvent` | output token counts | `_update_streaming_progress(...)` |

Beta content blocks add structured tool results and container/code execution
results. These blocks are preserved in `PromptMessageExtended` channels for
future expansion (citations, server tools, etc.).

## Reasoning and thinking blocks
Extended thinking (`thinking=...`) remains supported for Anthropic models that
declare `reasoning="anthropic_thinking"` in the model database.

When structured outputs are enabled:

- JSON outputs: extended thinking is allowed; the grammar applies only to the final output.
- Legacy `tool_use` fallback: thinking is disabled because tool choice is forced.

## Combined tools + structured outputs
Anthropic's current docs allow JSON outputs and strict tool use features to be used
together in a single request. Fast-agent partially reflects that today:

- JSON mode keeps normal tools enabled.
- Legacy `tool_use` fallback does not; it swaps in the synthetic structured-output
  tool and suppresses normal tools.

The provider now emits an explicit warning when a structured request with tools has to
fall back to legacy `tool_use` semantics.

## Testing notes
- Update e2e structured tests to use models that support JSON outputs.
- For legacy fallback coverage, set `?structured=tool_use` in the model string or use
  a legacy Anthropic model entry.

## Related files
- `src/fast_agent/llm/provider/anthropic/llm_anthropic.py`
- `src/fast_agent/llm/model_factory.py`
- `src/fast_agent/llm/model_database.py`
- `src/fast_agent/llm/usage_tracking.py`
