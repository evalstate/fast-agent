# Anthropic structured outputs (beta-only)

## Summary
Fast-agent now uses Anthropic **beta** APIs exclusively for structured output support.
This enables:

- **JSON outputs** via `output_format` (default for structured calls when supported)
- **Strict tool use** via `strict: true` (optional override)
- Beta-native streaming events and richer content blocks

The Anthropic provider no longer uses non-beta `client.messages.*` calls.

## API surface (Anthropic SDK 0.76.0)
We rely on:

- `AsyncAnthropic(...).beta.messages.create(...)`
- `AsyncAnthropic(...).beta.messages.stream(...)`

The beta API exposes:

- `output_format` (JSON schema response format)
- `betas` (beta feature flags)
- `BetaToolParam.strict`
- Beta content blocks / streaming events (`BetaTextBlock`, `BetaToolUseBlock`, etc.)

## Structured output modes
Structured calls (`agent.chat.structured(...)`) can operate in two modes:

| Mode | Description | Anthropic API usage |
| --- | --- | --- |
| `json` (default) | JSON outputs with schema validation | `output_format={"type": "json_schema", "schema": ...}` |
| `tool_use` | Strict tool-use, schema enforced by tool input | `tools=[{..., strict: true}]` + `tool_choice` |

### Default selection
The Anthropic provider selects **JSON outputs** by default when the model supports
Anthropic structured outputs. If the model is unsupported, we automatically fall back
to `tool_use`.

### Override (`structured=` query parameter)
You can override the mode in a model string query parameter:

- `claude-sonnet-4-5?structured=json`
- `claude-sonnet-4-5?structured=tool_use`

This is parsed in `ModelFactory.parse_model_string(...)` and passed to the provider
in the same style as `reasoning=`.

## Model database updates
Anthropic models are annotated for structured output support via `json_mode`.
For models that support JSON outputs, `json_mode="schema"` is retained. For
unsupported models, `json_mode=None`.

Supported in current Anthropic beta docs:

- `claude-sonnet-4-5`
- `claude-sonnet-4-5-20250929`
- `claude-opus-4-1`
- `claude-opus-4-5`
- `claude-haiku-4-5`
- `claude-haiku-4-5-20251001`

## Request construction
### JSON outputs
- Build schema from the Pydantic model.
- Use `anthropic.transform_schema()` for unsupported JSON Schema constraints.
- Pass through `output_format={"type": "json_schema", "schema": ...}`.
- Include beta flag: `betas=["structured-outputs-2025-11-13"]`.

### Strict tool use
- Define a synthetic tool named `return_structured_output`.
- Set `strict: true` on the tool definition.
- Apply `tool_choice` to force tool usage.
- Include beta flag: `betas=["structured-outputs-2025-11-13"]`.

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
- Strict tool use: thinking is disabled because tool choice is forced.

## Testing notes
- Update e2e structured tests to use models that support JSON outputs.
- For strict tool-use coverage, set `?structured=tool_use` in the model string.

## Related files
- `src/fast_agent/llm/provider/anthropic/llm_anthropic.py`
- `src/fast_agent/llm/model_factory.py`
- `src/fast_agent/llm/model_database.py`
- `src/fast_agent/llm/usage_tracking.py`
