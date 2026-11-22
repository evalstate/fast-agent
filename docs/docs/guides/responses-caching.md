# Responses prompt caching

OpenAI Responses and Codex Responses automatically send a stable
`prompt_cache_key`. This is a cache-routing hint, **not an explicit cache
breakpoint**, and does not guarantee a cache hit. Normal provider prompt-caching
requirements still apply.

The ChatGPT backend behind Codex Responses derives cache affinity from the
`session-id` header rather than the body field, so fast-agent also sends the key as
`session-id`, `thread-id` and `x-client-request-id`, as the Codex CLI does. It also
replays the `x-codex-turn-state` routing token from a turn's first response on that
turn's tool continuations (as a header over SSE, as client metadata over WebSockets),
and drops it at the next user turn. Copilot returns no turn-state token.

The key is retained in assistant history channels, including saved sessions.
Resuming that history restores the latest key; clearing the conversation creates a new
one. Independent LLM instances start with different keys. Copilot sends the key, and
the configuration updates below, for `gpt-6.2-astra` and `gpt-6.2-sol`. Other providers
keep their existing behavior, including xAI Responses.

## System prompt placement (GPT-6.2)

From GPT-6.2, top-level `instructions` are turn-scoped: the API appends them after
`input` and bills them at the uncached rate on every request. For `gpt-6.2-astra` and
`gpt-6.2-sol` on Responses, Codex Responses, and Copilot, fast-agent sends the system
prompt as the first `developer` input message instead, so it stays in the cached
prefix. Older models keep top-level `instructions`. Refreshing the system prompt
rewrites that message, which changes the prefix (as does changing tools).

See OpenAI's [Keep the prefix unchanged](https://developers.openai.com/api/docs/guides/keep-prefix-unchanged).

## Changing reasoning effort

For **`gpt-6-astra`, `gpt-6.2-astra`, and `gpt-6.2-sol` on Responses/Codex
in standard, single-agent mode only**, changing reasoning
effort preserves the original request-level `reasoning.effort`. Before the next
user message, fast-agent inserts:

```json
{"type": "configuration_update", "reasoning": {"effort": "high"}}
```

The effective effort persists until another change, including decreases.
Unchanged effort does not create an update. Changes during a tool-result
continuation (including error text and staged tool media) wait until the next
genuine user turn. Codex's `reasoning.context:
all_turns` remains unchanged.

Assistant channels record the **effective** effort used for each request:
`response.reasoning.effort` reports the original baseline, not the effective
effort. Replaying or resuming history reconstructs updates at their original
user-message boundaries, without adjacent updates.

The current configured effort is the desired effort for new turns, including
after loading history. If it differs from the saved effective effort, the new
turn gets an update; previous turns retain their original effort.

Other models continue to change request-level effort normally.

## Structured output

For the same models, structured output does not change the request-level
`text.format`. fast-agent sends the JSON schema as a `configuration_update` before
the user message, and switches back with `{"type": "text"}` on the next plain turn:

```json
{"type": "configuration_update", "text": {"format": {"type": "json_schema", "...": "..."}}}
```

When the schema applies only after tool calls (the `defer` structured tool policy),
the update is appended after the tool results instead. Effort and format changes at
the same boundary share one update item. Assistant channels record the effective
format, so replay and resume rebuild updates at their original positions. Formats
other than `json_schema` stay request-level.

See OpenAI's [Change the response format mid-conversation](https://developers.openai.com/api/docs/guides/structured-outputs#change-the-response-format-mid-conversation).

## Compaction and overrides

fast-agent uses provider-native checkpoints where supported and enabled, with
client-side summary compaction as a fallback; it never calls `/responses/compact`.
After replacing or truncating local history, surviving records establish a fresh
baseline; if none survive, the current desired effort becomes the baseline. Old
update positions are never reused against a shortened input array.

fast-agent currently rejects automatic provider compaction and truncation with
managed configuration updates. The newer API permits automatic compaction with
updates, but this integration continues to use explicit checkpoints.
OpenAI's standalone `/responses/compact`
also rejects histories containing updates. The API separately permits explicit
`compaction_trigger` items in `/responses`, which fast-agent uses for
[provider-native checkpoints](compaction.md#openai-provider-native-checkpoints)
on Responses, Codex Responses, and Copilot GPT models. A checkpoint resets the effective effort and format, so
fast-agent sends a fresh desired-effort update (and any non-text format) before the next
user message, as the API requires.

`RequestParams.metadata["prompt_cache_key"]` overrides the generated key.
The latest explicit key remains in use on subsequent turns and resume.
For these models, `metadata["reasoning"]["effort"]` selects the **desired** effort, not a
replacement baseline; other reasoning options are retained. Use direct metadata
for these options, not `extra_body`. `extra_body.model` overrides into or out of
these models are rejected. Raw `input` and `previous_response_id` overrides are rejected
for managed configuration-update history because they bypass the persisted turn boundaries.
WebSocket continuation remains automatic.

See OpenAI's [Change reasoning mid-conversation](https://developers.openai.com/api/docs/guides/reasoning#change-reasoning-mid-conversation).

## GPT-6.2 history preservation

GPT-6.2 may remove encrypted reasoning after edits or insertions before a reasoning
item, or after changing `safety_identifier`. Appending items or deleting complete
items does not by itself trigger a prefix mismatch. The provider reports removal
reasons in `input_transformations`. Local history rewrites and client-side summary
compaction do not guarantee reasoning preservation; provider-native checkpoints
are the appropriate option when preservation matters.

## Saved custom-tool calls

New Responses histories retain each tool call's provider IDs and function/custom
kind in an internal channel. Saving and loading that history into a fresh LLM
instance preserves custom calls and their result types, including raw input
whitespace. The metadata is not sent as tool arguments or added to tool schemas.

Older histories without this provenance retain the previous replay behavior;
fast-agent does not guess a custom-tool type from a tool name or an `input`
argument, which ordinary function tools can also use.
