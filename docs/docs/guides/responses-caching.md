# Responses prompt caching

OpenAI Responses and Codex Responses automatically send a stable
`prompt_cache_key`. This is a cache-routing hint, **not an explicit cache
breakpoint**, and does not guarantee a cache hit. Normal provider prompt-caching
requirements still apply.

The key is retained in assistant history channels, including saved sessions.
Resuming that history restores the latest key; clearing the conversation creates a new
one. Independent LLM instances start with different keys. Other providers keep
their existing behavior, including xAI Responses.

## Changing Astra effort

For **`gpt-6-astra` in standard, single-agent mode only**, changing reasoning
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

Other models continue to change request-level effort normally. This feature does
not implement configuration updates for other controls or execution modes.

## Compaction and overrides

Fast-agent uses summary compaction, not `/responses/compact`. After replacing or
truncating local history, surviving records establish a fresh baseline; if none
survive, the current desired effort becomes the baseline. Old update positions
are never reused against a shortened input array.

Automatic provider compaction and truncation are incompatible with Astra
configuration updates and are rejected. OpenAI's standalone `/responses/compact`
also rejects histories containing updates. The API separately permits explicit
`compaction_trigger` items in `/responses`; callers using that API must send a
fresh desired-effort update before the next user message. Fast-agent's managed
history path uses summary compaction and does not expose raw trigger items.

`RequestParams.metadata["prompt_cache_key"]` overrides the generated key.
The latest explicit key remains in use on subsequent turns and resume.
For Astra, `metadata["reasoning"]["effort"]` selects the **desired** effort, not a
replacement baseline; other reasoning options are retained. Use direct metadata
for these options, not `extra_body`. `extra_body.model` overrides into or out of
Astra are rejected. Raw `input` and `previous_response_id` overrides are rejected
for managed Astra history because they bypass the persisted turn boundaries.
WebSocket continuation remains automatic.

See OpenAI's [Change reasoning mid-conversation](https://developers.openai.com/api/docs/guides/reasoning#change-reasoning-mid-conversation).
