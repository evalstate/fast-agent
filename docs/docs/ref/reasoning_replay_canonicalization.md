---
title: Reasoning Replay Canonicalization
description: Persist and replay encrypted Responses reasoning with a stable, secure schema.
social:
  title: Reasoning Replay Canonicalization
  tagline: Preserve per-item reasoning state without persisting arbitrary SDK snapshots.
  description: Persist and replay encrypted Responses reasoning with a stable, secure schema.
  alt: fast-agent social card — Reasoning Replay Canonicalization
---

# Reasoning replay canonicalization

Status: engineering behavior note and implemented canonical replay design.

Fast-Agent uses OpenAI Responses-compatible providers with:

```text
store: false
include: ["reasoning.encrypted_content"]
```

To continue a stateless or Zero Data Retention conversation, completed
reasoning output items must be preserved and supplied as input on later
requests.

Fast-Agent changed how those items are persisted after v0.10.4:

```text
v0.10.4:
  persist a hand-built object containing:
    type
    encrypted_content
    optional id

v0.10.6 through v0.10.11:
  persist snapshot_json_value(output_item)
  remove only status
```

The v0.10.6 behavior fixes a real correctness problem: each reasoning item
retains its own raw `summary`. Reverting to the v0.10.4 minimal object would
restore lossy, message-wide summary reconstruction.

The design problem is narrower:

> A generic SDK object snapshot should not become Fast-Agent's unversioned
> persistence and replay schema.

Fast-Agent should preserve the provider-required per-item state through an
explicit, versioned, allowlisted envelope.

## Exact source change

The behavior was introduced by:

```text
commit 849325bacecfba3f81b1a89044541705a09c8e69
xai summaries&streaming (#919)
```

It first shipped in Fast-Agent v0.10.6.

Fast-Agent v0.10.4 constructed:

```python
payload = {
    "type": "reasoning",
    "encrypted_content": encrypted_content,
}
if item_id:
    payload["id"] = item_id
```

Fast-Agent v0.10.6 through v0.10.11 constructs:

```python
payload = snapshot_json_value(output_item)
if not isinstance(payload, dict) or not payload.get("encrypted_content"):
    continue
payload.pop("status", None)
```

Relevant implementation:

```text
src/fast_agent/llm/provider/openai/responses_output.py
```

Published v0.10.11 wheel:

```text
fast_agent_mcp-0.10.11-py3-none-any.whl
SHA-256:
a4b786b2d22271f3cd0347824f3b302b5b957fcae0993050687d6c704b6e7b4a
```

The relevant wheel sources match the audited release branch.

## Current development implementation

Fast-Agent now persists an explicit versioned envelope and reconstructs only
the provider input fields allowed by the reasoning item contract:

```text
type
id
summary
encrypted_content
optional validated nonempty content
```

The reader accepts only this canonical envelope. Earlier minimal records, raw
SDK snapshots, unknown versions, and malformed envelopes are non-resumable and
do not become provider input. Unknown SDK fields, lifecycle `status`, and null
or empty `content` do not cross the provider request boundary.

The xAI/Grok replay policy remains provider-specific:

- xAI WebSocket requests remain stateless and replay full context;
- repeated assistant-message IDs are still removed;
- exact duplicate reasoning state is dropped; and
- the same reasoning ID with different ciphertext is retained.

Relevant implementation:

```text
src/fast_agent/llm/provider/openai/reasoning_replay.py
```

As of August 28, 2026, Fast-Agent pins OpenAI Python SDK v3.5.0. Its reasoning
input and output item definitions are unchanged from v3.3.1. The v3.4.0
release contains SSE and WebSocket transport fixes, but no reasoning replay
schema change.

## Provider item contract

For the OpenAI SDK versions used by Fast-Agent v0.10.4 and v0.10.11, the
reasoning input item contract contains:

Required:

```text
type
id
summary
```

Optional:

```text
content
encrypted_content
status
```

For Fast-Agent's `store: false` replay path, `encrypted_content` is also
operationally required.

Provider guidance requires replaying the **completed** reasoning item. In a
streaming implementation, `encrypted_content` from an initial
`response.output_item.added` event may be incomplete; capture should use the
completed/done item.

### Field interpretation

`type`
: Provider item discriminator. Must be `reasoning`.

`id`
: Provider reasoning-item identity. Required for the input item.

`encrypted_content`
: Opaque resumable reasoning state. Treat as sensitive replay material.

`summary`
: Item-specific structured summary. Required and semantically significant.

`content`
: Optional reasoning content. Omit when null or empty unless a provider
  explicitly requires the empty value.

`status`
: Output lifecycle state. It should not be persisted or replayed as input.

## Why preserving per-item summary is correct

Fast-Agent stores two channels on an assistant history message.

### Display reasoning channel

```text
reasoning
```

This contains model-visible or UI-visible joined reasoning summary text.

### Replay channel

```text
openai-reasoning-encrypted
```

This contains JSON-encoded provider reasoning items.

When reconstructing a request, Fast-Agent reads the encrypted replay channel.
If a stored item lacks `summary`, it constructs a fallback from the entire
message's display reasoning channel.

Relevant implementation:

```text
src/fast_agent/llm/provider/openai/responses_content.py
```

### v0.10.4 legacy fallback problem

Because v0.10.4 did not persist item-specific summaries, request
reconstruction assigned the same message-wide joined summary to every
encrypted item in that message.

For a message containing multiple reasoning items:

```text
item A raw summary: A
item B raw summary: B
display summary:    A + B
```

Legacy reconstruction became:

```text
item A summary: A + B
item B summary: A + B
```

This:

- loses item boundaries;
- loses multipart summary structure;
- duplicates the combined summary for each item; and
- increases wire payload for multi-item messages.

The v0.10.11 change correctly preserves:

```text
item A summary: A
item B summary: B
```

The solution must keep this property.

## Why the generic SDK snapshot is still the wrong boundary

`snapshot_json_value(output_item)` serializes whatever fields the installed
SDK model currently exposes.

That is convenient for short-term provider fidelity but weak as a durable
history contract.

### No field allowlist

Only `status` is explicitly removed. Future SDK/provider fields may silently
enter:

- persisted session history;
- history exports;
- request replay;
- logs; and
- debug captures.

Fast-Agent should intentionally choose which fields cross each boundary.

### No schema version

The JSON string stored in `TextContent.text` has no Fast-Agent schema name or
version. A later reader cannot distinguish:

- v0.10.4 minimal records;
- v0.10.11 raw snapshots;
- provider-specific records;
- future canonical records; or
- malformed user-supplied channel data.

### Optional null/empty values are retained

The source test added with the snapshot change explicitly expects:

```json
{
  "content": null
}
```

The SDK input field is optional but is not necessarily nullable when present.
Omission is safer than retaining null.

In the audited Luna/max histories, every current item contained:

```text
content
encrypted_content
id
summary
type
```

and `content` was always an empty list. No current provider item contained
nonempty reasoning content.

### Future nonempty content may duplicate sensitive text

If a provider begins returning nonempty reasoning `content`, generic snapshot
storage may persist that text:

- inside the private replay item; and
- separately in the display reasoning channel.

This needs an explicit privacy and derivation policy.

### Forward compatibility is provider-controlled

A newly added SDK output field might not be accepted as an input field by:

- an older SDK;
- another Responses-compatible provider;
- SSE and WebSocket transports equally; or
- a future provider version.

Persisted history should not depend on arbitrary output-model expansion.

## Current reconstruction behavior

For each historical assistant message, Fast-Agent:

1. parses every JSON block in `openai-reasoning-encrypted`;
2. requires the exact canonical schema name and version;
3. requires an exact canonical item field set;
4. validates `type`, `id`, `summary`, `encrypted_content`, and optional
   nonempty `content`;
5. strips the envelope before provider input;
6. inserts replay items before message/tool items; and
7. applies provider-specific deduplication.

### ID-only deduplication weakness

Dropping every later item with the same `id` assumes IDs are globally unique
and immutable across:

- providers;
- sessions;
- retries;
- resumed histories; and
- ciphertext revisions.

The audited histories had:

```text
zero missing IDs
zero duplicate IDs
zero reused IDs with distinct ciphertext
```

So there was no observed loss. The generic contract remains fragile.

Prefer a dedupe identity including:

```text
provider
id
SHA-256(encrypted_content)
```

Drop only exact duplicates. If the same provider/ID appears with different
ciphertext, retain it or emit a hard diagnostic rather than silently
discarding state.

## Performance interpretation

The snapshot change increases persisted history size because each record now
contains item-specific summary structure and an empty `content` field.

Across the two 113-task Fast-Agent 0.10.11 Luna/max corpora, the extra stored
snapshot fields contributed approximately:

```text
run 1: 4.34 MB
run 2: 4.19 MB
```

However, this is not the primary explanation for the observed 14–21% total
input growth.

### Current representation is smaller on wire than legacy reconstruction

Applying the v0.10.4 message-wide fallback algorithm to the same current
histories would repeat combined summaries for multi-item messages.

The current per-item representation reduced reconstructed replay bytes by
approximately:

```text
8.9–9.2%
```

on those histories, including approximately:

```text
605–631 MB less cumulative replay across successful turns
```

than the legacy counterfactual.

Therefore:

> Preserve per-item summary. Do not attribute the broad benchmark token
> regression to that correction alone.

### What actually grew

Compared with the v0.10.4 run, current corpora had:

```text
about 10–15% more reasoning items
about 31–37% more encrypted ciphertext bytes
about 20–25% more completion reasoning tokens
more provider/tool turns in one run
```

Those are larger behavioral changes than the 4.2–4.4 MB of extra snapshot
fields.

The likely performance issue is more reasoning/trajectory growth overall, not
the fact that item-specific summaries are retained.

## Security and privacy

Encrypted reasoning state is opaque, but it is reusable conversation state.
Treat it like a bearer-capable secret rather than harmless telemetry.

### Persistence permissions

Current histories used:

```text
history files:      0600
session directories: 0700
```

This is appropriate.

Older copied histories were observed with group-readable/group-writable
permissions. Migration and export code should preserve or strengthen private
permissions.

### Normal debug logging

Responses request construction calls:

```python
self.logger.debug("Responses request", data=arguments)
```

The normal serializer redacts keys containing values such as:

```text
token
secret
password
auth
private_key
```

It does not currently classify:

```text
encrypted_content
```

as sensitive.

### Stream capture

When `FAST_AGENT_LLM_TRACE` is enabled, stream capture writes complete request
input to disk. That includes replay items and ciphertext.

Debug tooling should default to structural data:

```text
item count
field names
ciphertext byte length
ciphertext SHA-256
summary part count
```

Raw replay state should require an explicit unsafe opt-in and private file
permissions.

### History export

Exports that include replay channels should:

- clearly warn that resumable encrypted provider state is included;
- preserve private permissions;
- offer a redacted/non-resumable export mode; and
- avoid presenting ciphertext as ordinary diagnostics.

## Canonical replay envelope

Persist a versioned Fast-Agent envelope rather than the raw SDK model:

```json
{
  "schema": "fast-agent.openai-responses.reasoning-replay",
  "version": 1,
  "item": {
    "type": "reasoning",
    "id": "rs_...",
    "encrypted_content": "<opaque>",
    "summary": [
      {
        "type": "summary_text",
        "text": "..."
      }
    ]
  }
}
```

### Allowed provider item fields

Allow only:

```text
type
id
encrypted_content
summary
optional nonempty content
```

Never persist/replay:

```text
status
unknown SDK extras
transport event metadata
timestamps
usage
debug fields
```

### Canonicalization rules

1. Capture only completed reasoning output items.
2. Require nonempty `id` and `encrypted_content`.
3. Require a structured per-item `summary`, even when it is an empty list.
4. Preserve summary part order, types, and text exactly.
5. Omit `content` when null or empty.
6. Preserve nonempty `content` only if it matches an explicitly supported
   input schema.
7. Validate at history write and again at request reconstruction.
8. Strip the Fast-Agent envelope before sending provider input.
9. Reject or diagnose unknown envelope versions.
10. Keep provider-specific canonicalizers separate when contracts differ.

## Display reasoning versus replay reasoning

Fast-Agent currently persists both:

```text
reasoning
openai-reasoning-encrypted
```

This is intentional separation of display and replay concerns, but it
duplicates summary text in stored history.

Two valid designs are possible.

### Design A: persist both channels

Advantages:

- current UI/history consumers remain simple;
- display reasoning remains provider-independent;
- no decryption or provider item parsing is needed for display.

Requirements:

- mark display reasoning as derived;
- never use display text for canonical replay;
- ensure exports explain the duplication; and
- test that display and replay boundaries remain consistent.

### Design B: persist canonical replay, derive display summary on load

Advantages:

- one persisted source of truth;
- less duplicated history;
- item boundaries remain explicit.

Risks:

- provider-specific replay parsing enters display/history loading;
- redacted exports need a separately preserved display representation;
- future schema changes require an explicit new version.

The initial implementation should likely retain both channels for
compatibility while making the replay envelope authoritative.

## Noncanonical history

Fast-Agent does not migrate pre-canonical reasoning replay records. The
assistant display history remains available, but encrypted reasoning state is
not replayed unless it is stored in the canonical envelope.

This avoids:

- lossy reconstruction of per-item summaries;
- mixed persistence schemas;
- forwarding raw SDK snapshots;
- provider behavior that depends on upgrade history; and
- a permanent migration state machine in the request path.

## Proposed implementation boundaries

Introduce explicit functions/types rather than sharing generic object
serialization:

```python
canonicalize_completed_reasoning_output_item(...)
serialize_reasoning_replay_envelope(...)
parse_reasoning_replay_envelope(...)
reasoning_replay_item_to_provider_input(...)
reasoning_replay_dedupe_key(...)
redact_reasoning_replay_for_diagnostics(...)
```

Keep:

- SDK output parsing;
- durable history schema;
- provider input schema; and
- debug serialization

as separate boundaries.

## Telemetry

Safe per-turn telemetry:

```text
reasoning_replay_item_count
reasoning_replay_encrypted_bytes
reasoning_replay_summary_parts
reasoning_replay_summary_bytes
reasoning_replay_content_parts
reasoning_replay_content_bytes
replay_deduplicated_exact_count
replay_id_ciphertext_conflict_count
canonical_schema_version
```

Do not emit:

```text
encrypted_content
reasoning text
full request input
```

in ordinary telemetry.

## Required tests

### Canonicalization tests

1. Exact field allowlist.
2. Unknown SDK fields are dropped.
3. `status` is dropped.
4. Null/empty `content` is omitted.
5. Valid nonempty `content` is preserved.
6. Missing `id` is rejected.
7. Missing/empty `encrypted_content` is rejected.
8. Summary part order and boundaries round-trip exactly.
9. Capture is sourced from a completed output item.

### Multi-item replay tests

1. Two reasoning items retain distinct summaries.
2. No canonical item receives message-wide fallback.
3. Multipart summaries remain multipart.
4. Provider request order matches response/history order.
5. Display summary derivation does not alter replay items.

### Strict reader tests

1. v0.10.4 minimal records do not become provider input.
2. Raw SDK snapshots do not become provider input.
3. Unknown canonical envelope versions fail safely.
4. Unknown canonical fields fail safely.
5. Malformed channel JSON does not become provider input.

### Provider compatibility tests

1. OpenAI SDK 2.53.0 request serialization.
2. OpenAI SDK 3.3.1 request serialization.
3. SSE and WebSocket request builders produce the same canonical reasoning
   input.
4. Unknown future output fields never appear on provider input.
5. Null-versus-omitted optional-field matrix.

### Deduplication tests

1. Same provider, ID, and ciphertext is dropped as an exact duplicate.
2. Same ID with different ciphertext is not silently dropped.
3. Same ciphertext with different IDs follows explicit provider policy.
4. Same ID on different providers is not conflated.
5. Deduplication remains stable across session resume.

### Persistence and security tests

1. Session directory mode is `0700`.
2. History file mode is `0600`.
3. Normal debug logs do not contain ciphertext.
4. Default stream captures redact replay state.
5. Unsafe raw capture requires an explicit opt-in.
6. Redacted export is clearly non-resumable.
7. Resumable export warns and preserves private permissions.

### Payload regression tests

1. Golden canonical-envelope byte size.
2. No empty `content` field.
3. Multi-item per-item summaries are smaller than legacy repeated
   message-wide fallback.
4. Cumulative replay bytes stay bounded on a synthetic long tool loop.
5. Structural telemetry matches actual serialized bytes without exposing
   content.

### Mocked transport integration test

Without calling a model:

1. construct a completed response containing multiple reasoning items;
2. include distinct multipart summaries and encrypted values;
3. canonicalize and persist history;
4. reload history;
5. build SSE and WebSocket Responses requests;
6. assert exact allowlisted fields, ordering, summary association, and dedupe
   behavior;
7. assert no `status`, unknown SDK fields, or empty `content`; and
8. assert logs/captures contain only redacted structural metadata.

## Performance validation

No paid model call is required to validate canonicalization.

Use recorded or synthetic histories to compare:

```text
current raw snapshot
proposed canonical envelope
legacy v0.10.4 fallback
```

Measure:

- stored history bytes;
- one-final-request replay bytes;
- cumulative replay bytes over all turns;
- serialization/deserialization time;
- summary duplication;
- unknown-field leakage; and
- log/export redaction.

If a model evaluation is later authorized, reasoning replay should be tested
independently from shell auto-await and concurrency. Do not infer its effect
from a benchmark where those variables changed simultaneously.

## Interpretation of the Luna/max evidence

The Luna/max runs establish:

- the current histories contain more reasoning items and ciphertext;
- total prompt/context use increased substantially;
- full snapshots add modest storage overhead;
- current per-item summaries avoid larger legacy message-wide repetition; and
- the generic snapshot boundary creates schema/security risks.

They do **not** establish that preserving per-item summaries caused the score
decline.

The correct engineering response is:

```text
keep per-item summary fidelity
replace generic SDK snapshots with a canonical schema
redact replay state
version and validate history explicitly
measure payload growth structurally
```
