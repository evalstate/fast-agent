# Structured Tool Result Handling Plan

## Goal

Align LLM-facing tool result serialization with the MCP `structuredContent` field when it is present, while keeping provider-specific changes small and predictable.

Today there is a mismatch:

- the UI preview path prefers `structuredContent` for display in some cases
- provider serialization paths send `result.content` to the model
- the passthrough model narrows this even further and currently uses only `TextContent[0]`

That creates a bad failure mode:

- the user sees one payload in the tool result panel
- the model reasons over a different payload

This plan makes `structuredContent` the canonical source for LLM-facing text when present, but does so in a centralized helper layer instead of scattering policy across providers.

---

## Current State

### UI already treats `structuredContent` specially

The preview logic in:

- `src/fast_agent/ui/tool_display.py`

will replace multiple text blocks with pretty-printed JSON from `structuredContent` for display purposes.

This is a UI-only transformation.

### Providers currently ignore `structuredContent`

The current tool-result serialization paths all work from `result.content`:

- `src/fast_agent/llm/provider/openai/multipart_converter_openai.py`
- `src/fast_agent/llm/provider/openai/responses_content.py`
- `src/fast_agent/llm/provider/anthropic/multipart_converter_anthropic.py`
- `src/fast_agent/llm/provider/google/google_converter.py`
- `src/fast_agent/llm/provider/bedrock/llm_bedrock.py`
- `src/fast_agent/llm/provider/bedrock/multipart_converter_bedrock.py`
- `src/fast_agent/llm/internal/passthrough.py`

As a result, `structuredContent` is not currently part of the model-visible tool result unless some server already duplicated it into text blocks.

### MCP intent vs practice

The MCP shape expects the text content and `structuredContent` to agree semantically, but it does not enforce that invariant. In practice, tools sometimes:

- return several `TextContent` blocks instead of one
- return text that is stale, lossy, or human-oriented
- diverge from the actual `structuredContent`

We should therefore explicitly pick a canonical source rather than assuming these fields always agree.

---

## Recommendation

Introduce a centralized canonicalization helper for tool results in the content helper layer.

Rule:

- if `structuredContent` is absent, preserve current behavior and use `result.content`
- if `structuredContent` is present, synthesize a single canonical JSON `TextContent` for LLM text serialization
- preserve non-text blocks from `result.content` unchanged

This keeps the policy in one place and limits provider edits to swapping raw `result.content` iteration for helper output.

---

## Proposed Design

### 1. Add a canonical LLM-view helper

Add helper functions in:

- `src/fast_agent/mcp/helpers/content_helpers.py`

Recommended shape:

- `canonicalize_tool_result_content_for_llm(result, logger=None, source=None) -> list[ContentBlock]`
- optionally `tool_result_text_for_llm(result, logger=None, source=None) -> str`

Core behavior:

1. Inspect `result.content`
2. Inspect `getattr(result, "structuredContent", None)`
3. If no `structuredContent`:
   - return the original logical content view
4. If `structuredContent` is present:
   - gather all non-text content blocks from `result.content`
   - replace all text blocks with one synthesized `TextContent`
   - synthesized text is compact JSON from `structuredContent`
   - return `[synthetic_text_block, *non_text_blocks]`

This helper should not mutate the original `CallToolResult`.

### 2. JSON serialization policy

For the synthetic text block:

- use `json.dumps(structured_content, ensure_ascii=False, sort_keys=True, separators=(",", ":"))`

Rationale:

- compact JSON is cheaper for model input than indented JSON
- `ensure_ascii=False` preserves readable Unicode
- `sort_keys=True` improves determinism in tests and logs

The UI can continue to pretty-print separately for human readability. The LLM path should optimize for canonicality and compactness.

### 3. Warning policy

Emit a warning only in the narrow case that is most likely to indicate divergence or unexpected formatting:

- `structuredContent` is present
- there are more than one `TextContent` blocks in `result.content`

Warning message should make the behavior explicit:

- multiple text blocks were present alongside `structuredContent`
- fast-agent is ignoring those text blocks for LLM text serialization
- `structuredContent` is being used as the canonical text payload

This warning should be best-effort and non-fatal.

### 4. Comment the spec divergence clearly

The helper should contain a short explanatory comment stating:

- MCP intends text content and `structuredContent` to agree
- that invariant is not enforced in practice
- fast-agent prefers `structuredContent` for LLM-facing text to avoid divergence between model input and displayed structured preview

This comment matters because otherwise the helper will look like a surprising override of user-provided text blocks.

---

## Why the Content Layer Is the Right Place

This change is a policy decision about canonical representation of a tool result, not a provider capability decision.

Placing it in the content helper layer has these benefits:

- one place defines the rule
- providers remain thin serializers
- warning behavior is shared and consistent
- future provider implementations inherit the policy automatically
- tests can target the policy once rather than re-testing each provider in depth

This is the cleanest way to keep provider blast radius manageable.

---

## Provider Impact

Provider changes should be minimal and mechanical.

### OpenAI chat

Current file:

- `src/fast_agent/llm/provider/openai/multipart_converter_openai.py`

Change:

- when building tool response messages, iterate the canonicalized helper output rather than raw `tool_result.content`

### OpenAI Responses

Current file:

- `src/fast_agent/llm/provider/openai/responses_content.py`

Change:

- `_tool_result_to_text(...)` should use canonicalized content
- `_tool_result_to_input_parts(...)` should preserve non-text attachments while using canonical text

### Anthropic

Current file:

- `src/fast_agent/llm/provider/anthropic/multipart_converter_anthropic.py`

Change:

- build tool result blocks from canonicalized content

### Google

Current file:

- `src/fast_agent/llm/provider/google/google_converter.py`

Change:

- collect textual outputs from canonicalized content
- continue preserving media/resource parts as today

### Bedrock

Current files:

- `src/fast_agent/llm/provider/bedrock/llm_bedrock.py`
- `src/fast_agent/llm/provider/bedrock/multipart_converter_bedrock.py`

Change:

- use canonicalized content in both the instance-based and static conversion paths

### Passthrough

Current file:

- `src/fast_agent/llm/internal/passthrough.py`

Change:

- stop using `tool_result.content[0]`
- build passthrough text from the canonicalized text view instead

This is especially important because passthrough currently has the sharpest reduction of tool-result fidelity.

---

## Trade-offs

### Trade-off 1: prefer `structuredContent` over textual prose

Decision:

- when `structuredContent` is present, it becomes the canonical source for LLM-facing text

Pros:

- avoids user-visible/model-visible divergence
- aligns the model with the structured payload rather than potentially stale prose
- deterministic and easy to reason about

Cons:

- if a tool intentionally included useful narrative explanation in text blocks, that prose will no longer be the primary text passed to the model
- some tools may rely on descriptive text phrasing rather than pure data

Why this trade-off is acceptable:

- the presence of `structuredContent` is a strong signal that the structured payload is the authoritative result
- explanatory text can still be preserved in future if needed, but using divergent text as canonical input is the riskier default

### Trade-off 2: coerce structure to JSON string rather than provider-native structure

Decision:

- convert `structuredContent` to JSON text for current provider input paths

Pros:

- integrates cleanly with all existing providers
- avoids provider-specific schema branching
- keeps blast radius small

Cons:

- loses native structured semantics at the transport boundary
- model sees structure as text rather than an explicitly typed object

Why this trade-off is acceptable:

- all current tool-result serialization paths already operate on text/media/resource content
- introducing provider-native structured tool-result objects would be a much larger redesign

### Trade-off 3: warn on multiple text blocks, but do not attempt semantic comparison

Decision:

- warn only for the presence of multiple text blocks alongside `structuredContent`

Pros:

- low noise
- easy to implement
- catches a likely unexpected shape

Cons:

- does not detect divergence when there is exactly one misleading text block
- does not measure semantic mismatch directly

Why this trade-off is acceptable:

- semantic comparison would require parsing arbitrary text and would likely be noisy and brittle
- this plan favors a reliable canonicalization rule over speculative validation

### Trade-off 4: preserve non-text blocks

Decision:

- keep non-text `result.content` blocks intact when `structuredContent` is present

Pros:

- preserves multimodal behavior
- minimizes regression risk for image/document/resource tool results

Cons:

- canonicalized text plus preserved attachments can still produce mixed payloads

Why this trade-off is acceptable:

- we are changing only text canonicalization, not the broader multimodal contract
- removing non-text blocks would be a much more disruptive change

---

## Non-Goals

This plan does not attempt to:

- enforce MCP server correctness at the server boundary
- compare text content and `structuredContent` semantically
- introduce a new `CallToolResult` schema
- redesign provider APIs around native structured tool outputs
- change UI preview behavior beyond keeping it conceptually aligned with the new LLM policy

---

## Implementation Slices

### Slice 1: helper layer

Files:

- `src/fast_agent/mcp/helpers/content_helpers.py`

Changes:

- add canonical tool-result helper(s)
- add docstring/comment explaining MCP divergence and chosen canonicalization rule
- add optional warning support via passed logger

### Slice 2: provider adoption

Files:

- `src/fast_agent/llm/provider/openai/multipart_converter_openai.py`
- `src/fast_agent/llm/provider/openai/responses_content.py`
- `src/fast_agent/llm/provider/anthropic/multipart_converter_anthropic.py`
- `src/fast_agent/llm/provider/google/google_converter.py`
- `src/fast_agent/llm/provider/bedrock/llm_bedrock.py`
- `src/fast_agent/llm/provider/bedrock/multipart_converter_bedrock.py`
- `src/fast_agent/llm/internal/passthrough.py`

Changes:

- replace direct raw text iteration with canonical helper usage
- preserve current provider-specific handling for images/resources

### Slice 3: tests

Add focused tests for:

- helper behavior with no `structuredContent`
- helper behavior with matching `structuredContent`
- helper behavior with divergent text blocks
- warning emission when multiple text blocks are present
- one or two targeted provider-path tests to confirm canonical text is used

Recommended initial test locations:

- `tests/unit/fast_agent/mcp/...` for helper policy
- existing provider unit test files where tool-result serialization is already covered
- passthrough unit tests, since that path currently has the narrowest behavior

### Slice 4: verification

Run:

- targeted unit tests for helper/provider paths
- `uv run scripts/lint.py`
- `uv run scripts/typecheck.py`

---

## Risks

### Behavioral regression for tools that relied on text prose

Some tools may have been relying on prose-oriented text blocks as the model-facing summary even when `structuredContent` was present.

Mitigation:

- keep the rule narrow and explicit
- cover changed behavior in tests
- document the trade-off in code comments and changelog/PR notes if needed

### Provider inconsistency if one path is missed

If a provider path keeps using raw `result.content`, we would still have inconsistent model behavior.

Mitigation:

- centralize the policy in helpers
- audit all known tool-result serializers in this plan
- add at least one provider-path regression test plus passthrough coverage

### Log noise

Warnings could become noisy if many third-party tools emit multiple text blocks.

Mitigation:

- only warn in the narrow multi-text-block case
- do not warn when `structuredContent` is absent
- do not warn on every ordinary single-block structured tool result

---

## Recommendation Summary

Recommended implementation:

1. canonicalize tool-result text in the content helper layer
2. prefer `structuredContent` whenever it is present
3. preserve non-text attachments
4. warn when multiple text blocks accompany `structuredContent`
5. keep provider changes minimal and mechanical

This is the lowest-risk way to make displayed tool results and model-visible tool results agree without expanding the change into a provider-specific redesign.
