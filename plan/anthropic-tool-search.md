# Plan: Provider-neutral Tool Search Config with Anthropic Implementation (Option B)

## Summary

Implement a **provider-neutral `tool_search` configuration** on `AgentConfig` and AgentCard, with a **first implementation for Anthropic server-side tool search** and **MCP tools only**.

This keeps the product model clean (cross-provider feature shape), but limits initial execution scope and risk by:

- targeting Anthropic only,
- deferring only MCP tools (not local/function/shell/skill tools),
- preserving existing MCPAggregator ownership, permission flow, and tool execution behavior.

---

## Goals

1. Add a durable config model in `AgentConfig` + AgentCard for tool-search behavior.
2. Support Anthropic server-side tool search (`regex` / `bm25`) with deferred MCP tool loading.
3. Keep 3–5 frequently used tools eager using explicit overrides.
4. Reuse current tool filtering semantics and keep behavior predictable.
5. Make implementation extensible for future non-Anthropic providers and/or custom client-side search.

## Non-goals (v1)

- Deferring local/function/shell/skill/human_input tools.
- Switching to Anthropic `mcp_toolset` execution model.
- Implementing custom embedding-based search tools.
- Introducing global auto-tuning of eager/deferred sets.

---

## Why Option B

Option B gives us:

- A **stable product contract** (`tool_search`) independent of provider specifics.
- A clean path to add OpenAI/other provider equivalents later.
- Minimal coupling between AgentCard semantics and Anthropic wire details.

Anthropic-specific details remain in provider-layer translation.

---

## Current-state observations (relevant to implementation)

- `AgentConfig` is the canonical per-agent config source (`src/fast_agent/agents/agent_types.py`).
- AgentCard load/dump is centralized in `agent_card_loader.py` and already supports typed, conditional fields.
- Tool filtering already exists in `McpAgent` and uses server-scoped pattern matching (`fnmatch`) for `config.tools`.
- Anthropic provider already handles server-tool content blocks and replay channels robustly.
- Anthropic request assembly currently builds tool payloads from `Tool` objects in `_prepare_tools`.

---

## Proposed configuration model

### AgentConfig additions

Add a new nested config object:

```python
@dataclass
class ToolSearchConfig:
    enabled: bool = False
    scope: Literal["mcp"] = "mcp"  # reserve for future: "all", "provider", etc.
    strategy: Literal["auto", "bm25", "regex"] = "auto"
    eager_tools: dict[str, list[str]] = field(default_factory=dict)
```

Then add to `AgentConfig`:

```python
tool_search: ToolSearchConfig | None = None
```

### AgentCard shape

```yaml
tool_search:
  enabled: true
  scope: mcp
  strategy: bm25
  eager_tools:
    github:
      - "search_*"
      - "get_issue"
```

### Eager override matching semantics

**Yes — use the same server+pattern selection mechanism as existing tool selection.**

- Same map shape: `dict[server_name, list[pattern]]`
- Same matcher: `fnmatch`
- Same target: server-local tool names (not namespaced form)

This keeps mental model and docs consistent with `tools/resources/prompts` filtering.

---

## Behavioral model (v1)

1. Agent resolves available tools as today.
2. If `tool_search.enabled` and provider/model support it:
   - mark eligible MCP tools as deferred,
   - except tools matching `tool_search.eager_tools` patterns.
3. Anthropic provider injects tool-search server tool (`bm25` or `regex`) and forwards `defer_loading` on deferred tools.
4. Non-MCP tools remain eager.
5. If no tools end up deferred, provider sends normal tools payload (no search tool added).

---

## Provider-neutral policy layer (new helper)

Create a helper module (example):

`src/fast_agent/tools/tool_search_policy.py`

Responsibilities:

- Decide whether tool search should apply for this agent + provider context.
- Compute per-tool defer/eager decisions from:
  - tool origin (`MCP` namespaced vs local),
  - tool_search config,
  - eager override patterns.
- Store decision in tool metadata key, e.g.:
  - `"fast-agent/tool_search/defer_loading": True|False`

This avoids Anthropic-specific branching inside `McpAgent` logic and prevents duplication later.

---

## Anthropic provider wiring

### Request tool payload

Update Anthropic tool preparation to:

- read defer marker from tool metadata,
- include `defer_loading: true` on corresponding `ToolParam` payloads,
- add search tool definition when deferred tools exist.

Strategy mapping:

- `bm25` -> `tool_search_tool_bm25_20251119`
- `regex` -> `tool_search_tool_regex_20251119`
- `auto` -> default to `bm25` (configurable later)

### Compatibility gating

Before enabling at request time, require:

- provider = Anthropic,
- model is supported (Sonnet/Opus 4+ in model database policy),
- at least one eager tool remains and at least one deferred tool exists.

If unsupported or invalid, log one warning and continue with standard tool calling.

---

## Detailed implementation slices

### Slice 1: Types and config plumbing

Files:

- `src/fast_agent/agents/agent_types.py`
- `src/fast_agent/core/agent_card_loader.py`
- `src/fast_agent/core/agent_card_types.py` (if needed)
- `tests/unit/fast_agent/core/test_agent_card_loader.py`

Tasks:

- Add typed `ToolSearchConfig` dataclass.
- Add `tool_search` field to `AgentConfig`.
- Parse `tool_search` from AgentCard with strict validation.
- Include `tool_search` in allowed card fields for `agent`/`smart`/`router` as appropriate.
- Dump `tool_search` in card export when non-default.

### Slice 2: Policy engine and agent integration

Files:

- `src/fast_agent/tools/tool_search_policy.py` (new)
- `src/fast_agent/agents/mcp_agent.py`
- `tests/unit/fast_agent/agents/...` (new tests)

Tasks:

- Implement defer decision helper based on namespaced MCP tools + eager overrides.
- Apply metadata marker non-destructively when listing tools.
- Ensure local/function/shell/etc. remain eager in v1.

### Slice 3: Anthropic adapter support

Files:

- `src/fast_agent/llm/provider/anthropic/llm_anthropic.py`
- `src/fast_agent/llm/provider/anthropic/beta_types.py` (new aliases as needed)
- `src/fast_agent/llm/model_database.py` (capability helper)
- `tests/unit/fast_agent/llm/provider/anthropic/test_web_tools.py` (+ new file for tool search)

Tasks:

- Extend tool serialization to pass through `defer_loading`.
- Inject tool-search tool when applicable.
- Add model support helper(s).
- Keep behavior no-op when unsupported.

### Slice 4: Docs and examples

Files:

- AgentCard docs/examples under `docs/` and/or `examples/`

Tasks:

- Document `tool_search` config shape.
- Document eager override semantics (same as `tools` pattern matching).
- Note Anthropic-only v1 behavior and MCP-only scope.

---

## Validation rules (v1)

1. `tool_search.enabled=false` => feature inactive.
2. `scope` must be `mcp`.
3. `strategy` in `{auto,bm25,regex}`.
4. `eager_tools` keys must be non-empty server names; values non-empty string patterns.
5. On send:
   - If all tools would be deferred, force at least one eager (fallback) or disable search for that turn.
   - If no deferred tools remain, skip adding search tool.

---

## Testing plan

### Unit tests: config + cards

- parse valid `tool_search` block.
- reject malformed blocks (bad types/values).
- round-trip dump/load preserves expected values.

### Unit tests: policy engine

- MCP namespaced tools deferred by default when enabled.
- local tools remain eager.
- eager override patterns opt-in eager behavior.
- unmatched server override has no effect.

### Unit tests: Anthropic request build

- includes search tool + deferred flags when eligible.
- includes no search tool when no deferred tools.
- strategy mapping produces correct search tool type.
- unsupported model/provider gracefully no-ops.

### Regression tests

- existing web-search/web-fetch and server-tool replay tests continue to pass.
- no behavior regression for non-Anthropic providers.

---

## Rollout strategy

1. Ship behind config opt-in (`tool_search.enabled`).
2. Add telemetry/logging for:
   - deferred tool count,
   - eager tool count,
   - whether search tool injected.
3. Validate with real MCP catalogs (10s, 100s, 1000s of tools).
4. Expand scope only after stable behavior and docs.

---

## Risks and mitigations

### Risk: Namespaced tool-name collisions under truncation

- Current namespacing truncates to 64 chars; large catalogs increase collision risk.
- Mitigation: add detection/warning in aggregator map population; optionally add deterministic suffixing in later slice.

### Risk: all-tools-deferred Anthropic request error

- Mitigation: enforce eager fallback at tool payload build time.

### Risk: provider capability drift

- Mitigation: capability lookup via model database helper + safe no-op fallback.

---

## Open questions

1. Should `strategy=auto` always map to `bm25`, or select `regex` for smaller catalogs?
2. Should we expose per-turn toggles via slash command (`/model tool_search on|off`) later?
3. Should we add provider-level defaults in `Settings.anthropic` after per-agent path stabilizes?

---

## Acceptance criteria

- AgentCard can declare `tool_search` config and round-trip cleanly.
- Anthropic requests include `defer_loading` only for intended MCP tools.
- Anthropic request includes exactly one search tool when deferred set non-empty.
- Existing tool execution and MCP permissions remain unchanged.
- Non-Anthropic agents are unaffected.

