# Internal Resource Bundle Plan (`internal://`)

## Summary

Introduce a portable, read-only **internal resource bundle** for smart agents so we can keep large internal docs out of the system prompt while still making them discoverable and retrievable on demand.

### Desired UX

- Smart agent sees a compact index of internal docs (name + description + why it exists).
- Smart agent calls `resource_read` to fetch full content only when needed.
- Resources are packaged with fast-agent, so this works without `/skills add`.
- Optional: expose fast-agent skills-repo listings through the same mechanism.

---

## Goals

1. **Reduce system prompt bloat** by removing large inline docs.
2. **Keep docs portable/offline** by packaging with the distribution.
3. **Preserve discoverability** via listable resources with descriptions.
4. **Use read-only semantics** (no mutation, no arbitrary file access).
5. **Keep backward compatibility** with existing `{{internal:...}}` includes.

## Non-goals

- Replacing skill execution/installation workflows.
- Creating mutable knowledge stores.
- Exposing arbitrary local files under `internal://`.

---

## Current State (Relevant)

- Internal docs are currently included via template placeholders in instruction building:
  - `src/fast_agent/core/instruction.py` (`{{internal:...}}`)
  - `resources/shared/smart_prompt.md` currently includes `{{internal:smart_agent_cards}}`.
- Smart agents expose MCP resource helper tools for card execution paths:
  - `smart_list_resources`, `smart_get_resource`, `smart_with_resource` in
    `src/fast_agent/agents/smart_agent.py`.
- MCP aggregator currently returns **URI-only** resource listings in `list_resources()`:
  - `src/fast_agent/mcp/mcp_aggregator.py` (drops richer Resource metadata).

---

## Decision

Use a **native internal resource bundle** (resource semantics), not runtime-generated skills.

### Why

- Matches the use case (documentation retrieval) better than “skills as docs”.
- Avoids path/lifecycle complexity of runtime SKILL.md generation.
- Keeps portability simple: package resources once, read by URI.
- Allows explicit list/read behavior with minimal prompt tokens.

---

## Proposed Architecture

## 1) Internal Catalog + URI Scheme

Define a manifest for curated internal docs, e.g.:

- `uri`: `internal://fast-agent/smart-agent-cards`
- `title`
- `description`
- `why`
- `mime_type` (default `text/markdown`)
- `source` (packaged file path)
- optional tags (`smart`, `agent-cards`, `reference`)

Recommended location:

- Manifest: `resources/shared/internal_resources_manifest.json` (or `.yaml`)
- Content: `resources/shared/internal/*.md`

> Existing `resources/shared/*.md` can be migrated or referenced directly.

---

## 2) Internal Resource Provider (read-only)

Add a small provider layer that:

- lists catalog entries
- validates and resolves known `internal://` URIs
- returns MCP-compatible read results (`ReadResourceResult` with text contents)

### Safety constraints

- Allow-list only known manifest URIs.
- No arbitrary filesystem traversal.
- No writes.

---

## 3) Smart-Agent Tool Surface

Add smart-facing tools:

- `resource_list` → list internal resources with descriptions and “why” notes
- `resource_read` → read full resource content by URI

Notes:

- Keep existing `smart_*resource*` tools unchanged for card-oriented MCP inspection.
- `resource_read` should be optimized for internal docs; future extension can optionally read attached MCP resources too.

---

## 4) Prompt Integration

Add a generated prompt block similar in spirit to `{{agentSkills}}`, but for internal resources.

Candidate placeholder:

- `{{agentInternalResources}}`

Rendered format could be XML-ish:

```xml
<available_resources>
  <resource>
    <uri>internal://fast-agent/smart-agent-cards</uri>
    <description>AgentCard schema and field behavior reference.</description>
    <why>Use when creating or validating AgentCards.</why>
  </resource>
</available_resources>
```

Update smart prompt guidance to:

- include `{{agentInternalResources}}`
- instruct: “Use `resource_read` for details; do not inline large docs in memory unless needed.”

---

## 5) Skills Repo Visibility (Optional Phase)

Expose a read-only view of fast-agent’s skills repository as internal resources:

- `internal://fast-agent/skills/index`
- optionally `internal://fast-agent/skills/<skill-name>/SKILL.md`

This provides discoverability without requiring `/skills add`.

Packaging options:

1. package a snapshot of skill manifests/docs in resources
2. build index from packaged `marketplace.json`

---

## Why Not “Runtime Skill from `internal://`” (Primary Path)

It is feasible, but not the recommended default:

Pros:

- reuses existing `available_skills` prompt conventions
- leverages `read_skill`/`read_text_file`

Cons:

- semantic mismatch (docs != capabilities)
- requires absolute-path handling for packaged resources
- extra complexity for temp extraction/lifecycle
- can blur distinction between installed skills and built-in docs

Recommendation: keep this as a fallback experiment, not v1 core architecture.

---

## Implementation Plan

## Phase 1 (MVP)

1. Add manifest + packaged docs for internal resources.
2. Add internal resource provider module.
3. Add `resource_list` + `resource_read` tools to smart agents.
4. Add `{{agentInternalResources}}` template resolver.
5. Slim `resources/shared/smart_prompt.md` to index-first guidance.
6. Keep `{{internal:...}}` behavior unchanged.

## Phase 2 (Enhancements)

1. Add skills-repo index and optional per-skill reads under `internal://`.
2. Optionally expose richer resource metadata in aggregator APIs (if needed broadly).
3. Consider unifying naming between smart resource tools and generic resource tools.

---

## Testing Strategy (No mocks / no monkeypatch)

### Unit

- Manifest parsing and validation.
- URI allow-list enforcement.
- Internal resource read success/failure cases.
- Placeholder rendering for `{{agentInternalResources}}`.

### Integration

- Smart agent can list internal resources and read one.
- Prompt includes only index metadata, not full embedded docs.
- Existing `{{internal:...}}` includes still function.

### Regression

- Existing smart resource tools (`smart_list_resources`, etc.) unchanged.
- Existing skills behavior unchanged.

---

## Compatibility & Rollout

- Additive feature only in v1.
- Maintain existing smart prompt path and internal template includes.
- Gradual migration: move large docs from inline include to on-demand reads once tools are stable.

---

## Open Questions

1. Should `internal://` docs be available to **all MCP agents** or **smart-only** initially?
2. Do we want generic names (`resource_list`/`resource_read`) or smart-prefixed aliases for consistency?
3. Should skills-repo exposure include full SKILL bodies in v1, or index-only first?
4. Should catalog entries include token-size hints to help agent pick concise resources first?

---

## Recommendation

Proceed with **native internal resources (v1)** and keep runtime-generated skills as an optional follow-up. This gives the cleanest portability and the least implementation complexity while solving prompt bloat directly.
