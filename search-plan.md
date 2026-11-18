# PTC Tool Search

Status: implemented (v1).  
Date: 2026-11-17.

Code: [catalogue](src/fast_agent/ptc/catalog.py), [search](src/fast_agent/ptc/search.py),
[mentions](src/fast_agent/ptc/mentions.py), [PTC runtime](src/fast_agent/ptc/runtime.py),
[agent tool assembly](src/fast_agent/agents/mcp_agent.py), [tool surface](src/fast_agent/tools/tool_surface.py),
[`/tools` handler](src/fast_agent/commands/handlers/tools.py), [picker](src/fast_agent/ui/tool_picker.py).

## 1. Product decisions

1. **Tool Search exists only in programmatic tool calling (PTC).** It never changes the tool list sent to the model in traditional tool-calling mode.
2. **`--ptc` keeps built-ins and direct functions upfront** and defers other function stubs behind `search_tools` once search is active (section 3).
3. **`--ptc-all` lists every callable stub upfront and has no Tool Search.** It conflicts with `--no-ptc`.
4. Discovery follows **`/mcp connect`/disconnect**, including servers connected mid-conversation.
5. Users can **browse, select and add definitions** to their message (`/tools add`, `/tools select`, `^tool:` mentions).

`ProgrammaticToolCallingSettings.search` (default `true`) is the source of truth; `--ptc-all` turns it off for the run. Shell access stays conditional on shell being enabled.

## 2. Availability versus visibility

Every permitted function is registered in the sandbox and type-checked against its full stub; only the visibility of its definition changes. Searching neither activates nor authorizes a function: a known name is callable without searching, subject to the normal permissions, and execution still goes through the one code tool. `ToolSurface` marks deferred functions (`deferred`, `stub_tokens`, `source`) while keeping them programmatic.

## 3. Activation and the `search_tools` global

Search activates when the canonical vocabulary is in use and **more than three** extra functions exist (`len(extras) > PTC_SEARCH_EXTRA_THRESHOLD`, threshold `3`). Extras exclude built-ins and functions whose tools the model can also call directly; those always stay upfront. When active, the extras are deferred and the sandbox gains:

```python
await search_tools(
    query="list repository files",
    server="huggingface",
    limit=5,
    exclude=["huggingface__hf_models_search"],
)
```

- `query` omitted: list searchable sources (or one source's functions when `server` is given).
- `server` matches a roster source exactly, else case-insensitively or ignoring punctuation when unique.
- `limit` defaults to 5, capped at 20.
- `exclude` skips already-seen functions (matching Python name, raw tool name or unprefixed local name) before `limit` and byte budgets are applied.
- `search_tools` is a reserved sandbox name; a real tool with that name receives another Python name.

**Ranking** is deterministic lexical search: identifier-aware Unicode tokens (snake/camel case split), light plural folding and field-weighted BM25 (BM25F) over names, parameters and descriptions. Verbatim Python/tool names rank first, then case-insensitive names, then exact names without the server prefix, then identifier matches; filler words count only when nothing meaningful matches. Upfront functions match by exact name only. There is no regex mode in v1.

**Results** are the same canonical stubs used by the code tool description and `^tool:` mentions. The top match is shown in full, further matches in full until a byte budget, then as compact signatures; output fits the visible `exec_sandbox` result and is never cut partway through a definition. A query naming a **retired** function (one whose tool has gone) is answered with a "no longer available" note before any other matches.

## 4. Stable names and routing

- A **conversation name ledger** (`PtcNameLedger`) binds each Python name to its tool identity for the conversation. A name is never reassigned to a different tool, so old code and definitions in history cannot reach another tool; a name whose tool has gone is *retired* and fails with a "no longer available" message rather than reaching another tool.
- Each `PtcCatalog` snapshots **origin-qualified alias routes** (`routes: Mapping[str, ToolRoute]`), so a running program dispatches to the tool it was built against or fails; it never falls through to a different owner of the same raw name.
- Catalogues are immutable snapshots; mutations apply at the next build.

## 5. Advertising sources

When search is active, a compact `<fastagent:tool-sources>` roster (source name, deferred function count, unavailable/auth notes; bounded to 40 entries) is appended to the user turn **when it changes**, keyed by a fingerprint. It is re-announced after `/clear`, session resume or compaction, and a one-line retraction is sent when search becomes inactive. The code tool description stays stable so roster changes do not rewrite it.

## 6. User-selected definitions

- `^tool:<name>` in a user message inserts a `<fastagent:tool-definitions>` block with the canonical stubs. Inline mention tokens become backticked Python names in the prose; trailing mention tokens are stripped. Mentions neither call nor authorize tools, and resolve by Python name, then tool name.
- `/tools add <function>...` prefills mentions; `/tools select` opens a picker (source tabs, filter, multi-select, stub preview and token cost). ACP shows the draft text instead of editing input.
- `/tool <name>` is always a plain detail lookup, even for tools named like `/tools` actions (`add`, `select`, ...).
- Pinning (conversation-persistent definitions) is not implemented.

Generated roster/definition blocks stay in model history but are excluded from user-facing projections (previews, replay, rewind prefill).

## 7. Tests, evaluation and open questions

Contract coverage: traditional mode is unaffected; `--ptc-all` lists every stub with no search; deferred functions stay type-checkable and executable; search and mentions render identical stubs; exact/verbatim lookups win; connect/disconnect updates discovery without renaming or rerouting existing names; resume rebuilds availability rather than trusting history.

Still open:

1. Retrieval and query-guidance evaluation (hit@k, multi-tool all@k, exact-identity correctness, end-to-end task success, token cost) against captured catalogues.
2. Concise per-source capability summaries in the roster.
3. Whether the threshold, default result count and byte budgets are right.
4. Persistent pinning semantics.

## 8. Research and source references

### Research artefacts

- [Tool-search evaluation overview](../hf-mcp-tool-optimisations/tool-search/README.md): search engines, catalogue captures, suites and harness layout. Records the move from `hf-mcp-server/scripts/tool-search-eval` on 2026-10-31.
- [Oracle contract](../hf-mcp-tool-optimisations/tool-search/oracles/CONTRACT.md): direct/PTC mode distinctions and what retrieval metrics represent.
- [FX fidelity notes](../hf-mcp-tool-optimisations/tool-search/oracles/fx/FIDELITY.md): field-weighted BM25, exact identities, relevance filtering and result budgets.
- [OpenCode fidelity notes](../hf-mcp-tool-optimisations/tool-search/oracles/opencode/FIDELITY.md): Code Mode search, substring ranking, inline catalogue and instruction handling.
- [Codex fidelity notes](../hf-mcp-tool-optimisations/tool-search/oracles/codex/FIDELITY.md): BM25, `ALL_TOOLS`, optional code-mode search and truncation-sensitive output paths.
- [Cross-harness comparison](../hf-mcp-tool-optimisations/tool-search/pipeline/REPORT.compare-rv2.md): results and important comparability caveats.
- [Actual first-search query patterns](../hf-mcp-tool-optimisations/tool-search/pipeline/REPORT.query-patterns.md): vocabulary, server/brand mentions, name-based selection and very small requested limits.
- [Recovery spot-check](../hf-mcp-tool-optimisations/tool-search/pipeline/REPORT.recovery-spotcheck.md): limitations of assuming later turns recover from poor initial discovery.
- [Applied tool-search analysis](../hf-mcp-server/docs-internal/ptc-parity/tool-search-seo.md): retrieval effects of expanded toolsets, naming, descriptions and aggregation.
- [Recent HF PTC evaluation report](../hf-mcp-server/docs-internal/ptc-parity/eval/REPORT.md): follow-up evaluations of the revised toolset.

### Existing fast-agent design and code

- [PTC plan](plan/programmatic-tool-calling.md): large-catalogue discovery proposal, control-plane exceptions and tool-surface work.
- [PTC runtime](src/fast_agent/ptc/runtime.py): function bindings, stubs, execution description and type checking.
- [Agent tool assembly](src/fast_agent/agents/mcp_agent.py): permitted tool assembly, aliases and PTC runtime integration.
- [Tool surface](src/fast_agent/tools/tool_surface.py): representation used by tool listings.
- [Settings](src/fast_agent/config.py): `ProgrammaticToolCallingSettings`; keep defaults here.
- [Shared CLI options](src/fast_agent/cli/shared_options.py): `--ptc`, `--no-ptc`, `--ptc-all`.
- [MCP runtime commands](src/fast_agent/commands/handlers/mcp_runtime.py) and [aggregator](src/fast_agent/mcp/mcp_aggregator.py): connection and tool-list lifecycle.

### Local upstream/UI references

- [FX model catalogue](../../reference/fx/src/core/mcp/model_catalog.zig): server roster and change-notice implementation inspected for this plan.
- [discover plugin](../card-packs/plugins/discover/discover.py): popup, MCP connection and prompt-prefill patterns.

### Related research sessions

Under `../hf-mcp-server/.fast-agent/sessions/`:

- `2611161114-AFrGHf`: main PTC discussion and comparative evaluation follow-ups.
- `2611161728-P332Vf`: focused tool-search analysis.

Sibling-repository, reference-checkout and session paths are local research references; they are not bundled fast-agent resources. The fidelity notes pin the inspected research implementations. Revalidate those findings before relying on a newer upstream revision.
