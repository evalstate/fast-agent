# 2026-06-03 main merge / MCP registry handover

## Current branch state

Branch:

```bash
cleanup-pythonic-normalization-2026-06-03
```

At the time this handover was written, the branch was:

```text
ahead 7
```

Important recent commits on top of the pre-existing cleanup/refactor branch:

```text
73702143 Merge branch 'main' into cleanup-pythonic-normalization-2026-06-03
118fdd5b Rename default log file to fast-agent.jsonl
b0a53729 Avoid MCP skill scans during startup attach
709ca06f Constrain MCP registry probes to attached servers
684f722e Style startup agent counts consistently
5c098f0e Cache MCP skill registries during attach
```

The old `2026-06-03-cleanup-handover.md` is still useful historical context for
the earlier broad cleanup/refactor work, but this document describes the current
post-main-merge state.

## Main merge summary

Merged `main` at:

```text
10a996b6 Add MCP skill registry install support (#815)
```

The incoming main change added SEP-2640-style MCP skill registry support:

- MCP servers can advertise `io.modelcontextprotocol/skills`.
- Registry index is read from `skill://index.json`.
- Skills can be installed/updated from MCP-served `skill-md` or archive artifacts.
- Artifacts are SHA256 verified.
- `/skills registry`, `/skills available`, `/skills add`, and `/skills update`
  understand `mcp://<server>` registries.
- Runtime `/connect` summaries can report skills available from the server.

The merge was not taken mechanically. Main’s logic was ported into this branch’s
refactored style:

- shared command handlers instead of ACP/TUI duplication
- centralized command catalog / usage strings
- smaller formatting helpers
- branch-local `_McpAttachCounts` abstraction
- typed provenance/update helpers
- prompt-agent-info helpers that preserve literal text rendering

## Key integration decisions

### Skills over MCP registry

New/important files:

- `src/fast_agent/skills/mcp_registry.py`
- `src/fast_agent/mcp/tool_result_metadata.py`
- `tests/unit/fast_agent/skills/test_mcp_registry.py`
- `tests/unit/fast_agent/commands/test_skills_mcp_registry_handlers.py`

Important integrated paths:

- `src/fast_agent/commands/handlers/skills.py`
- `src/fast_agent/acp/slash/handlers/skills.py`
- `src/fast_agent/mcp/mcp_aggregator.py`
- `src/fast_agent/skills/provenance.py`
- `src/fast_agent/skills/operations.py`
- `src/fast_agent/ui/prompt/completer.py`
- `src/fast_agent/ui/prompt/agent_info.py`

### MCP skill registry discovery model

The current model is:

```text
attach_server()
  _refresh_attached_server_cache()
    _fetch_server_tools()
    _fetch_server_prompts()
    _scan_mcp_skill_registry()
      check server capabilities
      read skill://index.json through the already-attached server path
      cache McpSkillRegistry on the aggregator
  _attached_result()
    summarize tools/prompts from existing caches
    summarize skills from cached registry only
```

The registry cache is:

```python
self._mcp_skill_registries: dict[str, McpSkillRegistry]
```

This is intentionally cache-backed. `_attached_result()` must not perform live
MCP resource reads.

### Why this matters

During `fast-agent go --url https://huggingface.co/mcp`, servers are loaded
during startup:

```text
load_servers()
  attach_server()
```

The first port accidentally made `_attached_result()` call into
`scan_mcp_skill_registry()`, which read `skill://index.json` through the public
`get_resource()` API. Public `get_resource()` has this guard:

```python
if not self.initialized:
    await self.load_servers()
```

So startup entered this recursion:

```text
load_servers()
  attach_server()
    _attached_result()
      _mcp_skills_total()
        scan_mcp_skill_registry()
          get_resource()
            load_servers()
```

That caused the observed startup loop/flashing progress around:

```text
• Connecting    ⠠ dev        huggingface_co
```

The final design avoids this by:

- scanning the registry during attach cache refresh, not during result formatting
- using an attached-server scan adapter that calls `_get_resource_from_server()`
  directly, bypassing the public auto-initializing `get_resource()`
- making status/registry listing operate only on attached/cached state

### Attached-only probing

`list_mcp_skill_registries()` now scans/returns attached servers only:

```python
for server_name in self.list_attached_servers():
    ...
```

This avoids connecting configured-but-detached/deferred servers just to populate
skills UI.

`collect_server_status()` no longer calls `get_capabilities()` just to compute
the `Sk` / skills-capability flag. It uses cached or already-live capabilities
only.

## CLI/TUI output notes

Two different paths produce the outputs that were compared:

Startup / `fast-agent go --url ...`:

- `src/fast_agent/ui/prompt/agent_info.py`
- compact agent surface summary
- counts tools/prompts/resources
- resource count comes from `resources/list`

Runtime `/connect ...`:

- `src/fast_agent/commands/handlers/mcp_runtime.py`
- command result summary
- counts added tools/prompts
- skills count comes from cached/parsing `skill://index.json`

Example difference:

```text
MCP Server (8 tools, 4 prompts, 16 resources).
Added 8 tools and 4 prompts; 15 skills available from the server.
```

This is explainable:

- `16 resources` is the result of MCP `resources/list`.
- `15 skills` is the parsed count from `skill://index.json`.
- The extra resource is likely the index itself.

The display divergence was not treated as an intentional product decision.
Numeric count styling has been aligned so startup counts are also
`bold bright_cyan`, matching runtime connect summaries.

## Default log filename

Changed default logger path from:

```text
fastagent.jsonl
```

to:

```text
fast-agent.jsonl
```

Updated:

- `src/fast_agent/config.py`
- `docs/docs/ref/config_file.md`
- `examples/hf-toad-cards/skills/session-investigator/SKILL.md`

No old `fastagent.jsonl` references remained after the change.

## Tests / validation already run

Focused merge/MCP registry tests:

```bash
uv run pytest tests/unit/fast_agent/skills/test_mcp_registry.py \
  tests/unit/fast_agent/commands/test_skills_mcp_registry_handlers.py \
  tests/unit/fast_agent/tools/test_local_filesystem_runtime.py \
  tests/unit/fast_agent/ui/test_agent_info.py \
  tests/unit/fast_agent/ui/test_agent_completer.py -q
```

Result:

```text
225 passed
```

Broader command/UI/MCP subset:

```bash
uv run pytest tests/unit/fast_agent/commands/test_command_discovery.py \
  tests/unit/fast_agent/commands/test_mcp_runtime_handlers.py \
  tests/unit/fast_agent/ui/test_mcp_display.py \
  tests/unit/fast_agent/ui/test_tool_display_skybridge.py \
  tests/unit/fast_agent/ui/test_terminal_images.py -q
```

Result:

```text
166 passed
```

Startup/registry recursion guard tests:

```bash
uv run pytest tests/unit/fast_agent/mcp/test_mcp_aggregator_runtime_attach.py -q
```

Result after latest cache-backed refactor:

```text
10 passed
```

Latest broader registry/status/display validation:

```bash
uv run pytest tests/unit/fast_agent/mcp/test_mcp_aggregator_runtime_attach.py \
  tests/unit/fast_agent/commands/test_skills_mcp_registry_handlers.py \
  tests/unit/fast_agent/skills/test_mcp_registry.py \
  tests/unit/fast_agent/ui/test_mcp_display.py -q
```

Result:

```text
56 passed
```

Startup count styling validation:

```bash
uv run pytest tests/unit/fast_agent/ui/test_agent_info.py -q
```

Result:

```text
13 passed
```

Repo gates were run after each code change:

```bash
uv run scripts/lint.py
uv run scripts/typecheck.py
```

Latest result:

```text
All checks passed!
All checks passed!
```

## Important regression tests added/updated

`tests/unit/fast_agent/mcp/test_mcp_aggregator_runtime_attach.py` now covers:

- startup load routes through `attach_server`
- runtime server is registered before prompt discovery
- `_attached_result()` uses cached MCP registry data only
- refresh cache discovers and stores MCP skill registry
- status collection does not probe/connect detached server capabilities
- `list_mcp_skill_registries()` scans only attached servers

`tests/unit/fast_agent/ui/test_agent_info.py` now covers:

- count markup uses `bold bright_cyan` for numeric values
- plain text remains literal/safe
- MCP resource counts use aggregator surfaces rather than local runtime tools

## Known behavior / caveats

### Startup skills count

Startup currently shows tools/prompts/resources, not an explicit skills count.
Runtime `/connect` can show skills because attach refresh reads/caches the
registry index.

If product wants startup to show:

```text
15 skills
```

then use the cached registry created during `_refresh_attached_server_cache()`.
Do not call public `get_resource()` or active registry scans from display
formatters.

### `resources` versus `skills`

Resource count is from MCP `resources/list`.

Skill count is from `resources/read skill://index.json` and parsing the registry
index.

They are related but not identical. For Hugging Face MCP, `16 resources` and
`15 skills` can both be correct.

### Apply-patch error logs

Earlier `fastagent.jsonl` entries like:

```text
Error applying patch: Failed to find expected lines...
```

were not MCP transport failures. They came from failed local `apply_patch` tool
attempts during the merge conflict resolution. The default log file is now
`fast-agent.jsonl`.

## Suggested next checks before pushing/PR

Run:

```bash
git status --short --branch
git log --oneline --decorate --graph --max-count=12
uv run pytest tests/unit/fast_agent/mcp/test_mcp_aggregator_runtime_attach.py \
  tests/unit/fast_agent/commands/test_skills_mcp_registry_handlers.py \
  tests/unit/fast_agent/skills/test_mcp_registry.py \
  tests/unit/fast_agent/ui/test_mcp_display.py \
  tests/unit/fast_agent/ui/test_agent_info.py -q
uv run scripts/lint.py
uv run scripts/typecheck.py
```

Manual smoke to verify the original reported behavior:

```bash
fast-agent go --url https://huggingface.co/mcp
```

Expected:

- no repeated/flashing `Connecting ... huggingface_co` loop
- startup reaches prompt
- compact startup MCP counts render numeric values in bright cyan

Also verify runtime connect still works:

```text
/connect https://huggingface.co/mcp
```

Expected:

- connect succeeds
- summary includes added tools/prompts and, when advertised, skills available

## PR note

If preparing a PR, include the required answer to:

> You're given a calfskin wallet for your birthday. How would you feel about using it?
