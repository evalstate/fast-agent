# AgentCard Implementation Overview

## Summary
AgentCard is a text-first format (`.md` or `.yaml`) that compiles into
`AgentConfig`. The loader validates fields by card type, produces a structured
agent entry, and supports incremental reloads with lazy hot swap in the
runtime. This document describes what is implemented today.

## Terminology
- **Current agent**: the agent selected in the interactive prompt (TUI/ACP).
- **Default agent**: the agent used by CLI when no explicit target is specified.
- **Tool attach**: add child agent names to a BASIC agent's `child_agents` list
  so it is upgraded to Agents-as-Tools at runtime.

## Loader Entry Points
- `src/fast_agent/core/agent_card_loader.py` provides the core loader and dump
  helpers.
- `load_agent_cards(path)` accepts a single file or directory and returns
  `LoadedAgentCard` entries. Each entry contains:
  - `agent_data`: the dict used by `direct_factory`.
  - `message_files`: resolved history file paths.
  - `path` and `name` for diagnostics.

## Supported Formats
### YAML
- Single YAML mapping per file.
- `type` defaults to `agent` when omitted.
- `instruction` is required (string).

### Markdown
- YAML frontmatter parsed with `python-frontmatter`.
- `type` defaults to `agent` when omitted.
- The body becomes the instruction (unless `instruction` is set in frontmatter).
- Optional `---SYSTEM` marker at the start of the body is stripped.
- Inline history markers (`---USER`, `---ASSISTANT`, `---RESOURCE`) are rejected.
- UTF-8 BOM is tolerated.

## Examples (from the RFC)
### Basic agent (Markdown)
```md
---
type: agent
name: sizer
---
Given an object, respond only with an estimate of its size.
```

### Agent with MCP servers and child agents
```md
---
type: agent
name: PMO-orchestrator
servers:
  - time
  - github
agents:
  - NY-Project-Manager
  - London-Project-Manager
tools:
  time: [get_time]
  github: [search_*]
---
Get reports. Always use one tool call per project/news.
Responsibilities: NY projects: [OpenAI, Fast-Agent, Anthropic].
London news: [Economics, Art, Culture].
Aggregate results and add a one-line PMO summary.
```

### Agent with external history
```md
---
type: agent
name: analyst
messages: ./history.md
---
You are a concise analyst.
```

## Validation Rules
- Allowed fields are enforced per `type` via `_ALLOWED_FIELDS_BY_TYPE`.
- Unknown fields raise `AgentConfigError`.
- `schema_version` defaults to `1` and must be an integer if provided.
- `name` defaults to the filename stem when omitted.
- Required fields are enforced per type (for example `sequence` for `chain`).

## Instruction Resolution
- Only one source is allowed: `instruction` attribute or markdown body.
- `_resolve_instruction` is used to expand references.
- Empty instructions are rejected.

## History Preload (`messages`)
- History is external only; `messages` is a string or list of paths.
- Paths are resolved relative to the card file.
- Files must exist; missing paths are errors.
- Parsing and replay happens later via `prompt_load` (not in the loader).

## AgentConfig Mapping
`_build_agent_data()` converts raw fields into `AgentConfig` and extra runtime
keys:
- `config`: the `AgentConfig` instance (instruction, model, servers, tools,
  resources, prompts, skills, use_history, request_params, shell/cwd, etc.).
- `type`: agent type string (basic/chain/router/etc.).
- `source_path`: original card file path for reload/dump.
- `schema_version`: preserved for diagnostics and dumps.

Type-specific mappings:
- `agent`: `child_agents`, `function_tools`, `tool_hooks`,
  `agents_as_tools_options` (history_mode, max_parallel, child_timeout_sec,
  max_display_instances).
- `chain`: `sequence`, `cumulative`.
- `parallel`: `fan_out`, `fan_in`, `include_request`.
- `router`: `router_agents`.
- `orchestrator` / `iterative_planner`: `child_agents` and plan settings.
- `evaluator_optimizer`: generator/evaluator and refinement options.
- `MAKER`: worker and scoring parameters.

Defaults:
- `use_history` defaults per type (e.g., `router`/`orchestrator` default to
  `False`).
- `request_params` are validated and applied to the config if provided.

## Tool Description Support
- `description` is optional on cards.
- When agents are exposed as tools, descriptions prefer `description` and fall
  back to instruction if missing.

## Function Tools and Hooks
- `function_tools` is accepted as a string or list of strings.
- Paths are stored as-is; resolution happens later in the function tool loader.
- `tool_hooks` are passed through for the hook pipeline.

## CLI Integration
### `fast-agent go`
- `--agent-cards` / `--card` loads AgentCards.
- `--card-tool` loads AgentCards and attaches them to the default agent via
  Agents-as-Tools.
- Auto-load:
  - `.fast-agent/agent-cards/` is loaded if present.
  - `.fast-agent/tool-cards/` is loaded after agent-cards and attaches to the
    default agent.

### Runtime Commands
- `/card <path|url> [--tool [remove]]` loads cards at runtime. With `--tool`,
  loaded agents are attached to the current agent via Agents-as-Tools (scoped
  to the current agent only).
- `/agent <name> --tool [remove]` attaches or detaches an existing agent.
- `/agent [name] --dump` prints the selected agent's AgentCard.

## Export / Dump
- Dump helpers in `agent_card_loader.py` render AgentCards as Markdown or YAML.
- `FastAgent` exposes `--dump`, `--dump-yaml`, `--dump-agent`, and
  `--dump-agent-path` when invoked via the legacy CLI.
- The REPL uses a dump callback to support `/agent --dump`.

## Reload / Watch Behavior
- Card reloads are tracked per root with `(mtime_ns, size)` for each file.
- `--watch` uses OS file events when `watchfiles` is available; otherwise it
  falls back to polling.
- Reloads are incremental: only changed cards are re-parsed; removed cards are
  pruned from the registry.

### Lazy Hot Swap (Shared Instance)
- Registry changes are applied on the next request via `refresh_if_needed()`.
- Only impacted agents are rebuilt; unaffected agents stay in memory.
- Removed agents are detached and shut down.

## Rationale (brief)
- Prompt files are already commonly stored as Markdown; frontmatter keeps the
  format human-readable, diff-friendly, and tooling-friendly.
- The AgentCard spec covers common workflow archetypes and keeps agent
  definitions portable across runtimes.
- Agents-as-Tools is a core paradigm; exposing agents as tools without
  duplication keeps runtime wiring simple.

## Known Gaps
- Multi-card files are not implemented; one card per file is enforced by the
  loader.
- Advanced history routing (`history_source`, `history_merge_target`) is
  specified in the RFC but not implemented.

## References
- [AgentCard RFC](../plan/agent-card-rfc.md)
- [AgentCard at the Summit: The Multi-Agent Standardization Revolution](../plan/agentcard-standards-mini-article.md)
