# REPL Implementation Overview

## Executive Summary
The fast-agent REPL (Read-Eval-Print Loop) is an interactive TUI + ACP workflow
optimized for rapid agent iteration. You edit AgentCards on disk and the runtime
adopts changes without restart via lazy hot-reload and hot-swap, so the feedback
loop is effectively edit -> run -> observe. Architecturally, this keeps things
simple (file-based cards + incremental reloads) while staying fast: updated
agents appear on the next turn and only the impacted agents are rebuilt. The
same reload/watch pipeline is shared by ACP sessions, so `--watch` and `--reload`
apply there too.

## Terminology
- **Current agent**: the agent selected in the interactive prompt (TUI/ACP).
- **Default agent**: the agent used by CLI when no explicit target is specified.
- **Tool attach**: add child agent names to a BASIC agent's `child_agents` list
  so it is upgraded to Agents-as-Tools at runtime.

## Architecture Overview
### High-Level Flow (internal)
```
Filesystem change (watchfiles/poller)
    |
    v
FastAgent._watch_agent_cards()
    |
    v
FastAgent.reload_agents()
    |
    v
FastAgent._load_agent_cards_from_root()
    |
    v
agent_card_loader.load_agent_cards()
    |
    v
FastAgent._apply_agent_card_updates()
    |
    v
registry_version += 1
    |
    v
AgentApp._refresh_if_needed()  <-- user message / command
    |
    v
FastAgent.refresh_shared_instance()
    |
    v
partial rebuild of impacted agents
    |
    v
LLM
```

### Flow Notes
- The watcher runs in a background task and only mutates the registry.
- The swap into the live runtime happens lazily on the next request.
- Removed agents are pruned before rebuilding impacted agents.

### Key Components
- **CLI (`fast-agent go`)**: loads AgentCards, config, and sets up watch/reload.
- **FastAgent**: maintains the registry, reloads cards, and manages instances.
- **AgentApp**: routes messages, applies `refresh_if_needed()`, and exposes
  `/agent --dump`.
- **InteractivePrompt / EnhancedPrompt**: input loop and slash-command parsing.
- **AgentsAsToolsAgent**: upgrades basic agents when `child_agents` are declared.

## Runtime Behavior
### Reload / Watch Behavior
- `--reload`: manual reloads only; no filesystem watcher.
- `--watch`: background watcher triggers reload passes; uses OS events when
  `watchfiles` is available, otherwise falls back to mtime/size polling.
- Reloads are incremental: only changed card files are re-parsed; removed cards
  are pruned from the registry.
- Lazy swap happens on the next request/turn after the registry version changes.
- Without `--reload` or `--watch`, behavior stays as close as possible to the
  existing runtime for stability; new REPL features are opt-in.

### Lazy Hot Swap (Shared Instance)
#### Algorithm (shared instance)
1) Track changed/removed agent names plus dependency graph edges.
2) On the next request, check the registry version.
3) If changes are pending:
   - Compute `impacted = changed + dependents - removed`, then expand to
     transitive dependents.
   - Remove deleted agents from the active map and shut them down.
   - Rebuild only impacted agents in dependency order; leave others intact.
   - Re-apply AgentCard history and late-bound instruction context for impacted
     agents only.
4) Update the shared instance version and continue.

In-flight requests always complete on the old instance; swaps happen between
turns.

### AgentCard Reload
- Uses `agent_card_loader.load_agent_cards()` with strict field validation.
- Tracks per-file `(mtime_ns, size)` to determine incremental changes.
- Removed cards are pruned and detached on the next refresh.
- Parse failures on partial writes are logged as warnings and retried on the
  next change.

### ACP Integration
- ACP sessions use the same registry and lazy refresh logic.
- `--watch` and `--reload` update the registry; ACP sessions see updates on the
  next request/connection boundary depending on `instance_scope`.

## Command Surface
### Command Handling
- `/agents`: list available agents.
- `/card <path|url> [--tool [remove]]`: load cards at runtime; with `--tool`,
  attach loaded agents to the current agent via Agents-as-Tools (scoped to the
  current agent only).
- `/agent <name> --tool [remove]`: attach/detach an existing agent as a tool.
- `/agent [name] --dump`: print the selected agent's AgentCard.
- `/reload`: manual reload pass when `--reload` is enabled.

### Agents-as-Tools Integration
- Declaring `agents` on a basic agent upgrades it to `AgentsAsToolsAgent`.
- `/card --tool` and `/agent --tool` append to the current agent's
  `child_agents` and hot-swap using the same code path.
- Tool attach is a per-agent operation; it does not automatically attach to all
  loaded agents.
- Tool descriptions prefer `description` from the AgentCard, then fall back to
  the instruction text.

### Model Display Trimming
- Strip path-like prefixes (keep the last segment after `/`).
- Cap displayed model name length to a UI-safe maximum.
- Apply consistently in the console header, usage report, and enhanced prompt
  header.

## What's Implemented
- [x] Interactive REPL loop with slash commands and `@agent` switching.
- [x] AgentCard loading from files or directories.
- [x] `/card --tool` and `/agent --tool` attach agents via Agents-as-Tools.
- [x] Incremental reload with `mtime+size` checks and safe parse retry.
- [x] Removal handling: deleted agents are pruned from the registry and detached
  on refresh.
- [x] Lazy hot swap: rebuild only impacted agents for `instance_scope=shared`.
- [x] ACP uses the same reload/watch pipeline and lazy refresh model.

## Not Implemented (Yet)
- Multi-card files (one card per file is enforced).
- Inline history blocks inside AgentCard bodies.
- Advanced history routing (`history_source`, `history_merge_target`).
- Cross-file includes/imports beyond external `messages` history files.

### Low-Level Flow
- `watchfiles/poller`: emits filesystem events (or polls) when AgentCard roots change.
- `FastAgent._watch_agent_cards()`: background task that waits for events and triggers reload passes.
- `FastAgent.reload_agents()`: runs an incremental reload pass and bumps the registry version if changed.
- `FastAgent._load_agent_cards_from_root()`: diffs mtime/size per root and selects changed/removed files.
- `agent_card_loader.load_agent_cards()`: parses card files into validated entries for the registry.
- `FastAgent._apply_agent_card_updates()`: updates the registry, prunes removed agents, and records changes.
- `registry_version += 1`: marks that a newer registry is available for the REPL to pick up.
- `AgentApp._refresh_if_needed()`: runs at the start of each turn to check for pending registry updates.
- `FastAgent.refresh_shared_instance()`: rebuilds only impacted agents and swaps them into the shared instance.

#### partial rebuild of impacted agents
- Entry point: `FastAgent.refresh_shared_instance()` when `registry_version` changes.
- Compute `impacted` + transitive dependents via `get_agent_dependencies()` and
  `get_dependencies_groups()` (`fast_agent.core.validation`).
- Rebuild groups via `active_agents_in_dependency_group()` and
  `create_agents_by_type()` (`fast_agent.core.direct_factory`).
- Swap updated agents into `active_agents` and shut down old instances.
- Re-apply `_apply_agent_card_histories()` and `_apply_instruction_context()` to
  refreshed agents.
