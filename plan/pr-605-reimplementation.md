# PR 605 reapplication spec (based on PR 605)

## Background
PR 605 delivered three user-visible changes:
- /card --tool reuses Agents-as-Tools instead of a separate tool-card path.
- AgentCard watch refresh is incremental and resilient to partial writes.
- UI trims long model identifiers.

This spec defines the exact PR 605 behavior to reapply on top of the refactored codebase.

## Scope
- TUI and ACP: /card --tool attaches loaded cards as tools; remove detaches.
- TUI and ACP: /agent --tool attaches an existing agent as a tool; --dump prints its AgentCard.
- Watch mode: incremental reloads, tool file dependency watching, and safe parsing.
- UI: short model label formatting in headers and usage.
- Optional compatibility: keep --card-tool/.fast-agent/tool-cards behavior if present.

## Non-goals
- Attach tools to all agents at once.
- Change Agents-as-Tools runtime semantics.
- Redesign CLI command structure beyond PR 605 scope.

## Terminology
- **Current agent**: the agent selected in the interactive prompt (TUI/ACP).
- **Default agent**: the agent used by CLI when no explicit target is specified.
- **Tool attach**: add child agent names to a BASIC agent's `child_agents` list.

## Required behavior

### 1) /card --tool (TUI and ACP)
- `/card <path|url>` loads cards and refreshes active instances.
- `/card <path|url> --tool` attaches the loaded agents to the **current agent** only.
- `/card <path|url> --tool remove` detaches those loaded agents from the current agent.
- If the current agent is not BASIC, report a warning and skip attach/detach.
- Tool descriptions come from `description` if present; otherwise fall back to instruction.

### 2) /agent --tool and /agent --dump (TUI and ACP)
- `/agent <name> --tool` attaches the named agent to the current agent.
- `/agent <name> --tool remove` detaches the named agent from the current agent.
- `/agent <name> --dump` prints the AgentCard for that agent.
- Reject unknown agents with a clear error.

### 3) CLI behavior and default agent selection
- If `--card-tool` exists (compatibility), it **only** attaches to the default agent.
- `.fast-agent/tool-cards/` (if still supported) must be merged into `card_tools`
  and follow the same default-agent behavior as `--card-tool`.
- Default agent selection remains:
  - In `fast-agent go`, use the instruction-derived agent name (or "agent").
  - If that agent is not found, fall back to the first loaded agent.
- **Never** attach tools to all loaded agents automatically.

### 4) AgentCard watch reload
Reload logic must be safe under partial writes and large directories:
- Track card files and function tool files per root.
- Maintain an mtime/size cache and reload only changed files.
- Skip zero-byte or incomplete files during watch reloads.
- On invalid card parse during watch, keep the existing agent intact and log a warning.
- When a tool file changes, reload any cards that reference it.
- If a card is removed, prune its agent and remove it from any `child_agents` lists.

### 5) Model display trimming
- Strip path-like prefixes (keep last segment after `/`).
- Cap displayed model name length to a UI-safe maximum.
- Apply consistently in console header, usage report, and enhanced prompt header.

## Implementation notes (PR 605 anchors)
When reapplying, mirror the PR 605 intent:
- Prefer Agents-as-Tools for tool attachment; do not maintain a separate tool-card path.
- Unify TUI and ACP command paths through shared callbacks.
- Use the same /card and /agent grammar in TUI and ACP.

## Acceptance criteria
- /card --tool attaches only to the current agent in interactive sessions.
- CLI `--card-tool` (if present) attaches only to the default agent (not all agents).
- AgentCard watch reload skips invalid/empty writes and reacts to tool file changes.
- Model display is shortened consistently across UI surfaces.

## Tests (port from PR 605)
- AgentCard watch: tool file changes trigger reload.
- Invalid card during watch does not clobber existing agents.
- /card --tool remove detaches, /agent --tool attach/detach works, /agent --dump prints a card.
