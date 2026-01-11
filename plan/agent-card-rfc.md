# AgentCard RFC (Draft)

## Summary
AgentCard is a text-first format (`.md` / `.yaml`) that compiles into `AgentConfig`.
A loader validates fields based on `type` and loads a single file or a directory via
`load_agents(path)`. The default path is **one card per file**. Multi-card files are
optional/experimental and described in a separate spec.
AgentCards now support an optional `description` field used for tool descriptions when
agents are exposed as tools (MCP or agent-as-tool wiring).
AgentCards may enable local shell execution via `shell: true` with optional `cwd`.
AgentCards support `tool_only: true` to prevent exposure as first-class agents while
still allowing use as tools (useful for helper agents in `.fast-agent/tool-cards/`).
CLI `--card-tool` loads AgentCards and exposes them as tools on the default agent.
CLI runs also auto-load cards from `.fast-agent/agent-cards/` (agents) and
`.fast-agent/tool-cards/` (tool cards) when those directories exist and contain
supported card files.

## Agent vs Skill
- **Skill**: a reusable prompt fragment or capability description.
- **AgentCard**: a full runtime configuration (model, servers, tools, history source,
  and instruction) that can be instantiated as an agent.
- Formats can be compatible, but the semantics are different.

## Goals
- One canonical IR: `AgentConfig`.
- Strong validation: reject unknown fields for the given `type`.
- Deterministic parsing and minimal ambiguity.
- Simple authoring: one agent per file by default.

## Non-goals (for now)
- Cross-file imports/includes (beyond `messages` referencing external history files).
- A rich schema migration framework.

---

## Terminology
- **Card**: one AgentCard definition (`type` + attributes + instruction).
- **Frontmatter**: YAML header delimited by `---` lines in `.md` files.
- **Body**: markdown text following the frontmatter; used for instruction.
- **History file**: a separate file referenced by `messages` that seeds history.

---

## Minimal Attributes
- `type`: one of `agent`, `chain`, `parallel`, `evaluator_optimizer`, `router`,
  `orchestrator`, `iterative_planner`, `MAKER`
- `name`: unique card name within a load-set.
  - If a file contains a **single** card and `name` is omitted, it defaults to the
    filename (no extension).
  - Multi-card files are optional/experimental; in that case `name` is required.
- `description`: optional. Used as the tool description when exposing agents as tools.
- `tool_only`: optional boolean (default `false`). When `true`, the agent is hidden from
  agent listings and cannot be selected as the default agent, but remains usable as a
  tool by other agents. Mutually exclusive with `default: true`.
- `instruction`: required, and can be provided **either** in the body **or** as an
  `instruction` attribute (short one-line shortcut). If both are present, it is an error.

## Attribute Sets
- All attributes defined by the decorator for a given `type` are permitted.
- `type` determines the allowed attribute set.
- The loader enforces valid attributes and rejects unknown fields for that `type`.

## Schema Version
- `schema_version`: optional.
  - If present, must be an integer.
  - Loader should default to `1` when omitted.
  - Parser/loader must remain backwards-compatible within a major series when feasible.
  - Loader attaches `schema_version` to the in-memory agent entry for diagnostics/dumps.

---

## Supported File Formats

### YAML Card (`.yaml` / `.yml`)
A YAML card is a single YAML document whose keys map directly to the `AgentConfig`
schema. `type` is optional and defaults to `agent`. Use `instruction: |` for
multiline prompts.

Example:
```yaml
type: agent
name: sizer
instruction: |
  Given an object, respond only with an estimate of its size.
```

### Markdown Card (`.md` / `.markdown`)
A Markdown card is YAML frontmatter followed by an optional body. The body is treated
as the system instruction unless `instruction` is provided in frontmatter. `type`
is optional and defaults to `agent`.
UTF-8 BOM should be tolerated.

Example:
```md
---
type: agent
name: sizer
---
Given an object, respond only with an estimate of its size.
```

---

## 1:1 Card ↔ Decorator Mapping (Strict Validator)
Use this mapping to validate allowed fields for each `type`. Fields not listed for a
type are invalid. Card-only fields (`schema_version`, `messages`) are listed explicitly.

Code-only decorator args that are **not** representable in AgentCard:
- `instruction_or_kwarg` (positional instruction)
- `elicitation_handler` (callable)
- `tool_runner_hooks` (hook object)

### type: `agent` (maps to `@fast.agent`)
Allowed fields:
- `name`, `instruction`, `description`, `default`, `tool_only`
- `agents` (agents-as-tools)
- `servers`, `tools`, `resources`, `prompts`, `skills`
- `model`, `use_history`, `request_params`, `human_input`, `api_key`
- `history_mode`
- `max_parallel`, `child_timeout_sec`, `max_display_instances`
- `function_tools`, `tool_hooks` (see separate spec)
- `shell`, `cwd`
- `messages` (card-only history file)

### type: `chain` (maps to `@fast.chain`)
Allowed fields:
- `name`, `instruction`, `description`, `default`, `tool_only`
- `sequence`, `cumulative`

### type: `parallel` (maps to `@fast.parallel`)
Allowed fields:
- `name`, `instruction`, `description`, `default`, `tool_only`
- `fan_out`, `fan_in`, `include_request`

### type: `evaluator_optimizer` (maps to `@fast.evaluator_optimizer`)
Allowed fields:
- `name`, `instruction`, `description`, `default`, `tool_only`
- `generator`, `evaluator`
- `min_rating`, `max_refinements`, `refinement_instruction`
- `messages` (card-only history file)

### type: `router` (maps to `@fast.router`)
Allowed fields:
- `name`, `instruction`, `description`, `default`, `tool_only`
- `agents`
- `servers`, `tools`, `resources`, `prompts`
- `model`, `use_history`, `request_params`, `human_input`, `api_key`
- `messages` (card-only history file)

### type: `orchestrator` (maps to `@fast.orchestrator`)
Allowed fields:
- `name`, `instruction`, `description`, `default`, `tool_only`
- `agents`
- `model`, `use_history`, `request_params`, `human_input`, `api_key`
- `plan_type`, `plan_iterations`
- `messages` (card-only history file)

### type: `iterative_planner` (maps to `@fast.iterative_planner`)
Allowed fields:
- `name`, `instruction`, `description`, `default`, `tool_only`
- `agents`
- `model`, `request_params`, `api_key`
- `plan_iterations`
- `messages` (card-only history file)

### type: `MAKER` (maps to `@fast.maker`)
Allowed fields:
- `name`, `instruction`, `description`, `default`, `tool_only`
- `worker`
- `k`, `max_samples`, `match_strategy`, `red_flag_max_length`
- `messages` (card-only history file)

### Card-only fields (all types)
- `schema_version` (optional)

---

## Instruction Source
- **One source only**: either the body **or** the `instruction` attribute.
- If both are present, the loader must raise an error.
- If `instruction` is provided, the body must be empty (whitespace-only allowed).
- The body may start with an optional `---SYSTEM` marker to make the role explicit.

---

## History Preload (`messages`)
History is **external only**. Inline `---USER` / `---ASSISTANT` blocks inside the
AgentCard body are **not supported**.

### `messages` attribute shape
- `messages: ./history.md` (string)
- `messages: [./history.md, ./fewshot.json]` (list)

### Path resolution
- Relative paths are resolved relative to the card file directory.

### History file formats
History files use the same formats as `fast-agent` history save/load:
- **`.json`**: PromptMessageExtended JSON (`{"messages": [...]}`), including tool calls
  and other extended fields. This is the format written by `/save_history` when the
  filename ends in `.json`.
- **Text/Markdown (`.md`, `.txt`, etc.)**: delimited format with role markers:
  - `---USER`
  - `---ASSISTANT`
  - `---RESOURCE` (followed by JSON for embedded resources)
  If a file contains no delimiters, it is treated as a single user message.

History is its own file type; it is not embedded inside AgentCard files.

---

## Agents-as-Tools History Controls (Proposed)
These options define **where child clones fork history from** and **where merged
history lands**. This addresses the open questions from issue #202 about fork/merge
scope.

### Fields (AgentCard)
These fields are set on the **orchestrator** (parent) AgentCard because it
controls child invocation and the initial context passed to child agents.
They are **proposed** and not yet part of the validator or loader.

- `history_source`: `none` | `messages` | `child` | `orchestrator` | `cumulative` *)
- `history_merge_target`: `none` | `messages` **) | `child` | `orchestrator` | `cumulative` *)

Defaults (change current behavior to a cleaner baseline):
- `history_source`: `none`
- `history_merge_target`: `none`

Notes:
- `history_source=none`: no forked history is loaded (child starts empty).
- `history_source=messages`: fork base is loaded from the `messages` history file.
- `history_source=child`: fork base is the child template agent’s `message_history`.
- `history_source=orchestrator`: fork base is the parent/orchestrator `message_history`.
- `history_source=cumulative`: fork base is the session-wide merged transcript (not yet implemented).
- `history_merge_target=none`: no merge back occurs.
- `history_merge_target=messages`: merge back into the `messages` history file.
- `history_merge_target=child`: merge back into the child template agent’s `message_history`.
- `history_merge_target=orchestrator`: merge back into the parent/orchestrator `message_history`.
- `history_merge_target=cumulative`: merge back into the session-wide transcript (not yet implemented).

MVP path 1:
- `/card --tool` starts in **stateless** mode (`history_source=none`, `history_merge_target=none`),
  i.e. fresh clone per call with no history load or merge.
MVP path 2:
- Advanced history modes should be exercised first in the Agents-as-Tools workflow
  before being applied to `/card --tool`.

*) *Footnote:* there is no cumulative (session-wide merged) history store today;
it would need to be designed and implemented as a separate feature.
**) *Footnote:* writing merged history to a file-based `history_merge_target`
requires a read/write lock and is deferred to a separate implementation.

### Python API (proposed)
```python
AgentsAsToolsOptions(
    history_source="none",
    history_merge_target="none",
)
```

### CLI flags (proposed)
```
--child-history-source {none,messages,child,orchestrator,cumulative}
--child-history-merge-target {none,messages,child,orchestrator,cumulative}
```

### `/call` or tool invocation (proposed)
If a `/call` command (or MCP tool wrapper) is introduced for ad-hoc child calls,
it should accept the same options as overrides for the current invocation:
```
/call <agent> --history-source orchestrator --history-merge-target orchestrator
```

Rationale for open questions:
- There are **two plausible histories**: the child template history (stable per agent)
  and the orchestrator history (dynamic per session). Both are valid depending
  on whether you want a child to act with its own memory or respond in the
  orchestrator’s current context. A third option, **cumulative**, represents a
  session-wide merged transcript across agents (if implemented).
- Merge destination is ambiguous: merging back into the child template is useful for
  long-lived agent memory; merging into the orchestrator is useful for building a
  shared session transcript.

---

## MCP Servers and Tool Filters (YAML)
Match the existing decorator semantics:
- `servers`: list of MCP server names (strings), resolved via `fastagent.config.yaml`.
- `tools`: optional mapping `{server_name: [tool_name_or_pattern, ...]}`.
  - If omitted, all tools for that server are allowed.

Example:
```yaml
servers:
  - time
  - github
  - filesystem
tools:
  time: [get_time]
  github: [search_*]
```

---

## Precedence
1) CLI flags (highest priority)
2) AgentCard fields
3) `fastagent.config.yaml`

This applies to model selection, request params, servers, and other overlapping fields.

---

## Function Tools and Hooks (Separate Spec)
Function tool and hook wiring is evolving and documented separately.
See: [plan/hook-tool-declarative.md](plan/hook-tool-declarative.md) (current branch changes live there).
Relative `function_tools` paths resolve against the AgentCard file directory.

---

## Examples

### Basic agent card
```md
---
type: agent
name: sizer
---
Given an object, respond only with an estimate of its size.
```

### Agent with servers and child agents
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

### Tool-only agent (hidden from agent list)
```md
---
type: agent
name: code_formatter
tool_only: true
description: Formats code according to project style guidelines.
---
You are a code formatting helper. Format the provided code according to best practices.
```

When placed in `.fast-agent/tool-cards/`, this agent:
- Does **not** appear in `/agents` or `agent_names()`
- Cannot be selected as the default agent
- Can be used as a tool by other agents via `agents: [code_formatter]`
- Can still be accessed directly via `app["code_formatter"]` if needed

---

## Loading API
- `load_agents(path)` loads a file or a directory and returns the loaded agent names.
- CLI: `fast-agent go/serve/acp --card <path>` loads cards before starting.
- CLI: if `.fast-agent/agent-cards/` or `.fast-agent/tool-cards/` exists and contains
  `.md`/`.markdown`/`.yaml`/`.yml` files, those directories are loaded automatically
  (in addition to any explicit `--card`/`--card-tool` entries).
- `--agent-cards` remains as a legacy alias for `--card`.
- Loading is immediate (no deferred mode).
- All loaded agents are tracked with a name and source file path.
- If a subsequent `load_agents(path)` call does not include a previously loaded agent
  from that path, the agent is removed.
- TUI: `/card <path|url> [--tool]` loads cards at runtime. Autocomplete filters for
  AgentCard file extensions.
- ACP slash commands: `/card <path|url> [--tool]` loads cards at runtime and refreshes
  modes for the current session.

### Runtime tool injection (optional)
- `/card --tool` exposes the loaded agent as a tool on the **current** agent.
- CLI: `fast-agent go --card-tool <path>` loads cards and exposes them as tools on the default agent.
- Tool names default to `agent__{name}`.
- Tool descriptions prefer `description`; fall back to the agent instruction.
- Tool calls use a single `message` argument.
- Default behavior is **stateless**: fresh clone per call with no history load or merge
  (`history_source=none`, `history_merge_target=none`).

### Example: export AgentCards from a Python workflow
```bash
cd examples/workflows

uv run agents_as_tools_extended.py --dump ../workflows-md/agents_as_tools_extended
```

### Example: run interactive with hot lazy swap
```bash
cd examples/workflows-md

uv run fast-agent go --card agents_as_tools_extended --watch
```

Manual reload:
```bash
cd examples/workflows-md

uv run fast-agent go --card agents_as_tools_extended --reload
```

One-shot message:
```bash
cd examples/workflows-md

uv run fast-agent go --card agents_as_tools_extended --message "go"
```

### Example: load a directory in Python
```python
import asyncio

from fast_agent import FastAgent

fast = FastAgent("workflows-md")
fast.load_agents("/home/strato-space/fast-agent/examples/workflows-md/agents_as_tools_extended")


async def main() -> None:
    async with fast.run() as app:
        await app.interactive()


if __name__ == "__main__":
    asyncio.run(main())
```

## Export / Dump (CLI)
- Default export format is Markdown (frontmatter + body), matching SKILL.md style.
- `--dump <dir>` (alias: `--dump-agents`): after loading, export all loaded agents to `<dir>` as
  Markdown AgentCards (`<agent_name>.md`). Instruction is written to the body.
- `--dump-yaml <dir>` (alias: `--dump-agents-yaml`): export all loaded agents as YAML AgentCards
  (`<agent_name>.yaml`) with `instruction` in the YAML field.
- `--dump-agent <name> --dump-agent-path <file>`: export a single agent as Markdown
  (default) to a file.
- `--dump-agent-yaml`: export a single agent as YAML (used with `--dump-agent` and
  `--dump-agent-path`).
 - Optional future enhancement: after dumping, print a ready-to-run CLI example
   for the current directory (e.g. `fast-agent go --card <dir> --watch`).

## Interactive vs One-Shot CLI
- **Interactive**: `fast-agent go --card <dir>` launches the TUI, waits for
  user input, and keeps session state (history, tools, prompts) in memory.
- **One-shot**: `fast-agent go --card <dir> --message "..."` sends a single
  request and exits. `--prompt-file` loads a prompt/history file, runs it, then
  exits (or returns to interactive if explicitly invoked).

## Reload / Watch Behavior (Lazy Hot-Reload)
Both `--reload` and `--watch` use the same **lazy hot-reload** semantics. The loader
tracks `registry_version` (monotonic counter) and a per-file cache:
`path -> (mtime_ns, size, agent_name)`.

On each reload pass, only **changed** files are re-read:
- If `mtime_ns` or `size` differs, the file is re-parsed and its agents are updated.
- If a file disappears, its agents are removed from the registry.
- If a new file appears, its agents are added.

After a reload pass, `registry_version` is bumped if any changes were applied.
Runtime instances compare `instance_version` to the registry. If
`registry_version > instance_version`, a new instance is created on the next
eligible boundary.

### `--reload` (manual)
- No filesystem watcher.
- Reload is triggered explicitly (e.g. `/reload` in TUI; ACP/tool hooks pending).
- The loader performs an mtime-based incremental reload and updates the registry.

### `--watch` (automatic)
- OS file events trigger reload passes when `watchfiles` is available. Otherwise,
  the watcher falls back to mtime/size polling.
- Only changed files are re-read using the same mtime/size cache.
- No immediate restart; the swap happens lazily on the next request/connection.

### Instance scope behavior
- `instance_scope=shared`: on the **next request**, if version changed, the shared
  instance is recreated once (under lock), then reused for subsequent requests.
- `instance_scope=connection`: version check occurs when a new connection is opened;
  existing connections keep their old instance.
- `instance_scope=request`: a new instance is created per request, so the latest
  registry is always used.

### Force reload
- A “force” reload is a full runtime restart (process-level) to guarantee a clean
  Python module state.

## Tools Exposure (fast-agent-mcp)
Expose loader utilities via internal MCP tools:
- `fast-agent-mcp.load_agents(path)`

---

## Appendix: Multi-card Spec (Experimental)
See [plan/agent-card-rfc-multicard.md](plan/agent-card-rfc-multicard.md).

## Appendix: Current History Preload (Code)
- `save_messages(...)` and `load_messages(...)` in
  `src/fast_agent/mcp/prompt_serialization.py`
- Delimiter constants in `src/fast_agent/mcp/prompts/prompt_constants.py`
- `load_history_into_agent(...)` in `src/fast_agent/mcp/prompts/prompt_load.py`
- `/save_history` implementation in `src/fast_agent/llm/fastagent_llm.py`
- CLI `--prompt-file` loader in `src/fast_agent/cli/commands/go.py`

---

## Appendix: AgentCard Samples
See [agent-card-rfc-sample.md](agent-card-rfc-sample.md).

## Appendix: Issue Templates (from gpt-5.2-codex code review)

### Issue: prevent self-referential tool injection + dedupe tool names
**Summary:** `/card --tool` (and `--card-tool`) can register the current agent as a tool
when the loaded card set includes it. This creates a self-referential tool and can
recurse if the model calls it. Tool names can also collide silently.

**Evidence:** `add_tools_for_agents` is called with `loaded_names` as-is.
(`src/fast_agent/ui/interactive_prompt.py`, `src/fast_agent/acp/slash_commands.py`,
`src/fast_agent/core/fastagent.py`)

**Proposed fix:**
- Filter out the current agent from `loaded_names` before tool registration.
- Deduplicate tool names and warn on collisions (or skip with notice).

**Acceptance criteria:**
- Current agent is never exposed as its own tool.
- Duplicate tool names are surfaced to the user and do not silently override.

### Issue: injected tools are lost after reload/watch
**Summary:** Tool injection is **ephemeral**; after `--watch` refresh or manual reload,
injected tools disappear without warning.

**Evidence:** Tools are added to the active instance only; reload swaps the instance
with no re-application of injected tools.
(`src/fast_agent/ui/interactive_prompt.py`, `src/fast_agent/acp/slash_commands.py`,
`src/fast_agent/core/fastagent.py`)

**Proposed fix:**
- Persist injected tool intents and re-apply after reload, **or**
- Emit an explicit warning after reload that injected tools were dropped.

**Acceptance criteria:**
- After reload, either tools are restored or the user is notified.

### Issue: align `/card --tool` with Agents-as-Tools history options (future)
**Summary:** Once advanced history routing is added (history_source/history_merge_target),
`/card --tool` should either expose those options or explicitly lock to stateless mode.

**Evidence:** MVP path is stateless, but advanced history routing is under design.

**Proposed fix:**
- Reuse Agents-as-Tools helpers when advanced history routing lands, or document
  `/card --tool` as always-stateless.

**Acceptance criteria:**
- `/card --tool` behavior is explicit and consistent with Agents-as-Tools options.

## Appendix: How to solve all `/card --tool` issues

Conclusion: loading a card with `--tool` is **needed and useful**, but it should
**not** introduce a separate code path. In the Agents‑as‑Tools paradigm, any loaded
agent can be called by any other agent after declaration, so `--tool` should just
reuse the same flow.

To avoid cluttering the tool list (and to avoid creating tools for every loaded
agent by default), the orchestrator—like an agent with MCP servers/tools—declares
which agents it actually needs. Then, only those referenced in the orchestrator’s
`agents` attribute are turned into tools at runtime.
The list of agents available as tools can be extended dynamically via
`/card --tool` and `/agent --tool`.

Implementation rule:
`/card --tool` should **append** the loaded agent(s) to the current agent’s
`agents` list and hot‑swap using the existing Agents‑as‑Tools flow. That keeps the
implementation compact, avoids tool‑list clutter, and reuses the same code path.

Suggested help text:
`/card <path> [--tool]` — load an AgentCard; `--tool` attaches the loaded agent(s)
to the current agent’s `agents` list and hot‑swaps via Agents‑as‑Tools.

Suggested command:
`/agent` with options:
- `--tool`: attach the selected agent as a tool to the current agent.
- `--dump`: print the current agent’s AgentCard to screen.


## Appendix: Next-stage Work Items
- **Cumulative session history**: no shared, merged transcript exists today; requires
  a session-level history store and clear rules for when/what each agent writes.

## Appendix: History Mode Removal
`history_mode` is removed in this spec and replaced by the orthogonal pair
`history_source` + `history_merge_target`.

## Appendix: Open Questions / Remaining Work
- Decide whether `/card --subagent` is needed as a distinct primitive from tool injection.
- Define how (or if) advanced history routing is exposed outside agents-as-tools.
- Confirm whether a shared “cumulative” history across `@agent` switches is desired.

## Appendix: Open Issues

- AgentCard --watch: minimal incremental refresh (mtime/size, per-card reload, safe parse) [#603](https://github.com/evalstate/fast-agent/issues/603)

- Scope: apply to `--watch` for AgentCard roots.
- Detect changes using `mtime+size`; reload only changed card files.
- If a card fails to parse (empty/partial write), log a warning and skip; retry on the next change.
- If a tool file changes, reload only cards that reference that tool file.
- Handle removals by unregistering the agent and updating available agents without restarting the session.
- Refresh only the affected agent instances (no full app rebuild); for `instance_scope=shared`, swap updated
  agents; for `request/connection`, bump the registry version so new instances see the update.
- UX: emit a short “AgentCards reloaded” line and refresh the available agent list.
