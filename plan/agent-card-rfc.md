# AgentCard RFC (Draft)

## Summary
AgentCard is a text-first format (`.md` / `.yaml`) that compiles into `AgentConfig`.
A loader validates fields based on `type` and loads a single file or a directory via
`load_agents(path)`. The default path is **one card per file**. Multi-card files are
optional/experimental and described in a separate spec.

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
schema. Use `instruction: |` for multiline prompts.

Example:
```yaml
type: agent
name: sizer
instruction: |
  Given an object, respond only with an estimate of its size.
```

### Markdown Card (`.md` / `.markdown`)
A Markdown card is YAML frontmatter followed by an optional body. The body is treated
as the system instruction unless `instruction` is provided in frontmatter.
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
- `name`, `instruction`, `default`
- `agents` (agents-as-tools)
- `servers`, `tools`, `resources`, `prompts`, `skills`
- `model`, `use_history`, `request_params`, `human_input`, `api_key`
- `history_mode`, `max_parallel`, `child_timeout_sec`, `max_display_instances`
- `function_tools`, `tool_hooks` (see separate spec)
- `messages` (card-only history file)

### type: `chain` (maps to `@fast.chain`)
Allowed fields:
- `name`, `instruction`, `default`
- `sequence`, `cumulative`

### type: `parallel` (maps to `@fast.parallel`)
Allowed fields:
- `name`, `instruction`, `default`
- `fan_out`, `fan_in`, `include_request`

### type: `evaluator_optimizer` (maps to `@fast.evaluator_optimizer`)
Allowed fields:
- `name`, `instruction`, `default`
- `generator`, `evaluator`
- `min_rating`, `max_refinements`, `refinement_instruction`
- `messages` (card-only history file)

### type: `router` (maps to `@fast.router`)
Allowed fields:
- `name`, `instruction`, `default`
- `agents`
- `servers`, `tools`, `resources`, `prompts`
- `model`, `use_history`, `request_params`, `human_input`, `api_key`
- `messages` (card-only history file)

### type: `orchestrator` (maps to `@fast.orchestrator`)
Allowed fields:
- `name`, `instruction`, `default`
- `agents`
- `model`, `use_history`, `request_params`, `human_input`, `api_key`
- `plan_type`, `plan_iterations`
- `messages` (card-only history file)

### type: `iterative_planner` (maps to `@fast.iterative_planner`)
Allowed fields:
- `name`, `instruction`, `default`
- `agents`
- `model`, `request_params`, `api_key`
- `plan_iterations`
- `messages` (card-only history file)

### type: `MAKER` (maps to `@fast.maker`)
Allowed fields:
- `name`, `instruction`, `default`
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
See: `plan/hook-tool-declarative.md` (current branch changes live there).

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

---

## Loading API
- `load_agents(path)` loads a file or a directory.
- CLI: `fast-agent go --agent-cards <path>` loads cards before starting.
- Loading is immediate (no deferred mode).
- All loaded agents are tracked with a name and source file path.
- If a subsequent `load_agents(path)` call does not include a previously loaded agent
  from that path, the agent is removed.

### Example: export AgentCards from a Python workflow
```bash
cd examples/workflows

uv run agents_as_tools_extended.py --dump-agents ../workflows-md/agents_as_tools_extended
```

### Example: run interactive with hot lazy swap
```bash
cd examples/workflows-md

uv run fast-agent go --agent-cards agents_as_tools_extended --watch
```

Manual reload:
```bash
cd examples/workflows-md

uv run fast-agent go --agent-cards agents_as_tools_extended --reload
```

One-shot message:
```bash
cd examples/workflows-md

uv run fast-agent go --agent-cards agents_as_tools_extended --message "go"
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
- `--dump-agents <dir>`: after loading, export all loaded agents to `<dir>` as
  Markdown AgentCards (`<agent_name>.md`). Instruction is written to the body.
- `--dump-agents-yaml <dir>`: export all loaded agents as YAML AgentCards
  (`<agent_name>.yaml`) with `instruction` in the YAML field.
- `--dump-agent <name> --dump-agent-path <file>`: export a single agent as Markdown
  (default) to a file.
- `--dump-agent-yaml`: export a single agent as YAML (used with `--dump-agent` and
  `--dump-agent-path`).
 - Optional future enhancement: after dumping, print a ready-to-run CLI example
   for the current directory (e.g. `fast-agent go --agent-cards <dir> --watch`).

## Interactive vs One-Shot CLI
- **Interactive**: `fast-agent go --agent-cards <dir>` launches the TUI, waits for
  user input, and keeps session state (history, tools, prompts) in memory.
- **One-shot**: `fast-agent go --agent-cards <dir> --message "..."` sends a single
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
See `plan/agent-card-rfc-multicard.md`.

## Appendix: Current History Preload (Code)
- `save_messages(...)` and `load_messages(...)` in
  `src/fast_agent/mcp/prompt_serialization.py`
- Delimiter constants in `src/fast_agent/mcp/prompts/prompt_constants.py`
- `load_history_into_agent(...)` in `src/fast_agent/mcp/prompts/prompt_load.py`
- `/save_history` implementation in `src/fast_agent/llm/fastagent_llm.py`
- CLI `--prompt-file` loader in `src/fast_agent/cli/commands/go.py`

---

## Appendix: AgentCard Samples
See `plan/agent-card-rfc-sample.md`.
