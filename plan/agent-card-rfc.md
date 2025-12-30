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

### Markdown Card (`.md`)
A Markdown card is YAML frontmatter followed by an optional body. The body is treated
as the system instruction unless `instruction` is provided in frontmatter.

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
- Loading is immediate (no deferred mode).
- All loaded agents are tracked with a name and source file path.
- If a subsequent `load_agents(path)` call does not include a previously loaded agent
  from that path, the agent is removed.

## Reload / Watch Behavior
- `--watch`: enable OS file events (auto-reload on save).
- `--reload`: manual re-scan on demand (explicit refresh).

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
