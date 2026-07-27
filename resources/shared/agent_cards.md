<AgentCards>
---

# AgentCard

## Format
- **Markdown** with YAML frontmatter + body, or **YAML** only.
- **Body = system/developer instruction.**
  - Optional first non-empty line `---SYSTEM` is stripped.
- `instruction:` field **or** body may define the instruction (not both).
- If neither is present, the **default instruction** is used.
- **Invocation:** the card defines the agent; you invoke it later with a **user message** (first user turn).
  - `messages:` is **history files**, not the invocation message.

---

## Main fields (frontmatter, type = `agent`)
- `name` — string; defaults to file stem.
- `description` — optional summary.
- `type` — `"agent"` (default if omitted).
- `type: smart` — deprecated 0.10 compatibility alias for `type: agent` with
  `subagents: true` and `harness_tools: true`; remove it before 0.11.
- `model` — model ID.
- `instruction` — system/developer prompt string (mutually exclusive with body).
- `skills` — list of skills. **Disable all skills:** `skills: []`.
- `servers` — list of configured MCP server names.
- `tools` / `resources` / `prompts` — map: `server_name -> [allowed_items]`.
- `mcp_connect` — optional runtime MCP targets resolved at startup.
  - entries normally require `target` and may include optional `name`.
  - target forms: `https://...`, `@scope/pkg`, `npx ...`, `uvx ...`, or stdio command.
  - provider-managed remote MCP may add: `management: provider`, `description`, `access_token`, `defer_loading`.
  - provider-managed OpenAI connectors use structured entries with `name`, `management: provider`, `connector_id`, and `access_token`; omit `target` for connector-backed entries.
  - valid connector IDs come from the pinned OpenAI SDK; current IDs are: `connector_dropbox`, `connector_gmail`, `connector_googlecalendar`, `connector_googledrive`, `connector_microsoftteams`, `connector_outlookcalendar`, `connector_outlookemail`, `connector_sharepoint`.
  - for OpenAI Responses provider-managed remote MCP or connectors, `defer_loading: true` automatically enables server-side `tool_search` so tool definitions load lazily.
- `agents` — list of child agents (Agents-as-Tools).
- `tool_input_schema` — optional JSON Schema for this card when exposed as `agent__<name>`.
  - If omitted, parent agents use the legacy `message: string` tool schema.
  - Use `properties.<param>.description` for clear parameter guidance.
- `use_history` — bool (default `true`).
- `save_trajectory` — bool (default `false`); requires `use_history: false`.
  Saves one replay-oriented JSON trace per stateless invocation in the active
  session's `trajectories/` directory.
- `messages` — path or list of history files (relative to card directory).
- `request_params` — request/model overrides.
  - `tool_result_mode` controls what a caller receives after this agent uses tools.
  - `postprocess` means the agent uses tool outputs to compose a final reply.
  - `passthrough` means the tool result is returned directly instead of being rewritten into a final reply.
  - `selectable` means that, when this agent is exposed as a tool, callers can choose per invocation with `response_mode: inherit | postprocess | passthrough`.
- `human_input` — bool (enable human input tool).
- `shell` — bool (enable shell); `cwd` optional.
- `default` — marks this agent as the default runnable card when the path resolves multiple cards. First `default: true` non-`tool_only` agent wins; if none, the first non-`tool_only` agent is used.
- `tool_only` — excludes this agent from default selection; it can only be invoked by other agents as a tool.
- `subagents` — optional bool controlling the built-in `subagent` tool. Set
  `true` to enable it. Tool-only agents and built-in subagent children always
  disable it.
- `subagent_model` — optional non-empty model spec that every built-in
  subagent run must use.
- `harness_tools` — bool (default `false`). Adds the allow-listed
  `slash_command` and `get_resource` tools to a basic agent.

---

## Built-in subagent controls

```yaml
subagents: true
subagent_model: passthrough
```

When `subagent_model` is set, the `subagent` tool does not expose a `model`
argument and every child uses that model. Otherwise, a tool-call `model`
override wins; if omitted, the child inherits its parent's current model.
`subagent_model` has no effect when `subagents: false`.

Each invocation runs once in a clean conversation context. The child inherits
the parent's instruction and available capabilities except the built-in
`subagent` tool, cannot recursively delegate, and persists its full transcript
as a nested non-resumable session. Parallel calls run concurrently while the
parent waits for all results.

An exact standalone `fast-agent-subagents` line or
`<!-- fast-agent-subagents -->` comment in the resolved system instruction
enables the tool only when `subagents` is unset. The directive is stripped
before the model sees the instruction. Explicit `--no-subagents` and
`subagents: false` settings always win.

## Harness tools

```yaml
harness_tools: true
```

This installs `slash_command(command)` for selected read-only fast-agent
commands and `get_resource(uri, server_name?)` for bundled `internal://`
resources or attached MCP resources. Call `slash_command("/commands")` to
discover the available model-facing command surface. Harness tools can be
toggled at runtime and are not inherited by detached or built-in subagent
clones.

---

## Instruction templates (placeholders)
You can insert these in the **body** or `instruction:`.

| Placeholder | Meaning |
|---|---|
| `\{{currentDate}}` | Current date (e.g., “17 December 2025”) |
| `\{{hostPlatform}}` | Client host platform string |
| `\{{pythonVer}}` | Python version |
| `\{{workspaceRoot}}` | Host workspace root path (used for local file includes and skill discovery) |
| `\{{clientDisplay}}` | Client display name |
| `\{{executionEnvironment}}` | Active shell execution environment summary |
| `\{{executionEnvironmentName}}` | Active named environment, if selected |
| `\{{executionEnvironmentKind}}` | Active environment kind (`local`, `docker`, `remote`, etc.) |
| `\{{executionEnvironmentProvider}}` | Active environment provider (`docker`, `huggingface`, `wslc`, etc.) |
| `\{{executionEnvironmentShell}}` | Shell used by the active environment |
| `\{{executionEnvironmentCwd}}` | Working directory inside the active environment, if known |
| `\{{env}}` | Environment summary (host workspace, active execution environment, client and process ID, fast-agent Python runtime, client host platform) |
| `\{{agentName}}` | Current agent name |
| `\{{agentType}}` | Current agent type |
| `\{{agentCardPath}}` | Source AgentCard path (if loaded from card) |
| `\{{agentCardDir}}` | Directory containing the source AgentCard |
| `\{{modelReferences}}` | Model references, including the effective `$system.default` |
| `\{{model_specific}}` | Model-specific prompt guidance from the resolved model catalog entry or model overlay |
| `\{{serverInstructions}}` | MCP server instructions (if any) |
| `\{{agentSkills}}` | Formatted skill descriptions |
| `\{{agentInternalResources}}` | Bundled internal resource index |

---

## Content includes (inline)
- `\{{url:https://...}}` — fetch and inline URL content.
- `\{{file:relative/path}}` — inline file content (error if missing).
- `\{{file_silent:relative/path}}` — inline file content, **empty if missing**.

**Note:** file paths are **relative** (resolved against `workspaceRoot` when available).

---

## Minimal example (Markdown)

```md
---
name: my_agent
description: Focused helper
model: gpt-oss
skills: []   # disable skills
use_history: true
---

You are a concise assistant.

\{{env}}
\{{currentDate}}
\{{file:docs/house-style.md}}
```

---

</AgentCards>
