<AgentCards>
---

# Agent Card (type: `agent`)

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
- `model` — model ID.
- `instruction` — system/developer prompt string (mutually exclusive with body).
- `skills` — list of skills. **Disable all skills:** `skills: []`.
- `servers` — list of configured MCP server names.
- `tools` / `resources` / `prompts` — map: `server_name -> [allowed_items]`.
- `mcp_connect` — optional runtime MCP targets resolved at startup.
  - entries require `target` and may include optional `name`.
  - target forms: `https://...`, `@scope/pkg`, `npx ...`, `uvx ...`, or stdio command.
- `agents` — list of child agents (Agents-as-Tools).
- `tool_input_schema` — optional JSON Schema for this card when exposed as `agent__<name>`.
  - If omitted, parent agents use the legacy `message: string` tool schema.
  - Use `properties.<param>.description` for clear parameter guidance.
- `use_history` — bool (default `true`).
- `messages` — path or list of history files (relative to card directory).
- `request_params` — request/model overrides.
- `human_input` — bool (enable human input tool).
- `shell` — bool (enable shell); `cwd` optional.
- `default` — marks this agent as the `smart` tool target when the path resolves multiple cards. First `default: true` non-`tool_only` agent wins; if none, the first non-`tool_only` agent is used.
- `tool_only` — excludes this agent from default selection; it can only be invoked by other agents as a tool.

---

## Instruction templates (placeholders)
You can insert these in the **body** or `instruction:`.

| Placeholder | Meaning |
|---|---|
| `\{{currentDate}}` | Current date (e.g., “17 December 2025”) |
| `\{{hostPlatform}}` | Host platform string |
| `\{{pythonVer}}` | Python version |
| `\{{workspaceRoot}}` | Workspace root path (if available) |
| `\{{env}}` | Environment summary (client, host, workspace) |
| `\{{agentName}}` | Current agent name |
| `\{{agentType}}` | Current agent type |
| `\{{agentCardPath}}` | Source AgentCard path (if loaded from card) |
| `\{{agentCardDir}}` | Directory containing the source AgentCard |
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
