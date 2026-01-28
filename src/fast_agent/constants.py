"""
Global constants for fast_agent with minimal dependencies to avoid circular imports.
"""

# Canonical tool name for the human input/elicitation tool
HUMAN_INPUT_TOOL_NAME = "__human_input"
MCP_UI = "mcp-ui"
REASONING = "reasoning"
ANTHROPIC_THINKING_BLOCKS = "anthropic-thinking-raw"
"""Raw Anthropic thinking blocks with signatures for tool use passback."""
OPENAI_REASONING_ENCRYPTED = "openai-reasoning-encrypted"
"""Encrypted OpenAI reasoning items for stateless Responses API passback."""
FAST_AGENT_ERROR_CHANNEL = "fast-agent-error"
FAST_AGENT_REMOVED_METADATA_CHANNEL = "fast-agent-removed-meta"
FAST_AGENT_TIMING = "fast-agent-timing"
FAST_AGENT_TOOL_TIMING = "fast-agent-tool-timing"
FAST_AGENT_USAGE = "fast-agent-usage"

FORCE_SEQUENTIAL_TOOL_CALLS = False
"""Force tool execution to run sequentially even when multiple tool calls are present."""
# should we have MAX_TOOL_CALLS instead to constrain by number of tools rather than turns...?
DEFAULT_MAX_ITERATIONS = 99
"""Maximum number of User/Assistant turns to take"""

DEFAULT_STREAMING_TIMEOUT = 300.0
"""Default streaming timeout in seconds for provider streaming responses."""

DEFAULT_TERMINAL_OUTPUT_BYTE_LIMIT = 8192
"""Baseline byte limit for ACP terminal output when no model info exists."""

TERMINAL_OUTPUT_TOKEN_RATIO = 0.25
"""Target fraction of model max output tokens to budget for terminal output."""

TERMINAL_OUTPUT_TOKEN_HEADROOM_RATIO = 0.2
"""Leave headroom for tool wrapper text and other turn data."""

# Empirical observation from real shell outputs (135 samples, avg 3.33 bytes/token)
TERMINAL_BYTES_PER_TOKEN = 3.3
"""Bytes-per-token estimate for terminal output limits and display."""

MAX_TERMINAL_OUTPUT_BYTE_LIMIT = 32768
"""Hard cap on default ACP terminal output to avoid oversized tool payloads."""

DEFAULT_AGENT_INSTRUCTION = """You are a helpful AI Agent.

{{serverInstructions}}
{{agentSkills}}
{{file_silent:AGENTS.md}}
{{env}}

The current date is {{currentDate}}."""


# AUTODEV: add this to an internal: propmpt resolver
SMART_AGENT_INSTRUCTION = """You are a smart fast-agent helper.

Below is a **terse, LLM-oriented spec** for the default **agent** card.

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
- `servers` / `tools` / `resources` / `prompts` — map: `server_name -> [allowed_items]`.
- `agents` — list of child agents (Agents-as-Tools).
- `use_history` — bool (default `true`).
- `messages` — path or list of history files (relative to card directory).
- `request_params` — request/model overrides.
- `human_input` — bool (enable human input tool).
- `shell` — bool (enable shell); `cwd` optional.
- `default` / `tool_only` — booleans for default or tool-only behavior.

---

## Instruction templates (placeholders)
You can insert these in the **body** or `instruction:`.

| Placeholder | Meaning |
|---|---|
| `\\{{currentDate}}` | Current date (e.g., “17 December 2025”) |
| `\\{{hostPlatform}}` | Host platform string |
| `\\{{pythonVer}}` | Python version |
| `\\{{workspaceRoot}}` | Workspace root path (if available) |
| `\\{{env}}` | Environment summary (client, host, workspace) |
| `\\{{serverInstructions}}` | MCP server instructions (if any) |
| `\\{{agentSkills}}` | Formatted skill descriptions |

---

## Content includes (inline)
- `\\{{url:https://...}}` — fetch and inline URL content.
- `\\{{file:relative/path}}` — inline file content (error if missing).
- `\\{{file_silent:relative/path}}` — inline file content, **empty if missing**.

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

\\{{env}}
\\{{currentDate}}
\\{{file:docs/house-style.md}}
```

---

{{serverInstructions}}
{{agentSkills}}
{{file_silent:AGENTS.md}}
{{env}}

Fast-agent environment paths:
- Environment root: {{environmentDir}}
- Agent cards: {{environmentAgentCardsDir}}
- Tool cards: {{environmentToolCardsDir}}

Use the smart tool to load AgentCards temporarily when you need extra agents.
Use validate to check AgentCard files before running them.

The current date is {{currentDate}}."""

DEFAULT_GO_AGENT_TYPE = "smart"
"""Default agent type for CLI single-model runs ("smart" or "agent")."""

DEFAULT_SERVE_AGENT_TYPE = "agent"
"""Default agent type for CLI serve/acp single-model runs ("smart" or "agent")."""


DEFAULT_ENVIRONMENT_DIR = ".fast-agent"

DEFAULT_SKILLS_PATHS = [
    f"{DEFAULT_ENVIRONMENT_DIR}/skills",
    ".claude/skills",
]

CONTROL_MESSAGE_SAVE_HISTORY = "***SAVE_HISTORY"

FAST_AGENT_SHELL_CHILD_ENV = "FAST_AGENT_SHELL_CHILD"
"""Environment variable set when running fast-agent shell commands."""

SHELL_NOTICE_PREFIX = "[yellow][bold]Agents have shell[/bold][/yellow]"
