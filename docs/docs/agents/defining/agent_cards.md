---
title: Agent Cards
description: Define portable fast-agent agents with Markdown or YAML, including
  Python function tools, hooks, MCP servers, models, and tool-card loading.
social:
  title: Agent Cards
  tagline: Define portable agents with prompts, tools, hooks, MCP servers, and models.
  description: Define portable agents with prompts, tools, hooks, MCP servers, and models.
  alt: fast-agent social card — Agent Cards
---

# Agent Cards

Agent Cards are portable agent definitions. Use them when you want an agent that
can be checked in, shared, loaded by the CLI/TUI, or used from Python without
rewriting the definition as decorators.

An Agent Card can contain:

- the agent's instructions;
- model and request settings;
- MCP servers, tool/resource/prompt filters, and runtime MCP connections;
- local Python function tools;
- tool-loop hooks and lifecycle hooks;
- skills, shell access, history behaviour, and tool-card metadata.

Agent Cards and Python decorators feed the same fast-agent registry. The choice
is about authoring style and portability, not a separate runtime.

## Minimal Markdown card

Create `.fast-agent/agent-cards/support.md`:

```md
---
name: support
description: Answer customer support questions using the configured tools.
model: sonnet
servers:
  - filesystem
use_history: true
request_params:
  max_iterations: 8
---

You are a concise customer support assistant.

Ask for missing account details before taking actions.
```

Run it from the CLI:

```bash
fast-agent go --agent support
```

Or load it from Python:

```python
fast.load_agents(".fast-agent/agent-cards")

async with fast.run() as app:
    print(await app.support("Help me reset my password"))
```

The Harness API also loads cards from the active fast-agent home's `agent-cards/`
directory during startup:

```python
async with fast.harness() as harness:
    session = await harness.session("customer-123", agent_name="support")
    response = await session.generate("Help me reset my password")
```

## Card file format

Agent Cards can be written as Markdown with YAML frontmatter:

- `.md`
- `.markdown`

or as YAML-only files:

- `.yaml`
- `.yml`

Markdown cards are usually easier to read because the frontmatter contains
configuration and the body contains the instruction prompt.

## Configure built-in subagents

The built-in `subagent` tool is disabled unless the card sets
`subagents: true`. Tool-only agents and children created by the built-in tool
always keep it disabled.

Pin all built-in child runs to a model with `subagent_model`:

```md
---
name: coordinator
model: sonnet
subagents: true
subagent_model: gpt-oss
---

Delegate focused tasks to subagents.
```

With `subagent_model`, the model-visible `subagent` tool schema has no `model`
argument and every child uses the configured model. `subagent_model` has no
effect unless `subagents: true`.

The model calls `subagent(message, model?, label?, include_user_message?)`. Each call:

- starts a one-shot child with a clean conversation context;
- inherits the parent's instruction and available capabilities, except the
  built-in `subagent` tool, so children cannot recursively delegate;
- accepts an optional model override and short display label when
  `subagent_model` is not fixed;
- can optionally include only the latest external user message's text and
  attachments (never conversation history) in the child request. This can
  forward user content to another model or provider. Text is XML-escaped and
  appended after the explicit task inside
  `<included_user_context>...</included_user_context>`;
- can run concurrently with other subagent calls while the parent waits for
  all results; and
- persists its complete transcript as a nested, non-resumable child session.

Persisted runs receive a short alias scoped to the parent session:

```text
01_investigate_item
02_review_api_contract
```

The ordinal is allocated when the run starts. The slug comes from the supplied
label, or from the task when no label is supplied, and is normalized to bounded
lowercase ASCII. Use `/subagents` to list runs in the current session:

```text
/subagents
/subagents status
/subagents off
/subagents on
/subagents toggle
```

`on`, `off`, and `toggle` are runtime-only overrides for the current agent.
They do not rewrite its AgentCard, and `on` does not override an explicit
AgentCard or CLI disable.

The tool returns the child's final response first. When the active execution
environment supports private temporary artifacts, it also returns a temporary
path to a bounded, line-oriented transcript. The parent can search that file or
read selected line ranges without loading the whole transcript into context.
The path is valid only during the current runtime and is removed when the
parent shuts down. The transcript includes child messages and tool activity,
but excludes reasoning content.

For repository-level opt-in without editing an AgentCard, add this exact
standalone directive to the system prompt or an embedded `AGENTS.md`:

```md
<!-- fast-agent-subagents -->
```

fast-agent removes the directive before sending the instruction to the model.
It only enables subagents when the AgentCard leaves `subagents` unset.

The comment can include instructions intended only for the parent agent:

```md
<!-- fast-agent-subagents
use terra for analysis
-->
```

The body is included in the parent system instruction, while the complete
comment is excluded from built-in subagent instructions.
`--no-subagents` and `subagents: false` always win.

## Enable harness tools

Basic agents can opt into model-visible tools for inspecting and managing
fast-agent itself:

```yaml
harness_tools: true
```

This installs:

- `slash_command(command)` for an allow-listed slash-command surface, including
  `/mcp` and `/skills`;
- `get_resource(uri, server_name?)` for bundled `internal://` resources and
  resources from attached MCP servers.

Use `slash_command("/commands")` to list the commands available to the model.
The `/mcp` and `/skills` families can connect servers and install or remove
skills. Installing an active skill may also enable the shell and filesystem
tools needed to use it. Model-initiated OAuth is non-interactive; use an
explicit token or complete OAuth from a user-facing command. Ad-hoc stdio MCP
commands require shell access, while configured servers and MCP URLs do not.
Enable harness tools only for agents and sources you trust.
Harness tools are disabled by default, can be enabled or disabled on a live
agent, and are not inherited by detached or built-in subagent clones. The
setting is accepted only by basic `agent` cards.

## Add Python function tools

Cards can expose local Python functions as tools with `function_tools`. This is
often the quickest way to add deterministic behaviour without writing an MCP
server.

```python title=".fast-agent/tools.py"
def lookup_order(order_id: str) -> dict[str, str]:
    """Look up an order by ID."""
    return {"order_id": order_id, "status": "shipped"}


def refund_order(order_id: str, reason: str) -> str:
    """Request a refund for an order."""
    return f"Refund requested for {order_id}: {reason}"
```

```md title=".fast-agent/agent-cards/support.md"
---
name: support
model: sonnet
function_tools:
  - ../tools.py:lookup_order
  - ../tools.py:refund_order
---

You help customers with order status and refunds.
Use the available tools before answering order-specific questions.
```

Function specs are `path/to/file.py:function_name`. Relative paths are resolved
from the card's directory.

You can also use object entries for code-oriented tools:

```yaml
function_tools:
  - entrypoint: ../tools.py:run_python
    variant: code
    code_arg: code
    language: python
```

## Add Python hooks

Hooks let a card run Python at well-defined points without changing the core
agent implementation.

### Tool-loop hooks

Use `tool_hooks` to observe or mutate the agent's tool loop:

```python title=".fast-agent/hooks.py"
from fast_agent.hooks import HookContext


async def log_turn(ctx: HookContext) -> None:
    print(f"{ctx.agent_name}: {ctx.hook_type}")
```

```md
---
name: support
function_tools:
  - ../tools.py:lookup_order
tool_hooks:
  before_llm_call: ../hooks.py:log_turn
  after_llm_call: ../hooks.py:log_turn
  before_tool_call: ../hooks.py:log_turn
  after_tool_call: ../hooks.py:log_turn
  after_turn_complete: ../hooks.py:log_turn
---

You help customers with order status.
```

Supported `tool_hooks` phases:

| Hook | When it runs |
|---|---|
| `before_llm_call` | before a model call |
| `after_llm_call` | after a model response |
| `before_tool_call` | before a tool is executed |
| `after_tool_call` | after a tool result is received |
| `after_turn_complete` | after the agent turn completes |

Hook functions must be async and accept a `HookContext`.

### Lifecycle hooks

Use `lifecycle_hooks` for agent startup and shutdown:

```python title=".fast-agent/lifecycle.py"
from fast_agent.hooks import AgentLifecycleContext


async def record_lifecycle(ctx: AgentLifecycleContext) -> None:
    print(f"{ctx.agent_name}: {ctx.hook_type}")
```

```yaml
lifecycle_hooks:
  on_start: ../lifecycle.py:record_lifecycle
  on_shutdown: ../lifecycle.py:record_lifecycle
```

Supported lifecycle phases are `on_start` and `on_shutdown`.

## Configure MCP servers and filters

Use `servers` for MCP servers already configured in `fast-agent.yaml`:

```yaml
servers:
  - filesystem
tools:
  filesystem:
    - read_file
    - list_directory
resources:
  filesystem:
    - repo://readme
prompts:
  filesystem:
    - summarize_file
```

Use `mcp_connect` when a card needs MCP servers that are not preconfigured under
`mcp.servers` in `fast-agent.yaml`.

```yaml
mcp_connect:
  docs:
    target: "https://demo.hf.space"
    protocol_mode: modern
    headers:
      Authorization: "Bearer ${DEMO_TOKEN}"
    auth:
      oauth: true
  everything:
    target: "@modelcontextprotocol/server-everything"
```

- The canonical form is a mapping from server name to target settings.
- The legacy list form remains accepted and is preserved when a card is dumped.
- `target` (required): URL, `@pkg`, `npx ...`, `uvx ...`, or stdio command.
- `protocol_mode` (optional): `auto`, `modern`, or `legacy`.
- `headers` (optional): structured HTTP headers.
- `auth` (optional): structured auth settings, for example `oauth: true`.
- Process fields (`command`, `args`, `env`, and `cwd`) are not permitted in
  AgentCards. Declare a trusted `target`; fast-agent materializes process
  settings from that target.

For provider-managed remote MCP, use:

```yaml
mcp_connect:
  huggingface:
    target: "https://huggingface.co/mcp"
    management: provider
    access_token: "${HF_TOKEN}"
    description: "Hugging Face MCP"
```

- `management: provider` delegates remote MCP execution to the LLM provider.
- `target` must be a URL-based remote server when `management: provider` is used.
- `access_token` is the bearer token for the remote MCP server.
- `description` is optional provider-facing metadata.
- `defer_loading` is an OpenAI Responses hint for lazy remote tool loading.
- Do not use `headers` or `auth` with provider-managed entries; use
  `access_token` instead.

Provider-managed card targets are supported only for agents using:

- `anthropic`
- `responses`

They are not supported for `codexresponses`, Codex OAuth aliases,
`openresponses`, `anthropic-vertex`, or other providers.

OpenAI Responses connectors can also be declared as structured provider-managed
card entries. Use `connector_id` instead of `target`:

```yaml
mcp_connect:
  dropbox:
    management: provider
    connector_id: connector_dropbox
    access_token: "${DROPBOX_OAUTH_ACCESS_TOKEN}"
    description: "Dropbox connector"
    defer_loading: true
```

Connector-backed entries are supported only by the OpenAI `responses` provider.
They require a mapping key (or `name` in the compatible list form) and
`access_token`; omit `target`, `transport`, `headers`, and `auth`.

For provider-managed servers, use exact tool names in `tools.<server_name>`.
Wildcard tool filters, prompt filters, and resource filters are not supported.

`target` is a pure target string. Do not embed fast-agent CLI flags, such as
`--auth` or `--oauth`, in card targets. Use `headers`/`auth` fields instead.

When both target-derived values and explicit fields are present, explicit fields
(`headers`, `auth`, etc.) win.

If an inferred/provided name collides with another server using different
settings, startup fails with a collision error. Prefer explicit `name` values
for stability.

## AgentCards and ToolCards

In fast-agent, **ToolCards are AgentCards**. There is no separate schema.

The distinction is how the card is loaded:

- `--agent-cards` or `--card` loads cards as runnable agents.
- `--card-tool` loads cards, then attaches those loaded agents as tools to a
  parent agent.

Use `--agent-cards` for agents you want to run directly.

Use `--card-tool` for agents you primarily want another agent to invoke as a
tool. If a card should not appear in normal interactive agent lists, set:

```yaml
tool_only: true
```

When a card is attached as a tool, fast-agent uses the card's `description` as
the tool description the parent agent sees.

```md
---
name: reviewer
description: Review a proposed plan or patch for risks, missed tests, and unnecessary complexity.
tool_only: true
model: sonnet
---

You are a concise software reviewer. Focus on correctness, maintainability, and
test coverage.
```

## Default directories

By default, `fast-agent go` discovers cards from your home:

- `<home>/agent-cards/`
- `<home>/tool-cards/`

`<home>` defaults to `.fast-agent/` in your current workspace.
Use `--home` to point to a different home.
Use `--no-home` to disable implicit default directory discovery entirely.

## CLI examples

```bash
# Load runnable agents
fast-agent go --agent-cards ./agents

# Load cards as tools attached to the default/selected agent
fast-agent go --card-tool ./tool-cards

# Mix both
fast-agent go --agent-cards ./agents --card-tool ./tool-cards

# Ephemeral/no_home run: only explicit paths are loaded
fast-agent go --no-home --agent-cards ./agents --card-tool ./tool-cards

# Target a specific loaded agent
fast-agent go --agent-cards ./agents --agent researcher
```

## Notes on `--agent`

- `--agent` picks the target for `--message`, `--prompt-file`, and initial
  interactive mode.
- `--agent` can also target explicitly loaded tool-only agents when needed for
  testing.
