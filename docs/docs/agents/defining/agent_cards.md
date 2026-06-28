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

The Harness API also loads cards from the active environment's `agent-cards/`
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
  - target: "https://demo.hf.space"
    headers:
      Authorization: "Bearer ${DEMO_TOKEN}"
    auth:
      oauth: true
  - target: "@modelcontextprotocol/server-everything"
    name: "everything"
```

- `target` (required): URL, `@pkg`, `npx ...`, `uvx ...`, or stdio command.
- `name` (optional): explicit server alias; if omitted, fast-agent infers one.
- `headers` (optional): structured HTTP headers.
- `auth` (optional): structured auth settings, for example `oauth: true`.

For provider-managed remote MCP, use:

```yaml
mcp_connect:
  - target: "https://huggingface.co/mcp"
    name: "huggingface"
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
  - name: dropbox
    management: provider
    connector_id: connector_dropbox
    access_token: "${DROPBOX_OAUTH_ACCESS_TOKEN}"
    description: "Dropbox connector"
    defer_loading: true
```

Connector-backed entries are supported only by the OpenAI `responses` provider.
They require `name` and `access_token`; omit `target`, `transport`, `headers`,
and `auth`.

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

By default, `fast-agent go` discovers cards from your environment directory:

- `<env>/agent-cards/`
- `<env>/tool-cards/`

`<env>` defaults to `.fast-agent/` in your current project root.
Use `--env` to point to a different environment directory.
Use `--noenv` to disable implicit default directory discovery entirely.

## CLI examples

```bash
# Load runnable agents
fast-agent go --agent-cards ./agents

# Load cards as tools attached to the default/selected agent
fast-agent go --card-tool ./tool-cards

# Mix both
fast-agent go --agent-cards ./agents --card-tool ./tool-cards

# Ephemeral/noenv run: only explicit paths are loaded
fast-agent go --noenv --agent-cards ./agents --card-tool ./tool-cards

# Target a specific loaded agent
fast-agent go --agent-cards ./agents --agent researcher
```

## Notes on `--agent`

- `--agent` picks the target for `--message`, `--prompt-file`, and initial
  interactive mode.
- `--agent` can also target explicitly loaded tool-only agents when needed for
  testing.
