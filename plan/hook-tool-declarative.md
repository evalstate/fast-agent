# Hook Tool Declarative Spec (Experimental)

This spec adds a declarative API so any agent can mix MCP servers/tools, local Python
function tools, child agents-as-tools, and tool hooks in one place. The hook design is
nginx-style middleware: it can run before, instead, or after the original tool, and can
mutate args/results or skip execution.

Status: experimental (intended for maintainer review).

## Goals

- Declarative tools + hooks on `@fast.agent` (no custom ToolAgent subclass required).
- Mix in one declaration: `servers`, MCP `tools` filters, `function_tools`, `agents`
  (agents-as-tools), and `tool_hooks`.
- Hooks apply uniformly to all tool types (MCP, function, agent, built-ins).
- Hooks can inspect agent/tool identity, mutate args/results, or short-circuit execution.

## Declarative API (proposed, examples/workflows/agents_as_tools_extended)

`agents_as_tools_extended.py`
```python
@fast.agent(
    name="PMO-orchestrator",
    servers=["time"],
    tools={"time": ["get_time"]},  # MCP filters
    function_tools=[add_one],  # local Python tools
    agents=["NY-Project-Manager", "London-Project-Manager"],  # agents-as-tools
    tool_hooks=[audit_hook],  # applies to all tools
    instruction="Get project updates from the New York and London project managers and include the current time. Ask NY-Project-Manager three times about different projects: Anthropic, evalstate/fast-agent, and OpenAI, and London-Project-Manager for economics review. Return a brief, concise combined summary with clear city/time/topic labels.",
)
```

### AgentCard example (proposed, examples/workflows-md/agents_as_tools_extended)

`PMO-orchestrator.md`
```md
---
type: agent
name: PMO-orchestrator
default: true
servers:
  - time
tools:
  time: [get_time]
function_tools:
  - tools.py:add_one
agents:
  - NY-Project-Manager
  - London-Project-Manager
tool_hooks:
  - hooks.py:audit_hook
max_parallel: 128
child_timeout_sec: 120
max_display_instances: 20
---
Get project updates from the New York and London project managers and include the current time. Ask NY-Project-Manager three times about different projects: Anthropic, evalstate/fast-agent, and OpenAI, and London-Project-Manager for economics review. Return a brief, concise combined summary with clear city/time/topic labels.
```

Notes:
- `tools={...}` remains MCP tool filtering only.
- `function_tools=[...]` accepts callables or `"module.py:function"` strings.
- When loaded from an AgentCard, relative paths are resolved against the card directory.
- AgentCard supports **string specs only**; callables are only valid in Python decorators.
- `tool_hooks=[...]` is a new middleware layer around every tool call.

## Function tool loading (implemented)

The current implementation in `src/fast_agent/tools/function_tool_loader.py` loads
function tools as follows:

- `callable` entries are wrapped via `FastMCPTool.from_function`.
- String specs must be `module.py:function_name` and are loaded dynamically from file.
- Relative module paths are resolved against a `base_path` (AgentCard directory); if
  no base path is provided, `cwd` is used.
- Errors raise (invalid format, missing file, missing attribute, non-callable). The loader
  logs the failure and re-raises to avoid silent misconfiguration.
- The module name is generated uniquely (`_function_tool_<stem>_<id>`) to avoid collisions.

Implication for hooks: hook loaders should mirror this behavior to keep string specs
consistent across function_tools and tool_hooks. Hook loading is based on the same
Function tool loading implementation.

## Hook signature and behavior (nginx-style)

### Core types

```python
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Literal
from mcp.types import CallToolResult

ToolCallArgs = dict[str, Any] | None
ToolCallFn = Callable[[ToolCallArgs], Awaitable[CallToolResult]]

@dataclass(frozen=True)
class ToolHookContext:
    agent_name: str
    server_name: str | None
    tool_name: str
    tool_source: Literal["mcp", "function", "agent", "runtime"]
    tool_use_id: str | None
    correlation_id: str | None
    original_tool_func: ToolCallFn

ToolHookFn = Callable[[ToolHookContext, ToolCallArgs, ToolCallFn], Awaitable[CallToolResult]]
```

### Semantics

Each hook receives:
- `agent_name`, `server_name`, `tool_name` (identity)
- `original_tool_func` (the underlying tool callable)
- `args` (mutable tool arguments)
- `call_next` (the next hook in chain; last call invokes `original_tool_func`)

This supports **before / instead / after** behavior:

```python
async def safety_guard(ctx, args, call_next):
    # before: mutate args or enforce limits
    args = clamp_args(args)

    # instead: decide not to call the tool
    if ctx.tool_name == "shell.execute":
        return CallToolResult(isError=True, content=[text_content("blocked")])

    # call underlying tool (or next hook)
    result = await call_next(args)

    # after: mutate or log result
    return redact_result(result)
```

Multiple hooks compose like middleware; order is the order declared.

Error handling:
- If a hook raises, the exception bubbles; the tool loop records an error result.

## Tool identity mapping

`tool_hooks` must apply uniformly. Proposed mapping:

- MCP tools: `tool_source="mcp"`, `server_name=<mcp server>`, `tool_name=<namespaced tool>`
- Local function tools: `tool_source="function"`, `server_name=None` (or "local")
- Agents-as-tools: `tool_source="agent"`, `server_name="agent"`, `tool_name=agent__Child`
- Built-in runtimes (shell, filesystem, human-input, skill reader):
  `tool_source="runtime"`, `server_name="runtime"` (or specific runtime name)

Hooks can branch on `tool_source`, `server_name`, and `tool_name`.

## Current state (AgentCard branch)

- AgentCard loader + CLI: `--card` / `--agent-cards`, supports URL cards.
- `function_tools` supported in `@fast.agent`, `@fast.custom`, and AgentCards.
- `function_tool_loader.py` supports callable or `module.py:function` specs.
- `tool_hooks` implementation and tests were removed; hooks are currently missing.

## Implementation plan (compact)

1) Data model + parsing
- Add `ToolHookConfig` type alias in `src/fast_agent/agents/agent_types.py`.
- Extend AgentConfig with `tool_hooks: ToolHooksConfig | None`.
- Update `@fast.agent` and `@fast.custom` signatures to accept `tool_hooks`.
- Update `agent_card_loader.py` to parse `tool_hooks` from YAML/MD.
  - AgentCard supports **string specs only**; callables are only allowed in decorators.

2) Loader reuse
- Prefer reusing `function_tool_loader.py` with a generic callable loader.
- Load hook functions from `module.py:function` specs.
- Validate callability; raise with clear error on mismatch.

3) Runtime wiring
- Restore `ToolHookContext` + `run_tool_with_hooks` (module `agents/tool_hooks.py`).
- Reapply hook execution around tool calls:
  - `ToolAgent.call_tool` wraps local function tools.
  - `McpAgent.call_tool` wraps MCP tools, runtime tools, and agents-as-tools.
- Preserve `tool_use_id` and `correlation_id` in context.
- Keep ToolRunnerHooks unchanged; tool_hooks is independent middleware.

4) Tests
- Unit: hook chain order, before/instead/after behavior, skip execution.
- Integration: restore declarative hooks test and add AgentCard hook test.

5) Examples + docs
- Extend `examples/workflows/agents_as_tools_extended.py` with hook usage.
- Update `examples/workflows-md/agents_as_tools_extended` AgentCards to match.
- Review `examples/workflows-md/hf-api-agent` before adding hook samples.
- Update this spec and CLI README.

## Open question
- Do we want a strict signature check for hooks at load time?

## Examples (planned)

### 1) Mixed MCP + function tools + agents + hooks

```python
def add_one(x: int) -> int:
    return x + 1


async def audit_hook(ctx, args, call_next):
    # before: enforce limits
    if ctx.tool_name.endswith("add_one"):
        args = dict(args or {})
        args["x"] = min(int(args.get("x", 0)), 10)

    # instead: block unsafe tools
    if ctx.tool_source == "runtime" and ctx.tool_name == "shell.execute":
        return CallToolResult(isError=True, content=[text_content("blocked")])

    # call original tool
    result = await call_next(args)

    # after: log or modify result
    result.content.append(text_content("[audit]"))
    return result


@fast.agent(
    name="PMO-orchestrator",
    instruction="Get project updates from the New York and London project managers and include the current time. Ask NY-Project-Manager three times about different projects: Anthropic, evalstate/fast-agent, and OpenAI, and London-Project-Manager for economics review. Return a brief, concise combined summary with clear city/time/topic labels.",
    agents=["NY-Project-Manager", "London-Project-Manager"],
    servers=["time"],
    tools={"time": ["get_time"]},
    function_tools=[add_one],
    tool_hooks=[audit_hook],
    default=True,
)
async def main() -> None:
    async with fast.run() as agent:
        await agent("Run PMO report and add 1 to 3")
```

## Related examples (existing)

- Function tools: `examples/new-api/simple_llm.py` and `examples/tool-use-agent/agent.py`
- Tool runner hooks: `examples/tool-runner-hooks/tool_runner_hooks.py`
- Agents-as-tools: `examples/workflows/agents_as_tools_extended.py`
