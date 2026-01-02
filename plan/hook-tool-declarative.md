# Hook Tool Declarative Spec (Experimental)

This spec adds a declarative API so any agent can mix MCP servers/tools, local Python function tools, child agents-as-tools, and tool hooks in one place. The hook design is nginx-style middleware: it can run before, instead, or after the original tool, and can mutate args/results or skip execution.

Status: experimental (intended for maintainer review).

## Goals

- Declarative tools + hooks on `@fast.agent` (no custom ToolAgent subclass required).
- Mix in one declaration: `servers`, MCP `tools` filters, `function_tools`, `agents` (agents-as-tools), and `tool_hooks`.
- Hooks apply uniformly to all tool types (MCP, function, agent, built-ins).
- Hooks can inspect agent/tool identity, mutate args/results, or short-circuit execution.

## Declarative API (proposed)

```python
@fast.agent(
    name="PMO-orchestrator",
    servers=["time", "github"],
    tools={"time": ["get_time"], "github": ["search_*"]},  # MCP filters
    function_tools=[local_summarize, "tools.py:local_redact"],  # local Python tools
    agents=["NY-Project-Manager", "London-Project-Manager"], # agents-as-tools
    tool_hooks=[audit_hook, safety_guard],                    # applies to all tools
)
async def main():
    ...
```

Notes:
- `tools={...}` remains MCP tool filtering only.
- `function_tools=[...]` accepts callables or `"module.py:function"` strings.
- When loaded from an AgentCard, relative paths are resolved against the card directory.
- `tool_hooks=[...]` is a new middleware layer around every tool call.

## Function tool loading (implemented)

The current implementation in `src/fast_agent/tools/function_tool_loader.py` loads function
tools as follows:

- `callable` entries are wrapped via `FastMCPTool.from_function`.
- String specs must be `module.py:function_name` and are loaded dynamically from file.
- Relative module paths are resolved against a `base_path` (AgentCard directory); if
  no base path is provided, `cwd` is used.
- Errors raise (invalid format, missing file, missing attribute, non-callable). The loader
  logs the failure and re-raises to avoid silent misconfiguration.
- The module name is generated uniquely (`_function_tool_<stem>_<id>`) to avoid collisions.

Implication for hooks: hook loaders should mirror this behavior to keep string specs
consistent across function_tools and tool_hooks.

## Hook signature and behavior (nginx-style)

### Core types

```python
from dataclasses import dataclass
from typing import Awaitable, Callable, Literal
from mcp.types import CallToolResult

ToolCallArgs = dict
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

## Tool identity mapping

`tool_hooks` must apply uniformly. Proposed mapping:

- MCP tools: `tool_source="mcp"`, `server_name=<mcp server>`, `tool_name=<namespaced tool>`
- Local function tools: `tool_source="function"`, `server_name=None` (or "local")
- Agents-as-tools: `tool_source="agent"`, `server_name="agent"`, `tool_name=agent__Child`
- Built-in runtimes (shell, filesystem, human-input, skill reader): `tool_source="runtime"`, `server_name="runtime"` (or specific runtime name)

Hooks can branch on `tool_source`, `server_name`, and `tool_name`.

## What changes in `function_tools` vs previous

Previous:
- Local Python tools were attached via ToolAgent or custom subclasses.
- Standard `@fast.agent` did not accept local function tools.

Current (AgentCard branch):
- `function_tools=[...]` is supported by `@fast.agent`, `@fast.custom`, and AgentCards.
- Tools can be callables or `module.py:function` strings.
- Local tools are registered alongside MCP tools automatically.
- Allows mixing MCP servers + local tools + agents-as-tools in a single agent.

## What changes in hooks vs current

Current:
- ToolRunnerHooks exist but are only accessible via ToolAgent subclassing.
- Hooks operate at the tool loop level (before/after tool call messages), not the tool execution function itself.

Proposed:
- New declarative `tool_hooks` applied to any agent type.
- Hooks wrap actual tool execution and apply to all tool sources.
- Hooks can skip execution, mutate args/results, or call the original tool directly.

Why:
- Enables nginx-style middleware around tool calls (auth, redaction, retries, telemetry, policy).
- Consistent control plane for MCP tools, local tools, and agent tools.

## Implementation sketch (high level)

1) Decorator/API
- `function_tools` already implemented for `@fast.agent` / `@fast.custom` / AgentCard.
- Add `tool_hooks` alongside `function_tools` and pass through factory/constructor.
- For `@fast.custom`, `function_tools` / `tool_hooks` only apply if the custom class
  implements `ToolHookCapable` or subclasses `ToolAgent`; otherwise they are ignored.

2) Execution wiring
- Build a `ToolHookChain` around each tool call in the tool runner loop.
- Construct `ToolHookContext` with agent/tool identity and `original_tool_func`.
- The last `call_next` invokes the real tool implementation.

3) Apply to all tool types
- MCP tool calls wrap aggregator execution.
- Local function tools wrap FastMCPTool execution.
- Agents-as-tools wrap child agent call (detached instance).
- Built-ins (shell/filesystem/human-input/skills) go through the same hook chain.

4) Experimental note
- Keep docs labeled experimental; no config gate required.

## Examples (new, declarative)

Implemented example file:
- `examples/tool-hooks-declarative/mixed_tools_and_hooks.py`

### 1) Mixed MCP + function tools + agents + hooks

```python
@fast.agent(
    name="coordinator",
    servers=["time"],
    tools={"time": ["get_time"]},
    function_tools=[summarize_text, "tools.py:redact_text"],
    agents=["NY-Project-Manager", "London-Project-Manager"],
    tool_hooks=[audit_hook],
)
async def main():
    ...
```

### 2) Policy enforcement and redaction

```python
async def audit_hook(ctx, args, call_next):
    log_call(ctx.agent_name, ctx.server_name, ctx.tool_name)
    result = await call_next(args)
    return redact_result(result)
```

### 3) Timeout or bypass

```python
async def timeout_hook(ctx, args, call_next):
    if ctx.tool_source == "agent":
        return await asyncio.wait_for(call_next(args), timeout=5)
    return await call_next(args)
```

## Related examples (existing)

- Function tools: `examples/new-api/simple_llm.py` and `examples/tool-use-agent/agent.py`
- Tool runner hooks: `examples/tool-runner-hooks/tool_runner_hooks.py`
- Agents-as-tools: `examples/workflows/agents_as_tools_simple.py` and `examples/workflows/agents_as_tools_extended.py`

These should be referenced from the new declarative examples to show the evolution from current patterns.

## Still Missing in This Spec
- Explicit test plan / coverage checklist for tool hook middleware behavior.
- Detailed failure-mode guidance (e.g., hook exceptions, timeouts, cancellation).
