# Hook Tool Declarative v2 (post AgentCard + function_tools)

Status: draft plan.

## Comparison Summary

### feat/hook-tool-declarative (older branch)
- Implemented `ToolHookContext` + `run_tool_with_hooks`.
- Added `tool_hooks` to `ToolAgent` and `McpAgent`.
- Hooks wrap **all** tool sources (MCP, local function tools, runtime tools, agents-as-tools).
- Added declarative example `examples/tool-hooks-declarative/mixed_tools_and_hooks.py`.
- Added integration test `tests/integration/api/test_declarative_tools_and_hooks.py`.

### origin/feat/agent-card (current target)
- Added AgentCard loader + CLI: `--card` / `--agent-cards`, supports URL cards.
- Added `function_tools` to `AgentConfig` and loader for `module.py:function`.
- Added `function_tool_loader.py` and integration tests for function tools.
- **Removed** tool hook implementation and tests; `tool_hooks` module no longer exists.
- Example `mixed_tools_and_hooks.py` and declarative hook tests were removed.

**Result:** function tools are supported; tool hooks are not.

## Goal
Reintroduce declarative `tool_hooks` in a way that fits the AgentCard + function_tools
pipeline in `origin/feat/agent-card`, without disrupting existing ToolRunnerHooks.

## Plan

### 1) Data model + parsing
- Add `ToolHookConfig` type alias in `src/fast_agent/agents/agent_types.py`.
  - Same pattern as `FunctionToolsConfig`: list of callables or string specs.
- Extend AgentConfig with `tool_hooks: ToolHooksConfig | None`.
- Update `@fast.agent` and `@fast.custom` signatures to accept `tool_hooks`.
  - For `@fast.custom`, only apply when the class subclasses ToolAgent or implements
    a ToolHookCapable interface; otherwise ignore (explicit warning).
- Update `agent_card_loader.py` to parse `tool_hooks` from YAML/MD.
  - Accept string or list of strings, same as `function_tools`.

### 2) Hook loader
- Add `tool_hook_loader.py` or extend `function_tool_loader.py`:
  - Load hook functions from `module.py:function` specs.
  - Validate callability; raise with clear error on mismatch.

### 3) Runtime wiring
- Restore `ToolHookContext` + `run_tool_with_hooks` (module `agents/tool_hooks.py`).
- Reapply hook execution around tool calls:
  - `ToolAgent.call_tool` wraps local function tools.
  - `McpAgent.call_tool` wraps:
    - MCP tools via aggregator
    - runtime tools (shell, filesystem, human_input, skills)
    - agents-as-tools
- Preserve `tool_use_id` and `correlation_id` in context.
- Keep ToolRunnerHooks unchanged; tool_hooks should be independent middleware.

### 4) Tool identity mapping (same as original spec)
- MCP tools: `tool_source="mcp"`, `server_name=<mcp server>`, `tool_name=<namespaced>`.
- Function tools: `tool_source="function"`, `server_name=None`.
- Agents-as-tools: `tool_source="agent"`, `server_name="agent"`.
- Runtime tools: `tool_source="runtime"`, `server_name="shell"|"filesystem"|"skills"|...`.

### 5) Tests
- Unit: hook chain order, before/instead/after behavior, skip execution.
- Integration:
  - Restore `tests/integration/api/test_declarative_tools_and_hooks.py`.
  - Add AgentCard test: `tool_hooks` + `function_tools` in one card.

### 6) Examples + docs
- Restore `examples/tool-hooks-declarative/mixed_tools_and_hooks.py`.
- Add an AgentCard example mixing MCP + function_tools + hooks.
- Update `plan/hook-tool-declarative.md` to match new loader behavior.

## Open Questions
- Should hooks be allowed as import specs only (`module.py:function`), or also
  inline callables in `@fast.agent`? (Recommend: both; same as function_tools.)
- Do we want a strict signature check for hooks at load time?
- Should hook errors fail the tool call or be wrapped into error results?

## Next Steps
1) Reintroduce `tool_hooks` runtime wiring (ToolAgent + McpAgent).
2) Add loader + AgentCard parsing for `tool_hooks`.
3) Restore tests and examples.
4) Document in the spec and CLI README.
