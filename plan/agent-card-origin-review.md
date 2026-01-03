# Code Review: origin/feat/agent-card (pre-pull)

Branch status: on `feat/agent-card` with local uncommitted changes. The branch is behind `origin/feat/agent-card` by 4 commits; review is based on `git diff HEAD..origin/feat/agent-card` without pulling.

## Findings

- `src/fast_agent/agents/tool_agent.py:446` — `ToolExecutionHandler.on_tool_start` is called with `tool_use_id=None`. For ACP progress/streaming, this prevents correlating tool calls to the LLM `tool_use_id` (potential duplicate or unlinked tool notifications). If intentional, document it; otherwise plumb `tool_use_id` through `run_tools` → `call_tool`.
- `src/fast_agent/core/fastagent.py:446` and `src/fast_agent/core/fastagent.py:595` — CLI `--model` override is no longer applied to AgentCard-loaded agents. This changes precedence vs the RFC (CLI → AgentCard → config). If the intent is “AgentCard model wins,” update docs/spec; otherwise treat as regression.
- `src/fast_agent/agents/tool_agent.py:90` — `_clone_constructor_kwargs` passes `FastMCPTool` instances directly into clones. If any tool closures capture parent state (for example, `add_agent_tool`), clones will still dispatch to the original child template, not a per-clone graph. This may be fine, but it weakens the “detached per-call clones” story for nested agent-tools.

## Missing tests

- No test coverage for the new ToolExecutionHandler progress flow (`on_tool_start` / `on_tool_progress` / `on_tool_complete`) or for tool propagation into detached clones. A focused unit test would help guard ACP tool-progress behavior and clone tool availability.

## Questions / assumptions

- Is the CLI `--model` override supposed to lose precedence over AgentCard models now? If yes, update the RFC and CLI docs; if no, treat as a regression.
- Should local tool execution be correlated to LLM `tool_use_id` in ACP progress? If yes, thread it through `run_tools`/`call_tool`.

If you want me to pull and merge the origin changes, tell me whether to commit or stash the local modifications first.
