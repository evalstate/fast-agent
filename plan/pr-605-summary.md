## PR #605 Summary for Maintainer

**Key changes**
- `/card --tool` now reuses Agents-as-Tools by appending loaded agent(s) to the current agent’s `agents` list and hot-swapping (no separate tool-injection path).
- Restores `--card-tool` and `.fast-agent/tool-cards/` as normal AgentCard sources loaded after `--card` / `.fast-agent/agent-cards/`, then attached to the default agent via Agents-as-Tools.
- Adds `/agent --tool` (attach agent as tool) and `/agent --dump` (print selected/current AgentCard).
- Introduces `history_source` + `history_merge_target` (validation when `messages` is required); removes `history_mode` per RFC.
- Improves `--watch`/REPL quality: incremental reload (mtime+size), safe parse for empty/partial writes, tool-file watch, removal pruning, refreshed agent list, and model name trimming in UI.

**Implemented from RFC “Open Issues”**
- ✅ Incremental `--watch` reload (mtime+size, per-card)
- ✅ Safe parse (empty/partial writes -> warning + retry)
- ✅ Tool-file change reload for referencing cards
- ✅ Removal pruning (agents + attached agent-tools)
- ✅ Partial instance refresh (shared/request)
- ✅ UX: “AgentCards reloaded” line + refreshed agent list

**Tests added / updated**
- `tests/unit/fast_agent/core/test_agent_card_watch.py`
- `tests/unit/fast_agent/core/test_agents_as_tools_function_tools.py`
- `tests/unit/fast_agent/ui/test_interactive_prompt_agent_commands.py`
- `tests/unit/fast_agent/ui/test_interactive_prompt_refresh.py`
- `tests/unit/fast_agent/commands/test_go_command.py`
- `tests/unit/fast_agent/commands/test_serve_command.py`
- `tests/unit/fast_agent/commands/test_acp_command.py`

**Status**
- AgentCard RFC implementation is mostly complete (`plan/agent-card-rfc.md`).
- Remaining: finish `history_source` / `history_merge_target` execution paths (excluding `cumulative` + file merge), remove any residual `history_mode`.
- Manual REPL testing covered add/remove agents, `/agents`, `@switch`, tool attach/detach; most issues were converted into tests.

**Note**
- Tracking a pytest warning: “Task was destroyed but it is pending!” tied to AsyncEventBus shutdown (details in `plan/fix-async-event-bus.md`).
