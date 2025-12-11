# Agents-as-Tools — Fix Plan for Current Implementation

## 1. Scope

This document describes how to evolve and harden the current `AgentsAsToolsAgent` implementation in this repo:

- File: `src/fast_agent/agents/workflow/agents_as_tools_agent.py`
- Wiring:
  - `direct_decorators.agent(..., agents=[...])`
  - `direct_factory.create_agents_by_type` (BASIC agents with `child_agents`)
- Supporting components:
  - `ToolAgent`, `LlmAgent`
  - `McpAgent`, `MCPAggregator`
  - UI: `RichProgressDisplay`, `ConsoleDisplay`, `history_display`, `usage_display`
  - Stats: `UsageAccumulator`

Goal: keep this implementation **experimental but coherent**, good enough for real workflows and for an upstream-quality PR later.

---

## 2. Recovered Intended Design

From the module docstring and issue #458:

- **Concept**
  - Parent is a normal tool-calling LLM.
  - Each child agent is exposed as a tool: `agent__{child_name}`.
  - Parent delegates; it doesn't orchestrate explicitly.

- **Tool interface**
  - `list_tools()` → one tool per child, permissive schema:
    - `{ text?: string, json?: object, ... }`
  - `call_tool()`:
    - Routes tool name → child agent.
    - Normalizes arguments to a single `Prompt.user(text)`.
    - Executes `child.generate([...])` and returns `CallToolResult`.

- **Parallelism**
  - Parent LLM may emit multiple tool calls in one turn.
  - `run_tools()` should:
    - Validate tools against `list_tools()`.
    - Run all valid calls via `asyncio.gather`.
    - Associate each physical tool call with a **virtual instance** index: `[1]`, `[2]`.

- **Progress panel semantics** (Rich progress, left side)
  - Before fan-out: one line per *agent* (parent and children).
  - During fan-out:
    - Parent line shows `Ready` (waiting on children).
    - Each child instance shows its own line, with instance-index-suffixed name: `OriginalName[1]`, `OriginalName[2]`.
    - Lines disappear as soon as each instance finishes.
  - After fan-in:
    - Only base agent lines remain; original names restored.

- **Chat/log semantics**
  - Parent chat should show **tool request/result panels** for each instance.
  - Child chat should **not** stream to the panel when invoked as a tool.
  - Child **tool usage** (MCP tools, shell, etc.) should still be visible.

- **MCP initialization semantics**
  - Children are real agents (`McpAgent` or similar) with MCP clients & aggregators.
  - Multiple instances of the same child **share** one MCP aggregator.
  - Parent itself does **not** talk to MCP directly; it only calls children.

- **Stats semantics**
  - Token/tool stats are tracked per *agent* via `UsageAccumulator`.
  - Instances are **transient**; they may be visible in progress/chat but stats roll up per agent.

---

## 3. Current Implementation Review

### 3.1. What's already good

- **Tool naming & discovery**
  - `_make_tool_name(child_name)` → `agent__{child_name}`.
  - `list_tools()` returns Tool schemas with the minimal `{ text, json }` interface.

- **Routing & argument handling**
  - `call_tool()` resolves both `agent__Child` and bare `Child`.
  - Arguments → `text` precedence, then `json`, then `full args` JSON.
  - Child is called via `Prompt.user(...)` + `child.generate([...])`.

- **Error surfacing**
  - If child writes to the `FAST_AGENT_ERROR_CHANNEL`, those blocks are appended to the tool result contents and `CallToolResult.isError` is set.

- **Parallel fan-out**
  - `run_tools()` builds `call_descriptors` and `descriptor_by_id`.
  - Uses `asyncio.gather(..., return_exceptions=True)` to execute all calls concurrently.

- **Instance naming for UI**
  - For `pending_count > 1`, collects `original_names[tool_name] = child._name`.
  - In `call_with_instance_name()`:
    - Computes `instance_name = f"{original}[{instance}]"`.
    - Mutates `child._name` and `child._aggregator.agent_name`.
    - Emits a synthetic `ProgressEvent(CHATTING, target=instance_name, agent_name=instance_name)` to create a line in the progress panel.
  - On completion, hides that line by flipping `task.visible = False` in `RichProgressDisplay`.

- **Child display suppression**
  - `call_tool()` lazily creates:
    - `_display_suppression_count: { id(child) -> int }`.
    - `_original_display_configs: { id(child) -> ConsoleDisplayConfig }`.
  - On first use of a given child, makes a copy of `child.display.config`, sets:
    - `logger.show_chat = False`
    - `logger.show_tools = True`
  - Ensures **children don't spam chat**, but still show their own MCP tool usage.

- **Top/bottom panels**
  - `_show_parallel_tool_calls()` and `_show_parallel_tool_results()` correctly label tools as `tool_name[instance]` in chat panels and bottom status items.

Overall, the core mechanics of Agents-as-Tools are present and coherent.

### 3.2. Gaps and fragilities

1. [x] **Display config restoration logic is incomplete**

   - In `call_tool()` we:
     - Always increment `_display_suppression_count[child_id]`.
     - In `finally` we **only decrement** the counter, do **not** restore config.
   - In `run_tools()` we restore config **only if `pending_count > 1`**:
     - For each `child` in `original_names`:
       - Delete `_display_suppression_count[child_id]`.
       - Restore `display.config` from `_original_display_configs`.
   - Problems:
     - For a **single tool call** (the most common case!), `pending_count == 1`, so `original_names` is empty and **display configs are never restored**.
     - Even for `pending_count > 1`, restoration is decoupled from `_display_suppression_count[child_id]` (no 0→1 / 1→0 semantics).

   **Effect:** once a child is ever used as a tool, its chat may remain permanently suppressed for all subsequent uses, including direct runs, which is surprising.

2. [x] **Instance naming races on shared child objects**

   - Multiple tool calls to the **same child agent** share a single `child` object and a single `child._aggregator`.
   - `call_with_instance_name()` mutates `child._name` and `child._aggregator.agent_name` in each task.
   - Under concurrency, whichever task last mutates these fields wins; log lines from the child and from its aggregator may be attributed to the last instance, not this instance.

   **Effect:** progress rows are mostly correct (because we also emit explicit `ProgressEvent`s), but logs and transport stats that come from `MCPAggregator` may mix instance names.

3. [x] **Direct reliance on private internals of `RichProgressDisplay`**

   - `call_with_instance_name()` accesses:
     - `outer_progress_display._taskmap`
     - `outer_progress_display._progress.tasks`
     - and flips `task.visible = False`.

   **Risk:** this is brittle against internal refactors of the progress UI and difficult to test in isolation.

4. [x] **`MessageType` import is unused**

   - `from fast_agent.ui.message_primitives import MessageType` is imported but not used.
   - Indicates some UI scenarios were planned (e.g. structured tool headers) and not implemented.

5. [x] **Stats are per-agent only, not per-instance**

   - `UsageAccumulator` is owned by the LLM (via `LlmDecorator.usage_accumulator`).
   - Usage is aggregated per **agent** (e.g. `PM-1-DayStatusSummarizer`), not per `[i]` instance.
   - This matches the general fast-agent philosophy but does **not** match the stronger requirement separate rows in the stats panel per instance.

   **Current behavior is acceptable**, but the instance-per-row requirement should be documented as **out of scope** for the first implementation.

6. [ ] **Tool availability check and naming**

   - `run_tools()` validates tool names against `list_tools()` of `AgentsAsToolsAgent` (agent-tools only).
   - There is no support to **merge MCP tools and agent-tools** in `list_tools()`.

   **Status:** this matches a conservative interpretation of issue #458, but the design doc leaves the door open to unifying MCP tools and agent-tools; that needs an explicit decision.

---

## 4. Design Decisions to Lock In (for this branch)

Before making changes, clarify the intended semantics for this repo:

1. **Child chat visibility**
   - When a child agent is used as a tool via `AgentsAsToolsAgent`, its chat is **never** shown.
   - When a child is run directly (by the user), its chat **is** shown.

2. **Instance stats vs agent stats**
   - For this implementation, stats remain **per agent**, not per instance.
   - Instance-level visibility is provided by:
     - Progress panel (per-instance lines).
     - Chat log (tool headers `tool_name[i]`).

3. **MCP reuse model**
   - Child MCP aggregators are **shared** between all instances and all parents.
   - No per-instance MCP clients.

4. **Tool namespace composition**
   - For now, `AgentsAsToolsAgent.list_tools()` returns **only agent-tools**.
   - MCP tools, if any, must be accessed via separate agents (not through this orchestrator).

These decisions simplify the fix plan and keep surface area small.

---

## 5. Step-by-Step Fix Plan

### 5.1. Fix display suppression and restoration

**Goal:** implement correct reference counting per-child and always restore display config after the last instance completes, regardless of `pending_count`.

**Steps:**

1. [x] **Introduce explicit helpers on `AgentsAsToolsAgent`**

   - Private methods:
     - `_ensure_display_maps_initialized()`
     - `_suppress_child_display(child)`
     - `_release_child_display(child)`

   - Semantics:
     - `_suppress_child_display(child)`:
       - If `child_id` not in `_display_suppression_count`:
         - Snapshot `child.display.config` into `_original_display_configs[child_id]`.
         - Install a modified config with `show_chat=False, show_tools=True`.
         - Initialize counter to `0`.
       - Increment counter.
     - `_release_child_display(child)`:
       - Decrement counter.
       - If counter reaches `0`:
         - Restore original config from `_original_display_configs`.
         - Delete both entries for this `child_id`.

2. [x] **Apply helpers in `call_tool()`**

   - Replace direct manipulation with:
     - `_suppress_child_display(child)` before `await child.generate(...)`.
     - `_release_child_display(child)` in `finally`.

3. [x] **Remove display restoration from `run_tools()`**

   - The `_display_suppression_count` & `_original_display_configs` clean-up should be **entirely local** to `call_tool()`; `run_tools()` should not know about it.
   - This also makes `call_tool()` correct if it's ever used outside of `run_tools()`.

**Outcome:** display configs are always restored after the last parallel/sequential instance finishes, independent of how many tools or which code path called them.

---

### 5.2. Stabilize instance naming and progress UI

**Goal:** keep existing UX (progress lines + names `[i]`) but reduce reliance on private internals.

1. **Add a small public API to `RichProgressDisplay`**

   - In `rich_progress.py`:
     - Add methods:
       - `def hide_task(self, task_name: str) -> None:`
         - Look up `task_id` via `_taskmap.get(task_name)`.
         - If found, set `task.visible = False`.
       - Optionally `def ensure_task(self, event: ProgressEvent) -> TaskID:` to encapsulate `add_task` + update logic.

   - Refactor `update()` to use `ensure_task()` internally.

2. [x] **Use the public API in `AgentsAsToolsAgent`**

   - Replace direct access to `_taskmap` and `_progress.tasks` with:
     - `outer_progress_display.hide_task(instance_name)`.

3. **Document expected lifetime**

   - Comment in `AgentsAsToolsAgent`:
     - Instance lines are **ephemeral**; they are hidden immediately when each task completes but progress data continues to exist for the duration of the run.

**Outcome:** same UI behavior, less fragile coupling to UI internals.

---

### 5.3. Reduce naming races (best-effort for experimental phase)

Completely eliminating races around `child._name` and `child._aggregator.agent_name` would require:

- Either a per-instance `MCPAggregator`, or
- Making `MCPAggregator` fully stateless in terms of `agent_name`, or
- Augmenting all tool/progress logs with an explicit correlation/instance id.

That is a larger refactor than we want for the current experimental implementation. Instead, we can apply a **minimal mitigation**:

1. [x] **Minimize mutation window**

   - In `call_with_instance_name()`:
     - Set `child._name` and `child._aggregator.agent_name` **immediately** before `await self.call_tool(...)`.
     - Right after the `await`, restore them to the base `original_names[tool_name]` (inside the same task's `try/finally`).
   - `run_tools()` should **no longer perform name restoration** for children; it only needs to restore parent-level names (if we ever mutate them) and handle display.

2. **Clarify known limitation**

   - In the module docstring, add a short Limitations section explaining:
     - Under heavy concurrency, some low-level logs from MCP may still show mixed instance names; the progress panel and chat tool headers are the authoritative view.

**Outcome:** race window is strictly bounded to the duration of a single tool call in a single task; we no longer keep children renamed after the call completes.

---

### 5.4. Explicitly document stats behavior

**Goal:** align user expectations with current implementation.

1. **Update README / docs** (or a dedicated experimental note):

   - Describe that:
     - Token and tool usage stats are aggregated **per agent**.
     - Agents-as-Tools does **not** create per-instance stats rows; instead:
       - Per-instance work is visible in the progress panel.
       - Tool calls are visible in the history summary as `tool→` / `result→` rows.

2. **Optionally tag tool results with instance index in content**

   - For debug clarity, `AgentsAsToolsAgent` could prepend a short header block to each `CallToolResult` content:
     - e.g. `"[instance 1]"`.
   - This would make the instance index visible in `history_display` even outside the UI tool headers.

   This is optional and can be added behind a config flag if needed.

---

### 5.5. Tests and diagnostics

1. **Unit tests for `AgentsAsToolsAgent`**

   - Scenarios:
     - Single tool call to one child.
     - Two sequential tool calls in separate turns.
     - Two parallel tool calls to **different** children.
     - Two parallel tool calls to the **same** child.
     - Tool-not-found error path.
   - Assertions:
     - `list_tools()` returns expected tool names.
     - `call_tool()` forwards `text` and `json` correctly.
     - Display suppression:
       - `child.display.config.logger.show_chat` toggles to False during calls.
       - Restored to original after calls (check for all scenarios).

2. **Integration-style test with a fake `RichProgressDisplay`**

   - Inject a fake progress display with a deterministic in-memory representation.
   - Assert that for parallel calls:
     - Parent gets a `READY` event.
     - Each instance gets a `CHATTING` event with `target=OriginalName[i]`.
     - `hide_task()` is called exactly once per instance.

3. **Manual diagnostic recipe**

   - Document a small `fastagent.config.yaml` example that:
     - Defines N children representing mocked projects.
     - Defines a parent with `agents: [...]` using Agents-as-Tools.
   - Steps to reproduce and visually verify:
     - Instance lines in progress panel.
     - Tool rows in history summary.
     - Stats table showing aggregate per agent.

---

## 6. Future Enhancements (Beyond Fix Plan)

These are candidates for the from-scratch design rather than this incremental fix:

- **Per-instance stats**
  - Attach a lightweight `InstanceUsage` struct per tool call and aggregate it at run end.

- **Correlation IDs and structured logging**
  - Emit a unique correlation ID for each tool call and propagate it through:
    - Parent request → tool_call.
    - Child logs and progress events.
    - MCPAggregator transport tracking.

- **Cleaner abstraction boundary**
  - Extract an `AgentsAsToolsRuntime` helper that contains **no UI or LLM logic**, only:
    - Tool mapping.
    - Parallel execution.
    - Result collation.
  - A separate `AgentsAsToolsDisplayAdapter` layer would handle:
    - Progress events.
    - Display config changes.

These ideas are elaborated further in `agetns_as_tools_plan_scratcj.md`.
