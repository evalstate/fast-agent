# Agents-as-Tools — From-Scratch Design Plan (Upstream-Oriented)

## 1. Objectives

Design a clean, upstream-friendly implementation of the **Agents-as-Tools** pattern for `fast-agent`, starting from the upstream repository semantics:

- **Model**: a parent LLM agent exposes other agents as callable tools.
- **Behavior**: parent can invoke children in arbitrary order and in parallel, using normal tool-calling.
- **DX**: minimal new concepts; works naturally with existing decorators and config.
- **UX**: integrates with current progress panel, history, and usage stats without introducing ad hoc hacks.

This plan does **not** assume any existing WIP code; it re-derives the feature from first principles using the current architecture (decorators, factory, MCP, UI, stats).

---

## 2. Conceptual Model

### 2.1. Roles & responsibilities

- **Parent agent (Agents-as-Tools orchestrator)**
  - A normal LLM agent with tool-calling capability.
  - Exposes *child agents* as tools (`agent__ChildName`).
  - Delegates the actual work to children; no custom planning.

- **Child agent(s)**
  - Existing agents (typically `McpAgent`-based) with their own MCP servers, skills, tools, etc.
  - Own their own `UsageAccumulator`, history, and MCP aggregator.
  - Are reused as-is; we do not clone them per instance.

- **Virtual child instances**
  - Logical construct: per tool call, we treat it as an `Instance` of a child with an index `[i]`.
  - Instances are used purely for **UI and logging**, not for real objects.

### 2.2. Key invariants

- **Single source of truth for child agents**
  - One `LlmAgent` object per defined agent name.
  - All parents and instances refer to the same child objects.

- **LLM tool-loop compatibility**
  - The parents `generate()` uses the standard `ToolAgent` loop:
    - LLM → `stop_reason=TOOL_USE` → `run_tools()` → new USER message.

- **MCP reuse**
  - Each child has exactly one `MCPAggregator` that persists according to its config.
  - Instances never create or destroy MCP connections directly.

- **Stats aggregation per agent**
  - Usage summary is per *agent name* (parent + each child), not per instance.
  - Instances show up only in progress/historical views.

### 2.3. Alternative execution models (future options)

While the core plan intentionally reuses a single child object per agent, there are cases where **"honest" per-call isolation** is preferred. Two strategies can be layered onto this design later:

1. **Dedicated child agent per call**
   - Before dispatching a tool call, clone the target child (including MCP aggregator, LLM, memory) to form a short-lived agent.
   - Guarantees zero shared state: logs, history, MCP connections stay scoped to that instance.
   - Downsides: high startup cost (MCP discovery, model warm-up) for every call; extra resource usage if multiple calls run in parallel.

2. **Pre-warmed agent pool**
   - Keep `N` fully initialized child agents per name (each with its own MCP aggregator/LLM).
   - A call acquires a free agent from the pool; after completion it returns the instance for reuse.
   - Pros: isolates state without per-call bootstrap; allows true parallelism as long as pool capacity is available.
   - Cons: more memory + open MCP connections proportional to pool size; scheduling logic needed when pool is exhausted.

Both approaches can be integrated into the factory/runtime layer without rewriting the Agents-as-Tools surface: the parent would simply target a different acquisition strategy when resolving `self._children`. Documenting these options here keeps the plan aligned with future requirements around strict isolation.

### 2.4. Current implementation snapshot — Detached per-call clones (Nov 2025)

While §2.3 framed cloning/pooling as optional futures, the active codebase now runs with the **Dedicated child agent per call** strategy so we can guarantee honest per-instance state:

1. **Clone creation**
   - `AgentsAsToolsAgent.run_tools()` calls `child.spawn_detached_instance(name=f"{child}[i]")` before every tool dispatch.
   - `spawn_detached_instance` (added to `LlmDecorator`) deep-copies the agent config, re-attaches the same LLM factory/request params, and replays initialization hooks.

2. **MCP aggregator ownership**
   - Each detached clone constructs its own `MCPAggregator`, which in turn acquires a shared `MCPConnectionManager` from context.
   - To avoid tearing down the shared TaskGroup, `MCPAggregator` now tracks `_owns_connection_manager`; only the original agent that created the manager performs shutdown on `close()`.

3. **Lifecycle + cleanup**
   - After the tool call completes we `await clone.shutdown()` and merge its `UsageAccumulator` back into the parent child via `child.merge_usage_from(clone)`.
   - Progress entries remain visible by emitting `ProgressAction.FINISHED` events instead of hiding tasks, ensuring traceability per instance.

4. **Implications**
   - Logs, MCP events, and progress panel lines now display fully indexed names (for example, `PM-1-DayStatusSummarizer[2]`).
   - The CLI *Usage Summary* table still reports a single aggregated row per template agent (for example, `PM-1-DayStatusSummarizer`), not per `[i]` instance.
   - Resource cost is higher than the single-object model, but correctness (agent naming, MCP routing, and per-instance traceability in logs/UI) takes priority for the current StratoSpace workflows.

This snapshot should stay in sync with the actual code to document why the detached-instance path is the default today, even though the plan keeps the door open for lighter reuse models.

---

## 3. High-Level Architecture

### 3.1. New class: `AgentsAsToolsAgent`

Location: `src/fast_agent/agents/workflow/agents_as_tools_agent.py`.

Base class: **`McpAgent`** (inherits `ToolAgent` and manages MCP connections).

Responsibilities:

- Adapter between **LLM tool schema**, **child agents**, and **MCP tools**.
- `list_tools()` → MCP tools (from `McpAgent`) plus synthetic tools for children.
- `call_tool()` → executes child agents first; falls back to MCP/local tools.
- `run_tools()` → parallel fan-out for child agents plus integration with the base MCP `run_tools` for mixed batches.

Constructor:

```python
class AgentsAsToolsAgent(McpAgent):
    def __init__(
        self,
        config: AgentConfig,
        agents: list[LlmAgent],
        context: Context | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config=config, context=context, **kwargs)
        self._child_agents: dict[str, LlmAgent] = {}
        # Maps tool name -> child agent (keys are agent__ChildName)
```

### 3.2. Integration points

1. **Decorators (`direct_decorators.agent`)**
   - Add parameter `agents: List[str]` (already present upstream).
   - Store `child_agents=agents` in the agent metadata.

2. **Factory (`direct_factory.create_agents_by_type`)**
   - For `AgentType.BASIC`:
     - If `child_agents` is non-empty:
       - Resolve child names to **already-created** agents.
       - Construct `AgentsAsToolsAgent(config, context, agents=child_agents)`.
       - Attach LLM.
     - Else: create a normal `McpAgent` (as today).

3. **UI / CLI**
   - No CLI flags change.
   - New behavior is activated simply by specifying `agents:` in the decorator/config.

### 3.3. Minimal usage sample (for docs and examples)

This sample is used as a reference for both local testing and future docs/README updates.
It mirrors the standalone script in the Strato workspace (`fast/agent-as-tools.py`).

```python
import asyncio
from fast_agent import FastAgent

fast = FastAgent("Agents-as-Tools demo")


@fast.agent(
    name="NY-Time",
    instruction="Return current time in New York.",
    servers=["tm"],  # MCP server 'tm' configured in fastagent.config.yaml
    model="gpt-5-mini",
    tools={"tm": ["get_current_time"]},
)
@fast.agent(
    name="London-Time",
    instruction="Return current time in London.",
    servers=["tm"],
    model="gpt-5-mini",
    tools={"tm": ["get_current_time"]},
)
@fast.agent(
    name="time-orchestrator",
    instruction="Get current time in New York and London.",
    model="gpt-5-mini",
    default=True,
    agents=[
        "NY-Time",
        "London-Time",
    ],
)
async def main() -> None:
    async with fast.run() as agent:
        result = await agent("get time for NY and London")
        print(result)


if __name__ == "__main__":
    asyncio.run(main())
```

Key points:

- `NY-Time` and `London-Time` are normal MCP-enabled agents using the `tm` server.
- `time-orchestrator` is a BASIC agent with `agents=[...]`; the factory instantiates it
  as an `AgentsAsToolsAgent` under the hood, exposing each child as a tool.
- From the LLM's perspective, it simply sees additional tools (`agent__NY-Time`,
  `agent__London-Time`) alongside regular MCP tools.

---

## 4. Detailed Design by Concern

### 4.1. Tool exposure (`list_tools`)

**Goal:** make each child agent a callable tool with a permissive schema.

- Tool naming:
  - `tool_name = f"agent__{child.name}"`.
  - We store the mapping internally, not relying on `child.name` string matching later.

- Input schema:
  - Keep it minimal and robust:

    ```json
    {
      "type": "object",
      "properties": {
        "text": { "type": "string", "description": "Plain text input" },
        "json": { "type": "object", "description": "Arbitrary JSON payload" }
      },
      "additionalProperties": true
    }
    ```

- Implementation sketch:
  - For each child in `self._child_agents`:
    - Build an `mcp.Tool`:
      - `name = tool_name`
      - `description = child.instruction`
      - `inputSchema = schema_above`.

- In the current implementation these child tools are **merged** with the MCP tools exposed by `McpAgent.list_tools()`: `AgentsAsToolsAgent.list_tools()` returns a single combined surface (MCP tools + `agent__Child` tools), adding child tools only when their names do not conflict with existing MCP/local tool names.

### 4.2. Argument mapping (`call_tool`)

**Goal:** map tool arguments to a single child **user message**.

Rules:

- If `arguments["text"]` is a string → use as-is.
- Else if `"json" in arguments`:
  - If it is a dict → `json.dumps` (UTF-8, no ASCII-escaping).
  - Else → `str(...)`.
- Else:
  - If there are other arguments → `json.dumps(arguments)`.
  - Else → empty string.

Then:

- Build `PromptMessageExtended.user(input_text)` (or `Prompt.user` helper) and call:
  - `child.generate([user_message], request_params=None)`.

Error handling:

- Unknown tool name → `CallToolResult(isError=True, content=["Unknown agent-tool: {name}"])`.
- Unhandled exception in child → `CallToolResult(isError=True, content=["Error: {e}"])`.

Wire error-channel content:

- If childs response has `channels[FAST_AGENT_ERROR_CHANNEL]`, append those blocks to `CallToolResult.content` and set `isError=True`.

### 4.3. Display behavior for children

**Requirement:** when a child is used as a tool:

- Do **not** show its normal assistant chat blocks.
- Do show its **tool usage** (MCP tools, shell, etc.).

Design:

- Define a tiny utility in `AgentsAsToolsAgent`:

  - `self._display_suppression: dict[int, DisplayState]` where `DisplayState` holds:
    - `original_config: ConsoleDisplayConfig`.
    - `ref_count: int`.

- Methods:

  - `_suppress_child_display(child: LlmAgent)`
    - On first entry for this child:
      - Copy `child.display.config` → `original_config`.
      - Clone config and set `logger.show_chat = False`, `logger.show_tools = True`.
      - Assign cloned config to `child.display.config`.
    - Increment `ref_count`.

  - `_release_child_display(child: LlmAgent)`
    - Decrement `ref_count`.
    - If it reaches 0:
      - Restore `child.display.config = original_config`.
      - Remove entry from `_display_suppression`.

- Use these methods in `call_tool()` via `try/finally`.

Rationale: children can still be run standalone (outside Agents-as-Tools) with full chat visible; we only alter display while they are acting as tools.

### 4.4. Parallel `run_tools` semantics

**Goal:** replace `ToolAgent.run_tools` with a parallel implementation that preserves its contract but allows:

- multiple tool calls per LLM turn;
- concurrent execution via `asyncio.gather`;
- clear UI for each per-call instance.

#### 4.4.1. Data structures

- `call_descriptors: list[dict]`:
  - `{"id", "tool", "args", "status", "error_message"?}`.

- `descriptor_by_id: dict[correlation_id -> descriptor]`.
- `tasks: list[Task[CallToolResult]]`.
- `ids_in_order: list[str]` for stable correlation.

#### 4.4.2. Algorithm (current implementation)

1. **Validate tool calls**
   - Snapshot `available_tools` from `list_tools()`.
   - For each `request.tool_calls[correlation_id]`:
     - If name not in `available_tools` → create `CallToolResult(isError=True, ...)`, mark descriptor as `status="error"`, skip task.
     - Else → `status="pending"`, add to `ids_in_order`.

2. **Create detached instances and names**

   - For each `correlation_id` in `ids_in_order` assign `instance_index = 1..N`.
   - Resolve the template child from `_child_agents`.
   - Compute `base_name = child.name` and `instance_name = f"{base_name}[{instance_index}]"`.
   - Use `instance_name` consistently for:
     - the detached clone (`spawn_detached_instance(name=instance_name)`),
     - progress events (`agent_name=instance_name`, `target=instance_name`),
     - chat/tool headers (`tool_name[instance_index]`).

3. **Instance execution wrapper**

   Conceptually (simplified):

   ```python
   async def call_with_instance_name(tool_name, tool_args, instance_index) -> CallToolResult:
       child = resolve_template_child(tool_name)
       base_name = child.name
       instance_name = f"{base_name}[{instance_index}]"

       clone = await child.spawn_detached_instance(name=instance_name)

       progress_display.update(ProgressEvent(
           action=ProgressAction.CHATTING,
           target=instance_name,
           agent_name=instance_name,
       ))

       try:
           # Handles argument → text mapping, display suppression, error channel, etc.
           return await self._invoke_child_agent(clone, tool_args)
       finally:
           await clone.shutdown()
           child.merge_usage_from(clone)
           progress_display.update(ProgressEvent(
               action=ProgressAction.FINISHED,
               target=instance_name,
               agent_name=instance_name,
               details="Completed",
           ))
   ```

   - All interaction with the Rich progress panel goes through `ProgressEvent` objects and the shared `progress_display.update(...)` API.
   - `RichProgressDisplay.update` is responsible for marking `FINISHED` lines complete without hiding other tasks.

4. **Parallel execution and UI**

   - For each `correlation_id` with a valid tool call, create a task:

     ```python
     tasks.append(asyncio.create_task(
         call_with_instance_name(tool_name, tool_args, instance_index=i)
     ))
     ```

   - `_show_parallel_tool_calls(call_descriptors)` and `_show_parallel_tool_results(ordered_records)` use `tool_name[i]` labels in the chat panels and bottom status items, but do not touch `RichProgressDisplay` internals.
   - `results = await asyncio.gather(*tasks, return_exceptions=True)` collects all results and maps them back to `correlation_id` in input order.

5. **Finalize**

- Build ordered `records = [{"descriptor": ..., "result": ...}, ...]` in input order.
- Call `_show_parallel_tool_results(records)`.
- Return `self._finalize_tool_results(tool_results, tool_loop_error)` for consistency with `ToolAgent`.

6. **Mixed MCP + agent-tools batches**

- If `request.tool_calls` contains both child-agent tools and regular MCP tools:
  - Split `tool_calls` into two subsets: child-agent calls and remaining MCP/local tools.
  - Run child-agent calls via the parallel `call_with_instance_name(...)` path described above.
  - Delegate the remaining tools to the base `McpAgent.run_tools()` implementation.
  - Merge `tool_results` and error text from both branches (using the `FAST_AGENT_ERROR_CHANNEL` error channel) into a single `PromptMessageExtended`.

### 4.5. Stats and history integration

- Leave `UsageAccumulator` unchanged.
- Parent and each child agent track their own usage normally.
  - In the detached-clone implementation, each clone accrues usage on its own accumulator and then merges it back into the template child.
- History:
  - `PromptMessageExtended.tool_results` remains a flat mapping by correlation id.
  - `history_display` will show:
    - `tool→` and `result→` sections per tool call.
    - We can optionally prepend `tool_name[i]` into either:
      - the preview text, or
      - a dedicated text block in the tool result content.

No new data model types are needed for stats.

---

## 5. Engineering notes

In the current implementation `AgentsAsToolsAgent` combines both:

- core runtime concerns (tool mapping, argument normalization, `run_tools` orchestration), and
- UI wiring (progress events and chat/tool panels).

This keeps the surface area small and matches the needs of the CLI UI. A future refactor could still extract a pure runtime helper and a separate UI adapter (see §7), but that split is **not** required for the feature to work today.

---

## 6. Implementation Phasing

### Phase 0 — Skeleton

- Add `AgentsAsToolsAgent` class with:
  - Constructor storing children.
  - Basic `list_tools()` and `call_tool()` (no parallelism, no UI tweaks).
- Wire into `direct_factory` for BASIC agents with `child_agents`.
- Provide a minimal example in `examples/` using synchronous tool calls.

### Phase 1 — Parallel execution

- Implement `run_tools()` with `asyncio.gather` but **no special UI**:
  - Just run calls concurrently and aggregate results.
  - Keep the behavior as close as possible to `ToolAgent.run_tools`.

- Add tests:
  - Unit tests for argument mapping and error handling.
  - Concurrency tests with fake children that sleep.

### Phase 2 — UI integration (progress + instance naming)

- Introduce `AgentsAsToolsDisplayAdapter` to centralize Agents-as-Tools-specific progress behavior.
- Implement instance naming and FINISHED-based progress lines so instances remain visible after completion.
- Suppress child chat via ref-counted display config changes.

- Manual QA:
  - Validate panel behavior with 1, 2, N parallel tasks.
  - Validate that parent name & child names are restored.

### Phase 3 — Documentation & ergonomics

- Add docs page / section (for example, a `README.md` subsection
  "Agents-as-Tools (child agents as tools)"):
  - Concept explanation.
  - Minimal Python example from §3.3 (NY/London time orchestrator).
  - Comparison with Orchestrator / IterativePlanner / Parallel workflows.

- Keep the code sample in sync with the shipped example script
  (currently `fast/agent-as-tools.py` in the Strato workspace, upstream
  examples path TBD).

- Add clear notes about:
  - Stats aggregation semantics.
  - Reuse of MCP connections.
  - Limitations (e.g. no per-instance stats rows).

---

## 7. Potential Future Extensions

The current implementation is intentionally minimal. The items below are still **future** additions (not implemented as of Nov 2025).

1. **Per-instance stats & traces**

   - Extend the runtime to emit per-instance stats objects with `instance_id`, `start_time`, `end_time`, `duration_ms`.
   - Allow a richer UI (CLI or web) to display per-instance timing bars and aggregates.

2. **Recursive Agents-as-Tools**

   - Explicitly document and test scenarios where children are themselves `AgentsAsToolsAgent` instances.
   - Ensure nested tool calls remain readable in progress and history views.

3. **Correlation-friendly logging**

   - Standardize structured log fields for tools (`agent_name`, `instance_name`, `correlation_id`, `tool_name`).
   - Make `history_display` able to group tool rows per `(correlation_id, instance)` so parallel runs are easier to inspect.

## 8. Summary

This from-scratch plan defines Agents-as-Tools as a **lightweight adapter agent** that:

- Exposes existing agents as tools.
- Delegates execution to them, preserving their MCP connections and stats.
- Adds a small, well-encapsulated UI layer for:
  - Parallel instance progress lines.
  - Clear tool call/result labeling (`agent__Child[i]`).

By keeping a strict separation between core runtime, UI adapter, and factories, the feature remains understandable and testable, and it aligns with fast-agents existing engineering patterns and philosophy.
