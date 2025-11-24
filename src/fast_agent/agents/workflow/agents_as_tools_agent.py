"""
Agents as Tools Pattern Implementation
=======================================

Overview
--------
This module implements the "Agents as Tools" pattern, inspired by OpenAI's Agents SDK
(https://openai.github.io/openai-agents-python/tools). It allows child agents to be
exposed as callable tools to a parent agent, enabling hierarchical agent composition
without the complexity of traditional orchestrator patterns. The current implementation
goes a step further by spawning **detached per-call clones** of every child so that each
parallel execution has its own LLM + MCP stack, eliminating name overrides and shared
state hacks.

Rationale
---------
Traditional approaches to multi-agent systems often require:
1. Complex orchestration logic with explicit routing rules
2. Iterative planning mechanisms that add cognitive overhead
3. Tight coupling between parent and child agent implementations

The "Agents as Tools" pattern simplifies this by:
- **Treating agents as first-class tools**: Each child agent becomes a tool that the
  parent LLM can call naturally via function calling
- **Delegation, not orchestration**: The parent LLM decides which child agents to invoke
  based on its instruction and context, without hardcoded routing logic
- **Parallel execution**: Multiple child agents can run concurrently when the LLM makes
  parallel tool calls
- **Clean abstraction**: Child agents expose minimal schemas (text or JSON input),
  making them universally composable

Benefits over iterative_planner/orchestrator:
- Simpler codebase: No custom planning loops or routing tables
- Better LLM utilization: Modern LLMs excel at function calling
- Natural composition: Agents nest cleanly without special handling
- Parallel by default: Leverage asyncio.gather for concurrent execution

Algorithm
---------
1. **Initialization**
   - `AgentsAsToolsAgent` is itself an `McpAgent` (with its own MCP servers + tools) and receives a list of **child agents**.
   - Each child agent is mapped to a synthetic tool name: `agent__{child_name}`.
   - Child tool schemas advertise text/json input capabilities.

2. **Tool Discovery (list_tools)**
   - `list_tools()` starts from the base `McpAgent.list_tools()` (MCP + local tools).
   - Synthetic child tools `agent__ChildName` are added on top when their names do not collide with existing tools.
   - The parent LLM therefore sees a **merged surface**: MCP tools and agent-tools in a single list.

3. **Tool Execution (call_tool)**
   - If the requested tool name resolves to a child agent (either `child_name` or `agent__child_name`):
     - Convert tool arguments (text or JSON) to a child user message.
     - Execute via detached clones created inside `run_tools` (see below).
     - Responses are converted to `CallToolResult` objects (errors propagate as `isError=True`).
   - Otherwise, delegate to the base `McpAgent.call_tool` implementation (MCP tools, shell, human-input, etc.).

4. **Parallel Execution (run_tools)**
   - Collect all tool calls from the parent LLM response.
   - Partition them into **child-agent tools** and **regular MCP/local tools**.
   - Child-agent tools are executed in parallel:
     - For each child tool call, spawn a detached clone with its own LLM + MCP aggregator and suffixed name.
     - Emit `ProgressAction.CHATTING` / `ProgressAction.FINISHED` events for each instance and keep parent status untouched.
     - Merge each clone's usage back into the template child after shutdown.
   - Remaining MCP/local tools are delegated to `McpAgent.run_tools()`.
   - Child and MCP results (and their error text from `FAST_AGENT_ERROR_CHANNEL`) are merged into a single `PromptMessageExtended` that is returned to the parent LLM.

Progress Panel Behavior
-----------------------
To provide clear visibility into parallel executions, the progress panel (left status
table) undergoes dynamic updates:

**Before parallel execution:**
```
▎▶ Chatting      ▎ PM-1-DayStatusSummarizer     gpt-5 turn 1
```

**During parallel execution (2+ instances):**
- Parent line stays in whatever lifecycle state it already had; no forced "Ready" flips.
- New lines appear for each detached instance with suffixed names:
```
▎▶ Chatting      ▎ PM-1-DayStatusSummarizer[1]   gpt-5 turn 2
▎▶ Calling tool  ▎ PM-1-DayStatusSummarizer[2]   tg-ro (list_messages)
```

**Key implementation details:**
- Each clone advertises its own `agent_name` (e.g., `OriginalName[instance_number]`).
- MCP progress events originate from the clone's aggregator, so tool activity always shows under the suffixed name.
- Parent status lines remain visible for context while children run.

**As each instance completes:**
- We emit `ProgressAction.FINISHED` with elapsed time, keeping the line in the panel for auditability.
- Other instances continue showing their independent progress until they also finish.

**After all parallel executions complete:**
- Finished instance lines remain until the parent agent moves on, giving a full record of what ran.
- Parent and child template names stay untouched because clones carry the suffixed identity.

- **Instance line visibility**: We now leave finished instance lines visible (marked `FINISHED`)
  instead of hiding them immediately, preserving a full audit trail of parallel runs.
- **Chat log separation**: Each parallel instance gets its own tool request/result headers
  with instance numbers [1], [2], etc. for traceability.

Stats and Usage Semantics
-------------------------
- Each detached clone accrues usage on its own `UsageAccumulator`; after shutdown we
  call `child.merge_usage_from(clone)` so template agents retain consolidated totals.
- Runtime events (logs, MCP progress, chat headers) use the suffixed clone names,
  ensuring per-instance traceability even though usage rolls up to the template.
- The CLI *Usage Summary* table still reports one row per template agent
  (for example, `PM-1-DayStatusSummarizer`), not per `[i]` instance; clones are
  runtime-only and do not appear as separate agents in that table.

**Chat log display:**
Tool headers show instance numbers for clarity:
```
▎▶ orchestrator    [tool request - agent__PM-1-DayStatusSummarizer[1]]
▎◀ orchestrator    [tool result - agent__PM-1-DayStatusSummarizer[1]]
▎▶ orchestrator    [tool request - agent__PM-1-DayStatusSummarizer[2]]
▎◀ orchestrator    [tool result - agent__PM-1-DayStatusSummarizer[2]]
```

Bottom status bar shows all instances:
```
| agent__PM-1-DayStatusSummarizer[1] · running | agent__PM-1-DayStatusSummarizer[2] · running |
```

Implementation Notes
--------------------
- **Instance naming**: `run_tools` computes `instance_name = f"{child.name}[i]"` inside the
  per-call wrapper and passes it into `spawn_detached_instance`, so the template child object
  keeps its original name while each detached clone owns the suffixed identity.
- **Progress event routing**: Because each clone's `MCPAggregator` is constructed with the
  suffixed `agent_name`, all MCP/tool progress events naturally use
  `PM-1-DayStatusSummarizer[i]` without mutating base agent fields or using `ContextVar` hacks.
- **Display suppression with reference counting**: Multiple parallel instances of the same
  child agent share a single agent object. Use reference counting to track active instances:
  - `_display_suppression_count[child_id]`: Count of active parallel instances
  - `_original_display_configs[child_id]`: Stored original config
  - Only modify display config when first instance starts (count 0→1)
  - Only restore display config when last instance completes (count 1→0)
  - Prevents race condition where early-finishing instances restore config while others run
- **Child agent(s)**
  - Existing agents (typically `McpAgent`-based) with their own MCP servers, skills, tools, etc.
  - Serve as **templates**; `run_tools` now clones them before every tool call via
    `spawn_detached_instance`, so runtime work happens inside short-lived replicas.

- **Detached instances**
  - Each tool call gets an actual cloned agent with suffixed name `Child[i]`.
  - Clones own their MCP aggregator/LLM stacks and merge usage back into the template after shutdown.
- **Chat log separation**: Each parallel instance gets its own tool request/result headers
  with instance numbers [1], [2], etc. for traceability

Usage Example
-------------
```python
from fast_agent import FastAgent

fast = FastAgent("parent")

# Define child agents
@fast.agent(name="researcher", instruction="Research topics")
async def researcher(): pass

@fast.agent(name="writer", instruction="Write content")
async def writer(): pass

# Define parent with agents-as-tools
@fast.agent(
    name="coordinator",
    instruction="Coordinate research and writing",
    child_agents=["researcher", "writer"]  # Exposes children as tools
)
async def coordinator(): pass
```

The parent LLM can now naturally call researcher and writer as tools.

References
----------
- OpenAI Agents SDK: https://openai.github.io/openai-agents-python/tools
- GitHub Issue: https://github.com/evalstate/fast-agent/issues/XXX
"""

from __future__ import annotations

import asyncio
import json
from copy import copy
from typing import Any

from mcp import ListToolsResult, Tool
from mcp.types import CallToolResult

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.llm_agent import LlmAgent
from fast_agent.agents.mcp_agent import McpAgent
from fast_agent.core.logging.logger import get_logger
from fast_agent.core.prompt import Prompt
from fast_agent.constants import FAST_AGENT_ERROR_CHANNEL
from fast_agent.mcp.helpers.content_helpers import (
    get_text,
    is_text_content,
    text_content,
)
from fast_agent.ui.message_primitives import MessageType
from fast_agent.types import PromptMessageExtended, RequestParams

logger = get_logger(__name__)


class AgentsAsToolsAgent(McpAgent):
    """MCP-enabled agent that exposes child agents as additional tools.

    This hybrid agent:

    - Inherits all MCP behavior from :class:`McpAgent` (servers, MCP tool discovery, local tools).
    - Exposes each child agent as an additional synthetic tool (`agent__ChildName`).
    - Merges **MCP tools** and **agent-tools** into a single `list_tools()` surface.
    - Routes `call_tool()` to child agents when the name matches a child, otherwise delegates
      to the base `McpAgent.call_tool` implementation.
    - Overrides `run_tools()` to fan out child-agent tools in parallel using detached clones,
      while delegating any remaining MCP/local tools to the base `McpAgent.run_tools` and
      merging all results into a single tool-loop response.
    """

    def __init__(
        self,
        config: AgentConfig,
        agents: list[LlmAgent],
        context: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize AgentsAsToolsAgent.

        Args:
            config: Agent configuration for this parent agent (including MCP servers/tools)
            agents: List of child agents to expose as tools
            context: Optional context for agent execution
            **kwargs: Additional arguments passed through to :class:`McpAgent` and its bases
        """
        super().__init__(config=config, context=context, **kwargs)
        self._child_agents: dict[str, LlmAgent] = {}

        # Build tool name mapping for children
        for child in agents:
            tool_name = self._make_tool_name(child.name)
            if tool_name in self._child_agents:
                logger.warning(
                    f"Duplicate tool name '{tool_name}' for child agent '{child.name}', overwriting"
                )
            self._child_agents[tool_name] = child

    def _make_tool_name(self, child_name: str) -> str:
        """Generate a tool name for a child agent.

        Args:
            child_name: Name of the child agent

        Returns:
            Prefixed tool name to avoid collisions with MCP tools
        """
        return f"agent__{child_name}"

    async def initialize(self) -> None:
        """Initialize this agent and all child agents."""
        await super().initialize()
        for agent in self._child_agents.values():
            if not getattr(agent, "initialized", False):
                await agent.initialize()

    async def shutdown(self) -> None:
        """Shutdown this agent and all child agents."""
        await super().shutdown()
        for agent in self._child_agents.values():
            try:
                await agent.shutdown()
            except Exception as e:
                logger.warning(f"Error shutting down child agent {agent.name}: {e}")

    async def list_tools(self) -> ListToolsResult:
        """List MCP tools plus child agents exposed as tools."""

        base = await super().list_tools()
        tools = list(base.tools)
        existing_names = {tool.name for tool in tools}

        for tool_name, agent in self._child_agents.items():
            if tool_name in existing_names:
                continue

            input_schema: dict[str, Any] = {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Plain text input"},
                    "json": {"type": "object", "description": "Arbitrary JSON payload"},
                },
                "additionalProperties": True,
            }
            tools.append(
                Tool(
                    name=tool_name,
                    description=agent.instruction,
                    inputSchema=input_schema,
                )
            )
            existing_names.add(tool_name)

        return ListToolsResult(tools=tools)

    def _ensure_display_maps_initialized(self) -> None:
        """Lazily initialize display suppression tracking maps."""
        if not hasattr(self, "_display_suppression_count"):
            self._display_suppression_count = {}
            self._original_display_configs = {}

    def _suppress_child_display(self, child: LlmAgent) -> None:
        """Suppress child chat output while preserving tool logs."""
        self._ensure_display_maps_initialized()
        child_id = id(child)
        count = self._display_suppression_count.get(child_id, 0)
        if 0 == count:
            if (
                hasattr(child, "display")
                and child.display
                and getattr(child.display, "config", None)
            ):
                # Store original config for restoration later
                self._original_display_configs[child_id] = child.display.config
                temp_config = copy(child.display.config)
                if hasattr(temp_config, "logger"):
                    temp_logger = copy(temp_config.logger)
                    temp_logger.show_chat = False
                    temp_logger.show_tools = True  # Explicitly keep tools visible
                    temp_config.logger = temp_logger
                child.display.config = temp_config
        self._display_suppression_count[child_id] = count + 1

    def _release_child_display(self, child: LlmAgent) -> None:
        """Restore child display configuration when the last tool instance completes."""
        if not hasattr(self, "_display_suppression_count"):
            return
        child_id = id(child)
        if child_id not in self._display_suppression_count:
            return
        self._display_suppression_count[child_id] -= 1
        if self._display_suppression_count[child_id] <= 0:
            del self._display_suppression_count[child_id]
            original_config = self._original_display_configs.pop(child_id, None)
            if (
                original_config is not None
                and hasattr(child, "display")
                and child.display
            ):
                child.display.config = original_config

    async def _invoke_child_agent(
        self,
        child: LlmAgent,
        arguments: dict[str, Any] | None = None,
        *,
        suppress_display: bool = True,
    ) -> CallToolResult:
        """Shared helper to execute a child agent with standard serialization and display rules."""

        args = arguments or {}
        # Serialize arguments to text input
        if isinstance(args.get("text"), str):
            input_text = args["text"]
        elif "json" in args:
            input_text = (
                json.dumps(args["json"], ensure_ascii=False)
                if isinstance(args["json"], dict)
                else str(args["json"])
            )
        else:
            input_text = json.dumps(args, ensure_ascii=False) if args else ""

        child_request = Prompt.user(input_text)

        try:
            # Suppress child agent chat messages (keep tool calls visible)
            if suppress_display:
                self._suppress_child_display(child)

            response: PromptMessageExtended = await child.generate(
                [child_request], None
            )
            # Prefer preserving original content blocks for better UI fidelity
            content_blocks = list(response.content or [])

            from fast_agent.constants import FAST_AGENT_ERROR_CHANNEL

            error_blocks = None
            if response.channels and FAST_AGENT_ERROR_CHANNEL in response.channels:
                error_blocks = response.channels.get(FAST_AGENT_ERROR_CHANNEL) or []
                # Append error blocks so they are visible in the tool result panel
                if error_blocks:
                    content_blocks.extend(error_blocks)

            return CallToolResult(
                content=content_blocks,
                isError=bool(error_blocks),
            )
        except Exception as e:
            logger.error(f"Child agent {child.name} failed: {e}")
            return CallToolResult(content=[text_content(f"Error: {e}")], isError=True)
        finally:
            if suppress_display:
                self._release_child_display(child)

    def _resolve_child_agent(self, name: str) -> LlmAgent | None:
        return self._child_agents.get(name) or self._child_agents.get(
            self._make_tool_name(name)
        )

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        tool_use_id: str | None = None,
    ) -> CallToolResult:
        """Route tool execution to child agents first, then MCP/local tools.

        The signature matches :meth:`McpAgent.call_tool` so that upstream tooling
        can safely pass the LLM's ``tool_use_id`` as a positional argument.
        """

        child = self._resolve_child_agent(name)
        if child is not None:
            # Child agents don't currently use tool_use_id, they operate via
            # a plain PromptMessageExtended tool call.
            return await self._invoke_child_agent(child, arguments)

        return await super().call_tool(name, arguments, tool_use_id)

    def _show_parallel_tool_calls(self, descriptors: list[dict[str, Any]]) -> None:
        """Display tool call headers for parallel agent execution.

        Args:
            descriptors: List of tool call descriptors with metadata
        """
        if not descriptors:
            return

        status_labels = {
            "pending": "running",
            "error": "error",
            "missing": "missing",
        }

        # Show detailed call information for each agent
        for i, desc in enumerate(descriptors, 1):
            tool_name = desc.get("tool", "(unknown)")
            args = desc.get("args", {})
            status = desc.get("status", "pending")

            if status == "error":
                continue  # Skip display for error tools, will show in results

            # Always add individual instance number for clarity
            display_tool_name = f"{tool_name}[{i}]"

            # Build bottom item for THIS instance only (not all instances)
            status_label = status_labels.get(status, "pending")
            bottom_item = f"{display_tool_name} · {status_label}"

            # Show individual tool call with arguments
            self.display.show_tool_call(
                name=self.name,
                tool_name=display_tool_name,
                tool_args=args,
                bottom_items=[bottom_item],  # Only this instance's label
                max_item_length=28,
            )

    def _summarize_result_text(self, result: CallToolResult) -> str:
        for block in result.content or []:
            if is_text_content(block):
                text = (get_text(block) or "").strip()
                if text:
                    text = text.replace("\n", " ")
                    return text[:180] + "…" if len(text) > 180 else text
        return ""

    def _show_parallel_tool_results(self, records: list[dict[str, Any]]) -> None:
        """Display tool result panels for parallel agent execution.

        Args:
            records: List of result records with descriptor and result data
        """
        if not records:
            return

        # Show detailed result for each agent
        for i, record in enumerate(records, 1):
            descriptor = record.get("descriptor", {})
            result = record.get("result")
            tool_name = descriptor.get("tool", "(unknown)")

            if result:
                # Always add individual instance number for clarity
                display_tool_name = f"{tool_name}[{i}]"

                # Show individual tool result with full content
                self.display.show_tool_result(
                    name=self.name,
                    tool_name=display_tool_name,
                    result=result,
                )

    async def run_tools(self, request: PromptMessageExtended) -> PromptMessageExtended:
        """Handle mixed MCP + agent-tool batches."""

        if not request.tool_calls:
            logger.warning("No tool calls found in request", data=request)
            return PromptMessageExtended(role="user", tool_results={})

        child_ids: list[str] = []
        for correlation_id, tool_request in request.tool_calls.items():
            if self._resolve_child_agent(tool_request.params.name):
                child_ids.append(correlation_id)

        if not child_ids:
            return await super().run_tools(request)

        child_results, child_error = await self._run_child_tools(
            request, set(child_ids)
        )

        if len(child_ids) == len(request.tool_calls):
            return self._finalize_tool_results(
                child_results, tool_loop_error=child_error
            )

        # Execute remaining MCP/local tools via base implementation
        remaining_ids = [
            cid for cid in request.tool_calls.keys() if cid not in child_ids
        ]
        mcp_request = PromptMessageExtended(
            role=request.role,
            content=request.content,
            tool_calls={cid: request.tool_calls[cid] for cid in remaining_ids},
        )
        mcp_message = await super().run_tools(mcp_request)
        mcp_results = mcp_message.tool_results or {}
        mcp_error = self._extract_error_text(mcp_message)

        combined_results = {}
        combined_results.update(child_results)
        combined_results.update(mcp_results)

        tool_loop_error = child_error or mcp_error
        return self._finalize_tool_results(
            combined_results, tool_loop_error=tool_loop_error
        )

    async def _run_child_tools(
        self,
        request: PromptMessageExtended,
        target_ids: set[str],
    ) -> tuple[dict[str, CallToolResult], str | None]:
        """Run only the child-agent tool calls from the request."""

        if not target_ids:
            return {}, None

        tool_results: dict[str, CallToolResult] = {}
        tool_loop_error: str | None = None

        try:
            listed = await self.list_tools()
            available_tools = [t.name for t in listed.tools]
        except Exception as exc:
            logger.warning(f"Failed to list tools before execution: {exc}")
            available_tools = list(self._child_agents.keys())

        call_descriptors: list[dict[str, Any]] = []
        descriptor_by_id: dict[str, dict[str, Any]] = {}
        tasks: list[asyncio.Task] = []
        id_list: list[str] = []

        for correlation_id, tool_request in request.tool_calls.items():
            if correlation_id not in target_ids:
                continue

            tool_name = tool_request.params.name
            tool_args = tool_request.params.arguments or {}

            descriptor = {
                "id": correlation_id,
                "tool": tool_name,
                "args": tool_args,
            }
            call_descriptors.append(descriptor)
            descriptor_by_id[correlation_id] = descriptor

            if (
                tool_name not in available_tools
                and self._make_tool_name(tool_name) not in available_tools
            ):
                error_message = f"Tool '{tool_name}' is not available"
                tool_results[correlation_id] = CallToolResult(
                    content=[text_content(error_message)], isError=True
                )
                tool_loop_error = tool_loop_error or error_message
                descriptor["status"] = "error"
                continue

            descriptor["status"] = "pending"
            id_list.append(correlation_id)

        from fast_agent.event_progress import ProgressAction, ProgressEvent
        from fast_agent.ui.progress_display import (
            progress_display as outer_progress_display,
        )

        async def call_with_instance_name(
            tool_name: str, tool_args: dict[str, Any], instance: int
        ) -> CallToolResult:
            child = self._resolve_child_agent(tool_name)
            if not child:
                error_msg = f"Unknown agent-tool: {tool_name}"
                return CallToolResult(content=[text_content(error_msg)], isError=True)

            base_name = getattr(child, "_name", child.name)
            instance_name = f"{base_name}[{instance}]"

            try:
                clone = await child.spawn_detached_instance(name=instance_name)
            except Exception as exc:
                logger.error(
                    "Failed to spawn dedicated child instance",
                    data={
                        "tool_name": tool_name,
                        "agent_name": base_name,
                        "error": str(exc),
                    },
                )
                return CallToolResult(
                    content=[text_content(f"Spawn failed: {exc}")], isError=True
                )

            progress_started = False
            try:
                outer_progress_display.update(
                    ProgressEvent(
                        action=ProgressAction.CHATTING,
                        target=instance_name,
                        details="",
                        agent_name=instance_name,
                    )
                )
                progress_started = True
                return await self._invoke_child_agent(clone, tool_args)
            finally:
                try:
                    await clone.shutdown()
                except Exception as shutdown_exc:
                    logger.warning(
                        "Error shutting down dedicated child instance",
                        data={
                            "instance_name": instance_name,
                            "error": str(shutdown_exc),
                        },
                    )
                try:
                    child.merge_usage_from(clone)
                except Exception as merge_exc:
                    logger.warning(
                        "Failed to merge usage from child instance",
                        data={
                            "instance_name": instance_name,
                            "error": str(merge_exc),
                        },
                    )
                if progress_started and instance_name:
                    outer_progress_display.update(
                        ProgressEvent(
                            action=ProgressAction.FINISHED,
                            target=instance_name,
                            details="Completed",
                            agent_name=instance_name,
                        )
                    )

        for i, cid in enumerate(id_list, 1):
            tool_name = descriptor_by_id[cid]["tool"]
            tool_args = descriptor_by_id[cid]["args"]
            tasks.append(
                asyncio.create_task(call_with_instance_name(tool_name, tool_args, i))
            )

        self._show_parallel_tool_calls(call_descriptors)

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for i, result in enumerate(results):
                correlation_id = id_list[i]
                if isinstance(result, Exception):
                    msg = f"Tool execution failed: {result}"
                    tool_results[correlation_id] = CallToolResult(
                        content=[text_content(msg)], isError=True
                    )
                    tool_loop_error = tool_loop_error or msg
                    descriptor_by_id[correlation_id]["status"] = "error"
                    descriptor_by_id[correlation_id]["error_message"] = msg
                else:
                    tool_results[correlation_id] = result
                    descriptor_by_id[correlation_id]["status"] = (
                        "error" if result.isError else "done"
                    )

        ordered_records: list[dict[str, Any]] = []
        for cid in id_list:
            result = tool_results.get(cid)
            if result is None:
                continue
            descriptor = descriptor_by_id.get(cid, {})
            ordered_records.append({"descriptor": descriptor, "result": result})

        self._show_parallel_tool_results(ordered_records)

        return tool_results, tool_loop_error

    def _extract_error_text(self, message: PromptMessageExtended) -> str | None:
        if not message.channels:
            return None

        error_blocks = message.channels.get(FAST_AGENT_ERROR_CHANNEL)
        if not error_blocks:
            return None

        for block in error_blocks:
            text = get_text(block)
            if text:
                return text

        return None
