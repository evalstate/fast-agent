"""
Agents as Tools Pattern Implementation
=======================================

Overview
--------
This module implements the "Agents as Tools" pattern, inspired by OpenAI's Agents SDK
(https://openai.github.io/openai-agents-python/tools). It allows child agents to be
exposed as callable tools to a parent agent, enabling hierarchical agent composition
without the complexity of traditional orchestrator patterns.

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
   - Parent agent receives list of child agents
   - Each child agent is mapped to a tool name: `agent__{child_name}`
   - Tool schemas advertise text/json input capabilities

2. **Tool Discovery (list_tools)**
   - Parent LLM receives one tool per child agent
   - Each tool schema includes child agent's instruction as description
   - LLM decides which tools (child agents) to call based on user request

3. **Tool Execution (call_tool)**
   - Route tool name to corresponding child agent
   - Convert tool arguments (text or JSON) to child agent input
   - Suppress child agent's chat messages (show_chat=False) using reference counting
   - Keep child agent's tool calls visible (show_tools=True)
   - Track active instances per child agent to prevent race conditions
   - Only modify display config on first instance, restore on last instance
   - Execute child agent and return response as CallToolResult

4. **Parallel Execution (run_tools)**
   - Collect all tool calls from parent LLM response
   - Create asyncio tasks for each child agent call
   - Modify child agent names with instance numbers: `AgentName[1]`, `AgentName[2]`
   - Update both child._name and child._aggregator.agent_name for progress routing
   - Set parent agent to "Ready" status while instances run
   - Execute all tasks concurrently via asyncio.gather
   - Hide instance lines immediately as each task completes (via finally block)
   - Aggregate results and return to parent LLM

Progress Panel Behavior
-----------------------
To provide clear visibility into parallel executions, the progress panel (left status
table) undergoes dynamic updates:

**Before parallel execution:**
```
▎▶ Chatting      ▎ PM-1-DayStatusSummarizer     gpt-5 turn 1
```

**During parallel execution (2+ instances):**
- Parent line switches to "Ready" status to indicate waiting for children
- New lines appear for each instance:
```
▎ Ready          ▎ PM-1-DayStatusSummarizer      ← parent waiting
▎▶ Calling tool  ▎ PM-1-DayStatusSummarizer[1]   tg-ro (list_messages)
▎▶ Chatting      ▎ PM-1-DayStatusSummarizer[2]   gpt-5 turn 2
```

**Key implementation details:**
- Each instance gets unique agent_name: `OriginalName[instance_number]`
- Both child._name and child._aggregator.agent_name are updated for correct progress routing
- Tool progress events (CALLING_TOOL) use instance name, not parent name
- Each instance shows independent status: Chatting, Calling tool, turn count

**As each instance completes:**
- Instance line disappears immediately (task.visible = False in finally block)
- Other instances continue showing their independent progress
- No "stuck" status lines after completion

**After all parallel executions complete:**
- All instance lines hidden
- Parent line returns to normal agent lifecycle  
- Original agent names and display configs restored

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
- **Name modification timing**: Agent names are modified in a wrapper coroutine that
  executes at task runtime, not task creation time, to avoid race conditions
- **Original name caching**: Store original names before ANY modifications to prevent
  [1][2] bugs when the same agent is called multiple times
- **Progress event routing**: Must update both agent._name and agent._aggregator.agent_name
  since MCPAggregator caches agent_name for progress events
- **Display suppression with reference counting**: Multiple parallel instances of the same
  child agent share a single agent object. Use reference counting to track active instances:
  - `_display_suppression_count[child_id]`: Count of active parallel instances
  - `_original_display_configs[child_id]`: Stored original config
  - Only modify display config when first instance starts (count 0→1)
  - Only restore display config when last instance completes (count 1→0)
  - Prevents race condition where early-finishing instances restore config while others run
- **Instance line visibility**: Each instance line is hidden immediately in the task's
  finally block, not after all tasks complete. Uses consistent progress_display singleton
  reference to ensure visibility changes work correctly
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
from fast_agent.agents.tool_agent import ToolAgent
from fast_agent.core.logging.logger import get_logger
from fast_agent.core.prompt import Prompt
from fast_agent.mcp.helpers.content_helpers import get_text, is_text_content, text_content
from fast_agent.ui.message_primitives import MessageType
from fast_agent.types import PromptMessageExtended, RequestParams

logger = get_logger(__name__)


class AgentsAsToolsAgent(ToolAgent):
    """
    An agent that makes each child agent available as an MCP Tool to the parent LLM.

    - list_tools() advertises one tool per child agent
    - call_tool() routes execution to the corresponding child agent
    - run_tools() is overridden to process multiple tool calls in parallel
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
            config: Agent configuration
            agents: List of child agents to expose as tools
            context: Optional context for agent execution
            **kwargs: Additional arguments passed to ToolAgent
        """
        # Initialize as a ToolAgent but without local FastMCP tools; we'll override list_tools
        super().__init__(config=config, tools=[], context=context)
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
        """List all available tools (one per child agent).
        
        Returns:
            ListToolsResult containing tool schemas for all child agents
        """
        tools: list[Tool] = []
        for tool_name, agent in self._child_agents.items():
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
        return ListToolsResult(tools=tools)

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> CallToolResult:
        """Execute a child agent by name.
        
        Args:
            name: Tool name (agent name with prefix)
            arguments: Optional arguments to pass to the child agent
            
        Returns:
            CallToolResult containing the child agent's response
        """
        child = self._child_agents.get(name) or self._child_agents.get(self._make_tool_name(name))
        if child is None:
            return CallToolResult(content=[text_content(f"Unknown agent-tool: {name}")], isError=True)

        args = arguments or {}
        if isinstance(args.get("text"), str):
            input_text = args["text"]
        elif "json" in args:
            input_text = json.dumps(args["json"], ensure_ascii=False) if isinstance(args["json"], dict) else str(args["json"])
        else:
            input_text = json.dumps(args, ensure_ascii=False) if args else ""

        # Serialize arguments to text input
        child_request = Prompt.user(input_text)
        
        try:
            # Note: Display suppression is now handled in run_tools before parallel execution
            # This ensures all instances use the same suppressed config
            response: PromptMessageExtended = await child.generate([child_request], None)
            # Prefer preserving original content blocks for better UI fidelity
            content_blocks = list(response.content or [])

            # Mark error if error channel contains entries, and surface them
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

        # Show instance count if multiple agents
        instance_count = len([d for d in descriptors if d.get("status") != "error"])
        
        # Show detailed call information for each agent
        for i, desc in enumerate(descriptors, 1):
            tool_name = desc.get("tool", "(unknown)")
            args = desc.get("args", {})
            status = desc.get("status", "pending")
            
            if status == "error":
                continue  # Skip display for error tools, will show in results
            
            # Add individual instance number if multiple
            display_tool_name = tool_name
            if instance_count > 1:
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
                max_item_length=50,  # Increased from 28 to prevent truncation
            )

    def _summarize_result_text(self, result: CallToolResult) -> str:
        for block in result.content or []:
            if is_text_content(block):
                text = (get_text(block) or "").strip()
                if text:
                    text = text.replace("\n", " ")
                    return text[:180] + "…" if len(text) > 180 else text
        return ""

    def _show_parallel_tool_results(
        self, records: list[dict[str, Any]]
    ) -> None:
        """Display tool result panels for parallel agent execution.
        
        Args:
            records: List of result records with descriptor and result data
        """
        if not records:
            return

        instance_count = len(records)
        
        # Show detailed result for each agent
        for i, record in enumerate(records, 1):
            descriptor = record.get("descriptor", {})
            result = record.get("result")
            tool_name = descriptor.get("tool", "(unknown)")
            
            if result:
                # Add individual instance number if multiple
                display_tool_name = tool_name
                if instance_count > 1:
                    display_tool_name = f"{tool_name}[{i}]"
                
                # Show individual tool result with full content
                self.display.show_tool_result(
                    name=self.name,
                    tool_name=display_tool_name,
                    result=result,
                )

    async def run_tools(self, request: PromptMessageExtended) -> PromptMessageExtended:
        """
        Override ToolAgent.run_tools to execute multiple tool calls in parallel.
        """
        if not request.tool_calls:
            logger.warning("No tool calls found in request", data=request)
            return PromptMessageExtended(role="user", tool_results={})

        tool_results: dict[str, CallToolResult] = {}
        tool_loop_error: str | None = None

        # Snapshot available tools for validation and UI
        try:
            listed = await self.list_tools()
            available_tools = [t.name for t in listed.tools]
        except Exception as exc:
            logger.warning(f"Failed to list tools before execution: {exc}")
            available_tools = list(self._child_agents.keys())

        # Build aggregated view of all tool calls
        call_descriptors: list[dict[str, Any]] = []
        descriptor_by_id: dict[str, dict[str, Any]] = {}
        tasks: list[asyncio.Task] = []
        id_list: list[str] = []
        
        for correlation_id, tool_request in request.tool_calls.items():
            tool_name = tool_request.params.name
            tool_args = tool_request.params.arguments or {}

            descriptor = {
                "id": correlation_id,
                "tool": tool_name,
                "args": tool_args,
            }
            call_descriptors.append(descriptor)
            descriptor_by_id[correlation_id] = descriptor

            if tool_name not in available_tools and self._make_tool_name(tool_name) not in available_tools:
                error_message = f"Tool '{tool_name}' is not available"
                tool_results[correlation_id] = CallToolResult(
                    content=[text_content(error_message)], isError=True
                )
                tool_loop_error = tool_loop_error or error_message
                descriptor["status"] = "error"
                continue

            descriptor["status"] = "pending"
            id_list.append(correlation_id)

        # Collect original names and prepare for parallel execution
        pending_count = len(id_list)
        original_names = {}
        instance_map = {}  # Map correlation_id -> (child, instance_name, instance_number)
        suppressed_configs = {}  # Store original configs to restore later
        
        # Build instance map and suppress child progress events
        if pending_count > 1:
            for i, cid in enumerate(id_list, 1):
                tool_name = descriptor_by_id[cid]["tool"]
                child = self._child_agents.get(tool_name) or self._child_agents.get(self._make_tool_name(tool_name))
                if child:
                    # Store original name once
                    if tool_name not in original_names and hasattr(child, '_name'):
                        original_names[tool_name] = child._name
                    
                    # Create instance name
                    original = original_names.get(tool_name, child._name if hasattr(child, '_name') else tool_name)
                    instance_name = f"{original}[{i}]"
                    instance_map[cid] = (child, instance_name, i)
                    
                    # Suppress ALL child output/events to prevent duplicate panel rows
                    child_id = id(child)
                    if child_id not in suppressed_configs:
                        # Store original display, logger, and aggregator logger
                        suppressed_configs[child_id] = {
                            'display': child.display if hasattr(child, 'display') else None,
                            'logger': child.logger if hasattr(child, 'logger') else None,
                            'agg_logger': None
                        }
                        
                        # Also store aggregator logger if it exists
                        if hasattr(child, '_aggregator') and child._aggregator and hasattr(child._aggregator, 'logger'):
                            suppressed_configs[child_id]['agg_logger'] = child._aggregator.logger
                        
                        # Replace with null objects that do nothing
                        class NullDisplay:
                            """A display that suppresses ALL output and events"""
                            def __init__(self):
                                self.config = None
                            def __getattr__(self, name):
                                return lambda *args, **kwargs: None
                        
                        class NullLogger:
                            """A logger that suppresses ALL logging"""
                            def __getattr__(self, name):
                                return lambda *args, **kwargs: None
                        
                        # Replace child's display and logger
                        if hasattr(child, 'display'):
                            child.display = NullDisplay()
                        if hasattr(child, 'logger'):
                            child.logger = NullLogger()
                        
                        # CRITICAL: Also replace aggregator's logger (MCP tools emit progress here)
                        if hasattr(child, '_aggregator') and child._aggregator:
                            child._aggregator.logger = NullLogger()
                        
                        logger.info(f"Replaced display, logger & aggregator logger with null objects for {child._name}")
                    
                    logger.info(f"Mapped {cid} -> {instance_name}")
        
        # Import progress_display at outer scope to ensure same instance  
        from fast_agent.event_progress import ProgressAction, ProgressEvent
        from fast_agent.ui.progress_display import progress_display as outer_progress_display
        
        # Simple wrapper - NO renaming, just call the tool
        # Instance numbers already shown in display headers via _show_parallel_tool_calls
        async def call_with_instance_name(correlation_id: str, tool_name: str, tool_args: dict[str, Any]) -> CallToolResult:
            instance_info = instance_map.get(correlation_id)
            
            if instance_info:
                _, instance_name, _ = instance_info
                logger.info(f"[{instance_name}] Starting parallel execution")
                result = await self.call_tool(tool_name, tool_args)
                logger.info(f"[{instance_name}] Completed parallel execution")
                return result
            else:
                # Single instance - just call normally
                return await self.call_tool(tool_name, tool_args)
        
        # Create tasks with instance-specific wrappers
        for cid in id_list:
            tool_name = descriptor_by_id[cid]["tool"]
            tool_args = descriptor_by_id[cid]["args"]
            tasks.append(asyncio.create_task(call_with_instance_name(cid, tool_name, tool_args)))

        # Show aggregated tool call(s)
        self._show_parallel_tool_calls(call_descriptors)

        # Execute concurrently
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
                    descriptor_by_id[correlation_id]["status"] = "error" if result.isError else "done"

        # Show aggregated result(s)
        ordered_records: list[dict[str, Any]] = []
        for cid in request.tool_calls.keys():
            result = tool_results.get(cid)
            if result is None:
                continue
            descriptor = descriptor_by_id.get(cid, {})
            ordered_records.append({"descriptor": descriptor, "result": result})

        self._show_parallel_tool_results(ordered_records)

        # Restore original display, logger, and aggregator logger
        for child_id, originals in suppressed_configs.items():
            # Find the child agent by id
            for tool_name in original_names.keys():
                child = self._child_agents.get(tool_name) or self._child_agents.get(self._make_tool_name(tool_name))
                if child and id(child) == child_id:
                    if originals.get('display') and hasattr(child, 'display'):
                        child.display = originals['display']
                    if originals.get('logger') and hasattr(child, 'logger'):
                        child.logger = originals['logger']
                    if originals.get('agg_logger') and hasattr(child, '_aggregator') and child._aggregator:
                        child._aggregator.logger = originals['agg_logger']
                    logger.info(f"Restored original display, logger & aggregator logger for {child._name}")
                    break
        
        logger.info(f"Parallel execution complete for {len(id_list)} instances")

        return self._finalize_tool_results(tool_results, tool_loop_error=tool_loop_error)
