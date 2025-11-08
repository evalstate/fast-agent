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
            # Suppress only child agent chat messages (keep tool calls visible)
            original_config = None
            if hasattr(child, 'display') and child.display and child.display.config:
                original_config = child.display.config
                temp_config = copy(original_config)
                if hasattr(temp_config, 'logger'):
                    temp_logger = copy(temp_config.logger)
                    temp_logger.show_chat = False
                    temp_logger.show_tools = True  # Explicitly keep tools visible
                    temp_config.logger = temp_logger
                    child.display.config = temp_config
            
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
        finally:
            # Restore original config
            if original_config and hasattr(child, 'display') and child.display:
                child.display.config = original_config

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
        
        # Build bottom items with unique instance numbers if multiple
        bottom_items: list[str] = []
        for i, desc in enumerate(descriptors, 1):
            tool_label = desc.get("tool", "(unknown)")
            status = desc.get("status", "pending")
            status_label = status_labels.get(status, status)
            if instance_count > 1:
                bottom_items.append(f"{tool_label}[{i}] · {status_label}")
            else:
                bottom_items.append(f"{tool_label} · {status_label}")
        
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
            
            # Show individual tool call with arguments
            self.display.show_tool_call(
                name=self.name,
                tool_name=display_tool_name,
                tool_args=args,
                bottom_items=bottom_items,
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

        # Collect original names
        pending_count = len(id_list)
        original_names = {}
        if pending_count > 1:
            for cid in id_list:
                tool_name = descriptor_by_id[cid]["tool"]
                child = self._child_agents.get(tool_name) or self._child_agents.get(self._make_tool_name(tool_name))
                if child and hasattr(child, '_name') and tool_name not in original_names:
                    original_names[tool_name] = child._name
        
        # Create wrapper coroutine that sets name and emits progress for instance
        async def call_with_instance_name(tool_name: str, tool_args: dict[str, Any], instance: int) -> CallToolResult:
            from fast_agent.event_progress import ProgressAction, ProgressEvent
            from fast_agent.ui.progress_display import progress_display
            
            instance_name = None
            if pending_count > 1:
                child = self._child_agents.get(tool_name) or self._child_agents.get(self._make_tool_name(tool_name))
                if child and hasattr(child, '_name'):
                    original = original_names.get(tool_name, child._name)
                    instance_name = f"{original}[{instance}]"
                    child._name = instance_name
                    
                    # Also update aggregator's agent_name so tool progress events use instance name
                    if hasattr(child, '_aggregator') and child._aggregator:
                        child._aggregator.agent_name = instance_name
                    
                    # Emit progress event to create separate line in progress panel
                    progress_display.update(ProgressEvent(
                        action=ProgressAction.CHATTING,
                        target=instance_name,
                        details="",
                        agent_name=instance_name
                    ))
            
            return await self.call_tool(tool_name, tool_args)
        
        # Hide parent agent lines while instances run
        if pending_count > 1:
            from fast_agent.ui.progress_display import progress_display
            
            for tool_name in original_names.keys():
                original = original_names[tool_name]
                # Hide parent line from progress panel
                if original in progress_display._taskmap:
                    task_id = progress_display._taskmap[original]
                    for task in progress_display._progress.tasks:
                        if task.id == task_id:
                            task.visible = False
                            break
        
        # Create tasks with instance-specific wrappers
        for i, cid in enumerate(id_list, 1):
            tool_name = descriptor_by_id[cid]["tool"]
            tool_args = descriptor_by_id[cid]["args"]
            tasks.append(asyncio.create_task(call_with_instance_name(tool_name, tool_args, i)))

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

        # Restore original agent names and hide instance lines from progress panel
        if pending_count > 1:
            from fast_agent.ui.progress_display import progress_display
            
            for tool_name, original_name in original_names.items():
                child = self._child_agents.get(tool_name) or self._child_agents.get(self._make_tool_name(tool_name))
                if child:
                    child._name = original_name
                    # Restore aggregator's agent_name too
                    if hasattr(child, '_aggregator') and child._aggregator:
                        child._aggregator.agent_name = original_name
                
                # Show parent line again and hide instance lines
                if original_name in progress_display._taskmap:
                    task_id = progress_display._taskmap[original_name]
                    for task in progress_display._progress.tasks:
                        if task.id == task_id:
                            task.visible = True  # Restore parent line
                            break
                
                # Hide instance lines from progress panel
                for i in range(1, pending_count + 1):
                    instance_name = f"{original_name}[{i}]"
                    if instance_name in progress_display._taskmap:
                        task_id = progress_display._taskmap[instance_name]
                        for task in progress_display._progress.tasks:
                            if task.id == task_id:
                                task.visible = False
                                break
        else:
            # Single instance, just restore name
            for tool_name, original_name in original_names.items():
                child = self._child_agents.get(tool_name) or self._child_agents.get(self._make_tool_name(tool_name))
                if child:
                    child._name = original_name
                    if hasattr(child, '_aggregator') and child._aggregator:
                        child._aggregator.agent_name = original_name

        return self._finalize_tool_results(tool_results, tool_loop_error=tool_loop_error)
