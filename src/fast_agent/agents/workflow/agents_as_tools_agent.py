from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from mcp import ListToolsResult, Tool
from mcp.types import CallToolResult

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.llm_agent import LlmAgent
from fast_agent.agents.tool_agent import ToolAgent
from fast_agent.core.logging.logger import get_logger
from fast_agent.core.prompt import Prompt
from fast_agent.mcp.helpers.content_helpers import text_content
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
        agents: List[LlmAgent],
        context: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        # Initialize as a ToolAgent but without local FastMCP tools; we'll override list_tools
        super().__init__(config=config, tools=[], context=context)
        self._child_agents: Dict[str, LlmAgent] = {}
        self._tool_names: List[str] = []

        # Build tool name mapping for children
        for child in agents:
            tool_name = self._make_tool_name(child.name)
            if tool_name in self._child_agents:
                logger.warning(
                    f"Duplicate tool name '{tool_name}' for child agent '{child.name}', overwriting"
                )
            self._child_agents[tool_name] = child
            self._tool_names.append(tool_name)

    def _make_tool_name(self, child_name: str) -> str:
        # Use a distinct prefix to avoid collisions with MCP tools
        return f"agent__{child_name}"

    async def initialize(self) -> None:
        await super().initialize()
        # Initialize all child agents
        for agent in self._child_agents.values():
            if not getattr(agent, "initialized", False):
                await agent.initialize()

    async def shutdown(self) -> None:
        await super().shutdown()
        # Shutdown children, but do not fail the parent if any child errors
        for agent in self._child_agents.values():
            try:
                await agent.shutdown()
            except Exception as e:
                logger.warning(f"Error shutting down child agent {agent.name}: {e}")

    async def list_tools(self) -> ListToolsResult:
        # Dynamically advertise one tool per child agent
        tools: List[Tool] = []
        for tool_name, agent in self._child_agents.items():
            # Minimal permissive schema: accept either plain text or arbitrary JSON
            input_schema: Dict[str, Any] = {
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

    async def call_tool(self, name: str, arguments: Dict[str, Any] | None = None) -> CallToolResult:
        # Route the call to the corresponding child agent
        child = self._child_agents.get(name)
        if child is None:
            # Fallback: try to resolve without prefix in case the LLM omitted it
            alt = self._child_agents.get(self._make_tool_name(name))
            if alt is not None:
                child = alt
        if child is None:
            return CallToolResult(content=[text_content(f"Unknown agent-tool: {name}")], isError=True)

        args = arguments or {}
        # Prefer explicit text; otherwise serialize json; otherwise serialize entire dict
        input_text: str
        if isinstance(args.get("text"), str):
            input_text = args["text"]
        else:
            import json

            if "json" in args:
                try:
                    input_text = json.dumps(args["json"], ensure_ascii=False)
                except Exception:
                    input_text = str(args["json"])
            else:
                try:
                    input_text = json.dumps(args, ensure_ascii=False)
                except Exception:
                    input_text = str(args)

        # Build a single-user message to the child and execute
        child_request = Prompt.user(input_text)
        try:
            # We do not override child's request_params; pass None to use child's defaults
            response: PromptMessageExtended = await child.generate([child_request], None)
            return CallToolResult(
                content=[text_content(response.all_text() or "")],
                isError=False,
            )
        except Exception as e:
            logger.error(f"Child agent {child.name} failed: {e}")
            return CallToolResult(content=[text_content(f"Error: {e}")], isError=True)

    async def run_tools(self, request: PromptMessageExtended) -> PromptMessageExtended:
        """
        Override ToolAgent.run_tools to execute multiple tool calls in parallel.
        """
        if not request.tool_calls:
            logger.warning("No tool calls found in request", data=request)
            return PromptMessageExtended(role="user", tool_results={})

        tool_results: Dict[str, CallToolResult] = {}
        tool_loop_error: str | None = None

        # Snapshot available tools for validation and UI
        try:
            listed = await self.list_tools()
            available_tools = [t.name for t in listed.tools]
        except Exception as exc:
            logger.warning(f"Failed to list tools before execution: {exc}")
            available_tools = list(self._child_agents.keys())

        # Build tasks for parallel execution
        tasks: List[asyncio.Task] = []
        id_list: List[str] = []
        for correlation_id, tool_request in request.tool_calls.items():
            tool_name = tool_request.params.name
            tool_args = tool_request.params.arguments or {}

            if tool_name not in available_tools and self._make_tool_name(tool_name) not in available_tools:
                # Mark error in results but continue other tools
                error_message = f"Tool '{tool_name}' is not available"
                tool_results[correlation_id] = CallToolResult(
                    content=[text_content(error_message)], isError=True
                )
                tool_loop_error = tool_loop_error or error_message
                continue

            # UI: show planned tool call
            try:
                highlight_index = available_tools.index(tool_name)
            except ValueError:
                highlight_index = None
            self.display.show_tool_call(
                name=self.name,
                tool_args=tool_args,
                bottom_items=available_tools,
                tool_name=tool_name,
                highlight_index=highlight_index,
                max_item_length=12,
            )

            # Schedule execution
            id_list.append(correlation_id)
            tasks.append(asyncio.create_task(self.call_tool(tool_name, tool_args)))

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
                else:
                    tool_results[correlation_id] = result

        # UI: show results
        for cid, res in tool_results.items():
            # Try to infer the name shown in UI
            try:
                tool_name = request.tool_calls[cid].params.name
            except Exception:
                tool_name = None
            self.display.show_tool_result(name=self.name, result=res, tool_name=tool_name)

        return self._finalize_tool_results(tool_results, tool_loop_error=tool_loop_error)
