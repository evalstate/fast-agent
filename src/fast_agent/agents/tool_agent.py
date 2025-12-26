import time
from typing import Any, Callable, Dict, List, Sequence

from mcp.server.fastmcp.tools.base import Tool as FastMCPTool
from mcp.types import CallToolResult, ListToolsResult, Tool

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.llm_agent import LlmAgent
from fast_agent.agents.tool_runner import ToolRunner, ToolRunnerHooks, _ToolLoopAgent
from fast_agent.constants import (
    FAST_AGENT_ERROR_CHANNEL,
    FORCE_SEQUENTIAL_TOOL_CALLS,
    HUMAN_INPUT_TOOL_NAME,
)
from fast_agent.context import Context
from fast_agent.core.logging.logger import get_logger
from fast_agent.interfaces import ToolRunnerHookCapable
from fast_agent.mcp.helpers.content_helpers import text_content
from fast_agent.tools.elicitation import get_elicitation_fastmcp_tool
from fast_agent.types import PromptMessageExtended, RequestParams, ToolTimingInfo
from fast_agent.utils.async_utils import gather_with_cancel

logger = get_logger(__name__)


class ToolAgent(LlmAgent, _ToolLoopAgent):
    """
    A Tool Calling agent that uses FastMCP Tools for execution.

    Pass either:
    - FastMCP Tool objects (created via Tool.from_function)
    - Regular Python functions (will be wrapped as FastMCP Tools)
    """

    def __init__(
        self,
        config: AgentConfig,
        tools: Sequence[FastMCPTool | Callable] = [],
        context: Context | None = None,
    ) -> None:
        super().__init__(config=config, context=context)

        self._execution_tools: dict[str, FastMCPTool] = {}
        self._tool_schemas: list[Tool] = []

        # Build a working list of tools and auto-inject human-input tool if missing
        working_tools: list[FastMCPTool | Callable] = list(tools) if tools else []
        # Only auto-inject if enabled via AgentConfig
        if self.config.human_input:
            existing_names = {
                t.name if isinstance(t, FastMCPTool) else getattr(t, "__name__", "")
                for t in working_tools
            }
            if HUMAN_INPUT_TOOL_NAME not in existing_names:
                try:
                    working_tools.append(get_elicitation_fastmcp_tool())
                except Exception as e:
                    logger.warning(f"Failed to initialize human-input tool: {e}")

        for tool in working_tools:
            if isinstance(tool, FastMCPTool):
                fast_tool = tool
            elif callable(tool):
                fast_tool = FastMCPTool.from_function(tool)
            else:
                logger.warning(f"Skipping unknown tool type: {type(tool)}")
                continue

            self._execution_tools[fast_tool.name] = fast_tool
            # Create MCP Tool schema for the LLM interface
            self._tool_schemas.append(
                Tool(
                    name=fast_tool.name,
                    description=fast_tool.description,
                    inputSchema=fast_tool.parameters,
                )
            )

    async def generate_impl(
        self,
        messages: List[PromptMessageExtended],
        request_params: RequestParams | None = None,
        tools: List[Tool] | None = None,
    ) -> PromptMessageExtended:
        """
        Generate a response using the LLM, and handle tool calls if necessary.
        Messages are already normalized to List[PromptMessageExtended].
        """
        if tools is None:
            tools = (await self.list_tools()).tools

        runner = ToolRunner(
            agent=self,
            messages=messages,
            request_params=request_params,
            tools=tools,
            hooks=self._tool_runner_hooks(),
        )
        return await runner.until_done()

    def _tool_runner_hooks(self) -> ToolRunnerHooks | None:
        if isinstance(self, ToolRunnerHookCapable):
            return self.tool_runner_hooks
        return None

    async def _tool_runner_llm_step(
        self,
        messages: list[PromptMessageExtended],
        request_params: RequestParams | None = None,
        tools: list[Tool] | None = None,
    ) -> PromptMessageExtended:
        return await super().generate_impl(messages, request_params=request_params, tools=tools)

    # we take care of tool results, so skip displaying them
    def show_user_message(self, message: PromptMessageExtended) -> None:
        if message.tool_results:
            return
        super().show_user_message(message)

    async def run_tools(self, request: PromptMessageExtended) -> PromptMessageExtended:
        """Runs the tools in the request, and returns a new User message with the results"""
        if not request.tool_calls:
            logger.warning("No tool calls found in request", data=request)
            return PromptMessageExtended(role="user", tool_results={})

        tool_results: dict[str, CallToolResult] = {}
        tool_timings: dict[str, ToolTimingInfo] = {}
        tool_loop_error: str | None = None
        tool_schemas = (await self.list_tools()).tools
        available_tools = [t.name for t in tool_schemas]

        tool_call_items = list(request.tool_calls.items())
        should_parallel = (not FORCE_SEQUENTIAL_TOOL_CALLS) and len(tool_call_items) > 1

        planned_calls: list[tuple[str, str, dict[str, Any]]] = []
        for correlation_id, tool_request in tool_call_items:
            tool_name = tool_request.params.name
            tool_args = tool_request.params.arguments or {}

            if tool_name not in available_tools and tool_name not in self._execution_tools:
                error_message = f"Tool '{tool_name}' is not available"
                logger.error(error_message)
                tool_loop_error = self._mark_tool_loop_error(
                    correlation_id=correlation_id,
                    error_message=error_message,
                    tool_results=tool_results,
                )
                break
            planned_calls.append((correlation_id, tool_name, tool_args))

        if should_parallel and planned_calls:
            for correlation_id, tool_name, tool_args in planned_calls:
                highlight_index = None
                try:
                    highlight_index = available_tools.index(tool_name)
                except ValueError:
                    pass

                self.display.show_tool_call(
                    name=self.name,
                    tool_args=tool_args,
                    bottom_items=available_tools,
                    tool_name=tool_name,
                    highlight_index=highlight_index,
                    max_item_length=12,
                )

            async def run_one(
                correlation_id: str, tool_name: str, tool_args: dict[str, Any]
            ) -> tuple[str, CallToolResult, float]:
                start_time = time.perf_counter()
                result = await self.call_tool(tool_name, tool_args)
                end_time = time.perf_counter()
                return correlation_id, result, round((end_time - start_time) * 1000, 2)

            results = await gather_with_cancel(
                run_one(cid, name, args) for cid, name, args in planned_calls
            )

            for i, item in enumerate(results):
                correlation_id, tool_name, _ = planned_calls[i]
                if isinstance(item, BaseException):
                    msg = f"Error: {str(item)}"
                    result = CallToolResult(content=[text_content(msg)], isError=True)
                    duration_ms = 0.0
                else:
                    _, result, duration_ms = item

                tool_results[correlation_id] = result
                tool_timings[correlation_id] = ToolTimingInfo(
                    timing_ms=duration_ms,
                    transport_channel=None,
                )
                self.display.show_tool_result(
                    name=self.name, result=result, tool_name=tool_name, timing_ms=duration_ms
                )

            return self._finalize_tool_results(
                tool_results, tool_timings=tool_timings, tool_loop_error=tool_loop_error
            )

        for correlation_id, tool_name, tool_args in planned_calls:
            # Find the index of the current tool in available_tools for highlighting
            highlight_index = None
            try:
                highlight_index = available_tools.index(tool_name)
            except ValueError:
                # Tool not found in list, no highlighting
                pass

            self.display.show_tool_call(
                name=self.name,
                tool_args=tool_args,
                bottom_items=available_tools,
                tool_name=tool_name,
                highlight_index=highlight_index,
                max_item_length=12,
            )

            # Track timing for tool execution
            start_time = time.perf_counter()
            result = await self.call_tool(tool_name, tool_args)
            end_time = time.perf_counter()
            duration_ms = round((end_time - start_time) * 1000, 2)

            tool_results[correlation_id] = result
            # Store timing info (transport_channel not available for local tools)
            tool_timings[correlation_id] = ToolTimingInfo(
                timing_ms=duration_ms,
                transport_channel=None,
            )
            self.display.show_tool_result(
                name=self.name, result=result, tool_name=tool_name, timing_ms=duration_ms
            )

        return self._finalize_tool_results(
            tool_results, tool_timings=tool_timings, tool_loop_error=tool_loop_error
        )

    def _mark_tool_loop_error(
        self,
        *,
        correlation_id: str,
        error_message: str,
        tool_results: dict[str, CallToolResult],
    ) -> str:
        error_result = CallToolResult(
            content=[text_content(error_message)],
            isError=True,
        )
        tool_results[correlation_id] = error_result
        self.display.show_tool_result(name=self.name, result=error_result)
        return error_message

    def _finalize_tool_results(
        self,
        tool_results: dict[str, CallToolResult],
        *,
        tool_timings: dict[str, ToolTimingInfo] | None = None,
        tool_loop_error: str | None = None,
    ) -> PromptMessageExtended:
        import json

        from mcp.types import TextContent

        from fast_agent.constants import FAST_AGENT_TOOL_TIMING

        channels = None
        content = []
        if tool_loop_error:
            content.append(text_content(tool_loop_error))
            channels = {
                FAST_AGENT_ERROR_CHANNEL: [text_content(tool_loop_error)],
            }

        # Add tool timing data to channels
        if tool_timings:
            if channels is None:
                channels = {}
            channels[FAST_AGENT_TOOL_TIMING] = [
                TextContent(type="text", text=json.dumps(tool_timings))
            ]

        return PromptMessageExtended(
            role="user",
            content=content,
            tool_results=tool_results,
            channels=channels,
        )

    async def list_tools(self) -> ListToolsResult:
        """Return available tools for this agent. Overridable by subclasses."""
        return ListToolsResult(tools=list(self._tool_schemas))

    async def call_tool(self, name: str, arguments: Dict[str, Any] | None = None) -> CallToolResult:
        """Execute a tool by name using local FastMCP tools. Overridable by subclasses."""
        fast_tool = self._execution_tools.get(name)
        if not fast_tool:
            logger.warning(f"Unknown tool: {name}")
            return CallToolResult(
                content=[text_content(f"Unknown tool: {name}")],
                isError=True,
            )

        try:
            result = await fast_tool.run(arguments or {}, convert_result=False)
            return CallToolResult(
                content=[text_content(str(result))],
                isError=False,
            )
        except Exception as e:
            logger.error(f"Tool {name} failed: {e}")
            return CallToolResult(
                content=[text_content(f"Error: {str(e)}")],
                isError=True,
            )
