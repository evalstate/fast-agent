from typing import Any, Callable, Dict, List, Sequence

from mcp.server.fastmcp.tools.base import Tool as FastMCPTool
from mcp.types import CallToolResult, ListToolsResult, Tool

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.llm_agent import LlmAgent
from fast_agent.constants import (
    DEFAULT_MAX_ITERATIONS,
    FAST_AGENT_ERROR_CHANNEL,
    HUMAN_INPUT_TOOL_NAME,
)
from fast_agent.context import Context
from fast_agent.core.logging.logger import get_logger
from fast_agent.mcp.helpers.content_helpers import text_content
from fast_agent.tools.elicitation import get_elicitation_fastmcp_tool
from fast_agent.types import PromptMessageExtended, RequestParams
from fast_agent.types.llm_stop_reason import LlmStopReason

logger = get_logger(__name__)


class ToolAgent(LlmAgent):
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
            (tool)
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

        iterations = 0
        max_iterations = request_params.max_iterations if request_params else DEFAULT_MAX_ITERATIONS

        while True:
            result = await super().generate_impl(
                messages, request_params=request_params, tools=tools
            )

            if LlmStopReason.TOOL_USE == result.stop_reason:
                tool_message = await self.run_tools(result)
                error_channel_messages = (tool_message.channels or {}).get(FAST_AGENT_ERROR_CHANNEL)
                if error_channel_messages:
                    tool_result_contents = [
                        content
                        for tool_result in (tool_message.tool_results or {}).values()
                        for content in tool_result.content
                    ]
                    if tool_result_contents:
                        if result.content is None:
                            result.content = []
                        result.content.extend(tool_result_contents)
                    result.stop_reason = LlmStopReason.ERROR
                    break
                if self.config.use_history:
                    messages = [tool_message]
                else:
                    messages.extend([result, tool_message])
            else:
                break

            iterations += 1
            if iterations > max_iterations:
                logger.warning("Max iterations reached, stopping tool loop")
                break
        return result

    # we take care of tool results, so skip displaying them
    def show_user_message(self, message: PromptMessageExtended) -> None:
        if message.tool_results:
            return
        super().show_user_message(message)

    async def run_tools(self, request: PromptMessageExtended) -> PromptMessageExtended:
        """Runs the tools in the request, and returns a new User message with the results"""
        import time

        if not request.tool_calls:
            logger.warning("No tool calls found in request", data=request)
            return PromptMessageExtended(role="user", tool_results={})

        tool_results: dict[str, CallToolResult] = {}
        tool_timings: dict[str, float] = {}  # Track timing for each tool call
        tool_loop_error: str | None = None
        # TODO -- use gather() for parallel results, update display
        tool_schemas = (await self.list_tools()).tools
        available_tools = [t.name for t in tool_schemas]
        for correlation_id, tool_request in request.tool_calls.items():
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
            tool_timings[correlation_id] = duration_ms
            self.display.show_tool_result(name=self.name, result=result, tool_name=tool_name, timing_ms=duration_ms)

        return self._finalize_tool_results(tool_results, tool_timings=tool_timings, tool_loop_error=tool_loop_error)

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
        tool_timings: dict[str, float] | None = None,
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
