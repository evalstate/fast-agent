from typing import Any, Callable, Dict, List

from mcp.server.fastmcp.tools.base import Tool as FastMCPTool
from mcp.types import CallToolResult, ListToolsResult, Tool

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.llm_agent import LlmAgent
from fast_agent.constants import HUMAN_INPUT_TOOL_NAME
from fast_agent.context import Context
from fast_agent.mcp.helpers.content_helpers import text_content
from fast_agent.tools.elicitation import get_elicitation_fastmcp_tool
from fast_agent.types import PromptMessageExtended, RequestParams
from fast_agent.types.llm_stop_reason import LlmStopReason
from mcp_agent.logging.logger import get_logger

logger = get_logger(__name__)

DEFAULT_MAX_TOOL_CALLS = 20


# should we have MAX_TOOL_CALLS instead to constrain by number of tools rather than turns...?
DEFAULT_MAX_ITERATIONS = 20
"""Maximum number of User/Assistant turns to take"""


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
        tools: list[FastMCPTool | Callable] = [],
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

        iterations = 0

        while True:
            result = await super().generate_impl(
                messages, request_params=request_params, tools=tools
            )

            if LlmStopReason.TOOL_USE == result.stop_reason:
                messages = [await self.run_tools(result)]
            else:
                break

            iterations += 1
            if iterations > DEFAULT_MAX_ITERATIONS:
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
        if not request.tool_calls:
            logger.warning("No tool calls found in request", data=request)
            return PromptMessageExtended(role="user", tool_results={})

        tool_results: dict[str, CallToolResult] = {}
        # TODO -- use gather() for parallel results, update display
        available_tools = [t.name for t in (await self.list_tools()).tools]
        for correlation_id, tool_request in request.tool_calls.items():
            tool_name = tool_request.params.name
            tool_args = tool_request.params.arguments or {}
            self.display.show_tool_call(
                name=self.name,
                tool_args=tool_args,
                bottom_items=available_tools,
                tool_name=tool_name,
                max_item_length=12,
            )

            # Delegate to call_tool for execution (overridable by subclasses)
            result = await self.call_tool(tool_name, tool_args)
            tool_results[correlation_id] = result
            self.display.show_tool_result(name=self.name, result=result)

        return PromptMessageExtended(role="user", tool_results=tool_results)

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
