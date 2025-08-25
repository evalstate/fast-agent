from typing import Callable, List

from mcp.server.fastmcp.tools.base import Tool as FastMCPTool
from mcp.types import CallToolResult, Tool

from fast_agent.agents.llm_agent import LlmAgent
from fast_agent.context import Context
from fast_agent.types.llm_stop_reason import LlmStopReason
from mcp_agent.core.agent_types import AgentConfig
from mcp_agent.core.request_params import RequestParams
from mcp_agent.logging.logger import get_logger
from mcp_agent.mcp.helpers.content_helpers import text_content
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart

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
        tools: list[FastMCPTool | Callable] = [],
        context: Context | None = None,
    ) -> None:
        super().__init__(config=config, context=context)

        self._execution_tools: dict[str, FastMCPTool] = {}
        self._tool_schemas: list[Tool] = []

        for tool in tools:
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
        messages: List[PromptMessageMultipart],
        request_params: RequestParams | None = None,
        tools: List[Tool] | None = None,
    ) -> PromptMessageMultipart:
        """
        Generate a response using the LLM, and handle tool calls if necessary.
        Messages are already normalized to List[PromptMessageMultipart].
        """
        if tools is None:
            tools = self._tool_schemas

        while True:
            result = await super().generate_impl(
                messages, request_params=request_params, tools=tools
            )

            if LlmStopReason.TOOL_USE == result.stop_reason:
                messages = [await self.run_tools(result)]
            else:
                break

        return result

    # we take care of tool results, so skip displaying them
    def show_user_message(self, message: PromptMessageMultipart) -> None:
        if message.tool_results:
            return
        super().show_user_message(message)

    async def run_tools(self, request: PromptMessageMultipart) -> PromptMessageMultipart:
        """Runs the tools in the request, and returns a new User message with the results"""
        if not request.tool_calls:
            logger.warning("No tool calls found in request", data=request)
            return PromptMessageMultipart(role="user", tool_results={})

        tool_results: dict[str, CallToolResult] = {}
        # TODO -- use gather() for parallel results, update display
        for correlation_id, tool_request in request.tool_calls.items():
            tool_name = tool_request.params.name
            tool_args = tool_request.params.arguments or {}
            self.display.show_tool_call(
                name=self.name,
                tool_args=tool_args,
                available_tools=list(self._execution_tools.keys()),
                tool_name=tool_name,
            )

            fast_tool = self._execution_tools.get(tool_name)
            if fast_tool:
                try:
                    # Use FastMCP's run method without convert_result to get raw output
                    result = await fast_tool.run(tool_args, convert_result=False)

                    # Always wrap result in CallToolResult
                    tool_results[correlation_id] = CallToolResult(
                        content=[text_content(str(result))],
                        isError=False,
                    )

                    logger.debug(f"Tool {tool_name} executed successfully")
                except Exception as e:
                    logger.error(f"Tool {tool_name} failed: {e}")
                    tool_results[correlation_id] = CallToolResult(
                        content=[text_content(f"Error: {str(e)}")],
                        isError=True,
                    )
            else:
                logger.warning(f"Unknown tool: {tool_name}")
                tool_results[correlation_id] = CallToolResult(
                    content=[text_content(f"Unknown tool: {tool_name}")],
                    isError=True,
                )

            self.display.show_tool_result(name=self.name, result=tool_results[correlation_id])

        return PromptMessageMultipart(role="user", tool_results=tool_results)
