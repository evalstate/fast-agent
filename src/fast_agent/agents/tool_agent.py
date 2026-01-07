import asyncio
import time
from contextvars import ContextVar
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
from fast_agent.core.prompt import Prompt
from fast_agent.event_progress import ProgressAction
from fast_agent.interfaces import ToolRunnerHookCapable
from fast_agent.mcp.helpers.content_helpers import text_content
from fast_agent.mcp.tool_execution_handler import ToolExecutionHandler
from fast_agent.tools.elicitation import get_elicitation_fastmcp_tool
from fast_agent.types import LlmStopReason, PromptMessageExtended, RequestParams, ToolTimingInfo
from fast_agent.utils.async_utils import gather_with_cancel

logger = get_logger(__name__)

_tool_progress_context: ContextVar[tuple[ToolExecutionHandler, str] | None] = ContextVar(
    "tool_progress_context",
    default=None,
)


class _ToolLoopProgressEmitter:
    def __init__(self, handler: ToolExecutionHandler, agent_name: str) -> None:
        self._handler = handler
        self._agent_name = agent_name
        self._tool_call_id: str | None = None
        self._step = 0
        self._finished = False
        self._lock = asyncio.Lock()

    async def _ensure_started(self) -> str | None:
        if self._tool_call_id:
            return self._tool_call_id
        try:
            self._tool_call_id = await self._handler.on_tool_start(
                "agent_loop", self._agent_name, None
            )
        except Exception:
            self._tool_call_id = None
        return self._tool_call_id

    async def step(self, label: str) -> None:
        async with self._lock:
            if self._finished:
                return
            self._step += 1
            tool_call_id = await self._ensure_started()
            if not tool_call_id:
                return
            message = f"step {self._step}"
            if label:
                message = f"{message} ({label})"
            try:
                await self._handler.on_tool_progress(tool_call_id, float(self._step), None, message)
            except Exception:
                pass

    async def finish(self, success: bool, error: str | None = None) -> None:
        async with self._lock:
            if self._finished:
                return
            self._finished = True
            if not self._tool_call_id:
                return
            try:
                await self._handler.on_tool_complete(self._tool_call_id, success, None, error)
            except Exception:
                pass


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
        self._agent_tools: dict[str, LlmAgent] = {}
        self.tool_runner_hooks: ToolRunnerHooks | None = None

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

    def _clone_constructor_kwargs(self) -> dict[str, Any]:
        """Carry local tool definitions into detached clones."""
        if not self._execution_tools:
            return {}
        return {"tools": list(self._execution_tools.values())}

    def add_tool(self, tool: FastMCPTool, *, replace: bool = True) -> None:
        """Register a new execution tool and expose it to the LLM."""
        name = tool.name
        if not replace and name in self._execution_tools:
            raise ValueError(f"Tool '{name}' already exists")

        self._execution_tools[name] = tool
        self._tool_schemas = [schema for schema in self._tool_schemas if schema.name != name]
        self._tool_schemas.append(
            Tool(
                name=tool.name,
                description=tool.description,
                inputSchema=tool.parameters,
            )
        )

    def add_agent_tool(
        self,
        child: LlmAgent,
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> str:
        """Expose another agent as a tool on this agent."""
        tool_name = name or f"agent__{child.name}"
        if not description:
            config = getattr(child, "config", None)
            description = getattr(config, "description", None) or getattr(
                child, "instruction", None
            )
        tool_description = description or f"Send a message to the {child.name} agent"
        self._agent_tools[tool_name] = child

        async def call_agent(message: str) -> str:
            """Message to send to the child agent."""
            input_text = message
            clone = await child.spawn_detached_instance(name=f"{child.name}[tool]")
            progress_step = 0

            async def emit_progress(label: str | None = None) -> None:
                nonlocal progress_step
                progress_step += 1
                message = f"{child.name} step {progress_step}"
                if label:
                    message = f"{message} ({label})"

                ctx = _tool_progress_context.get()
                if ctx:
                    handler, tool_call_id = ctx
                    try:
                        await handler.on_tool_progress(
                            tool_call_id, float(progress_step), None, message
                        )
                    except Exception:
                        pass

                logger.info(
                    "Agent tool progress",
                    data={
                        "progress_action": ProgressAction.TOOL_PROGRESS,
                        "agent_name": self.name,
                        "progress": progress_step,
                        "total": None,
                        "details": message,
                    },
                )

            hooks_set = False
            if isinstance(clone, ToolAgent):
                existing_hooks = getattr(clone, "tool_runner_hooks", None)
                before_llm_call = existing_hooks.before_llm_call if existing_hooks else None
                before_tool_call = existing_hooks.before_tool_call if existing_hooks else None
                after_llm_call = existing_hooks.after_llm_call if existing_hooks else None
                after_tool_call = existing_hooks.after_tool_call if existing_hooks else None

                async def handle_before_llm_call(runner, messages):
                    if before_llm_call:
                        await before_llm_call(runner, messages)
                    await emit_progress("llm")

                async def handle_before_tool_call(runner, message):
                    if before_tool_call:
                        await before_tool_call(runner, message)
                    await emit_progress("tool")

                clone.tool_runner_hooks = ToolRunnerHooks(
                    before_llm_call=handle_before_llm_call,
                    after_llm_call=after_llm_call,
                    before_tool_call=handle_before_tool_call,
                    after_tool_call=after_tool_call,
                )
                hooks_set = True

            try:
                if not hooks_set:
                    await emit_progress("run")
                clone.load_message_history([])
                response = await clone.generate([Prompt.user(input_text)], None)
                return response.last_text() or ""
            finally:
                try:
                    await clone.shutdown()
                except Exception as exc:
                    logger.warning(f"Error shutting down tool clone for {child.name}: {exc}")
                try:
                    child.merge_usage_from(clone)
                except Exception as exc:
                    logger.warning(f"Failed to merge tool clone usage for {child.name}: {exc}")

        fast_tool = FastMCPTool.from_function(
            call_agent,
            name=tool_name,
            description=tool_description,
        )
        self.add_tool(fast_tool)
        return tool_name

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
            hooks=self._build_tool_runner_hooks(request_params),
        )
        return await runner.until_done()

    def _tool_runner_hooks(self) -> ToolRunnerHooks | None:
        if isinstance(self, ToolRunnerHookCapable):
            return self.tool_runner_hooks
        return None

    def _build_tool_runner_hooks(
        self, request_params: RequestParams | None
    ) -> ToolRunnerHooks | None:
        base_hooks = self._tool_runner_hooks()
        if (
            request_params is None
            or not request_params.emit_loop_progress
            or not request_params.tool_execution_handler
        ):
            return base_hooks

        progress_hooks = self._build_loop_progress_hooks(
            request_params.tool_execution_handler
        )
        return self._merge_tool_runner_hooks(base_hooks, progress_hooks)

    def _build_loop_progress_hooks(
        self, handler: ToolExecutionHandler
    ) -> ToolRunnerHooks:
        emitter = _ToolLoopProgressEmitter(handler, self.name)
        error_reasons = (
            LlmStopReason.ERROR.value,
            LlmStopReason.CANCELLED.value,
            LlmStopReason.TIMEOUT.value,
            LlmStopReason.SAFETY.value,
        )

        def tool_label(request: PromptMessageExtended) -> str:
            tool_calls = request.tool_calls or {}
            names = [call.params.name for call in tool_calls.values()]
            if len(names) == 1:
                return f"tool {names[0]}"
            if len(names) > 1:
                return f"tools x{len(names)}"
            return "tool"

        async def before_llm_call(runner, messages):
            await emitter.step("llm")

        async def before_tool_call(runner, request):
            await emitter.step(tool_label(request))

        async def after_llm_call(runner, message):
            if message.stop_reason == LlmStopReason.TOOL_USE:
                return
            stop_reason = message.stop_reason
            if stop_reason in error_reasons:
                if isinstance(stop_reason, LlmStopReason):
                    reason_label = stop_reason.value
                else:
                    reason_label = str(stop_reason) if stop_reason is not None else "unknown"
                await emitter.finish(False, error=f"stopped: {reason_label}")
            else:
                await emitter.finish(True)

        return ToolRunnerHooks(
            before_llm_call=before_llm_call,
            after_llm_call=after_llm_call,
            before_tool_call=before_tool_call,
        )

    @staticmethod
    def _merge_tool_runner_hooks(
        base: ToolRunnerHooks | None, extra: ToolRunnerHooks | None
    ) -> ToolRunnerHooks | None:
        if base is None:
            return extra
        if extra is None:
            return base

        def merge(one, two):
            if one is None:
                return two
            if two is None:
                return one

            async def merged(runner, payload):
                await one(runner, payload)
                await two(runner, payload)

            return merged

        return ToolRunnerHooks(
            before_llm_call=merge(base.before_llm_call, extra.before_llm_call),
            after_llm_call=merge(base.after_llm_call, extra.after_llm_call),
            before_tool_call=merge(base.before_tool_call, extra.before_tool_call),
            after_tool_call=merge(base.after_tool_call, extra.after_tool_call),
        )

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

    async def run_tools(
        self,
        request: PromptMessageExtended,
        request_params: RequestParams | None = None,
    ) -> PromptMessageExtended:
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
                result = await self.call_tool(
                    tool_name, tool_args, request_params=request_params
                )
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
            result = await self.call_tool(
                tool_name, tool_args, request_params=request_params
            )
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

    async def call_tool(
        self,
        name: str,
        arguments: Dict[str, Any] | None = None,
        tool_use_id: str | None = None,
        *,
        request_params: RequestParams | None = None,
    ) -> CallToolResult:
        """Execute a tool by name using local FastMCP tools. Overridable by subclasses."""
        fast_tool = self._execution_tools.get(name)
        if not fast_tool:
            logger.warning(f"Unknown tool: {name}")
            return CallToolResult(
                content=[text_content(f"Unknown tool: {name}")],
                isError=True,
            )

        tool_handler = self._get_tool_handler(request_params)
        tool_call_id = None
        if tool_handler:
            try:
                tool_call_id = await tool_handler.on_tool_start(
                    name, "local", arguments, tool_use_id
                )
            except Exception:
                tool_call_id = None

        token = None
        if tool_handler and tool_call_id:
            token = _tool_progress_context.set((tool_handler, tool_call_id))

        try:
            result = await fast_tool.run(arguments or {}, convert_result=False)
            tool_result = CallToolResult(
                content=[text_content(str(result))],
                isError=False,
            )
            if tool_handler and tool_call_id:
                try:
                    await tool_handler.on_tool_complete(
                        tool_call_id, True, tool_result.content, None
                    )
                except Exception:
                    pass
            return tool_result
        except Exception as e:
            logger.error(f"Tool {name} failed: {e}")
            tool_result = CallToolResult(
                content=[text_content(f"Error: {str(e)}")],
                isError=True,
            )
            if tool_handler and tool_call_id:
                try:
                    await tool_handler.on_tool_complete(tool_call_id, False, None, str(e))
                except Exception:
                    pass
            return tool_result
        finally:
            if token is not None:
                _tool_progress_context.reset(token)

    def _get_tool_handler(
        self, request_params: RequestParams | None = None
    ) -> ToolExecutionHandler | None:
        if request_params and request_params.tool_execution_handler:
            return request_params.tool_execution_handler
        context = getattr(self, "_context", None)
        acp = getattr(context, "acp", None) if context else None
        if acp is not None:
            progress_manager = getattr(acp, "progress_manager", None)
            if progress_manager is not None:
                return progress_manager
        return None
