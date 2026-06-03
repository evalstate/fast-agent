import asyncio
import json
import time
from collections.abc import Awaitable, Callable, Collection, Mapping, Sequence
from contextlib import suppress
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from typing import Any, cast

from fastmcp.tools import FunctionTool, ToolResult
from mcp.types import CallToolResult, ContentBlock, ListToolsResult, TextContent, Tool

from fast_agent.agents.agent_types import AgentConfig, AgentType
from fast_agent.agents.llm_agent import LlmAgent
from fast_agent.agents.tool_runner import ToolRunner, ToolRunnerHooks, _ToolLoopAgent
from fast_agent.constants import (
    FAST_AGENT_ERROR_CHANNEL,
    FAST_AGENT_PENDING_MEDIA_ATTACHMENTS,
    FAST_AGENT_TOOL_METADATA,
    FAST_AGENT_TOOL_TIMING,
    FAST_AGENT_URL_ELICITATION_CHANNEL,
    HUMAN_INPUT_TOOL_NAME,
    should_parallelize_tool_calls,
)
from fast_agent.context import Context
from fast_agent.core.logging.logger import get_logger
from fast_agent.core.prompt import Prompt
from fast_agent.event_progress import ProgressAction
from fast_agent.interfaces import LlmAgentProtocol, ToolRunnerHookCapable
from fast_agent.llm.structured_schema import validate_json_schema_definition
from fast_agent.mcp.helpers.content_helpers import text_content
from fast_agent.mcp.tool_execution_handler import ToolExecutionHandler
from fast_agent.mcp.url_elicitation_required import URLElicitationRequiredDisplayPayload
from fast_agent.tools.elicitation import get_elicitation_fastmcp_tool
from fast_agent.tools.function_tool_loader import build_default_function_tool
from fast_agent.types import LlmStopReason, PromptMessageExtended, RequestParams, ToolTimingInfo
from fast_agent.ui.message_display_helpers import resolve_highlight_index
from fast_agent.utils.async_utils import gather_with_cancel

logger = get_logger(__name__)

_tool_progress_context: ContextVar[tuple[ToolExecutionHandler, str] | None] = ContextVar(
    "tool_progress_context",
    default=None,
)


@dataclass(frozen=True)
class _PlannedToolCall:
    correlation_id: str
    name: str
    arguments: dict[str, Any]


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
            with suppress(Exception):
                await self._handler.on_tool_progress(tool_call_id, float(self._step), None, message)

    async def finish(self, success: bool, error: str | None = None) -> None:
        async with self._lock:
            if self._finished:
                return
            self._finished = True
            if not self._tool_call_id:
                return
            with suppress(Exception):
                await self._handler.on_tool_complete(self._tool_call_id, success, None, error)


class ToolAgent(LlmAgent, _ToolLoopAgent):
    """
    A Tool Calling agent that uses FastMCP Tools for execution.

    Pass either:
    - native FastMCP FunctionTool objects
    - regular Python functions (wrapped as FunctionTools)

    Naming note:
    ``tools`` here means executable local/function tools available to the
    agent. It does not refer to ``AgentConfig.tools``, which is the MCP
    filter map used by ``McpAgent``.
    """

    def __init__(
        self,
        config: AgentConfig,
        tools: Sequence[FunctionTool | Callable[..., Any]] = (),
        context: Context | None = None,
    ) -> None:
        """Create a tool-capable agent.

        Args:
            config: Agent configuration. ``config.tools`` remains the MCP
                filter map; it is separate from this ``tools`` argument.
            tools: Executable local/function tools to expose on the agent.
            context: Optional runtime context.
        """
        super().__init__(config=config, context=context)

        self._execution_tools: dict[str, FunctionTool] = {}
        self._tool_schemas: list[Tool] = []
        self._agent_tools: dict[str, LlmAgent] = {}
        self._card_tool_names: set[str] = set()
        self._smart_tool_names: set[str] = set()
        self._parallel_smart_tool_calls = False

        # Build a working list of tools and auto-inject human-input tool if missing
        working_tools: list[FunctionTool | Callable[..., Any]] = list(tools) if tools else []
        card_tool_source_ids = {id(tool) for tool in working_tools}
        # Only auto-inject if enabled via AgentConfig
        if self.config.human_input:
            existing_names = {
                t.name if isinstance(t, FunctionTool) else getattr(t, "__name__", "")
                for t in working_tools
            }
            if HUMAN_INPUT_TOOL_NAME not in existing_names:
                try:
                    working_tools.append(get_elicitation_fastmcp_tool())
                except Exception as e:
                    logger.warning(f"Failed to initialize human-input tool: {e}")

        for tool in working_tools:
            if isinstance(tool, FunctionTool):
                fast_tool = tool
            elif callable(tool):
                fast_tool = build_default_function_tool(tool)
            else:
                logger.warning(f"Skipping unknown tool type: {type(tool)}")
                continue

            self.add_tool(fast_tool)
            if id(tool) in card_tool_source_ids:
                self._card_tool_names.add(fast_tool.name)

    def _clone_constructor_kwargs(self) -> dict[str, Any]:
        """Carry local tool definitions into detached clones."""
        if not self._execution_tools:
            return {}
        return {"tools": list(self._execution_tools.values())}

    def add_tool(self, tool: FunctionTool, *, replace: bool = True) -> None:
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

    def _tool_display_metadata(self, tool_name: str) -> dict[str, Any] | None:
        tool = self._execution_tools.get(tool_name)
        if tool is None or not isinstance(tool.meta, Mapping):
            return None
        metadata = dict(tool.meta)
        return metadata or None

    def resolve_stream_tool_metadata(self, tool_name: str) -> Mapping[str, Any] | None:
        return self._jsonable_tool_metadata(self._tool_display_metadata(tool_name))

    @staticmethod
    def _jsonable_tool_metadata(metadata: dict[str, Any] | None) -> dict[str, Any] | None:
        if not metadata:
            return None
        try:
            json.dumps(metadata)
        except TypeError:
            return None
        return dict(metadata)

    @property
    def has_before_tool_call_hook(self) -> bool:
        """Return True if a before_tool_call hook is configured."""
        return (
            self.tool_runner_hooks is not None
            and self.tool_runner_hooks.before_tool_call is not None
        )

    @property
    def has_after_tool_call_hook(self) -> bool:
        """Return True if an after_tool_call hook is configured."""
        return (
            self.tool_runner_hooks is not None
            and self.tool_runner_hooks.after_tool_call is not None
        )

    @property
    def has_after_turn_complete_hook(self) -> bool:
        """Return True if an after_turn_complete hook is configured."""
        return (
            self.tool_runner_hooks is not None
            and self.tool_runner_hooks.after_turn_complete is not None
        )

    @property
    def agent_backed_tools(self) -> Mapping[str, LlmAgentProtocol]:
        """Return the public view of child agents exposed as tools."""
        return self._agent_tools

    @property
    def card_tool_names(self) -> Collection[str]:
        """Return the public view of card-sourced tool names."""
        return self._card_tool_names

    @property
    def smart_tool_names(self) -> Collection[str]:
        """Return the public view of smart tool names."""
        return self._smart_tool_names

    @smart_tool_names.setter
    def smart_tool_names(self, value: Collection[str]) -> None:
        self._smart_tool_names = set(value)

    @property
    def parallel_smart_tool_calls(self) -> bool:
        """Return whether a parallel smart-tool batch is active."""
        return self._parallel_smart_tool_calls

    @parallel_smart_tool_calls.setter
    def parallel_smart_tool_calls(self, value: bool) -> None:
        self._parallel_smart_tool_calls = value

    def _card_tools_label(self) -> str | None:
        if not self._card_tool_names:
            return None
        return "card_tools"

    def _card_tools_used(self, message: PromptMessageExtended) -> bool:
        if not self._card_tool_names or not message.tool_calls:
            return False
        return any(
            tool_request.params.name in self._card_tool_names
            for tool_request in message.tool_calls.values()
        )

    def _count_agent_tool_calls(self, tool_call_items: list[tuple[str, Any]]) -> int:
        if not tool_call_items:
            return 0
        agent_tool_names = set(self._agent_tools.keys())
        if self.config.agent_type == AgentType.SMART:
            agent_tool_names.add("smart")
        if not agent_tool_names:
            return 0
        return sum(
            1
            for _, tool_request in tool_call_items
            if tool_request.params.name in agent_tool_names
        )

    def _agent_tool_description(
        self,
        child: LlmAgent,
        description: str | None,
    ) -> str:
        if description:
            return description
        return (
            child.config.description
            or child.instruction
            or f"Send a message to the {child.name} agent"
        )

    async def _emit_agent_tool_progress(
        self,
        *,
        child_name: str,
        progress_step: int,
        label: str | None,
    ) -> None:
        message = f"{child_name} step {progress_step}"
        if label:
            message = f"{message} ({label})"

        ctx = _tool_progress_context.get()
        if ctx:
            handler, tool_call_id = ctx
            with suppress(Exception):
                await handler.on_tool_progress(
                    tool_call_id,
                    float(progress_step),
                    None,
                    message,
                )

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

    @staticmethod
    def _wrap_agent_tool_progress_hooks(
        clone: "ToolAgent",
        emit_progress: Callable[[str | None], Awaitable[None]],
    ) -> None:
        existing_hooks = clone.tool_runner_hooks
        before_llm_call = existing_hooks.before_llm_call if existing_hooks else None
        before_tool_call = existing_hooks.before_tool_call if existing_hooks else None
        after_llm_call = existing_hooks.after_llm_call if existing_hooks else None
        after_tool_call = existing_hooks.after_tool_call if existing_hooks else None
        after_turn_complete = existing_hooks.after_turn_complete if existing_hooks else None

        async def handle_before_llm_call(
            runner: ToolRunner,
            messages: list[PromptMessageExtended],
        ) -> None:
            if before_llm_call:
                await before_llm_call(runner, messages)
            await emit_progress("llm")

        async def handle_before_tool_call(
            runner: ToolRunner,
            message: PromptMessageExtended,
        ) -> None:
            if before_tool_call:
                await before_tool_call(runner, message)
            await emit_progress("tool")

        clone.tool_runner_hooks = ToolRunnerHooks(
            before_llm_call=handle_before_llm_call,
            after_llm_call=after_llm_call,
            before_tool_call=handle_before_tool_call,
            after_tool_call=after_tool_call,
            after_turn_complete=after_turn_complete,
        )

    @staticmethod
    async def _shutdown_agent_tool_clone(child: LlmAgent, clone: LlmAgent) -> None:
        try:
            await clone.shutdown()
        except Exception as exc:
            logger.warning(f"Error shutting down tool clone for {child.name}: {exc}")
        try:
            child.merge_usage_from(clone)
        except Exception as exc:
            logger.warning(f"Failed to merge tool clone usage for {child.name}: {exc}")

    def add_agent_tool(
        self,
        child: LlmAgent,
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> str:
        """Expose another agent as a tool on this agent."""
        tool_name = name or f"agent__{child.name}"
        tool_description = self._agent_tool_description(child, description)
        self._agent_tools[tool_name] = child

        async def call_agent(message: str) -> str:
            """Message to send to the child agent."""
            input_text = message
            clone = await child.spawn_detached_instance(name=f"{child.name}[tool]")
            progress_step = 0

            async def emit_progress(label: str | None = None) -> None:
                nonlocal progress_step
                progress_step += 1
                await self._emit_agent_tool_progress(
                    child_name=child.name,
                    progress_step=progress_step,
                    label=label,
                )

            hooks_set = False
            if isinstance(clone, ToolAgent):
                self._wrap_agent_tool_progress_hooks(clone, emit_progress)
                hooks_set = True

            try:
                if not hooks_set:
                    await emit_progress("run")
                clone.load_message_history([])
                response = await clone.generate([Prompt.user(input_text)], None)
                return response.last_text() or ""
            finally:
                await self._shutdown_agent_tool_clone(child, clone)

        fast_tool = build_default_function_tool(
            call_agent,
            name=tool_name,
            description=tool_description,
        )
        self.add_tool(fast_tool)
        return tool_name

    async def generate_impl(
        self,
        messages: list[PromptMessageExtended],
        request_params: RequestParams | None = None,
        tools: list[Tool] | None = None,
    ) -> PromptMessageExtended:
        """
        Generate a response using the LLM, and handle tool calls if necessary.
        Messages are already normalized to list[PromptMessageExtended].
        """
        use_history = request_params.use_history if request_params is not None else True
        has_tool_results = any(message.tool_results for message in messages)
        if use_history and not has_tool_results:
            history_state = ToolRunner.reconcile_interrupted_history(
                self,
                use_history=use_history,
            )
            if history_state.status == "appended_interrupted_tool_result":
                logger.warning(
                    "History ended with unanswered tool call; auto-healed by "
                    "appending interrupted tool result marker.",
                    data={
                        "history_before": history_state.history_before,
                        "history_after": history_state.history_after,
                    },
                )

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

    async def structured_schema_impl(
        self,
        messages: list[PromptMessageExtended],
        schema: dict[str, Any],
        request_params: RequestParams | None = None,
    ) -> tuple[Any | None, PromptMessageExtended]:
        """Run raw-schema structured output through the normal tool loop."""
        llm = self._require_llm()
        normalized_schema = validate_json_schema_definition(schema)
        structured_params = llm.get_request_params(request_params).model_copy(
            update={"structured_schema": normalized_schema}
        )

        response = await self.generate_impl(messages, structured_params)
        return llm.parse_structured_schema_response(response, normalized_schema)

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
            after_turn_complete=merge(
                base.after_turn_complete, extra.after_turn_complete
            ),
        )

    async def _tool_runner_llm_step(
        self,
        messages: list[PromptMessageExtended],
        request_params: RequestParams | None = None,
        tools: list[Tool] | None = None,
    ) -> PromptMessageExtended:
        return await super().generate_impl(messages, request_params=request_params, tools=tools)

    def should_finalize_deferred_structured_turn(
        self,
        messages: list[PromptMessageExtended],
        request_params: RequestParams | None,
        tools: list[Tool] | None,
        assistant_message: PromptMessageExtended,
    ) -> bool:
        del assistant_message
        if self.llm is None:
            return False
        final_params = self.llm.get_request_params(request_params)
        return (
            final_params.structured_schema is not None
            and bool(tools)
            and self.llm.resolve_structured_tool_policy(final_params) == "defer"
            and not any(message.tool_results for message in messages)
        )

    def should_suppress_tools_for_structured_turn(
        self,
        messages: list[PromptMessageExtended],
        request_params: RequestParams | None,
        tools: list[Tool] | None,
    ) -> bool:
        del messages
        if self.llm is None or not tools:
            return False
        final_params = self.llm.get_request_params(request_params)
        return (
            final_params.structured_schema is not None
            and self.llm.resolve_structured_tool_policy(final_params) == "no_tools"
        )

    def _should_display_user_message(self, message: PromptMessageExtended) -> bool:
        return not message.tool_results

    def _consume_pending_media_attachments(self) -> list[ContentBlock]:
        """Return pending media blocks to send as the next user input."""
        return []

    # we take care of tool results, so skip displaying them
    def show_user_message(self, message: PromptMessageExtended) -> None:
        if message.tool_results:
            return
        super().show_user_message(message)

    @staticmethod
    def _tool_names(tool_schemas: Sequence[Tool]) -> list[str]:
        return [tool.name for tool in tool_schemas]

    def _close_streaming_for_parallel_subagents(
        self,
        tool_call_items: list[tuple[str, Any]],
        *,
        should_parallel: bool,
    ) -> None:
        if not should_parallel or not tool_call_items:
            return
        subagent_calls = self._count_agent_tool_calls(tool_call_items)
        if subagent_calls <= 1:
            return

        did_close = self.close_active_streaming_display(
            reason="parallel subagent tool calls"
        )
        if did_close:
            logger.info(
                "Closing streaming display due to parallel subagent tool calls",
                agent_name=self.name,
                tool_call_count=len(tool_call_items),
                subagent_call_count=subagent_calls,
            )

    def _plan_tool_calls(
        self,
        tool_call_items: list[tuple[str, Any]],
        *,
        available_tools: Collection[str],
        should_parallel: bool,
        tool_results: dict[str, CallToolResult],
    ) -> tuple[list[_PlannedToolCall], str | None]:
        planned_calls: list[_PlannedToolCall] = []
        for correlation_id, tool_request in tool_call_items:
            tool_name = tool_request.params.name
            tool_args = tool_request.params.arguments or {}
            if tool_name not in available_tools and tool_name not in self._execution_tools:
                error_message = f"Tool '{tool_name}' is not available"
                logger.error(error_message)
                return planned_calls, self._mark_tool_loop_error(
                    correlation_id=correlation_id,
                    error_message=error_message,
                    tool_results=tool_results,
                    tool_call_id=correlation_id if should_parallel else None,
                )
            planned_calls.append(
                _PlannedToolCall(
                    correlation_id=correlation_id,
                    name=tool_name,
                    arguments=tool_args,
                )
            )
        return planned_calls, None

    def _show_planned_tool_call(
        self,
        planned_call: _PlannedToolCall,
        *,
        available_tools: list[str],
        tool_metadata: dict[str, dict[str, Any]],
        parallel: bool,
    ) -> None:
        metadata = self._jsonable_tool_metadata(
            self._tool_display_metadata(planned_call.name)
        )
        if metadata:
            tool_metadata[planned_call.correlation_id] = metadata

        self.display.show_tool_call(
            name=self.name,
            tool_args=planned_call.arguments,
            bottom_items=available_tools,
            tool_name=planned_call.name,
            highlight_index=resolve_highlight_index(available_tools, planned_call.name),
            max_item_length=12,
            metadata=metadata,
            tool_call_id=planned_call.correlation_id if parallel else None,
            show_hook_indicator=self.has_before_tool_call_hook,
        )

    async def _execute_planned_tool_call(
        self,
        planned_call: _PlannedToolCall,
        *,
        request_params: RequestParams | None,
    ) -> tuple[CallToolResult, float]:
        start_time = time.perf_counter()
        result = await self.call_tool(
            planned_call.name,
            planned_call.arguments,
            request_params=request_params,
        )
        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        return result, duration_ms

    async def _run_parallel_tool_calls(
        self,
        planned_calls: list[_PlannedToolCall],
        *,
        request_params: RequestParams | None,
    ) -> tuple[dict[str, CallToolResult], dict[str, ToolTimingInfo]]:
        async def run_one(
            planned_call: _PlannedToolCall,
        ) -> tuple[str, CallToolResult, float]:
            result, duration_ms = await self._execute_planned_tool_call(
                planned_call,
                request_params=request_params,
            )
            return planned_call.correlation_id, result, duration_ms

        results = await gather_with_cancel(run_one(call) for call in planned_calls)
        tool_results: dict[str, CallToolResult] = {}
        tool_timings: dict[str, ToolTimingInfo] = {}
        for planned_call, item in zip(planned_calls, results, strict=False):
            if isinstance(item, BaseException):
                result = CallToolResult(
                    content=[text_content(f"Error: {item!s}")],
                    isError=True,
                )
                duration_ms = 0.0
            else:
                _, result, duration_ms = item

            tool_results[planned_call.correlation_id] = result
            tool_timings[planned_call.correlation_id] = ToolTimingInfo(
                timing_ms=duration_ms,
                transport_channel=None,
            )
            self.display.show_tool_result(
                name=self.name,
                result=result,
                tool_name=planned_call.name,
                timing_ms=duration_ms,
                tool_call_id=planned_call.correlation_id,
                show_hook_indicator=self.has_after_tool_call_hook,
            )
        return tool_results, tool_timings

    async def _run_sequential_tool_calls(
        self,
        planned_calls: list[_PlannedToolCall],
        *,
        request_params: RequestParams | None,
    ) -> tuple[dict[str, CallToolResult], dict[str, ToolTimingInfo]]:
        tool_results: dict[str, CallToolResult] = {}
        tool_timings: dict[str, ToolTimingInfo] = {}
        for planned_call in planned_calls:
            result, duration_ms = await self._execute_planned_tool_call(
                planned_call,
                request_params=request_params,
            )
            tool_results[planned_call.correlation_id] = result
            tool_timings[planned_call.correlation_id] = ToolTimingInfo(
                timing_ms=duration_ms,
                transport_channel=None,
            )
            self.display.show_tool_result(
                name=self.name,
                result=result,
                tool_name=planned_call.name,
                timing_ms=duration_ms,
                show_hook_indicator=self.has_after_tool_call_hook,
            )
        return tool_results, tool_timings

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
        tool_metadata: dict[str, dict[str, Any]] = {}
        tool_loop_error: str | None = None
        tool_schemas = (await self.list_tools()).tools
        available_tools = self._tool_names(tool_schemas)

        tool_call_items = list(request.tool_calls.items())
        should_parallel = should_parallelize_tool_calls(len(tool_call_items))
        self._close_streaming_for_parallel_subagents(
            tool_call_items,
            should_parallel=should_parallel,
        )
        planned_calls, tool_loop_error = self._plan_tool_calls(
            tool_call_items,
            available_tools=available_tools,
            should_parallel=should_parallel,
            tool_results=tool_results,
        )

        if should_parallel and planned_calls:
            for planned_call in planned_calls:
                self._show_planned_tool_call(
                    planned_call,
                    available_tools=available_tools,
                    tool_metadata=tool_metadata,
                    parallel=True,
                )
            executed_results, executed_timings = await self._run_parallel_tool_calls(
                planned_calls,
                request_params=request_params,
            )
            tool_results.update(executed_results)
            tool_timings.update(executed_timings)

            return self._finalize_tool_results(
                tool_results,
                tool_timings=tool_timings,
                tool_metadata=tool_metadata,
                tool_loop_error=tool_loop_error,
            )

        for planned_call in planned_calls:
            self._show_planned_tool_call(
                planned_call,
                available_tools=available_tools,
                tool_metadata=tool_metadata,
                parallel=False,
            )
        executed_results, executed_timings = await self._run_sequential_tool_calls(
            planned_calls,
            request_params=request_params,
        )
        tool_results.update(executed_results)
        tool_timings.update(executed_timings)

        return self._finalize_tool_results(
            tool_results,
            tool_timings=tool_timings,
            tool_metadata=tool_metadata,
            tool_loop_error=tool_loop_error,
        )

    def _mark_tool_loop_error(
        self,
        *,
        correlation_id: str,
        error_message: str,
        tool_results: dict[str, CallToolResult],
        tool_call_id: str | None = None,
    ) -> str:
        error_result = CallToolResult(
            content=[text_content(error_message)],
            isError=True,
        )
        tool_results[correlation_id] = error_result
        self.display.show_tool_result(
            name=self.name,
            result=error_result,
            tool_call_id=tool_call_id,
            show_hook_indicator=self.has_after_tool_call_hook,
        )
        return error_message

    @staticmethod
    def _tool_result_channels(
        *,
        tool_timings: dict[str, ToolTimingInfo] | None,
        tool_metadata: dict[str, dict[str, Any]] | None,
        tool_loop_error: str | None,
        tool_results: Mapping[str, CallToolResult] | None = None,
    ) -> tuple[dict[str, Sequence[ContentBlock]] | None, list[ContentBlock]]:
        channels: dict[str, Sequence[ContentBlock]] = {}
        content: list[ContentBlock] = []
        if tool_loop_error:
            content.append(text_content(tool_loop_error))
            channels[FAST_AGENT_ERROR_CHANNEL] = [text_content(tool_loop_error)]
        if tool_results:
            fatal_errors = [
                str(error)
                for result in tool_results.values()
                if (error := getattr(result, "_fast_agent_fatal_tool_error", None))
            ]
            if fatal_errors:
                content.extend(text_content(error) for error in fatal_errors)
                channels[FAST_AGENT_ERROR_CHANNEL] = [
                    text_content("\n".join(fatal_errors))
                ]
        if tool_timings:
            channels[FAST_AGENT_TOOL_TIMING] = [
                TextContent(type="text", text=json.dumps(tool_timings))
            ]
        if tool_metadata:
            channels[FAST_AGENT_TOOL_METADATA] = [
                TextContent(type="text", text=json.dumps(tool_metadata))
            ]
        return channels or None, content

    @staticmethod
    def _deferred_url_elicitation_payloads(
        tool_results: Mapping[str, CallToolResult],
    ) -> list[dict[str, object]]:
        payloads: list[dict[str, object]] = []
        for result in tool_results.values():
            payload = getattr(result, "_fast_agent_url_elicitation_required", None)
            if isinstance(payload, URLElicitationRequiredDisplayPayload):
                payloads.append(asdict(payload))
        return payloads

    @staticmethod
    def _add_channel(
        channels: dict[str, Sequence[ContentBlock]] | None,
        name: str,
        content: Sequence[ContentBlock],
    ) -> dict[str, Sequence[ContentBlock]]:
        if channels is None:
            channels = {}
        channels[name] = content
        return channels

    def _finalize_tool_results(
        self,
        tool_results: dict[str, CallToolResult],
        *,
        tool_timings: dict[str, ToolTimingInfo] | None = None,
        tool_metadata: dict[str, dict[str, Any]] | None = None,
        tool_loop_error: str | None = None,
    ) -> PromptMessageExtended:
        channels, content = self._tool_result_channels(
            tool_timings=tool_timings,
            tool_metadata=tool_metadata,
            tool_loop_error=tool_loop_error,
            tool_results=tool_results,
        )

        deferred_url_elicitations = self._deferred_url_elicitation_payloads(tool_results)
        if deferred_url_elicitations:
            channels = self._add_channel(
                channels,
                FAST_AGENT_URL_ELICITATION_CHANNEL,
                [TextContent(type="text", text=json.dumps(deferred_url_elicitations))],
            )

        pending_media = self._consume_pending_media_attachments()
        if pending_media:
            channels = self._add_channel(
                channels,
                FAST_AGENT_PENDING_MEDIA_ATTACHMENTS,
                pending_media,
            )

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
        arguments: dict[str, Any] | None = None,
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
            native_result = await fast_tool.run(arguments or {})
            tool_result = self._native_tool_result_to_mcp_result(native_result)
            if tool_handler and tool_call_id:
                with suppress(Exception):
                    await tool_handler.on_tool_complete(
                        tool_call_id, True, tool_result.content, None
                    )
            return tool_result
        except Exception as e:
            logger.error(f"Tool {name} failed: {e}")
            tool_result = CallToolResult(
                content=[text_content(f"Error: {e!s}")],
                isError=True,
            )
            payload = getattr(e, "_fast_agent_url_elicitation_required", None)
            if payload is not None:
                with suppress(Exception):
                    tool_result_meta = cast("Any", tool_result)
                    tool_result_meta._fast_agent_url_elicitation_required = payload
            if tool_handler and tool_call_id:
                with suppress(Exception):
                    await tool_handler.on_tool_complete(tool_call_id, False, None, str(e))
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

    @staticmethod
    def _native_tool_result_to_mcp_result(result: ToolResult) -> CallToolResult:
        return CallToolResult(
            content=result.content,
            structuredContent=result.structured_content,
            _meta=result.meta,
            isError=False,
        )
