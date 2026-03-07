from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Awaitable,
    Callable,
    Literal,
    Protocol,
    Union,
)

from mcp.types import CallToolResult, ListToolsResult, TextContent

from fast_agent.constants import (
    DEFAULT_MAX_ITERATIONS,
    FAST_AGENT_ERROR_CHANNEL,
    FAST_AGENT_SYNTHETIC_FINAL_CHANNEL,
    FAST_AGENT_USAGE,
)
from fast_agent.core.logging.logger import get_logger
from fast_agent.interfaces import MessageHistoryAgentProtocol
from fast_agent.mcp.helpers.content_helpers import text_content
from fast_agent.types import PromptMessageExtended, RequestParams
from fast_agent.types.llm_stop_reason import LlmStopReason

if TYPE_CHECKING:
    from mcp import Tool


class _AgentConfig(Protocol):
    use_history: bool


class _ToolLoopAgent(MessageHistoryAgentProtocol, Protocol):
    config: _AgentConfig

    async def _tool_runner_llm_step(
        self,
        messages: list[PromptMessageExtended],
        request_params: RequestParams | None = None,
        tools: list[Tool] | None = None,
    ) -> PromptMessageExtended: ...

    async def run_tools(
        self,
        request: PromptMessageExtended,
        request_params: RequestParams | None = None,
    ) -> PromptMessageExtended: ...

    async def list_tools(self) -> ListToolsResult: ...


_logger = get_logger(__name__)


HistoryRollbackStatus = Literal[
    "history_disabled",
    "history_empty",
    "appended_interrupted_tool_result",
    "history_unchanged",
]


@dataclass(frozen=True)
class HistoryRollbackState:
    """Summary of how history was handled after an interrupted tool loop."""

    status: HistoryRollbackStatus
    history_before: int
    history_after: int
    removed_messages: int


@dataclass(frozen=True)
class ToolRunnerHooks:
    """
    Optional hook points for customizing the tool loop.

    These hooks are intentionally low-level and mutation-friendly: they can
    inspect and modify the agent history (via agent.load_message_history),
    tweak request params, or append extra messages via the runner.

    Hook points:
    - before_llm_call: Called before each LLM call with the messages to send
    - after_llm_call: Called after each LLM response is received
    - before_tool_call: Called before tools are executed
    - after_tool_call: Called after tool results are received
    - after_turn_complete: Called once after the entire turn completes (when stop_reason != TOOL_USE)
    """

    before_llm_call: (
        Callable[["ToolRunner", list[PromptMessageExtended]], Awaitable[None]] | None
    ) = None
    after_llm_call: Callable[["ToolRunner", PromptMessageExtended], Awaitable[None]] | None = None
    before_tool_call: Callable[["ToolRunner", PromptMessageExtended], Awaitable[None]] | None = None
    after_tool_call: Callable[["ToolRunner", PromptMessageExtended], Awaitable[None]] | None = None
    after_turn_complete: (
        Callable[["ToolRunner", PromptMessageExtended], Awaitable[None]] | None
    ) = None


class ToolRunner:
    """
    Async-iterable tool runner.

    Yields assistant messages (LLM responses). If the response requests tools,
    a tool response is prepared and sent on the next iteration.
    """

    def __init__(
        self,
        *,
        agent: _ToolLoopAgent,
        messages: list[PromptMessageExtended],
        request_params: RequestParams | None = None,
        tools: list[Tool] | None = None,
        hooks: ToolRunnerHooks | None = None,
    ) -> None:
        self._agent = agent
        self._delta_messages: list[PromptMessageExtended] = list(messages)
        self._request_params = request_params
        self._tools = tools
        self._hooks = hooks or ToolRunnerHooks()

        self._iteration = 0
        self._done = False
        self._last_message: PromptMessageExtended | None = None

        self._pending_tool_request: PromptMessageExtended | None = None
        self._pending_tool_response: PromptMessageExtended | None = None
        self._staged_terminal_response: PromptMessageExtended | None = None

    def __aiter__(self) -> "ToolRunner":
        return self

    async def __anext__(self) -> PromptMessageExtended:
        staged = self._consume_staged_terminal_response()
        if staged is not None:
            return staged

        if self._done:
            raise StopAsyncIteration

        await self._ensure_tool_response_staged()

        staged = self._consume_staged_terminal_response()
        if staged is not None:
            return staged

        if self._done:
            raise StopAsyncIteration

        await self._ensure_tools_ready()

        if self._hooks.before_llm_call is not None:
            await self._hooks.before_llm_call(self, self._delta_messages)

        assistant_message = await self._agent._tool_runner_llm_step(
            self._delta_messages,
            request_params=self._request_params,
            tools=self._tools,
        )

        self._last_message = assistant_message
        if self._hooks.after_llm_call is not None:
            await self._hooks.after_llm_call(self, assistant_message)

        if assistant_message.stop_reason == LlmStopReason.TOOL_USE:
            self._pending_tool_request = assistant_message
            self._pending_tool_response = None  # Clear cache for new request
        else:
            self._done = True

        return assistant_message

    async def until_done(self) -> PromptMessageExtended:
        last: PromptMessageExtended | None = None
        try:
            async for message in self:
                last = message
        except asyncio.CancelledError:
            rollback_state = self._reset_history_after_cancelled_turn()
            self._record_cancelled_turn(
                reason="cancelled",
                rollback_state=rollback_state,
            )
            await self._persist_cancelled_turn_state_after_task_cancel()
            raise
        except KeyboardInterrupt:
            rollback_state = self._reset_history_after_cancelled_turn()
            self._record_cancelled_turn(
                reason="interrupted",
                rollback_state=rollback_state,
            )
            await self._persist_cancelled_turn_state()
            raise
        if last is None:
            raise RuntimeError("ToolRunner produced no messages")

        if last.stop_reason == LlmStopReason.CANCELLED:
            rollback_state = self._reset_history_after_cancelled_turn()
            self._record_cancelled_turn(
                reason="cancelled",
                rollback_state=rollback_state,
            )
            await self._persist_cancelled_turn_state()
            return last

        # Fire after_turn_complete hook once the entire turn is done
        if self._hooks.after_turn_complete is not None:
            await self._hooks.after_turn_complete(self, last)

        return last

    def _record_cancelled_turn(
        self,
        *,
        reason: str,
        rollback_state: HistoryRollbackState,
    ) -> None:
        try:
            setattr(self._agent, "_last_turn_cancelled", True)
            setattr(self._agent, "_last_turn_cancel_reason", reason)
            setattr(self._agent, "_last_turn_history_state", rollback_state)
        except Exception:
            pass

    async def _persist_cancelled_turn_state(self) -> None:
        """Persist reconciled history for cancelled turns when session history is enabled."""
        history = self._agent.message_history
        if not history:
            return

        try:
            from fast_agent.hooks.hook_context import HookContext
            from fast_agent.hooks.session_history import save_session_history

            await save_session_history(
                HookContext(
                    runner=self,
                    agent=self._agent,
                    message=history[-1],
                    hook_type="after_turn_cancelled",
                )
            )
        except Exception as exc:
            _logger.warning(
                "Failed to persist cancelled turn state",
                error=str(exc),
                error_type=type(exc).__name__,
            )

    async def _persist_cancelled_turn_state_after_task_cancel(self) -> None:
        """Persist cancelled-turn history even when this task was externally cancelled."""
        task = asyncio.current_task()
        if task is None:
            await self._persist_cancelled_turn_state()
            return

        cancellation_requests = task.cancelling()
        if cancellation_requests == 0:
            await self._persist_cancelled_turn_state()
            return

        for _ in range(cancellation_requests):
            task.uncancel()

        try:
            await self._persist_cancelled_turn_state()
        finally:
            for _ in range(cancellation_requests):
                task.cancel()

    @staticmethod
    def reconcile_interrupted_history(
        agent: MessageHistoryAgentProtocol,
        *,
        use_history: bool,
    ) -> HistoryRollbackState:
        history = agent.message_history
        history_before = len(history)

        if not use_history:
            return HistoryRollbackState(
                status="history_disabled",
                history_before=history_before,
                history_after=history_before,
                removed_messages=0,
            )

        if not history:
            return HistoryRollbackState(
                status="history_empty",
                history_before=0,
                history_after=0,
                removed_messages=0,
            )

        last_message = history[-1]
        if (
            last_message.role == "assistant"
            and (last_message.tool_calls or {})
            and last_message.stop_reason == LlmStopReason.TOOL_USE
        ):
            interrupted_tool_message = ToolRunner._build_interrupted_tool_result(last_message)
            agent.load_message_history([*history, interrupted_tool_message])
            return HistoryRollbackState(
                status="appended_interrupted_tool_result",
                history_before=history_before,
                history_after=history_before + 1,
                removed_messages=0,
            )

        return HistoryRollbackState(
            status="history_unchanged",
            history_before=history_before,
            history_after=history_before,
            removed_messages=0,
        )

    def _reset_history_after_cancelled_turn(self) -> HistoryRollbackState:
        return ToolRunner.reconcile_interrupted_history(
            self._agent,
            use_history=self._agent.config.use_history,
        )

    @staticmethod
    def _build_interrupted_tool_result(
        pending_request: PromptMessageExtended,
    ) -> PromptMessageExtended:
        interrupted_text = "**The user interrupted this tool call**"
        tool_results: dict[str, CallToolResult] = {}
        for tool_id in (pending_request.tool_calls or {}).keys():
            tool_results[tool_id] = CallToolResult(
                content=[text_content(interrupted_text)],
                isError=True,
            )

        return PromptMessageExtended(
            role="user",
            content=[text_content(interrupted_text)],
            tool_results=tool_results,
        )

    def _build_tool_error_response(
        self, request: PromptMessageExtended, error_message: str
    ) -> PromptMessageExtended:
        tool_results: dict[str, CallToolResult] = {}
        for tool_id in (request.tool_calls or {}).keys():
            tool_results[tool_id] = CallToolResult(
                content=[text_content(error_message)],
                isError=True,
            )

        channels = {FAST_AGENT_ERROR_CHANNEL: [text_content(error_message)]}

        return PromptMessageExtended(
            role="user",
            content=[text_content(error_message)],
            tool_results=tool_results,
            channels=channels,
        )

    async def generate_tool_call_response(self) -> PromptMessageExtended | None:
        if self._pending_tool_request is None:
            return None
        if self._pending_tool_response is not None:
            return self._pending_tool_response

        try:
            hook_phase = "before_tool_call"
            if self._hooks.before_tool_call is not None:
                await self._hooks.before_tool_call(self, self._pending_tool_request)
            hook_phase = "run_tools"
            tool_message = await self._agent.run_tools(
                self._pending_tool_request, request_params=self._request_params
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            tool_calls = self._pending_tool_request.tool_calls or {}
            tool_call_ids = list(tool_calls.keys())
            tool_names = [call.params.name for call in tool_calls.values()]
            agent_name = getattr(self._agent, "name", None)
            tool_message = self._build_tool_error_response(
                self._pending_tool_request,
                f"Tool hook or execution failed during {hook_phase}: {exc}",
            )
            _logger.exception(
                "Tool hook or execution failed",
                agent_name=agent_name,
                hook_phase=hook_phase,
                tool_call_ids=tool_call_ids,
                tool_names=tool_names,
            )

        self._pending_tool_response = tool_message

        if self._hooks.after_tool_call is not None:
            try:
                await self._hooks.after_tool_call(self, tool_message)
            except Exception as exc:
                _logger.error("Tool hook failed after tool call", exc_info=exc)
        self._pending_tool_request = None

        return tool_message

    def set_request_params(self, params: RequestParams) -> None:
        self._request_params = params

    @property
    def request_params(self) -> RequestParams | None:
        """Current request params driving this tool-loop turn."""
        return self._request_params

    def append_messages(self, *messages: Union[str, PromptMessageExtended]) -> None:
        for message in messages:
            if isinstance(message, str):
                self._delta_messages.append(
                    PromptMessageExtended(
                        role="user",
                        content=[TextContent(type="text", text=message)],
                    )
                )
            else:
                self._delta_messages.append(message)

    @property
    def delta_messages(self) -> list[PromptMessageExtended]:
        """Messages to be sent in the next LLM call (not full history)."""
        return self._delta_messages

    @property
    def iteration(self) -> int:
        return self._iteration

    @property
    def is_done(self) -> bool:
        return self._done

    @property
    def last_message(self) -> PromptMessageExtended | None:
        return self._last_message

    @property
    def has_pending_tool_response(self) -> bool:
        return self._pending_tool_request is not None

    def _stage_tool_response(self, tool_message: PromptMessageExtended) -> None:
        if self._use_history_enabled():
            self._delta_messages = [tool_message]
        else:
            if self._last_message is not None:
                self._delta_messages.append(self._last_message)
            self._delta_messages.append(tool_message)

    def _consume_staged_terminal_response(self) -> PromptMessageExtended | None:
        staged = self._staged_terminal_response
        if staged is None:
            return None

        self._staged_terminal_response = None
        self._last_message = staged
        self._done = True
        return staged

    def _use_history_enabled(self) -> bool:
        if self._request_params is not None:
            return self._request_params.use_history
        return self._agent.config.use_history

    def _passthrough_enabled(self) -> bool:
        if self._request_params is None:
            return False
        return self._request_params.tool_result_passthrough

    def _append_history_messages(self, *messages: PromptMessageExtended) -> None:
        history = list(self._agent.message_history)
        history.extend(messages)
        self._agent.load_message_history(history)

    def _synthesize_passthrough_assistant(
        self,
        tool_message: PromptMessageExtended,
    ) -> PromptMessageExtended:
        content_blocks = [
            content
            for tool_result in (tool_message.tool_results or {}).values()
            for content in tool_result.content
        ]

        channels = {FAST_AGENT_SYNTHETIC_FINAL_CHANNEL: [text_content("tool_result_passthrough")]}
        if self._last_message is not None and self._last_message.channels:
            usage_blocks = self._last_message.channels.get(FAST_AGENT_USAGE)
            if usage_blocks:
                channels[FAST_AGENT_USAGE] = list(usage_blocks)

        return PromptMessageExtended(
            role="assistant",
            content=content_blocks,
            channels=channels,
            stop_reason=LlmStopReason.END_TURN,
        )

    async def _ensure_tools_ready(self) -> None:
        if self._tools is None:
            self._tools = (await self._agent.list_tools()).tools

    async def _ensure_tool_response_staged(self) -> None:
        if self._pending_tool_request is None:
            return

        tool_message = await self.generate_tool_call_response()
        if tool_message is None:
            return

        error_channel_messages = (tool_message.channels or {}).get(FAST_AGENT_ERROR_CHANNEL)
        if error_channel_messages and self._last_message is not None:
            tool_result_contents = [
                content
                for tool_result in (tool_message.tool_results or {}).values()
                for content in tool_result.content
            ]
            if tool_result_contents:
                if self._last_message.content is None:
                    self._last_message.content = []
                self._last_message.content.extend(tool_result_contents)
            self._last_message.stop_reason = LlmStopReason.ERROR
            self._done = True
            return

        self._iteration += 1
        max_iterations = (
            self._request_params.max_iterations
            if self._request_params is not None
            else DEFAULT_MAX_ITERATIONS
        )
        if self._iteration > max_iterations:
            self._done = True
            return

        if self._passthrough_enabled():
            terminal_message = self._synthesize_passthrough_assistant(tool_message)
            if self._use_history_enabled():
                self._append_history_messages(tool_message, terminal_message)

            if self._hooks.after_llm_call is not None:
                await self._hooks.after_llm_call(self, terminal_message)

            self._staged_terminal_response = terminal_message
            self._done = True
            return

        self._stage_tool_response(tool_message)
