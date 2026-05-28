from __future__ import annotations

import asyncio
import os
import time
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Awaitable,
    Callable,
    Literal,
    Protocol,
    Union,
    cast,
)

from mcp.types import CallToolResult, ContentBlock, ListToolsResult, TextContent

from fast_agent.constants import (
    DEFAULT_MAX_ITERATIONS,
    FAST_AGENT_ERROR_CHANNEL,
    FAST_AGENT_PENDING_MEDIA_ATTACHMENTS,
    FAST_AGENT_SYNTHETIC_FINAL_CHANNEL,
    FAST_AGENT_TIMING,
    FAST_AGENT_USAGE,
)
from fast_agent.core.logging.logger import get_logger
from fast_agent.interfaces import MessageHistoryAgentProtocol, TurnCancellationStateCapable
from fast_agent.llm.request_params import tool_result_mode_is_passthrough
from fast_agent.mcp.helpers.content_helpers import text_content
from fast_agent.types import PromptMessageExtended, RequestParams
from fast_agent.types.llm_stop_reason import LlmStopReason

if TYPE_CHECKING:
    from mcp import Tool

    from fast_agent.hooks.hook_context import HookAgentProtocol


# ── Context-overflow defence: cap oversized tool results pre-history ──────
#
# Incident 2026-05-17 (figma_read export_svg returning 777KB SVG markup):
# a single oversized tool result lands in message_history full-fat, every
# subsequent LLM call resends it, Anthropic API returns 400 "prompt is too
# long", error becomes assistant turn → poisons history → agent stuck.
#
# Defence-in-depth: regardless of any per-tool size cap, every tool result
# text block >MAX_TOOL_RESULT_BYTES is spilled to disk and replaced with a
# stub + 8KB preview BEFORE it ever reaches the staged history. Spill path
# lives under the workspace dir so the agent's filesystem MCP can read it
# back if it really needs the full content.
#
# Configurable via FAST_AGENT_MAX_TOOL_RESULT_BYTES env var; default 64KB.

_DEFAULT_MAX_TOOL_RESULT_BYTES = 64 * 1024
_PREVIEW_BYTES = 8 * 1024


def _max_tool_result_bytes() -> int:
    raw = os.environ.get("FAST_AGENT_MAX_TOOL_RESULT_BYTES")
    if not raw:
        return _DEFAULT_MAX_TOOL_RESULT_BYTES
    try:
        return max(1024, int(raw))
    except ValueError:
        return _DEFAULT_MAX_TOOL_RESULT_BYTES


def _spill_dir() -> Path:
    """Where to write oversized tool-result spills.

    Prefers ``TEAM_WORKSPACE`` (set by isolated_runner before MCP spawn), so
    the file is reachable via the agent's per-role filesystem MCP. Falls
    back to cwd, which after isolated_runner's chdir is also workspace_dir.
    """
    base = os.environ.get("TEAM_WORKSPACE") or os.getcwd()
    return Path(base) / ".tool-outputs"


def _sanitize_oversized_tool_results(
    message: PromptMessageExtended,
    *,
    agent_name: str = "agent",
) -> PromptMessageExtended:
    """Mutate-and-return: cap every TextContent inside tool_results.

    The mutation is in-place on each ``CallToolResult.content`` list — the
    caller's reference to ``message`` stays valid. We replace the oversized
    ``TextContent`` element entirely (rather than truncating its ``text``)
    so the stub is structurally identical to a normal text block and the
    LLM provider serialiser can't choke on a half-sized message.
    """
    if not getattr(message, "tool_results", None):
        return message

    max_bytes = _max_tool_result_bytes()
    spill_dir = _spill_dir()

    for tool_id, tr in list(message.tool_results.items()):
        content_list = tr.content or []
        for i, block in enumerate(content_list):
            if not isinstance(block, TextContent):
                continue
            text = block.text or ""
            raw_bytes = text.encode("utf-8", errors="replace")
            if len(raw_bytes) <= max_bytes:
                continue

            try:
                spill_dir.mkdir(parents=True, exist_ok=True)
                ts = int(time.time() * 1000)
                # Sanitise agent_name + tool_id for filesystem safety.
                safe_name = (agent_name or "agent").replace("/", "_").replace(" ", "_")
                safe_tid = str(tool_id).replace("/", "_")
                spill_path = spill_dir / f"{safe_name}-{safe_tid}-{ts}.txt"
                spill_path.write_text(text, encoding="utf-8")
                spill_target = str(spill_path)
            except OSError:
                # If we can't write to disk we still MUST cap — otherwise
                # the next LLM call dies. Fall back to "lost" stub.
                spill_target = "(spill failed — full content discarded)"

            preview = text[:_PREVIEW_BYTES]
            # Stub format: tell the agent (1) where the full file lives,
            # (2) the preview is bounded, (3) reading the file naively
            # will re-hit this cap. Trust the model to choose a bounded
            # read primitive (head/grep/awk via execute, or read_text_file
            # with head=/tail= if its filesystem MCP supports it) — don't
            # prescribe one. Phase 2 enhancement after the read-back
            # loopback concern raised 2026-05-18.
            stub_text = (
                f"[Output too large ({len(raw_bytes)} bytes) — saved to "
                f"{spill_target}. Read the file in bounded chunks if you "
                f"need more than this preview; an unbounded read hits the "
                f"same {_max_tool_result_bytes() // 1024}KB cap.]\n\n"
                f"--- preview (first {_PREVIEW_BYTES // 1024}KB) ---\n"
                f"{preview}"
            )
            content_list[i] = TextContent(type="text", text=stub_text)

        # Also cap any text block on the message-level content
        # (some providers serialise this too).
        if message.content:
            for j, block in enumerate(list(message.content)):
                if not isinstance(block, TextContent):
                    continue
                text = block.text or ""
                if len(text.encode("utf-8", errors="replace")) <= max_bytes:
                    continue
                message.content[j] = TextContent(
                    type="text",
                    text=text[:_PREVIEW_BYTES] + "\n\n[... truncated by Jarvis context-overflow guard]",
                )

    return message


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

    def should_finalize_deferred_structured_turn(
        self,
        messages: list[PromptMessageExtended],
        request_params: RequestParams | None,
        tools: list[Tool] | None,
        assistant_message: PromptMessageExtended,
    ) -> bool: ...

    def should_suppress_tools_for_structured_turn(
        self,
        messages: list[PromptMessageExtended],
        request_params: RequestParams | None,
        tools: list[Tool] | None,
    ) -> bool: ...


_logger = get_logger(__name__)

_HOOK_STATUS_BUCKET_BEFORE_LLM_CALL = "before_llm_call"
_HOOK_STATUS_BUCKET_AFTER_LLM_CALL = "after_llm_call"
_HOOK_STATUS_BUCKET_BEFORE_TOOL_CALL = "before_tool_call"
_HOOK_STATUS_BUCKET_AFTER_TOOL_RESULTS = "after_tool_results"
_HOOK_STATUS_BUCKET_AFTER_TURN_COMPLETE = "after_turn_complete"


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
    - on_pause_cancel: Optional async filter for ``CancelledError`` raised
      from inside the LLM call. Returns True to indicate the cancel was
      intentional pause (and the LLM call should be retried after the
      caller awaits resume); False to let CancelledError propagate as
      usual. Lets PauseController interrupt long-running LLM streams
      without tearing down the surrounding chat request.
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
    on_pause_cancel: Callable[["ToolRunner"], Awaitable[bool]] | None = None


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
        self._deferred_structured_finalization_started = False

    def _defer_hook_status_messages(self, bucket: str) -> AbstractContextManager[None]:
        # TODO: Replace this post-hook flush boundary with a first-class
        # streaming/display event path once hook output participates in the
        # live renderer instead of the status-line fallback.
        from fast_agent.agents.llm_agent import LlmAgent

        if isinstance(self._agent, LlmAgent):
            return self._agent.defer_hook_status_messages(bucket)
        return nullcontext()

    def _flush_deferred_hook_status_messages(self, bucket: str | None = None) -> None:
        from fast_agent.agents.llm_agent import LlmAgent

        if isinstance(self._agent, LlmAgent):
            self._agent.flush_deferred_hook_status_messages(bucket)

    def _clear_deferred_hook_status_messages(self, bucket: str | None = None) -> None:
        from fast_agent.agents.llm_agent import LlmAgent

        if isinstance(self._agent, LlmAgent):
            self._agent.clear_deferred_hook_status_messages(bucket)

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
            try:
                with self._defer_hook_status_messages(_HOOK_STATUS_BUCKET_BEFORE_LLM_CALL):
                    await self._hooks.before_llm_call(self, self._delta_messages)
            finally:
                self._flush_deferred_hook_status_messages(_HOOK_STATUS_BUCKET_BEFORE_LLM_CALL)

        tools_for_call = (
            []
            if self._agent.should_suppress_tools_for_structured_turn(
                self._delta_messages,
                self._request_params,
                self._tools,
            )
            else self._tools
        )

        # Retry loop: lets ``on_pause_cancel`` intercept a CancelledError
        # raised mid-LLM-call (e.g. when PauseController cancels the
        # task to interrupt a long-running stream). If the hook returns
        # True the LLM call is reissued with the same ``_delta_messages``
        # — effectively "pause + resume" without unwinding the chat
        # request task. ``task.uncancel()`` clears the pending-cancel
        # state so subsequent awaits don't immediately re-raise.
        # Falls through (raise) when no hook is registered or the hook
        # says the cancel was genuine.
        while True:
            try:
                assistant_message = await self._agent._tool_runner_llm_step(
                    self._delta_messages,
                    request_params=self._request_params,
                    tools=tools_for_call,
                )
            except asyncio.CancelledError:
                # Path A: provider re-raised CancelledError. Some providers
                # (anthropic, openai) catch it themselves and convert to
                # ``stop_reason=CANCELLED`` instead — handled in Path B below.
                hook = self._hooks.on_pause_cancel
                if hook is None:
                    raise
                should_retry = await hook(self)
                if not should_retry:
                    raise
                try:
                    asyncio.current_task().uncancel()
                except (AttributeError, RuntimeError):
                    pass  # uncancel unavailable / not a Task — re-raise next iter
                continue

            # Path B: provider swallowed CancelledError and returned a
            # ``stop_reason=CANCELLED`` response (anthropic + openai
            # graceful-cancel path). Call on_pause_cancel manually so
            # the pause hook still gets to await resume and decide
            # retry — without this dispatch the pause is invisible to
            # the controller and the UI is stuck on "Pausing…" because
            # ``agent_paused`` never emits.
            #
            # Asymmetry vs Path A (intentional)
            # ---------------------------------
            # On a genuine cancel (hook returns False or absent), Path A
            # ``raise``s — ``after_llm_call`` is skipped and
            # ``until_done`` enters its ``except asyncio.CancelledError``
            # branch (``_persist_cancelled_turn_state_after_task_cancel``).
            # Path B instead falls through to ``break`` → ``after_llm_call``
            # is invoked with the CANCELLED message → ``until_done``
            # enters its ``if last.stop_reason == LlmStopReason.CANCELLED``
            # branch (``_persist_cancelled_turn_state``). These are two
            # different persistence paths on purpose: Path A is "task was
            # cancelled mid-stream, partial state" while Path B is
            # "provider returned a complete CANCELLED response with
            # metadata". Hooks observing ``after_llm_call`` may see a
            # CANCELLED message on Path B; they will not on Path A.
            if assistant_message.stop_reason == LlmStopReason.CANCELLED:
                hook = self._hooks.on_pause_cancel
                if hook is not None and await hook(self):
                    # Hook awaited resume and asked for retry. Reissue
                    # the LLM call with the same delta_messages so the
                    # cancelled response is replaced by the real one.
                    continue
                # else: genuine cancel — fall through with the
                # CANCELLED message and let ``until_done`` route through
                # its ``last.stop_reason == LlmStopReason.CANCELLED``
                # rollback branch.
            break

        self._last_message = assistant_message
        if self._hooks.after_llm_call is not None:
            bucket = (
                _HOOK_STATUS_BUCKET_AFTER_TOOL_RESULTS
                if assistant_message.stop_reason == LlmStopReason.TOOL_USE
                else _HOOK_STATUS_BUCKET_AFTER_LLM_CALL
            )
            try:
                with self._defer_hook_status_messages(bucket):
                    await self._hooks.after_llm_call(self, assistant_message)
            except Exception:
                if assistant_message.stop_reason == LlmStopReason.TOOL_USE:
                    self._clear_deferred_hook_status_messages(bucket)
                else:
                    self._flush_deferred_hook_status_messages(bucket)
                raise
            if assistant_message.stop_reason != LlmStopReason.TOOL_USE:
                self._flush_deferred_hook_status_messages(bucket)

        if assistant_message.stop_reason == LlmStopReason.TOOL_USE:
            self._pending_tool_request = assistant_message
            self._pending_tool_response = None  # Clear cache for new request
        elif self._should_start_deferred_structured_finalization(assistant_message):
            self._start_deferred_structured_finalization(assistant_message)
        else:
            self._done = True

        return assistant_message

    async def until_done(self) -> PromptMessageExtended:
        last: PromptMessageExtended | None = None
        try:
            async for message in self:
                last = message
                if message.stop_reason == LlmStopReason.TOOL_USE:
                    await self._persist_tool_loop_checkpoint(message)
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
                try:
                    with self._defer_hook_status_messages(_HOOK_STATUS_BUCKET_AFTER_TURN_COMPLETE):
                        await self._hooks.after_turn_complete(self, last)
                finally:
                    self._flush_deferred_hook_status_messages(
                        _HOOK_STATUS_BUCKET_AFTER_TURN_COMPLETE
                    )

            return last
        except asyncio.CancelledError:
            self._clear_deferred_hook_status_messages()
            rollback_state = self._reset_history_after_cancelled_turn()
            self._record_cancelled_turn(
                reason="cancelled",
                rollback_state=rollback_state,
            )
            await self._persist_cancelled_turn_state_after_task_cancel()
            raise
        except KeyboardInterrupt:
            self._clear_deferred_hook_status_messages()
            rollback_state = self._reset_history_after_cancelled_turn()
            self._record_cancelled_turn(
                reason="interrupted",
                rollback_state=rollback_state,
            )
            await self._persist_cancelled_turn_state()
            raise
        except Exception:
            self._clear_deferred_hook_status_messages()
            await self._persist_exception_turn_state()
            raise

    def _record_cancelled_turn(
        self,
        *,
        reason: str,
        rollback_state: HistoryRollbackState,
    ) -> None:
        if isinstance(self._agent, TurnCancellationStateCapable):
            self._agent.record_last_turn_cancellation(
                reason=reason,
                rollback_state=rollback_state,
            )

    async def _persist_cancelled_turn_state(self) -> None:
        """Persist reconciled history for cancelled turns when session history is enabled."""
        await self._persist_session_history_best_effort(hook_type="after_turn_cancelled")

    async def _persist_tool_loop_checkpoint(self, message: PromptMessageExtended) -> None:
        """Persist in-progress tool-loop history after each tool-use response."""
        if not self._use_history_enabled():
            return
        await self._persist_session_history_best_effort(
            message=message,
            hook_type="after_tool_loop_iteration",
        )

    async def _persist_exception_turn_state(self) -> None:
        """Persist the last resumable tool-loop checkpoint on unhandled exceptions."""
        history_override = self._history_for_resumable_persistence()
        await self._persist_session_history_best_effort(
            hook_type="after_turn_error",
            history_override=history_override,
        )

    async def _persist_session_history_best_effort(
        self,
        *,
        hook_type: str,
        message: PromptMessageExtended | None = None,
        history_override: list[PromptMessageExtended] | None = None,
    ) -> None:
        """Best-effort session-history persistence for non-terminal tool-loop states."""
        history = history_override if history_override is not None else self._agent.message_history
        if not history:
            return

        try:
            from fast_agent.hooks.hook_context import HookContext
            from fast_agent.hooks.session_history import save_session_history

            await save_session_history(
                HookContext(
                    runner=self,
                    agent=cast("HookAgentProtocol", self._agent),
                    message=message if message is not None else history[-1],
                    hook_type=hook_type,
                    message_history_override=history_override,
                )
            )
        except Exception as exc:
            _logger.warning(
                "Failed to persist tool-loop session history",
                hook_type=hook_type,
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

        pending_request = ToolRunner._pending_tool_request_at_history_end(history)
        if pending_request is not None:
            interrupted_tool_message = ToolRunner._build_interrupted_tool_result(
                pending_request
            )
            updated_history = [*history, interrupted_tool_message]
            agent.load_message_history(updated_history)
            return HistoryRollbackState(
                status="appended_interrupted_tool_result",
                history_before=history_before,
                history_after=len(updated_history),
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
                try:
                    with self._defer_hook_status_messages(_HOOK_STATUS_BUCKET_BEFORE_TOOL_CALL):
                        await self._hooks.before_tool_call(self, self._pending_tool_request)
                finally:
                    self._flush_deferred_hook_status_messages(
                        _HOOK_STATUS_BUCKET_BEFORE_TOOL_CALL
                    )
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

        # Layer A — cap oversized tool results BEFORE they enter the staged
        # history. See _sanitize_oversized_tool_results for incident notes.
        tool_message = _sanitize_oversized_tool_results(
            tool_message,
            agent_name=getattr(self._agent, "name", "agent") or "agent",
        )

        self._pending_tool_response = tool_message

        if self._hooks.after_tool_call is not None:
            try:
                with self._defer_hook_status_messages(_HOOK_STATUS_BUCKET_AFTER_TOOL_RESULTS):
                    await self._hooks.after_tool_call(self, tool_message)
            except Exception as exc:
                _logger.error("Tool hook failed after tool call", exc_info=exc)
            finally:
                self._flush_deferred_hook_status_messages(_HOOK_STATUS_BUCKET_AFTER_TOOL_RESULTS)
        else:
            self._flush_deferred_hook_status_messages(_HOOK_STATUS_BUCKET_AFTER_TOOL_RESULTS)
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
        staged_messages = [tool_message]
        channels = tool_message.channels
        if channels and FAST_AGENT_PENDING_MEDIA_ATTACHMENTS in channels:
            pending_media = channels[FAST_AGENT_PENDING_MEDIA_ATTACHMENTS]
            visible_channels = dict(channels)
            del visible_channels[FAST_AGENT_PENDING_MEDIA_ATTACHMENTS]
            staged_messages = [
                tool_message.model_copy(update={"channels": visible_channels or None}),
                PromptMessageExtended(role="user", content=list(pending_media)),
            ]

        if self._use_history_enabled():
            self._delta_messages = staged_messages
        else:
            if self._last_message is not None:
                self._delta_messages.append(self._last_message)
            self._delta_messages.extend(staged_messages)

    def _should_start_deferred_structured_finalization(
        self,
        assistant_message: PromptMessageExtended,
    ) -> bool:
        if self._deferred_structured_finalization_started:
            return False
        return self._agent.should_finalize_deferred_structured_turn(
            self._delta_messages,
            self._request_params,
            self._tools,
            assistant_message,
        )

    def _start_deferred_structured_finalization(
        self,
        assistant_message: PromptMessageExtended,
    ) -> None:
        self._deferred_structured_finalization_started = True
        finalizer = PromptMessageExtended(
            role="user",
            content=[
                TextContent(
                    type="text",
                    text=(
                        "Now produce the final answer as structured JSON matching the "
                        "requested schema. Do not call any tools."
                    ),
                )
            ],
        )
        if self._use_history_enabled():
            self._delta_messages = [finalizer]
        else:
            self._delta_messages.append(assistant_message)
            self._delta_messages.append(finalizer)
        self._tools = []

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
        return tool_result_mode_is_passthrough(self._request_params.tool_result_mode)

    def _append_history_messages(self, *messages: PromptMessageExtended) -> None:
        history = list(self._agent.message_history)
        history.extend(messages)
        self._agent.load_message_history(history)

    @staticmethod
    def _pending_tool_request_at_history_end(
        history: list[PromptMessageExtended],
    ) -> PromptMessageExtended | None:
        if not history:
            return None

        last_message = history[-1]
        if (
            last_message.role == "assistant"
            and (last_message.tool_calls or {})
            and last_message.stop_reason == LlmStopReason.TOOL_USE
        ):
            return last_message
        return None

    def _history_for_resumable_persistence(self) -> list[PromptMessageExtended] | None:
        history = list(self._agent.message_history)
        if not history:
            return None

        if not self._use_history_enabled():
            return history

        pending_request = self._pending_tool_request_at_history_end(history)
        if (
            pending_request is None
            or self._pending_tool_response is None
            or self._pending_tool_request is not None
        ):
            return history

        return [*history, self._pending_tool_response.model_copy(deep=True)]

    def _synthesize_passthrough_assistant(
        self,
        tool_message: PromptMessageExtended,
    ) -> PromptMessageExtended:
        content_blocks = [
            content
            for tool_result in (tool_message.tool_results or {}).values()
            for content in tool_result.content
        ]

        channels: dict[str, list[ContentBlock]] = {
            FAST_AGENT_SYNTHETIC_FINAL_CHANNEL: [text_content("tool_result_passthrough")]
        }
        if self._last_message is not None and self._last_message.channels:
            for channel_name in (FAST_AGENT_TIMING, FAST_AGENT_USAGE):
                blocks = self._last_message.channels.get(channel_name)
                if blocks:
                    channels[channel_name] = list(blocks)

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
                try:
                    with self._defer_hook_status_messages(_HOOK_STATUS_BUCKET_AFTER_LLM_CALL):
                        await self._hooks.after_llm_call(self, terminal_message)
                finally:
                    self._flush_deferred_hook_status_messages(_HOOK_STATUS_BUCKET_AFTER_LLM_CALL)

            self._staged_terminal_response = terminal_message
            self._done = True
            return

        self._stage_tool_response(tool_message)
