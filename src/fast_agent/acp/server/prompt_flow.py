from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Callable, Protocol

from acp.exceptions import RequestError
from acp.helpers import ContentBlock as ACPContentBlock
from acp.helpers import update_agent_message_text, update_agent_thought_text
from acp.schema import PromptResponse, StopReason

from fast_agent.acp.content_conversion import (
    convert_acp_prompt_to_mcp_content_blocks,
    inline_resources_for_slash_command,
)
from fast_agent.acp.server.common import (
    CANCELLED,
    END_TURN,
    REFUSAL,
    clear_current_task_cancellation_requests,
    map_llm_stop_reason_to_acp,
)
from fast_agent.agents.tool_runner import ToolRunnerHooks
from fast_agent.core.exceptions import ProviderKeyError
from fast_agent.core.logging.logger import get_logger
from fast_agent.interfaces import AgentProtocol, StreamingAgentProtocol, ToolRunnerHookCapable
from fast_agent.mcp.helpers.content_helpers import is_text_content
from fast_agent.types import LlmStopReason, PromptMessageExtended
from fast_agent.ui.interactive_diagnostics import write_interactive_trace

if TYPE_CHECKING:
    from fast_agent.acp.server.models import ACPSessionState
    from fast_agent.core.fastagent import AgentInstance
    from fast_agent.llm.stream_types import StreamChunk

logger = get_logger(__name__)


class PromptFlowHost(Protocol):
    sessions: dict[str, Any]
    _session_lock: asyncio.Lock
    _prompt_locks: dict[str, asyncio.Lock]
    _active_prompts: set[str]
    _session_tasks: dict[str, asyncio.Task]
    _session_state: dict[str, ACPSessionState]
    _connection: Any
    primary_agent_name: str | None

    def _resolve_primary_agent_name(self, instance: AgentInstance) -> str | None: ...

    async def _build_session_request_params(
        self, agent: Any, session_state: ACPSessionState | None
    ) -> Any: ...

    def _merge_tool_runner_hooks(
        self, base: ToolRunnerHooks | None, extra: ToolRunnerHooks | None
    ) -> ToolRunnerHooks | None: ...

    def _build_status_line_meta(
        self, agent: Any, turn_start_index: int | None
    ) -> dict[str, Any] | None: ...

    async def _send_status_line_update(
        self, session_id: str, agent: Any, turn_start_index: int | None
    ) -> None: ...

    def _build_auth_required_data(
        self,
        error: ProviderKeyError,
        *,
        agent: AgentProtocol | object | None = None,
    ) -> dict[str, Any]: ...


class ACPPromptFlow:
    def __init__(self, host: PromptFlowHost) -> None:
        self._host = host

    async def get_prompt_lock(self, session_id: str) -> asyncio.Lock:
        """Get/create the lock used to serialize prompts for a session."""
        async with self._host._session_lock:
            lock = self._host._prompt_locks.get(session_id)
            if lock is None:
                lock = asyncio.Lock()
                self._host._prompt_locks[session_id] = lock
            return lock

    async def prompt(
        self,
        prompt: list[ACPContentBlock],
        session_id: str,
        message_id: str | None = None,
        **kwargs: Any,
    ) -> PromptResponse:
        """Serialize prompt turns per session to avoid interleaved ACP updates."""
        prompt_lock = await self.get_prompt_lock(session_id)
        async with prompt_lock:
            return await self.prompt_locked(
                prompt=prompt,
                session_id=session_id,
                message_id=message_id,
                **kwargs,
            )

    async def prompt_locked(
        self,
        prompt: list[ACPContentBlock],
        session_id: str,
        message_id: str | None = None,
        **kwargs: Any,
    ) -> PromptResponse:
        """
        Handle prompt request.

        Per ACP protocol, only one prompt can be active per session at a time.
        """
        _ = kwargs
        logger.info(
            "ACP prompt request",
            name="acp_prompt",
            session_id=session_id,
        )
        write_interactive_trace("acp.prompt.start", session_id=session_id)

        async with self._host._session_lock:
            self._host._active_prompts.add(session_id)
            current_task = asyncio.current_task()
            if current_task:
                self._host._session_tasks[session_id] = current_task

        try:
            async with self._host._session_lock:
                instance = self._host.sessions.get(session_id)

            if not instance:
                logger.error(
                    "ACP prompt error: session not found",
                    name="acp_prompt_error",
                    session_id=session_id,
                )
                return PromptResponse(stop_reason=REFUSAL)

            processed_prompt = inline_resources_for_slash_command(prompt)
            mcp_content_blocks = convert_acp_prompt_to_mcp_content_blocks(processed_prompt)
            prompt_message = PromptMessageExtended(
                role="user",
                content=mcp_content_blocks,
            )

            session_state = self._host._session_state.get(session_id)
            acp_context = session_state.acp_context if session_state else None
            current_agent_name = None
            if acp_context is not None:
                current_agent_name = acp_context.current_mode
            if not current_agent_name and session_state:
                current_agent_name = session_state.current_agent_name
            if not current_agent_name:
                current_agent_name = self._host._resolve_primary_agent_name(instance)

            slash_handler = session_state.slash_handler if session_state else None
            is_single_text_block = len(mcp_content_blocks) == 1 and is_text_content(
                mcp_content_blocks[0]
            )
            prompt_text = prompt_message.all_text() or ""
            if (
                slash_handler
                and is_single_text_block
                and slash_handler.is_slash_command(prompt_text)
            ):
                return await self._handle_slash_command(
                    slash_handler=slash_handler,
                    session_id=session_id,
                    current_agent_name=current_agent_name,
                    prompt_text=prompt_text,
                    message_id=message_id,
                )

            logger.info(
                "Sending prompt to fast-agent",
                name="acp_prompt_send",
                session_id=session_id,
                agent=current_agent_name,
                content_blocks=len(mcp_content_blocks),
            )

            acp_stop_reason: StopReason = END_TURN
            status_line_meta: dict[str, Any] | None = None
            active_agent: AgentProtocol | object | None = None
            try:
                if current_agent_name:
                    agent = instance.agents[current_agent_name]
                    active_agent = agent
                    stream_context = await self._prepare_streaming_context(
                        agent=agent,
                        session_id=session_id,
                    )

                    try:
                        session_request_params = await self._host._build_session_request_params(
                            agent, session_state
                        )
                        turn_start_index = None
                        if isinstance(agent, AgentProtocol) and agent.usage_accumulator is not None:
                            turn_start_index = len(agent.usage_accumulator.turns)

                        with_status_hooks = await self._run_with_status_hooks(
                            agent=agent,
                            session_id=session_id,
                            turn_start_index=turn_start_index,
                            prompt_message=prompt_message,
                            session_request_params=session_request_params,
                        )
                        result = with_status_hooks["result"]
                        response_text = result.last_text() or "No content generated"
                        status_line_meta = self._host._build_status_line_meta(
                            agent, turn_start_index
                        )

                        try:
                            acp_stop_reason = map_llm_stop_reason_to_acp(result.stop_reason)
                        except Exception as e:
                            logger.error(
                                f"Error mapping stop reason: {e}",
                                name="acp_stop_reason_error",
                                exc_info=True,
                            )
                            acp_stop_reason = END_TURN

                        logger.info(
                            "Received complete response from fast-agent",
                            name="acp_prompt_response",
                            session_id=session_id,
                            response_length=len(response_text),
                            llm_stop_reason=str(result.stop_reason) if result.stop_reason else None,
                            acp_stop_reason=acp_stop_reason,
                        )

                        await self._finalize_prompt_delivery(
                            session_id=session_id,
                            response_text=response_text,
                            streaming_tasks=stream_context["streaming_tasks"],
                            status_line_meta=status_line_meta,
                        )
                    except Exception as send_error:
                        await self._cleanup_stream_listener_after_error(
                            session_id=session_id,
                            stream_listener=stream_context["stream_listener"],
                            remove_listener=stream_context["remove_listener"],
                        )
                        raise send_error
                    finally:
                        await self._cleanup_stream_listener(
                            session_id=session_id,
                            stream_listener=stream_context["stream_listener"],
                            remove_listener=stream_context["remove_listener"],
                        )
                else:
                    logger.error("No primary agent available")
            except ProviderKeyError as e:
                logger.info(
                    "ACP prompt requires provider authentication",
                    name="acp_prompt_auth_required",
                    session_id=session_id,
                    agent=current_agent_name,
                    error=e.message,
                )
                raise RequestError.auth_required(
                    self._host._build_auth_required_data(e, agent=active_agent)
                ) from e
            except Exception as e:
                logger.error(
                    f"Error processing prompt: {e}",
                    name="acp_prompt_error",
                    exc_info=True,
                )
                import sys
                import traceback

                print(f"ERROR processing prompt: {e}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                raise

            return PromptResponse(
                stop_reason=acp_stop_reason,
                field_meta=status_line_meta,
                user_message_id=message_id,
            )
        except asyncio.CancelledError:
            clear_current_task_cancellation_requests(session_id=session_id)
            write_interactive_trace("acp.prompt.cancelled", session_id=session_id)
            logger.info(
                "Prompt cancelled by user",
                name="acp_prompt_cancelled",
                session_id=session_id,
            )
            return PromptResponse(
                stop_reason=CANCELLED,
                user_message_id=message_id,
            )
        finally:
            write_interactive_trace("acp.prompt.finally", session_id=session_id)
            async with self._host._session_lock:
                self._host._active_prompts.discard(session_id)
                self._host._session_tasks.pop(session_id, None)
            logger.debug(
                "Removed session from active prompts",
                name="acp_prompt_complete",
                session_id=session_id,
            )

    async def _handle_slash_command(
        self,
        *,
        slash_handler: Any,
        session_id: str,
        current_agent_name: str | None,
        prompt_text: str,
        message_id: str | None,
    ) -> PromptResponse:
        logger.info(
            "Processing slash command",
            name="acp_slash_command",
            session_id=session_id,
            prompt_text=prompt_text[:100],
        )
        slash_handler.set_current_agent(current_agent_name or "default")
        command_name, arguments = slash_handler.parse_command(prompt_text)
        response_text = await slash_handler.execute_command(command_name, arguments)

        if self._host._connection and response_text:
            try:
                message_chunk = update_agent_message_text(response_text)
                await self._host._connection.session_update(
                    session_id=session_id,
                    update=message_chunk,
                )
                logger.info(
                    "Sent slash command response",
                    name="acp_slash_command_response",
                    session_id=session_id,
                )
            except Exception as e:
                logger.error(
                    f"Error sending slash command response: {e}",
                    name="acp_slash_command_response_error",
                    exc_info=True,
                )

        return PromptResponse(
            stop_reason=END_TURN,
            user_message_id=message_id,
        )

    async def _prepare_streaming_context(
        self,
        *,
        agent: Any,
        session_id: str,
    ) -> dict[str, Any]:
        stream_listener = None
        remove_listener: Callable[[], None] | None = None
        streaming_tasks: list[asyncio.Task] = []
        if self._host._connection and isinstance(agent, StreamingAgentProtocol):
            connection = self._host._connection
            update_lock = asyncio.Lock()

            async def send_stream_update(chunk: StreamChunk) -> None:
                if not chunk.text:
                    return
                try:
                    async with update_lock:
                        if chunk.is_reasoning:
                            message_chunk = update_agent_thought_text(chunk.text)
                        else:
                            message_chunk = update_agent_message_text(chunk.text)
                        await connection.session_update(
                            session_id=session_id,
                            update=message_chunk,
                        )
                except Exception as e:
                    logger.error(
                        f"Error sending stream update: {e}",
                        name="acp_stream_error",
                        exc_info=True,
                    )

            def on_stream_chunk(chunk: StreamChunk) -> None:
                if not chunk or not chunk.text:
                    return
                task = asyncio.create_task(send_stream_update(chunk))
                streaming_tasks.append(task)

            stream_listener = on_stream_chunk
            remove_listener = agent.add_stream_listener(stream_listener)

            logger.info(
                "Streaming enabled for prompt",
                name="acp_streaming_enabled",
                session_id=session_id,
            )

        return {
            "stream_listener": stream_listener,
            "remove_listener": remove_listener,
            "streaming_tasks": streaming_tasks,
        }

    async def _run_with_status_hooks(
        self,
        *,
        agent: Any,
        session_id: str,
        turn_start_index: int | None,
        prompt_message: PromptMessageExtended,
        session_request_params: Any,
    ) -> dict[str, Any]:
        previous_hooks = None
        restore_hooks = False
        tool_hook_agent: ToolRunnerHookCapable | None = None
        if (
            self._host._connection
            and isinstance(agent, ToolRunnerHookCapable)
            and turn_start_index is not None
        ):
            tool_hook_agent = agent

            async def after_llm_call(_runner: Any, message: Any) -> None:
                if message.stop_reason != LlmStopReason.TOOL_USE:
                    return
                await self._host._send_status_line_update(
                    session_id, agent, turn_start_index
                )

            status_hook = ToolRunnerHooks(after_llm_call=after_llm_call)
            try:
                previous_hooks = tool_hook_agent.tool_runner_hooks
                tool_hook_agent.tool_runner_hooks = self._host._merge_tool_runner_hooks(
                    previous_hooks, status_hook
                )
                restore_hooks = True
            except AttributeError:
                previous_hooks = None
                restore_hooks = False

        try:
            result = await agent.generate(
                prompt_message,
                request_params=session_request_params,
            )
        finally:
            if restore_hooks and tool_hook_agent is not None:
                tool_hook_agent.tool_runner_hooks = previous_hooks

        return {"result": result}

    async def _finalize_prompt_delivery(
        self,
        *,
        session_id: str,
        response_text: str,
        streaming_tasks: list[asyncio.Task],
        status_line_meta: dict[str, Any] | None,
    ) -> None:
        if streaming_tasks:
            try:
                await asyncio.gather(*streaming_tasks)
                logger.debug(
                    f"All {len(streaming_tasks)} streaming tasks completed",
                    name="acp_streaming_complete",
                    session_id=session_id,
                    task_count=len(streaming_tasks),
                )
            except Exception as e:
                logger.error(
                    f"Error waiting for streaming tasks: {e}",
                    name="acp_streaming_wait_error",
                    exc_info=True,
                )

        if not streaming_tasks and self._host._connection and response_text:
            try:
                message_chunk = update_agent_message_text(response_text)
                if status_line_meta:
                    await self._host._connection.session_update(
                        session_id=session_id,
                        update=message_chunk,
                        **status_line_meta,
                    )
                else:
                    await self._host._connection.session_update(
                        session_id=session_id,
                        update=message_chunk,
                    )
                logger.info(
                    "Sent final sessionUpdate with complete response (no streaming)",
                    name="acp_final_update",
                    session_id=session_id,
                )
            except Exception as e:
                logger.error(
                    f"Error sending final update: {e}",
                    name="acp_final_update_error",
                    exc_info=True,
                )
        elif streaming_tasks and self._host._connection and status_line_meta:
            try:
                message_chunk = update_agent_message_text("")
                await self._host._connection.session_update(
                    session_id=session_id,
                    update=message_chunk,
                    **status_line_meta,
                )
                logger.debug(
                    "Sent status line metadata update after streaming",
                    name="acp_status_line_update",
                    session_id=session_id,
                )
            except Exception as e:
                logger.error(
                    f"Error sending status line update: {e}",
                    name="acp_status_line_update_error",
                    exc_info=True,
                )

    async def _cleanup_stream_listener_after_error(
        self,
        *,
        session_id: str,
        stream_listener: Any,
        remove_listener: Callable[[], None] | None,
    ) -> None:
        if stream_listener and remove_listener:
            try:
                remove_listener()
                logger.info(
                    "Removed stream listener after error",
                    name="acp_streaming_cleanup_error",
                    session_id=session_id,
                )
            except Exception:
                logger.warning("Failed to remove ACP stream listener after error")

    async def _cleanup_stream_listener(
        self,
        *,
        session_id: str,
        stream_listener: Any,
        remove_listener: Callable[[], None] | None,
    ) -> None:
        if stream_listener and remove_listener:
            try:
                remove_listener()
            except Exception:
                logger.warning("Failed to remove ACP stream listener")
            else:
                logger.info(
                    "Removed stream listener",
                    name="acp_streaming_cleanup",
                    session_id=session_id,
                )
