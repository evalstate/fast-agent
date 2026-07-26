"""Built-in one-shot subagent tool."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from typing import TYPE_CHECKING, Annotated, Protocol, cast

from fastmcp.tools import ToolResult
from pydantic import Field

from fast_agent.agents.subagent_directive import resolve_subagent_directive
from fast_agent.agents.subagent_labels import (
    SubagentLabel,
    generate_subagent_label,
    resolve_subagent_label,
)
from fast_agent.agents.tool_agent import ToolAgent
from fast_agent.agents.tool_runner import ToolRunnerHooks
from fast_agent.constants import (
    BUILTIN_SUBAGENT_TOOL_NAME,
    FAST_AGENT_SUBAGENT_RESULT_METADATA,
)
from fast_agent.core.exceptions import AgentConfigError
from fast_agent.core.logging.logger import get_logger
from fast_agent.event_progress import ProgressAction, ProgressEvent
from fast_agent.mcp.helpers.content_helpers import text_content
from fast_agent.mcp.prompt import Prompt
from fast_agent.session import (
    Session,
    SessionChildLinkSnapshot,
    SessionExecutionStatus,
    get_active_session_manager,
)
from fast_agent.session.history_agent import HistoryAgent
from fast_agent.tools.function_tool_loader import build_default_function_tool
from fast_agent.tools.invocation_context import get_local_tool_invocation_context
from fast_agent.ui.display_suppression import suppress_interactive_display

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from fast_agent.agents.tool_runner import ToolRunner
    from fast_agent.interfaces import AgentProtocol
    from fast_agent.session.session_manager import SessionManager
    from fast_agent.types import PromptMessageExtended

SUBAGENT_TOOL_NAME = BUILTIN_SUBAGENT_TOOL_NAME
SUBAGENT_TOOL_METADATA = {"fast_agent": {"builtin": SUBAGENT_TOOL_NAME}}
logger = get_logger(__name__)
_SHUTDOWN_TIMEOUT_SECONDS = 10.0
_FINALIZATION_TIMEOUT_SECONDS = 15.0

_DESCRIPTION = (
    "Run a focused subagent in a clean context. It inherits your instruction and available "
    "capabilities, except it does not receive this subagent tool. Give it a complete task and "
    "all necessary context. The tool returns only its final answer. The optional model overrides "
    "the current model for this run. The optional label is a short display name for this run."
)


class ProgressEventDisplay(Protocol):
    """The small progress-display surface used by a subagent run."""

    def update(self, event: ProgressEvent) -> None: ...


class _SubagentMonitorCoordinator:
    """Own the live subagent rows for one installed parent tool."""

    def __init__(self, *, display: ProgressEventDisplay, parent_name: str) -> None:
        self._display = display
        self._parent_name = parent_name
        self._active: dict[str, _SubagentProgress] = {}

    def start(
        self,
        *,
        label: str,
        child_name: str,
        parent_tool_call_id: str | None,
    ) -> "_SubagentProgress":
        progress = _SubagentProgress(
            coordinator=self,
            label=label,
            child_name=child_name,
            parent_tool_call_id=parent_tool_call_id,
            row_id=(
                f"{self._parent_name}::subagent::"
                f"{parent_tool_call_id or child_name}"
            ),
        )
        self._active[child_name] = progress
        self._emit_parent()
        progress.running(0)
        return progress

    def update_child(self, progress: "_SubagentProgress", details: str) -> None:
        self._emit(
            action=ProgressAction.RUNNING,
            target=progress.label,
            agent_name=progress.child_name,
            details=details,
            parent_tool_call_id=progress.parent_tool_call_id,
            row_id=progress.row_id,
        )

    def finish(self, progress: "_SubagentProgress", status: str) -> None:
        self._emit(
            action=ProgressAction.READY,
            target=progress.label,
            agent_name=progress.child_name,
            details=status,
            parent_tool_call_id=progress.parent_tool_call_id,
            row_id=progress.row_id,
        )
        self._active.pop(progress.child_name, None)
        if self._active:
            self._emit_parent()
        else:
            self._emit(
                action=ProgressAction.READY,
                target=self._parent_name,
                agent_name=self._parent_name,
                details="",
                parent_tool_call_id=None,
                row_id=None,
            )

    def _emit_parent(self) -> None:
        labels = [progress.label for progress in self._active.values()]
        noun = "subagent" if len(labels) == 1 else "subagents"
        self._emit(
            action=ProgressAction.MONITORING,
            target=self._parent_name,
            agent_name=self._parent_name,
            details=f"{len(labels)} {noun} · {', '.join(labels)}",
            parent_tool_call_id=None,
            row_id=None,
        )

    def _emit(
        self,
        *,
        action: ProgressAction,
        target: str,
        agent_name: str,
        details: str,
        parent_tool_call_id: str | None,
        row_id: str | None,
    ) -> None:
        self._display.update(
            ProgressEvent(
                action=action,
                target=target,
                details=details,
                agent_name=agent_name,
                correlation_id=parent_tool_call_id,
                instance_name=row_id or agent_name,
                tool_name=SUBAGENT_TOOL_NAME,
                tool_event="subagent_monitor" if row_id is not None else None,
            )
        )


class _SubagentProgress:
    """One live child row managed by its installed parent's coordinator."""

    def __init__(
        self,
        *,
        coordinator: _SubagentMonitorCoordinator,
        label: str,
        child_name: str,
        parent_tool_call_id: str | None,
        row_id: str,
    ) -> None:
        self._coordinator = coordinator
        self.label = label
        self.child_name = child_name
        self.parent_tool_call_id = parent_tool_call_id
        self.row_id = row_id
        self._agent: ToolAgent | None = None
        self._turn = 0
        self._tool_count = 0
        self._current_tool_name: str | None = None
        self._activity = "starting"

    def attach(self, agent: ToolAgent) -> None:
        self._agent = agent

    def running(self, turn: int) -> None:
        self._turn = turn
        self._coordinator.update_child(self, self._details())

    def before_llm_call(self, turn: int) -> None:
        self._current_tool_name = None
        self._activity = "thinking"
        self.running(turn)

    def after_llm_call(self, turn: int) -> None:
        self._activity = "processing"
        self.running(turn)

    def before_tool_call(self, turn: int, tool_names: list[str]) -> None:
        self._tool_count += len(tool_names)
        self._current_tool_name = (
            tool_names[0] if len(tool_names) == 1 else f"{len(tool_names)} tools"
        )
        self._activity = "tool"
        self.running(turn)

    def after_tool_call(self, turn: int) -> None:
        self._current_tool_name = None
        self._activity = "processing"
        self.running(turn)

    def finalizing(self) -> None:
        self._activity = "finalizing"
        self.running(self._turn)

    def finish(self, status: str) -> None:
        self._coordinator.finish(self, status)

    def _details(self) -> str:
        tool_details = f"tools {self._tool_count}"
        if self._current_tool_name:
            tool_details = f"{tool_details} ({self._current_tool_name})"
        return " · ".join(
            (f"turn {self._turn}", self._activity, self._usage_details(), tool_details)
        )

    def _usage_details(self) -> str:
        if self._agent is None or self._agent.usage_accumulator is None:
            return "in 0 out 0 cache 0"
        summary = self._agent.usage_accumulator.summary
        cache_total = _cache_total(summary.prompt.cache_read, summary.prompt.cache_write) or 0
        return " ".join(
            (
                f"in {summary.prompt.total or 0}",
                f"out {summary.completion.total or 0}",
                f"cache {cache_total}",
            )
        )


def _cache_total(cache_read: int | None, cache_write: int | None) -> int | None:
    values = [value for value in (cache_read, cache_write) if value is not None]
    return sum(values) if values else None


def _default_progress_display() -> ProgressEventDisplay:
    from fast_agent.ui.progress_display import progress_display

    return progress_display


def install_subagent_tool(
    agent: object,
    *,
    progress_display: ProgressEventDisplay | None = None,
    label_generator: Callable[[], str] | None = None,
) -> bool:
    """Install the built-in subagent tool on a compatible top-level agent."""
    if not isinstance(agent, ToolAgent):
        return False
    directive = resolve_subagent_directive(agent.instruction)
    if directive.found:
        agent.set_instruction(directive.instruction)
        agent.config.instruction = directive.instruction
        if agent.config.subagents is None and not agent.config.tool_only:
            agent.config.subagents = True
            agent.config.subagent_activation_source = "instruction"
    if agent.config.tool_only or agent.config.subagents is not True:
        existing = agent._execution_tools.get(SUBAGENT_TOOL_NAME)
        if existing is not None and existing.meta == SUBAGENT_TOOL_METADATA:
            agent.remove_tool(SUBAGENT_TOOL_NAME)
        return False

    existing = agent._execution_tools.get(SUBAGENT_TOOL_NAME)
    if existing is not None:
        if existing.meta == SUBAGENT_TOOL_METADATA:
            return True
        raise AgentConfigError(f"Tool name '{SUBAGENT_TOOL_NAME}' is reserved by fast-agent")

    logger.info(
        "Enabled built-in subagent tool",
        data={
            "agent_name": agent.name,
            "activation_source": agent.config.subagent_activation_source or "configuration",
        },
    )

    used_labels: set[str] = set()
    generate_label = label_generator or generate_subagent_label
    row_display = progress_display or _default_progress_display()
    monitor = _SubagentMonitorCoordinator(display=row_display, parent_name=agent.name)

    async def _run_subagent(
        message: Annotated[
            str,
            Field(
                min_length=1,
                description="A complete task for the subagent, including all required context.",
            ),
        ],
        model: Annotated[
            str | None,
            Field(
                description=(
                    "Optional model override. Omit to inherit the parent's current model."
                )
            ),
        ] = None,
        label: Annotated[
            SubagentLabel | None,
            Field(
                description=(
                    "Optional short display label for this subagent. It is not a durable "
                    "identity and must be 1-32 ASCII letters, digits, spaces, underscores, "
                    "or hyphens, beginning and ending with a letter or digit."
                )
            ),
        ] = None,
        *,
        model_source: str,
    ) -> ToolResult:
        """Run a complete task in a fresh one-shot child agent."""
        resolved_label = resolve_subagent_label(
            label,
            used_labels=used_labels,
            generator=generate_label,
        )
        context = get_local_tool_invocation_context()
        parent_tool_call_id = context.tool_use_id if context is not None else None
        child_session = _create_child_session(agent)
        child_name = f"{agent.name}[{resolved_label}]"
        clone: ToolAgent | None = None
        status: SessionExecutionStatus = "running"
        response_text = ""
        is_error = False
        cancellation: asyncio.CancelledError | None = None
        progress = monitor.start(
            label=resolved_label,
            child_name=child_name,
            parent_tool_call_id=parent_tool_call_id,
        )
        try:
            clone = await agent.spawn_isolated_instance(
                name=child_name,
                model=model,
            )
            clone.config.subagents = False
            clone.config.subagent_activation_source = None
            clone.set_session_history_persistence_enabled(False)
            clone.remove_tool(SUBAGENT_TOOL_NAME)
            clone.load_message_history([])
            progress.attach(clone)
            clone.tool_runner_hooks = ToolAgent._merge_tool_runner_hooks(
                clone.tool_runner_hooks,
                _subagent_progress_hooks(progress),
            )
            with _child_chat_suppressed(clone):
                response = await clone.generate([Prompt.user(message)])
            status = "completed"
            response_text = response.last_text() or ""
        except asyncio.CancelledError as exc:
            status = "cancelled"
            cancellation = exc
        except Exception as exc:
            status = "failed"
            response_text = f"Error: {exc!s}"
            is_error = True
        finally:
            finalizer = asyncio.create_task(
                _finalize_subagent_run(
                    parent=agent,
                    clone=clone,
                    child_session=child_session,
                    child_name=child_name,
                    status=status,
                    progress=progress,
                )
            )
            while not finalizer.done():
                try:
                    await asyncio.shield(finalizer)
                except asyncio.CancelledError as exc:
                    cancellation = cancellation or exc
            try:
                finalizer.result()
            except TimeoutError:
                logger.warning(
                    "Timed out while finalizing subagent",
                    data={"agent_name": child_name, "status": status},
                )
            except Exception as exc:
                logger.warning(
                    "Failed to finalize subagent",
                    data={"agent_name": child_name, "status": status, "error": str(exc)},
                )

        if cancellation is not None:
            raise cancellation

        return ToolResult(
            content=[text_content(response_text)],
            meta={
                FAST_AGENT_SUBAGENT_RESULT_METADATA: _subagent_result_metadata(
                    child_session=child_session,
                    child_name=child_name,
                    requested_label=label,
                    label=resolved_label,
                    clone=clone,
                    status=status,
                    parent_tool_call_id=parent_tool_call_id,
                    model_source=model_source,
                )
            },
            is_error=is_error,
        )

    forced_model = agent.config.subagent_model
    if forced_model is None:

        async def subagent(
            message: Annotated[
                str,
                Field(
                    min_length=1,
                    description="A complete task for the subagent, including all required context.",
                ),
            ],
            model: Annotated[
                str | None,
                Field(
                    description=(
                        "Optional model override. Omit to inherit the parent's current model."
                    )
                ),
            ] = None,
            label: Annotated[
                SubagentLabel | None,
                Field(
                    description=(
                        "Optional short display label for this subagent. It is not a durable "
                        "identity and must be 1-32 ASCII letters, digits, spaces, underscores, "
                        "or hyphens, beginning and ending with a letter or digit."
                    )
                ),
            ] = None,
        ) -> ToolResult:
            return await _run_subagent(
                message,
                model,
                label,
                model_source="tool_override" if model is not None else "parent",
            )

        description = _DESCRIPTION
    else:

        async def subagent(
            message: Annotated[
                str,
                Field(
                    min_length=1,
                    description="A complete task for the subagent, including all required context.",
                ),
            ],
            model: str | None = None,
            label: Annotated[
                SubagentLabel | None,
                Field(
                    description=(
                        "Optional short display label for this subagent. It is not a durable "
                        "identity and must be 1-32 ASCII letters, digits, spaces, underscores, "
                        "or hyphens, beginning and ending with a letter or digit."
                    )
                ),
            ] = None,
        ) -> ToolResult:
            del model
            return await _run_subagent(
                message,
                forced_model,
                label,
                model_source="agent_card",
            )

        description = (
            "Run a focused subagent in a clean context. It inherits your instruction and "
            "available capabilities, except it does not receive this subagent tool. Give it a "
            f"complete task and all necessary context. Each run uses the fixed model "
            f"`{forced_model}`. The optional label is a short display name for this run."
        )

    tool = build_default_function_tool(
        subagent,
        name=SUBAGENT_TOOL_NAME,
        description=description,
        metadata=SUBAGENT_TOOL_METADATA,
    )
    if forced_model is not None:
        tool.parameters["properties"].pop("model", None)
    agent.add_tool(tool, replace=False)
    return True


def _create_child_session(agent: ToolAgent) -> Session | None:
    """Create a child persistence target when this invocation has an active parent."""
    manager: SessionManager | None = (
        agent.context.session_manager if agent.context is not None else None
    )
    if manager is None or manager.current_session is None:
        try:
            active_manager = get_active_session_manager()
        except RuntimeError:
            active_manager = None
        if active_manager is not None and active_manager.current_session is not None:
            manager = active_manager

    if manager is None:
        return None
    parent = manager.current_session
    if parent is None:
        return None

    context = get_local_tool_invocation_context()
    try:
        return manager.create_child_session(
            parent,
            SessionChildLinkSnapshot(
                parent_session_id=parent.info.name,
                parent_agent_name=agent.name,
                parent_tool_call_id=context.tool_use_id if context is not None else None,
            ),
        )
    except Exception as exc:
        logger.warning(
            "Failed to create subagent child session",
            data={"parent_session": parent.info.name, "error": str(exc)},
        )
        return None


@contextmanager
def _child_chat_suppressed(agent: ToolAgent) -> Iterator[None]:
    """Hide nested panels while retaining tool execution and progress events."""
    settings = agent.display.logger_settings
    agent.display.update_logger_settings(
        settings.model_copy(update={"show_chat": False, "show_tools": False})
    )
    try:
        with suppress_interactive_display("monitor_only"):
            yield
    finally:
        agent.display.update_logger_settings(settings)


async def _finalize_subagent_run(
    *,
    parent: ToolAgent,
    clone: ToolAgent | None,
    child_session: Session | None,
    child_name: str,
    status: SessionExecutionStatus,
    progress: _SubagentProgress,
) -> None:
    """Persist and release one child without inheriting caller cancellation."""
    progress.finalizing()
    try:
        async with asyncio.timeout(_FINALIZATION_TIMEOUT_SECONDS):
            if child_session is not None:
                try:
                    child_session.set_execution_status(status)
                except Exception as exc:
                    logger.warning(
                        "Failed to update subagent child session status",
                        data={
                            "session": child_session.info.name,
                            "status": status,
                            "error": str(exc),
                        },
                    )
            if clone is not None and child_session is not None:
                try:
                    history_agent = HistoryAgent(
                        clone,
                        [
                            message.model_copy(deep=True)
                            for message in (clone.message_history or clone.last_turn_messages)
                        ],
                    )
                    await child_session.save_history(cast("AgentProtocol", history_agent))
                except Exception as exc:
                    logger.warning(
                        "Failed to persist subagent child session",
                        data={"session": child_session.info.name, "error": str(exc)},
                    )
            if clone is not None:
                try:
                    async with asyncio.timeout(_SHUTDOWN_TIMEOUT_SECONDS):
                        await clone.shutdown()
                except TimeoutError:
                    logger.warning(
                        "Timed out while shutting down subagent",
                        data={"agent_name": child_name},
                    )
                except Exception as exc:
                    logger.warning("Failed to shut down subagent", data={"error": str(exc)})
                try:
                    parent.merge_usage_from(clone)
                except Exception as exc:
                    logger.warning("Failed to merge subagent usage", data={"error": str(exc)})
    finally:
        progress.finish(status)


def _subagent_result_metadata(
    *,
    child_session: Session | None,
    child_name: str,
    requested_label: str | None,
    label: str,
    clone: ToolAgent | None,
    status: str,
    parent_tool_call_id: str | None,
    model_source: str,
) -> dict[str, object]:
    """Build durable, display-oriented details without changing model text."""
    model_spec: str | None = None
    provider: str | None = None
    usage: dict[str, int | None] = {
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
    }
    turn_count = 0

    if clone is not None:
        llm = clone.llm
        if llm is not None:
            resolved_model = llm.resolved_model
            model_spec = (
                resolved_model.selected_model_name
                or resolved_model.wire_model_name
                or clone.config.model
            )
            provider = resolved_model.provider.value

        accumulator = clone.usage_accumulator
        if accumulator is not None:
            summary = accumulator.summary
            usage = {
                "prompt_tokens": summary.prompt.total,
                "completion_tokens": summary.completion.total,
                "total_tokens": summary.total,
            }
            turn_count = len(accumulator.turns)

    return {
        "child_session_id": child_session.info.name if child_session is not None else None,
        "child_agent_name": child_name,
        "requested_label": requested_label,
        "label": label,
        "parent_tool_call_id": parent_tool_call_id,
        "model_spec": model_spec,
        "model_source": model_source,
        "provider": provider,
        "status": status,
        "usage": usage,
        "turn_count": turn_count,
    }


def _subagent_progress_hooks(progress: _SubagentProgress) -> ToolRunnerHooks:
    async def before_llm_call(runner: ToolRunner, _messages: list[PromptMessageExtended]) -> None:
        progress.before_llm_call(runner.iteration + 1)

    async def after_llm_call(runner: ToolRunner, _message: PromptMessageExtended) -> None:
        progress.after_llm_call(runner.iteration + 1)

    async def before_tool_call(runner: ToolRunner, request: PromptMessageExtended) -> None:
        tool_names = [call.params.name for call in (request.tool_calls or {}).values()]
        progress.before_tool_call(runner.iteration + 1, tool_names)

    async def after_tool_call(runner: ToolRunner, _message: PromptMessageExtended) -> None:
        progress.after_tool_call(runner.iteration + 1)

    async def after_turn_complete(runner: ToolRunner, _message: PromptMessageExtended) -> None:
        progress.after_llm_call(runner.iteration + 1)

    return ToolRunnerHooks(
        before_llm_call=before_llm_call,
        after_llm_call=after_llm_call,
        before_tool_call=before_tool_call,
        after_tool_call=after_tool_call,
        after_turn_complete=after_turn_complete,
    )
