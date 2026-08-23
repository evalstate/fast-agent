"""Built-in one-shot subagent tool."""

from __future__ import annotations

import asyncio
import time
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Annotated, Protocol, cast, runtime_checkable

from fastmcp.tools import ToolResult
from mcp_types import TextContent
from pydantic import Field

from fast_agent.agents.current_user_message import (
    CurrentUserMessage,
    get_current_user_message,
)
from fast_agent.agents.mcp_agent import McpAgent
from fast_agent.agents.subagent_directive import resolve_subagent_directive
from fast_agent.agents.subagent_labels import (
    SubagentLabel,
    generate_subagent_label,
    resolve_subagent_label,
)
from fast_agent.agents.subagent_transcript import (
    SubagentTranscriptMetadata,
    render_subagent_input,
    render_subagent_transcript,
)
from fast_agent.agents.tool_agent import ToolAgent
from fast_agent.agents.tool_runner import ToolRunnerHooks
from fast_agent.constants import (
    BUILTIN_SUBAGENT_TOOL_NAME,
    FAST_AGENT_SUBAGENT_RESULT_METADATA,
)
from fast_agent.core.exceptions import AgentConfigError
from fast_agent.core.logging.logger import get_logger
from fast_agent.event_progress import ProgressAction, ProgressEvent, SubagentMonitorSnapshot
from fast_agent.llm.model_display_name import resolve_llm_display_name, resolve_model_display_name
from fast_agent.mcp.helpers.content_helpers import text_content
from fast_agent.mcp.prompt import Prompt
from fast_agent.session import (
    Session,
    SessionChildLinkSnapshot,
    SessionExecutionStatus,
    get_active_session_manager,
    subagent_alias_slug,
    subagent_task_preview,
)
from fast_agent.session.history_agent import HistoryAgent
from fast_agent.session.subagent_runs import SUBAGENT_ALIAS_KEY, SUBAGENT_ORDINAL_KEY
from fast_agent.tools.function_tool_loader import build_default_function_tool
from fast_agent.tools.invocation_context import get_local_tool_invocation_context
from fast_agent.ui.display_suppression import suppress_interactive_display

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from fast_agent.agents.tool_runner import ToolRunner
    from fast_agent.interfaces import AgentProtocol
    from fast_agent.llm.stream_types import StreamChunk
    from fast_agent.session.session_manager import SessionManager
    from fast_agent.tools.transient_artifacts import TransientArtifactResult
    from fast_agent.types import PromptMessageExtended

SUBAGENT_TOOL_NAME = BUILTIN_SUBAGENT_TOOL_NAME
SUBAGENT_TOOL_METADATA = {"fast_agent": {"builtin": SUBAGENT_TOOL_NAME}}
logger = get_logger(__name__)
_SHUTDOWN_TIMEOUT_SECONDS = 10.0
_FINALIZATION_TIMEOUT_SECONDS = 15.0
_STREAM_UPDATE_TOKEN_INTERVAL = 8

_DESCRIPTION = (
    "Run a focused subagent in a clean context. It inherits your instruction and available "
    "capabilities, except it does not receive this subagent tool. Give it a complete task and "
    "all necessary context. The tool returns its final answer and, when available, a temporary "
    "path to its complete transcript. The optional model overrides "
    "the current model for this run. The optional label is a short display name for this run. "
    "The optional include_user_message forwards the latest external user text and attachments, "
    "but no history, to the subagent; this may send content to another model or provider."
)


class ProgressEventDisplay(Protocol):
    """The small progress-display surface used by a subagent run."""

    def update(self, event: ProgressEvent) -> None: ...


@dataclass(frozen=True, slots=True)
class _SubagentFinalizationResult:
    transcript: TransientArtifactResult | None = None


@runtime_checkable
class FoldableProgressEventDisplay(Protocol):
    """A progress display that can fold generic child rows into a monitor."""

    def fold_agent_progress(self, agent_name: str) -> None: ...


class _SubagentMonitorCoordinator:
    """Own the live subagent rows for one installed parent tool."""

    def __init__(self, *, display: ProgressEventDisplay, parent_name: str) -> None:
        self._display = display
        self._parent_name = parent_name

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
            row_id=(f"{self._parent_name}::subagent::{parent_tool_call_id or child_name}"),
        )
        if isinstance(self._display, FoldableProgressEventDisplay):
            self._display.fold_agent_progress(child_name)
        progress.running(0)
        return progress

    def update_child(self, progress: "_SubagentProgress") -> None:
        self._emit(
            action=ProgressAction.RUNNING,
            target=progress.label,
            agent_name=progress.child_name,
            details=progress.details(),
            activity=progress.activity,
            parent_tool_call_id=progress.parent_tool_call_id,
            row_id=progress.row_id,
            elapsed_seconds=progress.elapsed_seconds,
            snapshot=progress.snapshot(),
        )

    def finish(self, progress: "_SubagentProgress", status: str) -> None:
        self._emit(
            action=ProgressAction.READY,
            target=progress.label,
            agent_name=progress.child_name,
            details=status,
            parent_tool_call_id=progress.parent_tool_call_id,
            row_id=progress.row_id,
            elapsed_seconds=progress.elapsed_seconds,
            snapshot=progress.snapshot(),
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
        elapsed_seconds: float | None,
        activity: str | None = None,
        snapshot: SubagentMonitorSnapshot | None = None,
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
                elapsed_seconds=elapsed_seconds,
                activity=activity,
                subagent_monitor=snapshot,
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
        self._activity = "Starting"
        self._started_at = time.monotonic()
        self._streamed_chars = 0
        self._estimated_output_tokens = 0
        self._last_emitted_output_estimate = 0
        self._remove_stream_listener: Callable[[], None] | None = None

    def attach(self, agent: ToolAgent) -> None:
        self._agent = agent
        if agent.llm is not None:
            self._remove_stream_listener = agent.llm.add_stream_listener(self._observe_stream_chunk)

    @property
    def elapsed_seconds(self) -> float:
        return time.monotonic() - self._started_at

    @property
    def activity(self) -> str:
        return self._activity

    def running(self, turn: int) -> None:
        self._turn = turn
        self._coordinator.update_child(self)

    def before_llm_call(self, turn: int) -> None:
        self._reset_stream_estimate()
        self._current_tool_name = None
        self._activity = "Thinking"
        self.running(turn)

    def after_llm_call(self, turn: int) -> None:
        self._reset_stream_estimate()
        self._activity = "Processing"
        self.running(turn)

    def before_tool_call(self, turn: int, tool_names: list[str]) -> None:
        self._tool_count += len(tool_names)
        self._current_tool_name = (
            tool_names[0] if len(tool_names) == 1 else f"{len(tool_names)} tools"
        )
        self._activity = "Tool"
        self.running(turn)

    def after_tool_call(self, turn: int) -> None:
        self._current_tool_name = None
        self._activity = "Processing"
        self.running(turn)

    def finalizing(self) -> None:
        self._activity = "Finalizing"
        self.running(self._turn)

    def finish(self, status: str) -> None:
        if self._remove_stream_listener is not None:
            self._remove_stream_listener()
            self._remove_stream_listener = None
        self._coordinator.finish(self, status)

    def _observe_stream_chunk(self, chunk: StreamChunk) -> None:
        if chunk.event == "rollback":
            had_estimate = self._estimated_output_tokens > 0
            self._reset_stream_estimate()
            if had_estimate:
                self._coordinator.update_child(self)
            return
        if chunk.event != "delta" or not chunk.text:
            return

        self._streamed_chars += len(chunk.text)
        estimate = max(1, (self._streamed_chars + 3) // 4)
        if estimate == self._estimated_output_tokens:
            return
        self._estimated_output_tokens = estimate
        if (
            self._last_emitted_output_estimate > 0
            and estimate - self._last_emitted_output_estimate < _STREAM_UPDATE_TOKEN_INTERVAL
        ):
            return
        self._last_emitted_output_estimate = estimate
        self._coordinator.update_child(self)

    def _reset_stream_estimate(self) -> None:
        self._streamed_chars = 0
        self._estimated_output_tokens = 0
        self._last_emitted_output_estimate = 0

    def snapshot(self) -> SubagentMonitorSnapshot:
        input_tokens, output_tokens, output_estimated = self._usage()
        state = self._activity
        if self._current_tool_name is not None:
            state = f"tool: {self._current_tool_name}"
        return SubagentMonitorSnapshot(
            model=self._model_display_name(),
            context_percentage=self._context_percentage(),
            state=state,
            turn=self._turn,
            input_tokens=input_tokens,
            cache_percentage=self._cache_percentage(input_tokens),
            output_tokens=output_tokens,
            output_estimated=output_estimated,
        )

    def details(self) -> str:
        tool_details = f"tools {self._tool_count}"
        if self._current_tool_name:
            tool_details = f"{tool_details} ({self._current_tool_name})"
        parts = [f"turn {self._turn:>2}"]
        if model := self._model_display_name():
            parts.append(f"model {model}")
        input_tokens, output_tokens, output_estimated = self._usage()
        parts.extend(
            (
                _format_usage_details(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cache_percentage=self._cache_percentage(input_tokens),
                    output_estimated=output_estimated,
                ),
                tool_details,
            )
        )
        return " · ".join(parts)

    def _model_display_name(self) -> str | None:
        if self._agent is None:
            return None
        model_spec = self._agent.config.model
        accumulator = self._agent.usage_accumulator
        if not model_spec and accumulator is not None and accumulator.turns:
            model_spec = accumulator.turns[-1].model
        return resolve_llm_display_name(
            self._agent.llm,
            max_len=24,
        ) or resolve_model_display_name(
            model_spec,
            max_len=24,
        )

    def _usage(self) -> tuple[int, int, bool]:
        if self._agent is None or self._agent.usage_accumulator is None:
            return 0, self._estimated_output_tokens, self._estimated_output_tokens > 0
        summary = self._agent.usage_accumulator.summary
        input_tokens = summary.prompt.total or 0
        return (
            input_tokens,
            (summary.completion.total or 0) + self._estimated_output_tokens,
            self._estimated_output_tokens > 0,
        )

    def _cache_percentage(self, input_tokens: int) -> float | None:
        if self._agent is None or self._agent.usage_accumulator is None:
            return None
        return _cache_percentage(
            cache_read=self._agent.usage_accumulator.summary.prompt.cache_read,
            input_tokens=input_tokens,
        )

    def _context_percentage(self) -> float | None:
        if self._agent is None or self._agent.usage_accumulator is None:
            return None
        return self._agent.usage_accumulator.context_usage_percentage


def _format_usage_details(
    *,
    input_tokens: int,
    output_tokens: int,
    cache_percentage: float | None,
    output_estimated: bool = False,
) -> str:
    return (
        f"in {input_tokens:>7,} "
        f"out {'~' if output_estimated else ' '}{output_tokens:>6,} "
        f"cache {_format_cache_percentage(cache_percentage):>4}"
    )


def _format_cache_percentage(cache_percentage: float | None) -> str:
    if cache_percentage is None:
        return "—"
    if cache_percentage < 100 and round(cache_percentage) == 100:
        return ">99%"
    return f"{cache_percentage:.0f}%"


def _cache_percentage(*, cache_read: int | None, input_tokens: int) -> float | None:
    if cache_read is None or input_tokens == 0:
        return None
    return (cache_read / input_tokens) * 100


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
    if isinstance(agent, McpAgent):
        agent.set_instruction(agent.process_rendered_instruction(agent.instruction))
        directive_found = agent.subagent_directive_found
    else:
        directive = resolve_subagent_directive(agent.instruction)
        agent.set_instruction(directive.instruction)
        directive_found = directive.found
    if directive_found:
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
                description=("Optional model override. Omit to inherit the parent's current model.")
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
        include_user_message: Annotated[
            bool,
            Field(
                description=(
                    "Include the latest external user text and attachments, but no history. "
                    "This may forward content to another model or provider."
                )
            ),
        ] = False,
        *,
        model_source: str,
    ) -> ToolResult:
        """Run a complete task in a fresh one-shot child agent."""
        current_user_message = get_current_user_message() if include_user_message else None
        if include_user_message and current_user_message is None:
            return ToolResult(
                content=[
                    text_content(
                        "Error: include_user_message requires an active external user message."
                    )
                ],
                is_error=True,
            )
        child_input = _subagent_child_input(message, current_user_message)
        rendered_child_input = render_subagent_input(child_input)
        resolved_label = resolve_subagent_label(
            label,
            used_labels=used_labels,
            generator=generate_label,
        )
        context = get_local_tool_invocation_context()
        parent_tool_call_id = context.tool_use_id if context is not None else None
        child_session = _create_child_session(
            agent,
            alias_slug=subagent_alias_slug(label=label, task=message),
            label=resolved_label,
            task_preview=subagent_task_preview(message),
        )
        child_alias = _child_subagent_alias(child_session)
        child_name = f"{agent.name}[{child_alias or resolved_label}]"
        clone: ToolAgent | None = None
        status: SessionExecutionStatus = "running"
        response_text = ""
        is_error = False
        cancellation: asyncio.CancelledError | None = None
        cancellation_requested = asyncio.Event()
        finalization_result = _SubagentFinalizationResult()
        started_at = datetime.now(timezone.utc).isoformat()
        progress = monitor.start(
            label=resolved_label,
            child_name=child_name,
            parent_tool_call_id=parent_tool_call_id,
        )
        try:
            clone = await agent.spawn_isolated_instance(
                name=child_name,
                model=model,
                for_subagent=True,
            )
            clone.set_session_history_persistence_enabled(False)
            clone.remove_tool(SUBAGENT_TOOL_NAME)
            clone.load_message_history([])
            progress.attach(clone)
            clone.tool_runner_hooks = ToolAgent._merge_tool_runner_hooks(
                clone.tool_runner_hooks,
                _subagent_progress_hooks(progress),
            )
            with _child_chat_suppressed(clone):
                response = await clone.generate([child_input])
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
                    message=message,
                    child_input=child_input,
                    rendered_child_input=rendered_child_input,
                    requested_model=model,
                    label=resolved_label,
                    include_user_message=include_user_message,
                    parent_tool_call_id=parent_tool_call_id,
                    started_at=started_at,
                    cancellation_requested=cancellation_requested,
                )
            )
            while not finalizer.done():
                try:
                    await asyncio.shield(finalizer)
                except asyncio.CancelledError as exc:
                    cancellation = cancellation or exc
                    cancellation_requested.set()
            try:
                finalization_result = finalizer.result()
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
            finalization_result = await _remove_subagent_transcript(
                parent=agent,
                result=finalization_result,
            )
            raise cancellation

        transcript = finalization_result.transcript
        if transcript is not None:
            response_text = f"{response_text}\n\n{transcript.notice}"
        return ToolResult(
            content=[text_content(response_text)],
            meta={
                FAST_AGENT_SUBAGENT_RESULT_METADATA: _subagent_result_metadata(
                    child_session=child_session,
                    child_name=child_name,
                    child_alias=child_alias,
                    requested_label=label,
                    label=resolved_label,
                    clone=clone,
                    status=status,
                    parent_tool_call_id=parent_tool_call_id,
                    model_source=model_source,
                    transcript=transcript,
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
            include_user_message: Annotated[
                bool,
                Field(
                    description=(
                        "Include the latest external user text and attachments, but no history. "
                        "This may forward content to another model or provider."
                    )
                ),
            ] = False,
        ) -> ToolResult:
            return await _run_subagent(
                message,
                model,
                label,
                include_user_message,
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
            include_user_message: Annotated[
                bool,
                Field(
                    description=(
                        "Include the latest external user text and attachments, but no history. "
                        "This may forward content to another model or provider."
                    )
                ),
            ] = False,
        ) -> ToolResult:
            del model
            return await _run_subagent(
                message,
                forced_model,
                label,
                include_user_message,
                model_source="agent_card",
            )

        description = (
            "Run a focused subagent in a clean context. It inherits your instruction and "
            "available capabilities, except it does not receive this subagent tool. Give it a "
            f"complete task and all necessary context. Each run uses the fixed model "
            f"`{forced_model}`. The optional label is a short display name for this run. The "
            "optional include_user_message forwards the latest external user text and "
            "attachments, but no history, and may send content to another model or provider."
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


def subagent_tool_enabled(agent: object) -> bool:
    if not isinstance(agent, ToolAgent):
        return False
    tool = agent._execution_tools.get(SUBAGENT_TOOL_NAME)
    return tool is not None and tool.meta == SUBAGENT_TOOL_METADATA


def set_subagent_tool_enabled(agent: object, enabled: bool) -> bool:
    """Apply a runtime-only subagent tool override."""
    if not isinstance(agent, ToolAgent) or agent.config.tool_only or agent.config.subagent_child:
        return False
    source = agent.config.subagent_activation_source
    if enabled and agent.config.subagents is False and source in {"configuration", "cli"}:
        return False
    existing = agent._execution_tools.get(SUBAGENT_TOOL_NAME)
    if enabled and existing is not None and existing.meta != SUBAGENT_TOOL_METADATA:
        return False
    if not enabled and agent.config.subagents is False and source in {"configuration", "cli"}:
        install_subagent_tool(agent)
        return not subagent_tool_enabled(agent)
    agent.config.subagents = enabled
    agent.config.subagent_activation_source = "runtime"
    install_subagent_tool(agent)
    return subagent_tool_enabled(agent) is enabled


def _subagent_child_input(
    message: str,
    current_user_message: CurrentUserMessage | None,
) -> PromptMessageExtended:
    """Build the child user message, preserving external multipart content."""
    from fast_agent.types import PromptMessageExtended

    if current_user_message is None:
        return Prompt.user(message)

    content = current_user_message.content
    text_blocks = [block for block in content if isinstance(block, TextContent)]
    if len(text_blocks) == len(content):
        user_text = "\n".join(block.text for block in text_blocks)
        return PromptMessageExtended(
            role="user",
            content=[
                text_content(
                    f"{message}\n\n<included_user_context>\n"
                    f"{_escape_user_text(user_text)}\n"
                    "</included_user_context>"
                )
            ],
        )

    included_content = [
        text_content(_escape_user_text(block.text))
        if isinstance(block, TextContent)
        else deepcopy(block)
        for block in content
    ]
    return PromptMessageExtended(
        role="user",
        content=[
            text_content(f"{message}\n\n<included_user_context>\n"),
            *included_content,
            text_content("\n</included_user_context>"),
        ],
    )


def _escape_user_text(text: str) -> str:
    """Escape only the XML characters used by the context envelope."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _create_child_session(
    agent: ToolAgent,
    *,
    alias_slug: str,
    label: str,
    task_preview: str,
) -> Session | None:
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
            alias_slug=alias_slug,
            label=label,
            task_preview=task_preview,
        )
    except Exception as exc:
        logger.warning(
            "Failed to create subagent child session",
            data={"parent_session": parent.info.name, "error": str(exc)},
        )
        return None


def _child_subagent_alias(child_session: Session | None) -> str | None:
    if child_session is None:
        return None
    alias = child_session.info.metadata.get(SUBAGENT_ALIAS_KEY)
    return alias if isinstance(alias, str) else None


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
    message: str,
    child_input: PromptMessageExtended | None = None,
    rendered_child_input: str | None = None,
    requested_model: str | None,
    label: str,
    include_user_message: bool = False,
    parent_tool_call_id: str | None,
    started_at: str,
    cancellation_requested: asyncio.Event,
    finalization_timeout_seconds: float = _FINALIZATION_TIMEOUT_SECONDS,
) -> _SubagentFinalizationResult:
    """Persist and release one child without inheriting caller cancellation."""
    progress.finalizing()
    result = _SubagentFinalizationResult()
    effective_status = status
    effective_child_input = child_input or _subagent_child_input(message, None)
    effective_rendered_child_input = rendered_child_input or render_subagent_input(
        effective_child_input
    )
    try:
        try:
            async with asyncio.timeout(finalization_timeout_seconds):
                child_messages = (
                    [
                        item.model_copy(deep=True)
                        for item in (clone.message_history or clone.last_turn_messages)
                    ]
                    if clone is not None
                    else []
                )
                if cancellation_requested.is_set():
                    effective_status = "cancelled"
                if effective_status != "cancelled":
                    result = await _write_subagent_transcript(
                        parent=parent,
                        clone=clone,
                        child_messages=child_messages,
                        child_name=child_name,
                        status=effective_status,
                        child_input=effective_child_input,
                        rendered_child_input=effective_rendered_child_input,
                        label=label,
                    )
                if cancellation_requested.is_set():
                    effective_status = "cancelled"
                    result = await _remove_subagent_transcript(parent=parent, result=result)
                await _persist_subagent_run(
                    parent=parent,
                    clone=clone,
                    child_session=child_session,
                    child_name=child_name,
                    status=effective_status,
                    message=message,
                    rendered_child_input=effective_rendered_child_input,
                    requested_model=requested_model,
                    label=label,
                    include_user_message=include_user_message,
                    parent_tool_call_id=parent_tool_call_id,
                    started_at=started_at,
                    child_messages=child_messages,
                )
                if cancellation_requested.is_set():
                    if effective_status != "cancelled" and child_session is not None:
                        child_session.set_execution_status("cancelled")
                    effective_status = "cancelled"
                    result = await _remove_subagent_transcript(parent=parent, result=result)
        finally:
            await _release_subagent_clone(
                parent=parent,
                clone=clone,
                child_name=child_name,
            )
            if child_session is not None and child_session.manager is not None:
                child_session.manager.release_session(child_session.info.name)
    finally:
        progress.finish(effective_status)
    return result


async def _write_subagent_transcript(
    *,
    parent: ToolAgent,
    clone: ToolAgent | None,
    child_messages: list[PromptMessageExtended],
    child_name: str,
    status: SessionExecutionStatus,
    child_input: PromptMessageExtended,
    rendered_child_input: str,
    label: str,
) -> _SubagentFinalizationResult:
    store = parent.transient_artifact_store()
    if store is None:
        logger.info(
            "Subagent transcript is unavailable; omitting it from the tool result",
            data={"agent_name": child_name, "status": status},
        )
        return _SubagentFinalizationResult()
    try:
        model_name, provider = _subagent_model_metadata(clone)
        content = render_subagent_transcript(
            delegated_input=rendered_child_input,
            delegated_message=child_input,
            messages=child_messages,
            metadata=SubagentTranscriptMetadata(
                child_agent=child_name,
                label=label,
                status=status,
                model=model_name,
                provider=provider,
            ),
        )
        transcript = await store.write_text(
            producer="subagent",
            suffix=".log",
            content=content,
            description="subagent transcript",
        )
    except Exception as exc:
        logger.warning(
            "Failed to create subagent transcript artifact",
            data={"agent_name": child_name, "status": status, "error": str(exc)},
        )
        return _SubagentFinalizationResult()
    return _SubagentFinalizationResult(transcript=transcript)


async def _remove_subagent_transcript(
    *,
    parent: ToolAgent,
    result: _SubagentFinalizationResult,
) -> _SubagentFinalizationResult:
    transcript = result.transcript
    if transcript is None:
        return result
    store = parent.transient_artifact_store()
    if store is not None:
        try:
            await store.remove(transcript.artifact)
        except Exception as exc:
            logger.warning(
                "Failed to remove cancelled subagent transcript artifact",
                data={"path": transcript.artifact.path, "error": str(exc)},
            )
    return _SubagentFinalizationResult()


async def _persist_subagent_run(
    *,
    parent: ToolAgent,
    clone: ToolAgent | None,
    child_session: Session | None,
    child_name: str,
    status: SessionExecutionStatus,
    message: str,
    rendered_child_input: str,
    requested_model: str | None,
    label: str,
    include_user_message: bool,
    parent_tool_call_id: str | None,
    started_at: str,
    child_messages: list[PromptMessageExtended],
) -> None:
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
            history_agent = HistoryAgent(clone, child_messages)
            await child_session.save_history(cast("AgentProtocol", history_agent))
        except Exception as exc:
            logger.warning(
                "Failed to persist subagent child session",
                data={"session": child_session.info.name, "error": str(exc)},
            )
    elif clone is not None and child_messages and parent.subagent_trajectory_capture_enabled:
        from fast_agent.session.trajectory import TrajectoryRecord, new_trajectory_id

        tool_arguments: dict[str, object] = {
            "message": message,
            "label": label,
            "include_user_message": include_user_message,
        }
        if requested_model is not None:
            tool_arguments["model"] = requested_model
        usage = clone.usage_accumulator
        model_name, provider = _subagent_model_metadata(clone)
        parent.record_subagent_trajectory(
            TrajectoryRecord(
                trajectory_id=new_trajectory_id(),
                session_id="",
                parent_agent_name=parent.name,
                agent_name=child_name,
                template_agent_name=parent.name,
                tool_name=SUBAGENT_TOOL_NAME,
                parent_tool_call_id=parent_tool_call_id,
                use_history=clone.config.use_history,
                started_at=started_at,
                completed_at=datetime.now(timezone.utc).isoformat(),
                tool_input_schema=None,
                tool_arguments=tool_arguments,
                effective_tool_arguments=dict(tool_arguments),
                rendered_child_input=rendered_child_input,
                messages=child_messages,
                usage_summary=usage.get_summary() if usage is not None else None,
                model_name=model_name,
                provider=provider,
            )
        )


async def _release_subagent_clone(
    *,
    parent: ToolAgent,
    clone: ToolAgent | None,
    child_name: str,
) -> None:
    if clone is None:
        return
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
        parent.merge_subagent_usage_from(clone)
    except Exception as exc:
        logger.warning("Failed to merge subagent usage", data={"error": str(exc)})


def _subagent_result_metadata(
    *,
    child_session: Session | None,
    child_name: str,
    child_alias: str | None,
    requested_label: str | None,
    label: str,
    clone: ToolAgent | None,
    status: str,
    parent_tool_call_id: str | None,
    model_source: str,
    transcript: TransientArtifactResult | None,
) -> dict[str, object]:
    """Build durable, display-oriented details without changing model text."""
    model_spec, provider = _subagent_model_metadata(clone)
    usage: dict[str, int | None] = {
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
    }
    turn_count = 0

    if clone is not None:
        accumulator = clone.usage_accumulator
        if accumulator is not None:
            summary = accumulator.summary
            usage = {
                "prompt_tokens": summary.prompt.total,
                "completion_tokens": summary.completion.total,
                "total_tokens": summary.total,
            }
            turn_count = len(accumulator.turns)

    metadata: dict[str, object] = {
        "child_session_id": child_session.info.name if child_session is not None else None,
        "child_agent_name": child_name,
        "alias": child_alias,
        "ordinal": (
            child_session.info.metadata.get(SUBAGENT_ORDINAL_KEY)
            if child_session is not None
            else None
        ),
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
    if transcript is not None:
        metadata.update(
            {
                "transcript_path": transcript.artifact.path,
                "transcript_bytes": transcript.artifact.retained_bytes,
                "transcript_complete": transcript.artifact.complete,
            }
        )
    return metadata


def _subagent_model_metadata(clone: ToolAgent | None) -> tuple[str | None, str | None]:
    if clone is None or clone.llm is None:
        return None, None
    resolved_model = clone.llm.resolved_model
    model_name = (
        resolved_model.selected_model_name or resolved_model.wire_model_name or clone.config.model
    )
    return model_name, resolved_model.provider.value


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
