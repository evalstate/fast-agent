from __future__ import annotations

import asyncio
import json
import math
import posixpath
import shutil
import tempfile
import time
from collections import deque
from contextlib import nullcontext, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypedDict, cast

from mcp_types import CallToolResult, TextContent, Tool
from rich.text import Text

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from fast_agent.config import Settings
    from fast_agent.tools.execution_environment import ShellEnvironment, ShellExecutionResult

# Import tool progress context for reporting shell execution progress
from fast_agent.agents.tool_agent import _tool_progress_context
from fast_agent.constants import (
    DEFAULT_DURABLE_PROCESS_OUTPUT_RETENTION_BYTES,
    DEFAULT_TERMINAL_OUTPUT_BYTE_LIMIT,
    MAX_FOREGROUND_AUTO_AWAIT_SECONDS,
    MAX_MANAGED_SHELL_PROCESSES,
    MAX_TERMINAL_OUTPUT_BYTE_LIMIT,
    MIN_PROCESS_POLL_WAIT_SECONDS,
    TERMINAL_BYTES_PER_TOKEN,
)
from fast_agent.event_progress import ProgressAction
from fast_agent.mcp.tool_result_metadata import (
    update_tool_result_display_metadata,
)
from fast_agent.tools.durable_processes import (
    DurableProcessError,
    DurableProcessOutput,
    DurableProcessRecordError,
    DurableProcessSnapshot,
    DurableProcessStore,
    DurableProcessStream,
)
from fast_agent.tools.execution_environment import (
    ShellExecution,
    ShellExecutionRequest,
    ShellRuntimeInfo,
    execute_shell,
)
from fast_agent.tools.local_shell_executor import LocalShellExecutor
from fast_agent.tools.output_truncation import (
    format_output_truncation_notice,
    split_output_byte_limit,
)
from fast_agent.tools.process_resources import (
    ProcessResourceSnapshot,
    observe_resource_changes,
    sample_process_resources,
)
from fast_agent.tools.shell_command import (
    ShellDetachmentKind as ShellDetachmentKind,
)
from fast_agent.tools.shell_command import (
    classify_shell_detachment as classify_shell_detachment,
)
from fast_agent.tools.shell_output import ShellOutputBuffer, process_output_preview
from fast_agent.tools.shell_process import (
    ActiveProcessPoll,
    ForegroundAutoAwaitMetadata,
    ManagedProcessSnapshot,
    ManagedShellProcess,
    ProcessResultMetadata,
    ShellDisplayState,
    ShellRuntimeCallbacks,
    build_managed_process_result,
    process_result,
    process_result_metadata,
)
from fast_agent.tools.shell_profiles import (
    ResolvedShellToolProfile,
    ShellToolProfile,
    resolve_shell_tool_profile,
)
from fast_agent.tools.shell_progress import ShellProgressReporter
from fast_agent.tools.shell_tool_definitions import (
    PROCESS_OUTPUT_DEBOUNCE_SECONDS,
    MinimalProcessReadOutputArguments,
    ShellExecuteArguments,
    build_execute_tool,
    build_grok_shell_tool,
    build_luna_exec_tool,
    build_minimal_bash_tool,
    build_minimal_process_tool,
    build_poll_process_tool,
    build_terminate_process_tool,
    parse_execute_arguments,
    parse_grok_shell_arguments,
    parse_luna_exec_arguments,
    parse_minimal_bash_arguments,
    parse_minimal_process_arguments,
    parse_poll_process_arguments,
    parse_terminate_process_arguments,
    set_poll_process_tool_default_wait_seconds,
)
from fast_agent.tools.tool_sources import SHELL_TOOL_SOURCE, set_tool_source
from fast_agent.ui import console
from fast_agent.ui.console_display import ConsoleDisplay
from fast_agent.ui.display_suppression import display_tools_enabled
from fast_agent.ui.progress_display import progress_display
from fast_agent.ui.shell_output_truncation import (
    SHELL_OUTPUT_TRUNCATION_MARKER,
    split_shell_output_line_limit,
)
from fast_agent.utils.path_display import format_relative_path
from fast_agent.utils.text import summarize_command
from fast_agent.utils.tool_names import (
    BASH_TOOL_NAME,
    EXECUTE_TOOL_NAME,
    GROK_SHELL_TOOL_NAME,
    LUNA_EXEC_TOOL_NAME,
    POLL_PROCESS_TOOL_NAME,
    PROCESS_TOOL_NAME,
    TERMINATE_PROCESS_TOOL_NAME,
)

_IO_DRAIN_TIMEOUT_SECONDS = 2.0
_DEFAULT_IDLE_YIELD_SECONDS = 10
_DEFAULT_FOREGROUND_YIELD_SECONDS = 30
_DEFAULT_MINIMAL_PROCESS_WAIT_SECONDS = 30
_PROCESS_OUTPUT_DEBOUNCE_SECONDS = PROCESS_OUTPUT_DEBOUNCE_SECONDS


def _default_max_process_poll_seconds() -> int:
    from fast_agent.config import ShellSettings

    return ShellSettings().process_poll_max_wait_seconds


def _default_foreground_auto_await_max_seconds() -> int:
    from fast_agent.config import ShellSettings

    return ShellSettings().foreground_auto_await_max_seconds


_RESOURCE_OBSERVATION_TIMEOUT_SECONDS = 0.075


def _text_result(message: str, *, is_error: bool) -> CallToolResult:
    return CallToolResult(
        is_error=is_error,
        content=[TextContent(type="text", text=message)],
    )


@dataclass(slots=True)
class _ShellRuntimeExecution:
    execution: ShellExecution
    output_state: ShellOutputBuffer
    display_state: ShellDisplayState


@dataclass(frozen=True, slots=True)
class _ManagedProcessOperation:
    kind: Literal["list", "status", "wait", "stop", "read_output"]
    process_id: str | None
    wait_sec: int | None


ProcessTerminationState = Literal[
    "terminated",
    "stop_requested",
    "stop_already_requested",
    "already_exited",
    "unavailable",
    "not_found",
    "termination_failed",
]


@dataclass(frozen=True, slots=True)
class ProcessTerminationOutcome:
    process_id: str
    state: ProcessTerminationState
    error: str | None = None


class _ProcessListEntry(TypedDict):
    process_id: str
    status: str
    lifecycle: Literal["session", "persistent"]
    command: str
    working_directory: str
    elapsed_seconds: float
    total_output_bytes: int
    exit_code: int | None


class _ProcessListResult(TypedDict):
    processes: list[_ProcessListEntry]


def _coerce_output_byte_limit(output_byte_limit: int | None) -> int:
    if type(output_byte_limit) is not int or output_byte_limit <= 0:
        return DEFAULT_TERMINAL_OUTPUT_BYTE_LIMIT
    return min(output_byte_limit, MAX_TERMINAL_OUTPUT_BYTE_LIMIT)


class ShellRuntime:
    """Helper for managing the optional shell execute tool."""

    def __init__(
        self,
        activation_reason: str | None,
        logger,
        timeout_seconds: float = 90,
        warning_interval_seconds: int = 30,
        working_directory: Path | None = None,
        output_byte_limit: int | None = None,
        process_poll_default_wait_seconds: int = 0,
        config: Settings | None = None,
        agent_name: str | None = None,
        shell_environment: ShellEnvironment | None = None,
        idle_yield_seconds: float = _DEFAULT_IDLE_YIELD_SECONDS,
        foreground_yield_seconds: float = _DEFAULT_FOREGROUND_YIELD_SECONDS,
        minimal_shell_tool_name: str = BASH_TOOL_NAME,
        minimal_shell_tool_requires_description: bool = False,
        extended_guidance: bool = False,
        tool_profile: ShellToolProfile | None = None,
        model_tool_profile: ResolvedShellToolProfile | None = None,
        foreground_auto_await_max_seconds: float | None = None,
        durable_process_root: Path | None = None,
        session_id_provider: Callable[[], str | None] | None = None,
    ) -> None:
        self._working_directory = str(working_directory) if working_directory is not None else None
        self._environment = shell_environment or LocalShellExecutor(
            logger=logger,
            timeout_seconds=timeout_seconds,
            warning_interval_seconds=warning_interval_seconds,
            working_directory=working_directory,
            config=config,
        )
        self._activation_reason = activation_reason
        self._logger = logger
        self._progress = ShellProgressReporter(logger, agent_name)
        self._timeout_seconds = timeout_seconds
        self._warning_interval_seconds = warning_interval_seconds
        self._output_byte_limit = DEFAULT_TERMINAL_OUTPUT_BYTE_LIMIT
        self.set_output_byte_limit(output_byte_limit)
        self.enabled: bool = activation_reason is not None
        self._tool: Tool | None = None
        self._display = ConsoleDisplay(config=config)
        self._config = config
        self._agent_name = agent_name
        self._idle_yield_seconds = idle_yield_seconds
        self._foreground_yield_seconds = foreground_yield_seconds
        self._foreground_auto_await_max_seconds = float(
            _default_foreground_auto_await_max_seconds()
        )
        self._minimal_shell_tool_name = minimal_shell_tool_name
        self._minimal_shell_tool_requires_description = minimal_shell_tool_requires_description
        self._extended_guidance = extended_guidance
        self._managed_processes: dict[str, ManagedShellProcess] = {}
        self._durable_process_store: DurableProcessStore | None = None
        if durable_process_root is not None and isinstance(self._environment, LocalShellExecutor):
            try:
                self._durable_process_store = DurableProcessStore(durable_process_root)
            except (DurableProcessError, OSError) as exc:
                self._logger.warning(
                    "Durable process storage is unavailable at %s: %s",
                    durable_process_root,
                    exc,
                )
        self._attached_durable_processes: set[str] = set()
        self._durable_output_offsets: dict[str, int] = {}
        self._durable_output_total_offsets: dict[str, int] = {}
        self._durable_output_dropped_offsets: dict[str, int] = {}
        self._durable_observed_output_bytes: dict[str, int] = {}
        self._durable_last_output_times: dict[str, float] = {}
        self._durable_poll_locks: dict[str, asyncio.Lock] = {}
        self._session_id_provider = session_id_provider
        self._next_process_id = 1
        self._processes_lock = asyncio.Lock()
        self._output_display_lines: int | None = None
        self._show_bash_output = True
        self._prefer_local_shell = False
        self._max_process_poll_seconds = _default_max_process_poll_seconds()
        configured_profile = tool_profile or "auto"
        self._retained_output_directory: Path | None = None
        self._retained_output_max_bytes = 0
        self._durable_output_max_bytes = DEFAULT_DURABLE_PROCESS_OUTPUT_RETENTION_BYTES
        self._retained_output_next_id = 1
        self._retained_output_via_process = False
        retain_truncated_output = False
        retained_output_parent: Path | None = None
        if config is not None:
            shell_config = config.shell_execution
            self._output_display_lines = shell_config.output_display_lines
            self._show_bash_output = shell_config.show_bash
            self._prefer_local_shell = shell_config.prefer_local_shell
            self._max_process_poll_seconds = shell_config.process_poll_max_wait_seconds
            self._foreground_auto_await_max_seconds = float(
                shell_config.foreground_auto_await_max_seconds
            )
            configured_profile = tool_profile or shell_config.tool_profile
            self._retained_output_max_bytes = shell_config.retained_output_max_bytes
            self._durable_output_max_bytes = shell_config.durable_output_max_bytes
            retain_truncated_output = shell_config.retain_truncated_output
            retained_output_parent = shell_config.retained_output_temp_directory
        if foreground_auto_await_max_seconds is not None:
            auto_await_max_seconds = float(foreground_auto_await_max_seconds)
            if (
                isinstance(foreground_auto_await_max_seconds, bool)
                or not math.isfinite(auto_await_max_seconds)
                or auto_await_max_seconds < 0
                or auto_await_max_seconds > MAX_FOREGROUND_AUTO_AWAIT_SECONDS
            ):
                raise ValueError(
                    "foreground_auto_await_max_seconds must be finite and between "
                    f"0 and {MAX_FOREGROUND_AUTO_AWAIT_SECONDS}"
                )
            self._foreground_auto_await_max_seconds = auto_await_max_seconds
        self._minimal_process_profile = False
        self._grok_shell_profile = False
        self._luna_exec_profile = False
        self._process_poll_default_wait_seconds = min(
            process_poll_default_wait_seconds,
            self._max_process_poll_seconds,
        )
        self._resource_observations_enabled = self.runtime_info().kind == "local"
        self._poll_process_tool: Tool | None = None
        self._terminate_process_tool: Tool | None = None
        self.set_tool_profile(
            configured_profile,
            model_profile=model_tool_profile,
        )
        process_readback_supported = (
            self._minimal_process_profile or self._grok_shell_profile or self._luna_exec_profile
        )
        runtime_kind = self.runtime_info().kind
        if retain_truncated_output and (runtime_kind == "local" or process_readback_supported):
            if retained_output_parent is not None:
                retained_output_parent.mkdir(
                    mode=0o700,
                    parents=True,
                    exist_ok=True,
                )
            self._retained_output_directory = Path(
                tempfile.mkdtemp(
                    prefix="fast-agent-output-",
                    dir=(
                        str(retained_output_parent) if retained_output_parent is not None else None
                    ),
                )
            )
            self._retained_output_directory.chmod(0o700)
            self._retained_output_via_process = runtime_kind != "local"

    @property
    def tool(self) -> Tool | None:
        return self._tool

    @property
    def tools(self) -> list[Tool]:
        """Return all model-facing shell and process lifecycle tools."""
        return [
            tool
            for tool in (
                self._tool,
                self._poll_process_tool,
                self._terminate_process_tool,
            )
            if tool is not None
        ]

    def owns_tool(self, name: str) -> bool:
        """Return whether this runtime owns a model-facing tool name."""
        return any(tool.name == name for tool in self.tools)

    def set_tool_profile(
        self,
        profile: ShellToolProfile,
        *,
        model_profile: ResolvedShellToolProfile | None = None,
    ) -> None:
        """Replace model-facing shell tools using config and model metadata."""
        resolved_profile = resolve_shell_tool_profile(profile, model_profile)
        self._minimal_process_profile = resolved_profile == "minimal_process"
        self._grok_shell_profile = resolved_profile == "grok_shell"
        self._luna_exec_profile = resolved_profile == "luna_exec"
        if not self.enabled:
            self._tool = None
            self._poll_process_tool = None
            self._terminate_process_tool = None
            return
        shell_name = self.runtime_info().name
        if self._grok_shell_profile or self._luna_exec_profile:
            shell_tool = (
                build_luna_exec_tool(shell_name=shell_name)
                if self._luna_exec_profile
                else build_grok_shell_tool(shell_name=shell_name)
            )
            shell_tool_name = (
                LUNA_EXEC_TOOL_NAME if self._luna_exec_profile else GROK_SHELL_TOOL_NAME
            )
            self._tool = set_tool_source(
                shell_tool,
                SHELL_TOOL_SOURCE,
            )
            self._poll_process_tool = set_tool_source(
                build_minimal_process_tool(
                    default_wait_seconds=self._minimal_process_wait_seconds(),
                    max_wait_seconds=self._max_process_poll_seconds,
                    shell_tool_name=shell_tool_name,
                    extended_guidance=False,
                ),
                SHELL_TOOL_SOURCE,
            )
            self._terminate_process_tool = None
        elif self._minimal_process_profile:
            self._tool = set_tool_source(
                build_minimal_bash_tool(
                    shell_name=shell_name,
                    tool_name=self._minimal_shell_tool_name,
                    require_description=self._minimal_shell_tool_requires_description,
                    extended_guidance=self._extended_guidance,
                ),
                SHELL_TOOL_SOURCE,
            )
            self._poll_process_tool = set_tool_source(
                build_minimal_process_tool(
                    default_wait_seconds=self._minimal_process_wait_seconds(),
                    max_wait_seconds=self._max_process_poll_seconds,
                    shell_tool_name=self._minimal_shell_tool_name,
                    extended_guidance=self._extended_guidance,
                ),
                SHELL_TOOL_SOURCE,
            )
            self._terminate_process_tool = None
        else:
            self._tool = set_tool_source(
                build_execute_tool(shell_name=shell_name),
                SHELL_TOOL_SOURCE,
            )
            self._poll_process_tool = set_tool_source(
                build_poll_process_tool(
                    default_wait_seconds=self._process_poll_default_wait_seconds,
                    max_wait_seconds=self._max_process_poll_seconds,
                ),
                SHELL_TOOL_SOURCE,
            )
            self._terminate_process_tool = set_tool_source(
                build_terminate_process_tool(),
                SHELL_TOOL_SOURCE,
            )

    @property
    def active_process_count(self) -> int:
        """Return the number of managed processes that are currently alive."""
        return (
            sum(not process.task.done() for process in self._managed_processes.values())
            + self._active_durable_process_count()
        )

    def _active_durable_process_count(self) -> int:
        store = self._durable_process_store
        if store is None:
            return 0
        active = 0
        for process_id in self._attached_durable_processes:
            try:
                snapshot = store.get(process_id)
            except (DurableProcessRecordError, OSError):
                continue
            if self._durable_status(snapshot) == "running":
                active += 1
        return active

    def _stored_active_durable_process_count(self) -> int:
        store = self._durable_process_store
        if store is None:
            return 0
        try:
            snapshots = store.discover()
        except OSError:
            return 0
        return sum(self._durable_status(snapshot) == "running" for snapshot in snapshots)

    def set_extended_guidance(self, enabled: bool) -> None:
        """Refresh model-facing minimal tools when model guidance policy changes."""
        self.set_minimal_shell_tool_contract(
            tool_name=self._minimal_shell_tool_name,
            require_description=self._minimal_shell_tool_requires_description,
            extended_guidance=enabled,
        )

    def set_minimal_shell_tool_contract(
        self,
        *,
        tool_name: str,
        require_description: bool,
        extended_guidance: bool,
    ) -> None:
        """Refresh the catalog-driven model-facing minimal shell contract."""
        if (
            self._minimal_shell_tool_name == tool_name
            and self._minimal_shell_tool_requires_description == require_description
            and self._extended_guidance == extended_guidance
        ):
            return
        self._minimal_shell_tool_name = tool_name
        self._minimal_shell_tool_requires_description = require_description
        self._extended_guidance = extended_guidance
        if not self.enabled or not self._minimal_process_profile:
            return
        shell_name = self.runtime_info().name
        self._tool = set_tool_source(
            build_minimal_bash_tool(
                shell_name=shell_name,
                tool_name=tool_name,
                require_description=require_description,
                extended_guidance=extended_guidance,
            ),
            SHELL_TOOL_SOURCE,
        )
        self._poll_process_tool = set_tool_source(
            build_minimal_process_tool(
                default_wait_seconds=self._minimal_process_wait_seconds(),
                max_wait_seconds=self._max_process_poll_seconds,
                shell_tool_name=tool_name,
                extended_guidance=extended_guidance,
            ),
            SHELL_TOOL_SOURCE,
        )

    async def process_snapshots(self) -> tuple[ManagedProcessSnapshot, ...]:
        """Return retained process state for interactive status displays."""
        async with self._processes_lock:
            processes = tuple(self._managed_processes.values())
            attached_durable_processes = tuple(self._attached_durable_processes)
        now = time.monotonic()
        snapshots = [self._process_snapshot(process, now=now) for process in processes]
        store = self._durable_process_store
        if store is not None:
            for process_id in sorted(attached_durable_processes):
                try:
                    durable = await asyncio.to_thread(store.get, process_id)
                except DurableProcessRecordError:
                    continue
                snapshots.append(self._durable_managed_snapshot(durable))
        return tuple(snapshots)

    async def discover_durable_processes(self) -> tuple[DurableProcessSnapshot, ...]:
        """Return all durable local processes visible in this fast-agent home."""

        store = self._durable_process_store
        if store is None:
            return ()
        try:
            return tuple(await asyncio.to_thread(store.discover))
        except (DurableProcessRecordError, OSError) as exc:
            self._logger.warning(f"Could not discover durable processes: {exc}")
            return ()

    async def attach_durable_process(
        self,
        process_id: str,
        *,
        session_id: str | None = None,
    ) -> DurableProcessSnapshot:
        """Adopt durable process observation and management in this runtime."""

        store = self._durable_process_store
        if store is None:
            raise DurableProcessRecordError("Durable local processes are not available.")
        snapshot = await asyncio.to_thread(store.get, process_id)
        if session_id is not None:
            snapshot = await asyncio.to_thread(
                store.link_session,
                process_id,
                session_id=session_id,
            )
        async with self._processes_lock:
            self._attached_durable_processes.add(process_id)
            self._durable_output_offsets.setdefault(process_id, 0)
            observed_bytes = self._durable_observed_output_bytes.get(process_id, 0)
            if snapshot.output_total_bytes > observed_bytes:
                self._durable_last_output_times[process_id] = time.monotonic()
            self._durable_observed_output_bytes[process_id] = snapshot.output_total_bytes
            self._durable_output_total_offsets.setdefault(process_id, 0)
            self._durable_output_dropped_offsets.setdefault(process_id, 0)
            self._durable_poll_locks.setdefault(process_id, asyncio.Lock())
        return snapshot

    @property
    def durable_process_root(self) -> Path | None:
        """Return the active durable process root, when supported."""

        store = self._durable_process_store
        return store.root if store is not None else None

    @staticmethod
    def _durable_managed_snapshot(
        snapshot: DurableProcessSnapshot,
    ) -> ManagedProcessSnapshot:
        status = ShellRuntime._durable_status(snapshot)
        started_at = snapshot.status.started_at or snapshot.spec.created_at
        elapsed_at = snapshot.status.updated_at if status != "running" else time.time()
        return ManagedProcessSnapshot(
            process_id=snapshot.spec.process_id,
            command=snapshot.spec.command,
            working_directory=str(snapshot.spec.cwd),
            status=status,
            elapsed_seconds=max(elapsed_at - started_at, 0.0),
            os_process_id=snapshot.status.child_pid,
            total_output_bytes=snapshot.output_total_bytes,
            exit_code=snapshot.status.exit_code,
            lifecycle="persistent",
            output_spool_path=None,
        )

    @staticmethod
    def _durable_status(snapshot: DurableProcessSnapshot) -> str:
        state = snapshot.status.state
        if state in {"created", "starting", "running", "stopping"}:
            return "running"
        if state == "stopped":
            return "terminated"
        if state == "unavailable":
            return "unavailable"
        return "completed" if snapshot.status.exit_code == 0 else "failed"

    @staticmethod
    def _process_snapshot(
        process: ManagedShellProcess,
        *,
        now: float,
    ) -> ManagedProcessSnapshot:
        exit_code: int | None = None
        if not process.task.done():
            status = "running"
        elif process.task.cancelled():
            status = "terminated" if process.terminated else "cancelled"
        elif process.task.exception() is not None:
            status = "failed"
        else:
            exit_code = process.task.result().result.exit_code
            status = "completed" if exit_code == 0 else "failed"
        if process.task.done() and process.completed_at is None:
            process.completed_at = now
        elapsed_at = process.completed_at if process.completed_at is not None else now
        return ManagedProcessSnapshot(
            process_id=process.process_id,
            command=process.command,
            working_directory=process.working_directory,
            status=status,
            elapsed_seconds=max(elapsed_at - process.started_at, 0),
            os_process_id=process.callbacks.os_process_id,
            total_output_bytes=process.output_state.lifetime_output_bytes,
            exit_code=exit_code,
            lifecycle=process.lifecycle,
            output_spool_path=process.request.output_spool_path,
        )

    @property
    def prefer_local_shell(self) -> bool:
        """Whether ACP mode should keep this local shell runtime instead of client terminal."""
        return self._prefer_local_shell

    @property
    def output_byte_limit(self) -> int:
        """Return the current byte limit used to retain command output."""
        return self._output_byte_limit

    @property
    def timeout_seconds(self) -> float:
        """Return the idle/no-output timeout used for shell execution."""
        return self._timeout_seconds

    def set_output_byte_limit(self, output_byte_limit: int | None) -> None:
        """Set output retention byte limit, honoring global defaults and hard cap."""
        self._output_byte_limit = _coerce_output_byte_limit(output_byte_limit)

    def announce(self) -> None:
        """Inform the user why the local shell tool is active."""
        if not self.enabled or not self._activation_reason:
            return

        message = f"Shell execute tool enabled {self._activation_reason}."
        self._logger.info(message)

    def _render_display_line(self, text: str, style: str | None) -> Text:
        display_text = text.rstrip("\n").expandtabs()
        renderable = Text(display_text, style=style or "")
        renderable.no_wrap = True
        width = max(1, console.console.size.width)
        if len(display_text) > width:
            renderable.truncate(width, overflow="ellipsis")
        return renderable

    def working_directory(self) -> Path:
        """Return the working directory used for shell execution."""
        from pathlib import Path

        return Path(self._working_directory or self._environment.cwd)

    def set_working_directory(self, working_directory: Path | None) -> None:
        """Set the working directory used for shell execution."""
        self._working_directory = str(working_directory) if working_directory is not None else None

    def runtime_info(self) -> ShellRuntimeInfo:
        """Best-effort detection of the shell runtime used for execution.

        Prefers modern shells like pwsh (PowerShell 7+) and bash.
        """
        info = self._environment.runtime_info()
        if info.environment_name is not None:
            return info

        from fast_agent.tools.environment_registry import environment_name

        name = environment_name(self._environment)
        if name is None:
            return info
        return ShellRuntimeInfo(
            name=info.name,
            path=info.path,
            kind=info.kind,
            provider=info.provider,
            environment_name=name,
        )

    def metadata(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Build metadata for display when the shell tool is invoked."""
        info = self.runtime_info()
        try:
            parsed = (
                parse_minimal_bash_arguments(arguments)
                if self._minimal_process_profile
                else parse_luna_exec_arguments(arguments)
                if self._luna_exec_profile
                else parse_grok_shell_arguments(arguments)
                if self._grok_shell_profile
                else parse_execute_arguments(arguments)
            )
        except ValueError:
            parsed = None
        command = parsed.command if parsed is not None else arguments.get("command")
        working_dir = Path(
            self._resolve_managed_working_directory(parsed.cwd if parsed is not None else None)
        )
        idle_yield_seconds = (
            parsed.yield_after_idle_sec
            if parsed is not None and parsed.yield_after_idle_sec is not None
            else self._idle_yield_seconds
        )
        output_byte_limit = (
            parsed.output_byte_limit
            if parsed is not None and parsed.output_byte_limit is not None
            else self._output_byte_limit
        )

        return {
            "variant": "shell",
            "command": command,
            "shell_name": info.name,
            "shell_path": info.path,
            "shell_kind": info.kind,
            "shell_provider": info.provider,
            "working_dir": str(working_dir),
            "working_dir_display": format_relative_path(working_dir),
            "idle_yield_seconds": idle_yield_seconds,
            "foreground_yield_seconds": self._foreground_yield_seconds,
            "foreground_auto_await_max_seconds": self._foreground_auto_await_max_seconds,
            "background": parsed.background if parsed is not None else False,
            "lifecycle": parsed.lifecycle if parsed is not None else "session",
            "timeout_seconds": (parsed.hard_timeout_seconds if parsed is not None else None),
            "output_byte_limit": output_byte_limit,
            "streams_output": True,
            "returns_exit_code": True,
        }

    def process_tool_metadata(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Build compact display metadata for process lifecycle tools."""
        operation = self._managed_process_operation(tool_name, arguments)
        metadata: dict[str, Any] = {
            "variant": "shell_process",
            "action": (
                "list"
                if operation.kind == "list"
                else "terminate"
                if operation.kind == "stop"
                else "read_output"
                if operation.kind == "read_output"
                else "poll"
            ),
            "process_id": operation.process_id,
            "wait_sec": operation.wait_sec,
        }
        process_id = operation.process_id
        process = self._managed_processes.get(process_id) if isinstance(process_id, str) else None
        if process is None:
            if (
                isinstance(process_id, str)
                and process_id in self._attached_durable_processes
                and self._durable_process_store is not None
            ):
                try:
                    snapshot = self._durable_process_store.get(process_id)
                except DurableProcessRecordError:
                    return metadata
                managed_snapshot = self._durable_managed_snapshot(snapshot)
                metadata.update(
                    {
                        "command": snapshot.spec.command,
                        "command_summary": summarize_command(snapshot.spec.command),
                        "elapsed_seconds": managed_snapshot.elapsed_seconds,
                        "os_process_id": snapshot.status.child_pid,
                        "total_output_bytes": snapshot.output_total_bytes,
                        "stdout_bytes": snapshot.stdout_total_bytes,
                        "stderr_bytes": snapshot.stderr_total_bytes,
                        "retained_output_bytes": snapshot.output_bytes,
                        "dropped_output_bytes": snapshot.output_dropped_bytes,
                        "output_truncated": snapshot.output_dropped_bytes > 0,
                        "process_status": managed_snapshot.status,
                        "has_observed_output": bool(
                            snapshot.stdout_total_bytes or snapshot.stderr_total_bytes
                        ),
                    }
                )
            return metadata

        snapshot = self._process_snapshot(process, now=time.monotonic())
        metadata.update(
            {
                "command": snapshot.command,
                "command_summary": summarize_command(snapshot.command),
                "elapsed_seconds": snapshot.elapsed_seconds,
                "os_process_id": snapshot.os_process_id,
                "total_output_bytes": snapshot.total_output_bytes,
                "stdout_bytes": process.output_state.lifetime_stdout_bytes,
                "stderr_bytes": process.output_state.lifetime_stderr_bytes,
                "process_status": snapshot.status,
                "seconds_since_last_output": max(
                    time.monotonic() - process.callbacks.last_output_time,
                    0.0,
                ),
                "has_observed_output": process.output_state.had_stream_output,
            }
        )
        now = time.monotonic()
        if process.callbacks.last_stdout_time is not None:
            metadata["seconds_since_last_stdout"] = max(
                now - process.callbacks.last_stdout_time,
                0.0,
            )
        if process.callbacks.last_stderr_time is not None:
            metadata["seconds_since_last_stderr"] = max(
                now - process.callbacks.last_stderr_time,
                0.0,
            )
        return metadata

    def _managed_process_operation(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> _ManagedProcessOperation:
        if tool_name.casefold() == PROCESS_TOOL_NAME and (self._uses_unified_process_profile()):
            try:
                parsed = parse_minimal_process_arguments(
                    arguments,
                    min_wait_seconds=MIN_PROCESS_POLL_WAIT_SECONDS,
                    max_wait_seconds=self._max_process_poll_seconds,
                )
            except ValueError:
                return _ManagedProcessOperation(
                    kind="status",
                    process_id=None,
                    wait_sec=0,
                )
            if parsed.action == "list":
                return _ManagedProcessOperation(
                    kind="list",
                    process_id=None,
                    wait_sec=None,
                )
            if isinstance(parsed, MinimalProcessReadOutputArguments):
                return _ManagedProcessOperation(
                    kind="read_output",
                    process_id=parsed.process_id,
                    wait_sec=None,
                )
            return _ManagedProcessOperation(
                kind=parsed.action,
                process_id=parsed.process_id,
                wait_sec=(
                    None
                    if parsed.action in {"stop", "read_output"}
                    else 0
                    if parsed.action == "status"
                    else (
                        parsed.wait_sec
                        if parsed.wait_sec is not None
                        else self._minimal_process_wait_seconds()
                    )
                ),
            )

        process_id = arguments.get("process_id")
        return _ManagedProcessOperation(
            kind="stop" if tool_name == TERMINATE_PROCESS_TOOL_NAME else "wait",
            process_id=process_id if isinstance(process_id, str) else None,
            wait_sec=(
                None
                if tool_name == TERMINATE_PROCESS_TOOL_NAME
                else arguments.get("wait_sec", self._process_poll_default_wait_seconds)
            ),
        )

    async def list_processes(self) -> CallToolResult:
        """Return retained managed-process handles in creation order."""
        snapshots = await self.process_snapshots()
        if not snapshots:
            return _text_result("No managed processes.", is_error=False)

        payload: _ProcessListResult = {
            "processes": [
                {
                    "process_id": snapshot.process_id,
                    "status": snapshot.status,
                    "lifecycle": snapshot.lifecycle,
                    "command": snapshot.command,
                    "working_directory": snapshot.working_directory,
                    "elapsed_seconds": round(snapshot.elapsed_seconds, 3),
                    "total_output_bytes": snapshot.total_output_bytes,
                    "exit_code": snapshot.exit_code,
                }
                for snapshot in snapshots
            ]
        }
        return _text_result(json.dumps(payload, indent=2), is_error=False)

    async def _read_durable_output_delta(
        self,
        process_id: str,
        snapshot: DurableProcessSnapshot,
        output_preview_limit: int | None = None,
    ) -> tuple[str, int]:
        store = self._durable_process_store
        if store is None:
            return "", 0
        offset = self._durable_output_offsets.get(process_id, 0)
        unread_bytes = max(snapshot.output_bytes - offset, 0)
        if unread_bytes == 0:
            self._durable_output_total_offsets[process_id] = snapshot.output_total_bytes
            self._durable_output_dropped_offsets[process_id] = snapshot.output_dropped_bytes
            return "", 0
        output_limit = min(
            snapshot.spec.output_byte_limit,
            MAX_TERMINAL_OUTPUT_BYTE_LIMIT,
        )
        if output_preview_limit is not None:
            output = await asyncio.to_thread(
                store.read_output,
                process_id,
                stream=DurableProcessStream.COMBINED,
                offset=offset,
                # Read through a possible UTF-8 boundary before capping the preview.
                limit=min(unread_bytes, min(output_preview_limit, output_limit) + 3),
            )
            text = process_output_preview(
                output.text.encode("utf-8"),
                limit=min(output_preview_limit, output_limit),
                total_bytes=unread_bytes,
            )
        elif unread_bytes <= output_limit:
            output = await asyncio.to_thread(
                store.read_output,
                process_id,
                stream=DurableProcessStream.COMBINED,
                offset=offset,
                limit=unread_bytes,
            )
            text = output.text
        else:
            window = split_output_byte_limit(output_limit)
            head, tail = await asyncio.gather(
                asyncio.to_thread(
                    store.read_output,
                    process_id,
                    stream=DurableProcessStream.COMBINED,
                    offset=offset,
                    limit=max(window.head_bytes, 1),
                ),
                asyncio.to_thread(
                    store.read_output,
                    process_id,
                    stream=DurableProcessStream.COMBINED,
                    offset=snapshot.output_bytes - window.tail_bytes,
                    limit=max(window.tail_bytes, 1),
                ),
            )
            text = (
                head.text
                + format_output_truncation_notice(
                    label="Output",
                    total_bytes=unread_bytes,
                    head_bytes=window.head_bytes,
                    tail_bytes=window.tail_bytes,
                    guidance=(
                        "Use the process tool with action='read_output' to inspect "
                        "the durable output spool."
                    ),
                )
                + "\n"
                + tail.text
            )
        self._durable_output_offsets[process_id] = snapshot.output_bytes
        self._durable_output_total_offsets[process_id] = snapshot.output_total_bytes
        self._durable_output_dropped_offsets[process_id] = snapshot.output_dropped_bytes
        return text.rstrip("\n"), unread_bytes

    async def _poll_durable_process(
        self,
        process_id: str,
        *,
        wait_sec: int,
        wake_on_output: bool,
        output_preview_limit: int | None = None,
    ) -> CallToolResult:
        store = self._durable_process_store
        if store is None or process_id not in self._attached_durable_processes:
            return _text_result(
                f"Error: managed shell process {process_id!r} was not found",
                is_error=True,
            )
        poll_started_at = time.monotonic()
        try:
            snapshot = await asyncio.to_thread(store.get, process_id)
            should_wait = wait_sec > 0 and self._durable_status(snapshot) == "running"
            output_wake = False
            unread_offset = self._durable_output_total_offsets.get(process_id, 0)
            observed_bytes = self._durable_observed_output_bytes.get(process_id, 0)
            if snapshot.output_total_bytes > observed_bytes:
                self._durable_last_output_times[process_id] = poll_started_at
            self._durable_observed_output_bytes[process_id] = snapshot.output_total_bytes
            if snapshot.output_total_bytes > unread_offset:
                self._durable_last_output_times.setdefault(process_id, poll_started_at)
            if should_wait:
                snapshot, output_wake = await self._wait_for_durable_process_poll(
                    store,
                    process_id,
                    snapshot,
                    unread_offset=unread_offset,
                    wait_sec=wait_sec,
                    wake_on_output=wake_on_output,
                )
            output_bytes = max(snapshot.output_total_bytes - unread_offset, 0)
            retained_output_offset = self._durable_output_offsets.get(process_id, 0)
            retained_output_bytes = max(snapshot.output_bytes - retained_output_offset, 0)
            dropped_output_offset = self._durable_output_dropped_offsets.get(process_id, 0)
            dropped_output_bytes = max(
                snapshot.output_dropped_bytes - dropped_output_offset,
                0,
            )
            output, _ = await self._read_durable_output_delta(
                process_id,
                snapshot,
                output_preview_limit,
            )
        except (DurableProcessRecordError, OSError) as exc:
            return _text_result(
                f"Error: durable process {process_id!r} could not be read: {exc}",
                is_error=True,
            )

        status = self._durable_status(snapshot)
        if status != "running":
            poll_yield_reason = "completion"
        elif output_wake:
            poll_yield_reason = "output"
        elif should_wait:
            poll_yield_reason = "deadline"
        else:
            poll_yield_reason = "nonblocking"
        started_at = snapshot.status.started_at or snapshot.spec.created_at
        elapsed = max(
            (time.time() if status == "running" else snapshot.status.updated_at) - started_at,
            0.0,
        )
        poll_elapsed_seconds = time.monotonic() - poll_started_at
        last_output_time = self._durable_last_output_times.get(process_id)
        seconds_since_last_output = (
            max(time.monotonic() - last_output_time, 0.0)
            if last_output_time is not None
            else max(time.time() - snapshot.spec.created_at, 0.0)
        )
        output_observed = bool(snapshot.stdout_total_bytes or snapshot.stderr_total_bytes)
        lines = [output] if output else []
        lines.extend(
            [
                f"process_id: {process_id}",
                f"process status: {status}",
                f"elapsed_seconds: {elapsed:.1f}",
            ]
        )
        if snapshot.status.exit_code is not None:
            lines.append(f"process exit code was {snapshot.status.exit_code}")
        elif status == "running":
            lines.append("This durable process remains available across fast-agent invocations.")
        result = process_result(
            "\n".join(lines),
            is_error=status in {"failed", "unavailable"},
            metadata={
                "process_id": process_id,
                "lifecycle": "persistent",
                "process_status": status,
                "process_yield_reason": poll_yield_reason,
                "process_elapsed_seconds": elapsed,
                "os_process_id": snapshot.status.child_pid,
                "output_bytes_since_last_poll": output_bytes,
                "retained_output_bytes_since_last_poll": retained_output_bytes,
                "dropped_output_bytes_since_last_poll": dropped_output_bytes,
                "seconds_since_last_output": seconds_since_last_output,
                "total_output_bytes": snapshot.output_total_bytes,
                "stdout_bytes": snapshot.stdout_total_bytes,
                "stderr_bytes": snapshot.stderr_total_bytes,
                "retained_output_bytes": snapshot.output_bytes,
                "dropped_output_bytes": snapshot.output_dropped_bytes,
                "output_truncated": snapshot.output_dropped_bytes > 0,
                "has_observed_output": output_observed,
                "poll_wait_sec": wait_sec,
                "poll_wake_on_output": wake_on_output,
                "poll_elapsed_seconds": poll_elapsed_seconds,
                **(
                    {
                        "poll_deadline_overshoot_seconds": max(
                            poll_elapsed_seconds - wait_sec,
                            0.0,
                        )
                    }
                    if poll_yield_reason == "deadline"
                    else {}
                ),
                **(
                    {"exit_code": snapshot.status.exit_code}
                    if snapshot.status.exit_code is not None
                    else {}
                ),
            },
        )
        self._append_poll_output_activity(
            result,
            output_bytes=output_bytes,
            output_lines=len(output.splitlines()),
            seconds_since_last_output=seconds_since_last_output,
            output_observed=output_observed,
        )
        return result

    async def _wait_for_durable_process_poll(
        self,
        store: DurableProcessStore,
        process_id: str,
        snapshot: DurableProcessSnapshot,
        *,
        unread_offset: int,
        wait_sec: int,
        wake_on_output: bool,
    ) -> tuple[DurableProcessSnapshot, bool]:
        """Wait for durable completion/deadline, optionally after output settles."""
        observed_bytes = snapshot.output_total_bytes
        deadline = time.monotonic() + wait_sec
        while self._durable_status(snapshot) == "running":
            now = time.monotonic()
            remaining = deadline - now
            if remaining <= 0:
                return snapshot, False

            pending_output = observed_bytes > unread_offset
            last_output_time = self._durable_last_output_times.get(process_id, now)
            seconds_since_last_output = max(now - last_output_time, 0.0)
            if (
                wake_on_output
                and pending_output
                and seconds_since_last_output >= _PROCESS_OUTPUT_DEBOUNCE_SECONDS
            ):
                return snapshot, True

            quiet_wait = (
                _PROCESS_OUTPUT_DEBOUNCE_SECONDS - seconds_since_last_output
                if wake_on_output and pending_output
                else remaining
            )
            await asyncio.sleep(min(0.1, remaining, quiet_wait))
            snapshot = await asyncio.to_thread(store.get, process_id)
            if snapshot.output_total_bytes > observed_bytes:
                self._durable_last_output_times[process_id] = time.monotonic()
            observed_bytes = snapshot.output_total_bytes
            self._durable_observed_output_bytes[process_id] = observed_bytes

        return snapshot, False

    async def _read_durable_process_output(
        self,
        parsed: MinimalProcessReadOutputArguments,
    ) -> CallToolResult:
        store = self._durable_process_store
        if store is None or parsed.process_id not in self._attached_durable_processes:
            return _text_result(
                f"Error: managed shell process {parsed.process_id!r} was not found",
                is_error=True,
            )
        limit = parsed.limit or min(
            self._output_byte_limit,
            MAX_TERMINAL_OUTPUT_BYTE_LIMIT,
        )
        try:
            snapshot = await asyncio.to_thread(store.get, parsed.process_id)
            output = await asyncio.to_thread(
                store.read_output,
                parsed.process_id,
                stream=DurableProcessStream.COMBINED,
                offset=parsed.offset,
                limit=limit,
                query=parsed.query,
            )
        except (DurableProcessRecordError, OSError, ValueError) as exc:
            return _text_result(
                f"Error: durable process output could not be read: {exc}",
                is_error=True,
            )
        status = self._durable_status(snapshot)
        payload = {
            "process_id": parsed.process_id,
            "process_status": status,
            "retained_output_bytes": snapshot.output_bytes,
            "dropped_output_bytes": snapshot.output_dropped_bytes,
            "output_truncated": snapshot.output_dropped_bytes > 0,
            **self._durable_output_payload(output),
        }
        return process_result(
            json.dumps(payload, indent=2),
            is_error=False,
            metadata={
                "process_id": parsed.process_id,
                "process_status": status,
                "retained_output_bytes": snapshot.output_bytes,
                "retained_output_complete": (
                    status != "running" and snapshot.output_dropped_bytes == 0
                ),
                "dropped_output_bytes": snapshot.output_dropped_bytes,
                "output_truncated": snapshot.output_dropped_bytes > 0,
                "output_read_offset": parsed.offset,
                "output_read_bytes": output.returned_bytes,
                "output_read_has_more": not output.at_end,
                **(
                    {
                        "output_query": parsed.query,
                        "output_match_count": output.match_count or 0,
                    }
                    if parsed.query is not None
                    else {}
                ),
            },
        )

    @staticmethod
    def _durable_output_payload(output: DurableProcessOutput) -> dict[str, object]:
        return {
            "offset": output.offset,
            "next_offset": output.next_offset,
            "at_end": output.at_end,
            "content": output.text,
            **({"match_count": output.match_count} if output.match_count is not None else {}),
        }

    async def read_process_output(
        self,
        parsed: MinimalProcessReadOutputArguments,
    ) -> CallToolResult:
        """Read bounded retained output owned by one managed process."""
        process = await self._get_managed_process(parsed.process_id)
        if process is None:
            if parsed.process_id in self._attached_durable_processes:
                return await self._read_durable_process_output(parsed)
            return _text_result(
                f"Error: managed shell process {parsed.process_id!r} was not found",
                is_error=True,
            )

        retained_path = process.output_state.retained_output_path
        if retained_path is None or not retained_path.is_file():
            return process_result(
                (
                    f"process_id: {parsed.process_id}\n"
                    "retained_output: unavailable\n"
                    "No retained full output exists for this process. Retention begins "
                    "only after model-facing output is truncated; use status or wait "
                    "for current unread output."
                ),
                is_error=True,
                metadata={
                    "process_id": parsed.process_id,
                    "process_status": self._process_snapshot(
                        process,
                        now=time.monotonic(),
                    ).status,
                    "retained_output_bytes": 0,
                    "retained_output_complete": True,
                },
            )

        try:
            retained_blob = retained_path.read_bytes()
        except OSError as exc:
            return _text_result(
                f"Error: retained output for process {parsed.process_id!r} could not be read: "
                f"{exc.__class__.__name__}",
                is_error=True,
            )

        offset = min(parsed.offset, len(retained_blob))
        limit = parsed.limit or min(
            self._output_byte_limit,
            MAX_TERMINAL_OUTPUT_BYTE_LIMIT,
        )
        query = parsed.query
        match_count = 0
        if query is None:
            content_blob = retained_blob[offset : offset + limit]
            next_offset = offset + len(content_blob)
            has_more = next_offset < len(retained_blob)
            content = content_blob.decode("utf-8", errors="replace")
        else:
            matching_lines = [
                line
                for line in retained_blob[offset:]
                .decode(
                    "utf-8",
                    errors="replace",
                )
                .splitlines(keepends=True)
                if query in line
            ]
            match_count = len(matching_lines)
            selected = bytearray()
            returned_matches = 0
            truncated_match = False
            for line in matching_lines:
                line_blob = line.encode("utf-8", errors="replace")
                remaining = limit - len(selected)
                if remaining <= 0:
                    break
                selected.extend(line_blob[:remaining])
                returned_matches += 1
                if len(line_blob) > remaining:
                    truncated_match = True
                    break
            content_blob = bytes(selected)
            content = content_blob.decode("utf-8", errors="replace")
            next_offset = offset
            has_more = truncated_match or returned_matches < match_count

        snapshot = self._process_snapshot(process, now=time.monotonic())
        payload = {
            "process_id": parsed.process_id,
            "process_status": snapshot.status,
            "retained_output_bytes": len(retained_blob),
            "retained_output_complete": process.output_state.retained_output_complete,
            "offset": offset,
            "returned_bytes": len(content_blob),
            "next_offset": next_offset,
            "has_more": has_more,
            "query": query,
            "match_count": match_count if query is not None else None,
            "content": content,
        }
        return process_result(
            json.dumps(payload, indent=2),
            is_error=False,
            metadata={
                "process_id": parsed.process_id,
                "process_status": snapshot.status,
                "retained_output_bytes": len(retained_blob),
                "retained_output_complete": process.output_state.retained_output_complete,
                "output_read_offset": offset,
                "output_read_bytes": len(content_blob),
                "output_read_has_more": has_more,
                **({"output_query": query, "output_match_count": match_count} if query else {}),
            },
        )

    def _invalid_execute_result(self, message: str) -> CallToolResult:
        return _text_result(message, is_error=True)

    def _minimal_process_wait_seconds(self) -> int:
        configured_wait = (
            self._process_poll_default_wait_seconds
            if self._process_poll_default_wait_seconds > 0
            else _DEFAULT_MINIMAL_PROCESS_WAIT_SECONDS
        )
        return min(
            max(configured_wait, MIN_PROCESS_POLL_WAIT_SECONDS),
            self._max_process_poll_seconds,
        )

    def _uses_unified_process_profile(self) -> bool:
        return self._minimal_process_profile or self._grok_shell_profile or self._luna_exec_profile

    def set_process_poll_default_wait_seconds(self, value: int) -> None:
        """Update the model-specific default used when wait_sec is omitted."""
        default_wait = value if type(value) is int and value >= 0 else 0
        self._process_poll_default_wait_seconds = min(
            default_wait,
            self._max_process_poll_seconds,
        )
        if self._poll_process_tool is not None:
            set_poll_process_tool_default_wait_seconds(
                self._poll_process_tool,
                default_wait_seconds=(
                    self._minimal_process_wait_seconds()
                    if self._uses_unified_process_profile()
                    else self._process_poll_default_wait_seconds
                ),
            )

    def _build_display_state(
        self,
        *,
        defer_display_to_tool_result: bool,
        display_line_limit: int | None = None,
        respect_tool_display: bool = True,
    ) -> ShellDisplayState:
        compact_summary_only = respect_tool_display and self._compact_shell_summary_only()
        use_live_shell_display = (
            self._show_bash_output
            and not defer_display_to_tool_result
            and not compact_summary_only
            and display_tools_enabled()
        )
        state = ShellDisplayState(
            use_live_shell_display=use_live_shell_display,
            display_line_limit=display_line_limit,
        )
        if display_line_limit is not None and display_line_limit > 0:
            display_window = split_shell_output_line_limit(display_line_limit)
            state.display_head_limit = display_window.head_lines
            state.display_tail_limit = display_window.tail_lines
            state.display_tail_buffer = deque(maxlen=max(display_window.tail_lines, 1))
        return state

    def _compact_shell_summary_only(self) -> bool:
        return self._config is not None and (
            self._display.tool_display_layout == "compact"
            and self._display.tool_display_settings.results != "all"
        )

    def _maybe_print_truncation_notice(
        self,
        *,
        output_state: ShellOutputBuffer,
        display_state: ShellDisplayState,
    ) -> None:
        if output_state.truncation_notice_printed or not output_state.output_truncated:
            return
        if display_state.use_live_shell_display and (
            display_state.display_line_limit is None or display_state.display_line_limit > 0
        ):
            estimated_tokens = int(output_state.output_byte_limit / TERMINAL_BYTES_PER_TOKEN)
            console.console.print(
                " ".join(
                    [
                        "▶ Shell to agent output reached",
                        f"{output_state.output_byte_limit} bytes",
                        f"(~{estimated_tokens} tokens);",
                        "additional output omitted from tool result.",
                    ]
                ),
                style=self._truncation_notice_style(output_state),
            )
        output_state.truncation_notice_printed = True

    @staticmethod
    def _truncation_notice_style(output_state: ShellOutputBuffer) -> str:
        return "black on blue" if output_state.output_byte_limit_requested else "black on red"

    def _print_timeout_notice(
        self,
        display_state: ShellDisplayState,
        *,
        timeout_seconds: float | None = None,
    ) -> None:
        if not display_state.use_live_shell_display or display_state.timeout_notice_printed:
            return
        message = "▶ Timeout exceeded - terminating process"
        if timeout_seconds is not None:
            message = f"▶ Timeout after {timeout_seconds:g}s - process terminated"
        console.console.print(message, style="black on red")
        display_state.timeout_notice_printed = True

    def _render_live_shell_output(
        self,
        text: str,
        style: str | None,
        *,
        display_state: ShellDisplayState,
    ) -> None:
        if not display_state.use_live_shell_display:
            return
        if display_state.display_line_limit is None:
            console.console.print(
                self._render_display_line(text, style),
                markup=False,
            )
            return
        if display_state.display_line_limit <= 0:
            return

        display_state.display_total_line_count += 1
        current_line_index = display_state.display_total_line_count
        if display_state.displayed_head_count < display_state.display_head_limit:
            console.console.print(
                self._render_display_line(text, style),
                markup=False,
            )
            display_state.displayed_head_count += 1
            return

        if display_state.display_tail_limit > 0:
            display_state.display_tail_buffer.append((current_line_index, text, style))
        if current_line_index > display_state.display_line_limit:
            display_state.display_overflowed = True
            if not display_state.display_ellipsis_printed:
                console.console.print(
                    SHELL_OUTPUT_TRUNCATION_MARKER,
                    style="dim",
                    markup=False,
                )
                display_state.display_ellipsis_printed = True

    def _record_stream_output(
        self,
        text: str,
        *,
        style: str | None,
        output_state: ShellOutputBuffer,
        display_state: ShellDisplayState,
        is_stderr: bool,
        count_bytes: bool = True,
    ) -> None:
        output_state.had_stream_output = True
        output_state.unread_output_activity = True
        output_state.output_line_count += 1
        output_state.unread_output_line_count += 1
        output_state.append_stream(
            text,
            is_stderr=is_stderr,
            count_bytes=count_bytes,
        )
        self._maybe_print_truncation_notice(
            output_state=output_state,
            display_state=display_state,
        )
        self._render_live_shell_output(
            text,
            style,
            display_state=display_state,
        )

    async def _emit_watchdog_progress(self, elapsed: float) -> None:
        ctx = _tool_progress_context.get()
        if not ctx:
            return
        handler, tool_call_id = ctx
        try:
            await handler.on_tool_progress(
                tool_call_id,
                0.5,
                None,
                f"Waiting for output ({int(elapsed)}) seconds ...",
            )
        except Exception:
            return

    def _flush_live_display_tail(self, display_state: ShellDisplayState) -> None:
        if (
            not display_state.use_live_shell_display
            or display_state.display_line_limit is None
            or display_state.display_line_limit <= 0
        ):
            return
        if display_state.display_overflowed and not display_state.display_ellipsis_printed:
            console.console.print(
                SHELL_OUTPUT_TRUNCATION_MARKER,
                style="dim",
                markup=False,
            )
        for buffered_index, buffered_text, buffered_style in display_state.display_tail_buffer:
            if (
                display_state.display_overflowed
                and buffered_index <= display_state.display_line_limit
            ):
                continue
            console.console.print(
                self._render_display_line(buffered_text, buffered_style),
                markup=False,
            )

    def _finalize_shell_result_display(
        self,
        result: CallToolResult,
        *,
        shell_result: ShellExecutionResult,
        output_state: ShellOutputBuffer,
        display_state: ShellDisplayState,
        tool_use_id: str | None,
        show_tool_call_id: bool,
        defer_display_to_tool_result: bool,
    ) -> CallToolResult:
        self._flush_live_display_tail(display_state)
        if display_state.use_live_shell_display:
            self._display.show_shell_exit_code(
                shell_result.exit_code,
                no_output=not output_state.had_stream_output,
                output_line_count=output_state.output_line_count
                if output_state.had_stream_output
                else None,
                tool_call_id=tool_use_id if show_tool_call_id else None,
            )

        suppress_display = True
        compact_summary_only = self._compact_shell_summary_only()
        if (defer_display_to_tool_result or compact_summary_only) and self._show_bash_output:
            suppress_display = False
        update_tool_result_display_metadata(
            result,
            {
                "suppress_display": suppress_display,
                "exit_code": shell_result.exit_code,
                "output_line_count": output_state.output_line_count,
            },
        )
        return result

    async def _execute_shell_command(
        self,
        command: str,
        *,
        cwd: str | Path | None,
        env: Mapping[str, str] | None,
        timeout: float | None,
        output_byte_limit: int | None,
        defer_display_to_tool_result: bool,
        display_line_limit: int | None,
        respect_tool_display: bool = True,
    ) -> _ShellRuntimeExecution:
        output_state = ShellOutputBuffer(
            output_byte_limit=(
                self._output_byte_limit if output_byte_limit is None else output_byte_limit
            ),
            output_byte_limit_requested=output_byte_limit is not None,
            retained_output_path=self._next_retained_output_path(),
            retained_output_max_bytes=self._retained_output_max_bytes,
            retained_output_via_process=self._retained_output_via_process,
            extended_guidance=self._extended_guidance,
        )
        display_state = self._build_display_state(
            defer_display_to_tool_result=defer_display_to_tool_result,
            display_line_limit=display_line_limit,
            respect_tool_display=respect_tool_display,
        )
        execution = await self._environment.execute(
            ShellExecutionRequest(
                command=command,
                cwd=str(cwd) if cwd is not None else self._working_directory,
                env=env,
                timeout=self._timeout_seconds if timeout is None else timeout,
            ),
            callbacks=ShellRuntimeCallbacks(
                runtime=self,
                progress=self._progress,
                output_state=output_state,
                display_state=display_state,
            ),
        )
        return _ShellRuntimeExecution(
            execution=execution,
            output_state=output_state,
            display_state=display_state,
        )

    async def _start_managed_process(
        self,
        parsed: ShellExecuteArguments,
        *,
        defer_display_to_tool_result: bool,
    ) -> ManagedShellProcess:
        output_state = ShellOutputBuffer(
            output_byte_limit=(
                self._output_byte_limit
                if parsed.output_byte_limit is None
                else parsed.output_byte_limit
            ),
            output_byte_limit_requested=parsed.output_byte_limit is not None,
            retained_output_path=self._next_retained_output_path(),
            retained_output_max_bytes=self._retained_output_max_bytes,
            retained_output_via_process=self._retained_output_via_process,
            extended_guidance=self._extended_guidance,
        )
        display_state = self._build_display_state(
            defer_display_to_tool_result=defer_display_to_tool_result,
            display_line_limit=self._output_display_lines,
        )
        callbacks = ShellRuntimeCallbacks(
            runtime=self,
            progress=self._progress,
            output_state=output_state,
            display_state=display_state,
        )
        working_directory = self._resolve_managed_working_directory(parsed.cwd)

        async with self._processes_lock:
            completed_ids = [
                process_id
                for process_id, process in self._managed_processes.items()
                if process.task.done()
            ]
            while len(self._managed_processes) >= MAX_MANAGED_SHELL_PROCESSES and completed_ids:
                self._managed_processes.pop(completed_ids.pop(0))
            active_managed_processes = sum(
                not process.task.done() for process in self._managed_processes.values()
            )
            active_durable_processes = await asyncio.to_thread(
                self._stored_active_durable_process_count
            )
            if active_managed_processes + active_durable_processes >= MAX_MANAGED_SHELL_PROCESSES:
                raise RuntimeError(
                    f"at most {MAX_MANAGED_SHELL_PROCESSES} managed shell processes may run at once"
                )

            process_id = f"process-{self._next_process_id}"
            self._next_process_id += 1
            request = ShellExecutionRequest(
                command=parsed.command,
                cwd=working_directory,
                env=None,
                timeout=None,
                terminate_after_idle=False,
                retain_output=False,
                terminate_on_cancel=parsed.lifecycle == "session",
                detach=parsed.lifecycle == "persistent",
            )
            task = asyncio.create_task(
                self._environment.execute(request, callbacks=callbacks),
                name=f"fast-agent-{process_id}",
            )
            process = ManagedShellProcess(
                process_id=process_id,
                command=parsed.command,
                working_directory=working_directory,
                started_at=time.monotonic(),
                task=task,
                request=request,
                lifecycle=parsed.lifecycle,
                intentional_persistent_background=(
                    parsed.background and parsed.lifecycle == "persistent"
                ),
                callbacks=callbacks,
                output_state=output_state,
                display_state=display_state,
            )
            callbacks.process = process
            task.add_done_callback(
                lambda completed_task: self._record_managed_process_completion(
                    process,
                    completed_task,
                )
            )
            self._managed_processes[process_id] = process
            return process

    async def _start_durable_process(
        self,
        parsed: ShellExecuteArguments,
    ) -> DurableProcessSnapshot:
        store = self._durable_process_store
        if store is None or not isinstance(self._environment, LocalShellExecutor):
            raise RuntimeError("Durable local processes are not available.")
        session_id = self._session_id_provider() if self._session_id_provider is not None else None
        async with self._processes_lock:
            active_managed_processes = sum(
                not process.task.done() for process in self._managed_processes.values()
            )
            durable_capacity = MAX_MANAGED_SHELL_PROCESSES - active_managed_processes
            if durable_capacity <= 0:
                raise RuntimeError(
                    f"at most {MAX_MANAGED_SHELL_PROCESSES} managed shell processes may run at once"
                )
            launch_task = asyncio.create_task(
                asyncio.to_thread(
                    self._environment.start_durable_process,
                    store,
                    command=parsed.command,
                    cwd=Path(self._resolve_managed_working_directory(parsed.cwd)),
                    origin_session_id=session_id,
                    agent_name=self._agent_name,
                    output_byte_limit=(
                        parsed.output_byte_limit
                        if parsed.output_byte_limit is not None
                        else self._output_byte_limit
                    ),
                    output_retention_byte_limit=self._durable_output_max_bytes,
                    max_active_processes=durable_capacity,
                ),
                name="fast-agent-durable-process-launch",
            )
            cancelled: asyncio.CancelledError | None = None
            try:
                snapshot = await asyncio.shield(launch_task)
            except asyncio.CancelledError as exc:
                cancelled = exc
                while True:
                    try:
                        snapshot = await asyncio.shield(launch_task)
                    except asyncio.CancelledError:
                        if launch_task.cancelled():
                            raise exc from None
                        continue
                    except Exception as launch_error:
                        raise exc from launch_error
                    break
            self._register_durable_process(snapshot)
            if cancelled is not None:
                raise cancelled
        return snapshot

    def _register_durable_process(self, snapshot: DurableProcessSnapshot) -> None:
        process_id = snapshot.spec.process_id
        self._attached_durable_processes.add(process_id)
        self._durable_output_offsets[process_id] = 0
        self._durable_output_total_offsets[process_id] = 0
        self._durable_output_dropped_offsets[process_id] = 0
        self._durable_observed_output_bytes[process_id] = snapshot.output_total_bytes
        if snapshot.output_total_bytes > 0:
            self._durable_last_output_times[process_id] = time.monotonic()
        self._durable_poll_locks[process_id] = asyncio.Lock()

    def _durable_launch_result(
        self,
        snapshot: DurableProcessSnapshot,
    ) -> CallToolResult:
        process_id = snapshot.spec.process_id
        status = self._durable_status(snapshot)
        store = self._durable_process_store
        output_spool_path = str(store.directory(process_id)) if store is not None else None
        if status == "running":
            status_lines = [
                "Managed background process is still running.",
                (
                    "This durable process remains available across fast-agent invocations. "
                    "Use `process` with action='status' to inspect it or action='stop' "
                    "to request termination."
                ),
            ]
        elif status == "completed":
            status_lines = [
                "Managed background process completed before launch acknowledgement.",
                f"process exit code was {snapshot.status.exit_code}",
            ]
        elif status == "failed":
            status_lines = [
                "Managed background process failed before launch acknowledgement.",
                f"process exit code was {snapshot.status.exit_code}",
            ]
        elif status == "terminated":
            status_lines = ["Managed background process was stopped during launch."]
        else:
            status_lines = ["Managed background process supervisor is unavailable."]
        message = "\n".join(
            [
                status_lines[0],
                "effective_lifecycle: persistent",
                f"process_id: {process_id}",
                f"os_pid: {snapshot.status.child_pid}",
                *(
                    [f"output_spool_path: {output_spool_path}"]
                    if output_spool_path is not None
                    else []
                ),
                *status_lines[1:],
            ]
        )
        return process_result(
            message,
            is_error=status in {"failed", "unavailable"},
            metadata={
                "process_id": process_id,
                "lifecycle": "persistent",
                "process_status": status,
                "process_yield_reason": "background",
                "process_elapsed_seconds": 0.0,
                "os_process_id": snapshot.status.child_pid,
                "total_output_bytes": snapshot.output_total_bytes,
                "stdout_bytes": snapshot.stdout_total_bytes,
                "stderr_bytes": snapshot.stderr_total_bytes,
                "retained_output_bytes": snapshot.output_bytes,
                "dropped_output_bytes": snapshot.output_dropped_bytes,
                "output_truncated": snapshot.output_dropped_bytes > 0,
                **(
                    {"exit_code": snapshot.status.exit_code}
                    if snapshot.status.exit_code is not None
                    else {}
                ),
                **(
                    {"output_spool_path": output_spool_path}
                    if output_spool_path is not None
                    else {}
                ),
            },
        )

    @staticmethod
    def _record_managed_process_completion(
        process: ManagedShellProcess,
        completed_task: asyncio.Task[ShellExecution],
    ) -> None:
        del completed_task
        if process.completed_at is None:
            process.completed_at = time.monotonic()

    def _resolve_managed_working_directory(self, requested_cwd: str | None) -> str:
        base_cwd = self._working_directory or self._environment.cwd
        runtime_info = self.runtime_info()
        if runtime_info.kind == "local":
            base_path = Path(base_cwd)
            if not base_path.is_absolute():
                base_path = Path(self._environment.cwd) / base_path
            if requested_cwd is None:
                candidate = str(base_path)
            else:
                requested_path = Path(requested_cwd)
                candidate = str(
                    requested_path if requested_path.is_absolute() else base_path / requested_path
                )
        else:
            resolved_base = (
                base_cwd
                if posixpath.isabs(base_cwd)
                else posixpath.join(self._environment.cwd, base_cwd)
            )
            if requested_cwd is None:
                candidate = resolved_base
            else:
                candidate = (
                    requested_cwd
                    if posixpath.isabs(requested_cwd)
                    else posixpath.join(resolved_base, requested_cwd)
                )

        resolver = getattr(self._environment, "resolve_path", None)
        if callable(resolver):
            return str(resolver(candidate))
        if runtime_info.kind == "local":
            return str(Path(candidate).resolve())
        return posixpath.normpath(candidate)

    def _next_retained_output_path(self) -> Path | None:
        if self._retained_output_directory is None:
            return None
        path = self._retained_output_directory / (f"output-{self._retained_output_next_id}.log")
        self._retained_output_next_id += 1
        return path

    async def _get_managed_process(self, process_id: str) -> ManagedShellProcess | None:
        async with self._processes_lock:
            return self._managed_processes.get(process_id)

    async def _sample_managed_process_resources(
        self,
        process: ManagedShellProcess,
    ) -> ProcessResourceSnapshot | None:
        if not self._resource_observations_enabled:
            return None
        pid = process.callbacks.os_process_id if not process.task.done() else None
        try:
            async with asyncio.timeout(_RESOURCE_OBSERVATION_TIMEOUT_SECONDS):
                return await sample_process_resources(process.working_directory, pid)
        except Exception:
            return None

    async def _capture_process_resource_baseline(
        self,
        process: ManagedShellProcess,
    ) -> None:
        snapshot = await self._sample_managed_process_resources(process)
        if snapshot is not None:
            observe_resource_changes(process.resource_observations, snapshot)

    async def _wait_for_initial_process_result(
        self,
        process: ManagedShellProcess,
        *,
        idle_yield_seconds: float,
    ) -> Literal["idle", "foreground"] | None:
        if process.task.done():
            return None
        foreground_deadline = process.started_at + self._foreground_yield_seconds

        while not process.task.done():
            now = time.monotonic()
            idle_deadline = process.callbacks.last_output_time + idle_yield_seconds
            deadline = min(idle_deadline, foreground_deadline)
            if now >= deadline:
                return "idle" if idle_deadline <= foreground_deadline else "foreground"

            process.callbacks.activity_event.clear()
            if process.task.done():
                return None
            activity_task = asyncio.create_task(process.callbacks.activity_event.wait())
            try:
                done, _ = await asyncio.wait(
                    (process.task, activity_task),
                    timeout=max(deadline - time.monotonic(), 0),
                    return_when=asyncio.FIRST_COMPLETED,
                )
            finally:
                if not activity_task.done():
                    activity_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await activity_task
            if process.task in done:
                return None

        return None

    async def _auto_await_foreground_process(
        self,
        process: ManagedShellProcess,
        *,
        initial_yield_reason: Literal["idle", "foreground"],
    ) -> ForegroundAutoAwaitMetadata:
        """Continue one foreground invocation without creating a model action."""
        auto_await_started_at = time.monotonic()
        initial_yield_elapsed_seconds = max(
            auto_await_started_at - process.started_at,
            0.0,
        )
        deadline_at = process.started_at + self._foreground_auto_await_max_seconds
        remaining_seconds = max(deadline_at - auto_await_started_at, 0.0)
        if self._foreground_auto_await_max_seconds > 0:
            await asyncio.wait(
                (process.task,),
                timeout=remaining_seconds,
                return_when=asyncio.FIRST_COMPLETED,
            )
        finished_at = time.monotonic()
        outcome: Literal[
            "process_finished",
            "cap_reached",
            "terminated",
            "cancelled",
            "disabled",
        ]
        if self._foreground_auto_await_max_seconds <= 0:
            outcome = "disabled"
        elif process.task.cancelled():
            outcome = "terminated" if process.terminated else "cancelled"
        elif process.task.done():
            outcome = "process_finished"
        else:
            outcome = "cap_reached"
        metadata = ForegroundAutoAwaitMetadata(
            initial_yield_reason=initial_yield_reason,
            max_total_seconds=self._foreground_auto_await_max_seconds,
            initial_yield_elapsed_seconds=initial_yield_elapsed_seconds,
            awaited_seconds=(
                0.0 if outcome == "disabled" else max(finished_at - auto_await_started_at, 0.0)
            ),
            total_elapsed_seconds=max(finished_at - process.started_at, 0.0),
            outcome=outcome,
        )
        process.foreground_auto_await = metadata
        return metadata

    @staticmethod
    async def _terminate_managed_process_task(process: ManagedShellProcess) -> None:
        """Request process-group termination and wait for the environment contract."""
        if process.task.done():
            return
        process.terminated = True
        process.request.terminate_on_cancel = True
        process.task.cancel()
        await asyncio.gather(process.task, return_exceptions=True)

    def _record_buffered_process_result(self, process: ManagedShellProcess) -> None:
        if process.buffered_result_recorded or not process.task.done():
            return
        process.buffered_result_recorded = True
        if process.task.cancelled() or process.task.exception() is not None:
            return
        execution = process.task.result()
        if process.output_state.had_stream_output:
            return
        recorded_output = False
        recorded_at = time.monotonic()
        if execution.result.stdout:
            self._record_stream_output(
                execution.result.stdout,
                style=None,
                output_state=process.output_state,
                display_state=process.display_state,
                is_stderr=False,
            )
            process.callbacks.last_stdout_time = recorded_at
            recorded_output = True
        if execution.result.stderr:
            self._record_stream_output(
                execution.result.stderr,
                style="red",
                output_state=process.output_state,
                display_state=process.display_state,
                is_stderr=True,
            )
            process.callbacks.last_stderr_time = recorded_at
            recorded_output = True
        if recorded_output:
            process.callbacks.last_output_time = recorded_at

    @staticmethod
    def _append_poll_output_activity(
        result: CallToolResult,
        *,
        output_bytes: int,
        output_lines: int,
        seconds_since_last_output: float,
        output_observed: bool,
    ) -> None:
        activity = (
            f"{seconds_since_last_output:.1f}s since last output"
            if output_observed
            else f"no output observed for {seconds_since_last_output:.1f}s"
        )
        line = (
            f"output_activity: {output_lines} lines / {output_bytes} bytes "
            f"since last poll; {activity}"
        )
        for block in result.content:
            if isinstance(block, TextContent):
                block.text = f"{block.text}\n{line}"
                return

    @staticmethod
    def _append_resource_observation(
        result: CallToolResult,
        observation: str,
    ) -> None:
        for block in result.content:
            if isinstance(block, TextContent):
                block.text = f"{block.text}\nresource_observation: {observation}"
                return

    def _managed_process_result(
        self,
        process: ManagedShellProcess,
        *,
        yielded_reason: str | None = None,
        output_preview_limit: int | None = None,
    ) -> CallToolResult:
        self._record_buffered_process_result(process)
        result = build_managed_process_result(
            process,
            yielded_reason=yielded_reason,
            minimal_process_profile=self._minimal_process_profile,
            aligned_shell_tool_name=(
                LUNA_EXEC_TOOL_NAME
                if self._luna_exec_profile
                else GROK_SHELL_TOOL_NAME
                if self._grok_shell_profile
                else None
            ),
            io_drain_timeout_seconds=_IO_DRAIN_TIMEOUT_SECONDS,
            output_preview_limit=output_preview_limit,
        )
        metadata = process_result_metadata(result)
        if metadata is not None and process.foreground_auto_await is not None:
            metadata["foreground_auto_await"] = process.foreground_auto_await
        return result

    async def poll_process(
        self,
        arguments: dict[str, Any] | None = None,
        *,
        progress_tool_use_id: str | None = None,
        output_preview_limit: int | None = None,
    ) -> CallToolResult:
        """Return incremental output and status for a managed process."""
        poll_started_at = time.monotonic()
        try:
            parsed = parse_poll_process_arguments(
                arguments,
                default_wait_seconds=self._process_poll_default_wait_seconds,
                max_wait_seconds=self._max_process_poll_seconds,
            )
        except ValueError as exc:
            return _text_result(str(exc), is_error=True)

        process = await self._get_managed_process(parsed.process_id)
        if process is None:
            async with self._processes_lock:
                durable_poll_lock = self._durable_poll_locks.get(parsed.process_id)
            if durable_poll_lock is not None:
                async with durable_poll_lock:
                    return await self._poll_durable_process(
                        parsed.process_id,
                        wait_sec=parsed.wait_sec,
                        wake_on_output=parsed.wake_on_output,
                        output_preview_limit=output_preview_limit,
                    )
            return _text_result(
                f"Error: managed shell process {parsed.process_id!r} was not found",
                is_error=True,
            )

        async with process.poll_lock:
            async with process.lock:
                should_wait = parsed.wait_sec > 0 and not process.task.done()

            waited = False
            output_wake = False
            poll_started_at_monotonic = time.monotonic()
            active_poll = (
                ActiveProcessPoll(
                    tool_use_id=progress_tool_use_id,
                    deadline_at=poll_started_at_monotonic + parsed.wait_sec,
                    started_at=poll_started_at_monotonic,
                )
                if should_wait and progress_tool_use_id is not None
                else None
            )
            process.active_poll = active_poll
            if active_poll is not None:
                # Quiet processes (sleep) never stream output; heartbeat keeps
                # the live countdown bar visible for the whole wait.
                active_poll.heartbeat_task = asyncio.create_task(
                    self._progress.poll_heartbeat(process, active_poll),
                    name=f"fast-agent-{process.process_id}-poll-heartbeat",
                )
            try:
                if should_wait:
                    waited = True
                    output_wake = await self._wait_for_managed_process_poll(
                        process,
                        wait_sec=parsed.wait_sec,
                        wake_on_output=parsed.wake_on_output,
                    )
            finally:
                if process.active_poll is active_poll:
                    process.active_poll = None
                if active_poll is not None:
                    if active_poll.pending_progress_task is not None:
                        active_poll.pending_progress_task.cancel()
                    if active_poll.heartbeat_task is not None:
                        active_poll.heartbeat_task.cancel()
                        with suppress(asyncio.CancelledError):
                            await active_poll.heartbeat_task

            poll_elapsed_seconds = time.monotonic() - poll_started_at
            resource_snapshot = await self._sample_managed_process_resources(process)

            async with process.lock:
                if process.task.done():
                    poll_yield_reason = "completion"
                elif output_wake:
                    poll_yield_reason = "output"
                elif waited:
                    poll_yield_reason = "deadline"
                else:
                    poll_yield_reason = "nonblocking"
                self._record_buffered_process_result(process)
                output_bytes_since_last_poll = process.output_state.total_output_bytes
                output_lines_since_last_poll = process.output_state.unread_output_line_count
                seconds_since_last_output = max(
                    time.monotonic() - process.callbacks.last_output_time,
                    0.0,
                )
                output_observed = process.output_state.had_stream_output
                result = self._managed_process_result(
                    process, output_preview_limit=output_preview_limit
                )
                metadata = cast("ProcessResultMetadata", process_result_metadata(result))
                metadata["process_yield_reason"] = poll_yield_reason
                metadata["output_bytes_since_last_poll"] = output_bytes_since_last_poll
                metadata["seconds_since_last_output"] = seconds_since_last_output
                metadata["has_observed_output"] = output_observed
                metadata["poll_wait_sec"] = parsed.wait_sec
                metadata["poll_wake_on_output"] = parsed.wake_on_output
                metadata["poll_elapsed_seconds"] = poll_elapsed_seconds
                if poll_yield_reason == "deadline":
                    metadata["poll_deadline_overshoot_seconds"] = max(
                        poll_elapsed_seconds - parsed.wait_sec,
                        0.0,
                    )
                if resource_snapshot is not None:
                    metadata["resource_snapshot"] = resource_snapshot.metadata()
                    observation = observe_resource_changes(
                        process.resource_observations,
                        resource_snapshot,
                    )
                    if observation is not None:
                        metadata["resource_observation"] = observation
                        self._append_resource_observation(result, observation)
                self._append_poll_output_activity(
                    result,
                    output_bytes=output_bytes_since_last_poll,
                    output_lines=output_lines_since_last_poll,
                    seconds_since_last_output=seconds_since_last_output,
                    output_observed=output_observed,
                )
                return result

    @staticmethod
    async def _wait_for_managed_process_poll(
        process: ManagedShellProcess,
        *,
        wait_sec: int,
        wake_on_output: bool,
    ) -> bool:
        """Wait for completion/deadline, optionally returning after output settles."""
        if not wake_on_output:
            await asyncio.wait(
                (process.task,),
                timeout=wait_sec,
                return_when=asyncio.FIRST_COMPLETED,
            )
            return False

        deadline = time.monotonic() + wait_sec
        while not process.task.done():
            now = time.monotonic()
            remaining = deadline - now
            if remaining <= 0:
                return False

            process.callbacks.activity_event.clear()
            pending_output = (
                process.output_state.total_output_bytes > 0
                or process.output_state.unread_output_activity
            )
            seconds_since_last_output = max(
                now - process.callbacks.last_output_time,
                0.0,
            )
            if pending_output and seconds_since_last_output >= _PROCESS_OUTPUT_DEBOUNCE_SECONDS:
                return True

            quiet_wait = (
                _PROCESS_OUTPUT_DEBOUNCE_SECONDS - seconds_since_last_output
                if pending_output
                else remaining
            )
            activity_task = asyncio.create_task(process.callbacks.activity_event.wait())
            try:
                await asyncio.wait(
                    (process.task, activity_task),
                    timeout=min(remaining, quiet_wait),
                    return_when=asyncio.FIRST_COMPLETED,
                )
            finally:
                if not activity_task.done():
                    activity_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await activity_task

        return False

    async def terminate_process(
        self,
        arguments: dict[str, Any] | None = None,
    ) -> CallToolResult:
        """Terminate one managed process through the environment cancellation contract."""
        try:
            process_id = parse_terminate_process_arguments(arguments)
        except ValueError as exc:
            return _text_result(str(exc), is_error=True)

        outcome = await self._terminate_process_by_id(
            process_id,
            include_unattached_durable=False,
        )
        return self._termination_tool_result(outcome)

    async def terminate_interactive_process(
        self,
        process_id: str,
    ) -> ProcessTerminationOutcome:
        """Terminate a managed or discoverable durable process by explicit user request."""

        return await self._terminate_process_by_id(
            process_id,
            include_unattached_durable=True,
        )

    async def _terminate_process_by_id(
        self,
        process_id: str,
        *,
        include_unattached_durable: bool,
    ) -> ProcessTerminationOutcome:
        process = await self._get_managed_process(process_id)
        if process is None:
            store = self._durable_process_store
            if store is not None and (
                include_unattached_durable or process_id in self._attached_durable_processes
            ):
                try:
                    snapshot = await asyncio.to_thread(store.get, process_id)
                    status = self._durable_status(snapshot)
                    if status == "unavailable":
                        return ProcessTerminationOutcome(
                            process_id=process_id,
                            state="unavailable",
                        )
                    if status != "running":
                        return ProcessTerminationOutcome(
                            process_id=process_id,
                            state="already_exited",
                        )
                    created = await asyncio.to_thread(store.request_stop, process_id)
                except ValueError:
                    return ProcessTerminationOutcome(
                        process_id=process_id,
                        state="not_found",
                    )
                except (DurableProcessRecordError, OSError) as exc:
                    return ProcessTerminationOutcome(
                        process_id=process_id,
                        state="termination_failed",
                        error=f"durable process {process_id!r} could not be stopped: {exc}",
                    )
                return ProcessTerminationOutcome(
                    process_id=process_id,
                    state="stop_requested" if created else "stop_already_requested",
                )
            return ProcessTerminationOutcome(
                process_id=process_id,
                state="not_found",
            )

        async with process.lock:
            if process.task.done():
                return ProcessTerminationOutcome(
                    process_id=process_id,
                    state="already_exited",
                )
            await self._terminate_managed_process_task(process)
            if not process.task.cancelled():
                exception = process.task.exception()
                if exception is not None:
                    process.terminated = False
                    return ProcessTerminationOutcome(
                        process_id=process_id,
                        state="termination_failed",
                        error=str(exception),
                    )
            return ProcessTerminationOutcome(
                process_id=process_id,
                state="terminated",
            )

    @staticmethod
    def _termination_tool_result(outcome: ProcessTerminationOutcome) -> CallToolResult:
        process_id = outcome.process_id
        if outcome.state == "not_found":
            return _text_result(
                f"Error: managed shell process {process_id!r} was not found",
                is_error=True,
            )
        if outcome.state == "termination_failed":
            error = outcome.error or "process termination failed"
            prefix = (
                "Error: "
                if error.startswith("durable process ")
                else f"process_id: {process_id}\noutcome: termination_failed\nerror: "
            )
            if prefix == "Error: ":
                return _text_result(f"{prefix}{error}", is_error=True)
            return process_result(
                f"{prefix}{error}",
                is_error=True,
                metadata={
                    "process_id": process_id,
                    "process_status": "termination_failed",
                },
            )
        if outcome.state == "unavailable":
            return process_result(
                f"process_id: {process_id}\noutcome: unavailable\n"
                "The supervisor is no longer available; no stop request was sent.",
                is_error=True,
                metadata={
                    "process_id": process_id,
                    "process_status": "unavailable",
                },
            )
        process_status = (
            "stopping"
            if outcome.state in {"stop_requested", "stop_already_requested"}
            else outcome.state
        )
        return process_result(
            f"process_id: {process_id}\noutcome: {outcome.state}",
            is_error=False,
            metadata={
                "process_id": process_id,
                "process_status": process_status,
            },
        )

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        tool_use_id: str | None = None,
        *,
        show_tool_call_id: bool = False,
        defer_display_to_tool_result: bool = False,
    ) -> CallToolResult:
        """Dispatch one model-facing shell lifecycle tool."""
        if (
            self._tool is not None
            and name.casefold() == self._tool.name.casefold()
            and self._minimal_process_profile
        ):
            try:
                parsed = parse_minimal_bash_arguments(
                    arguments,
                    tool_name=self._minimal_shell_tool_name,
                    require_description=self._minimal_shell_tool_requires_description,
                )
            except ValueError as exc:
                return self._invalid_execute_result(str(exc))
            return await self._execute_parsed(
                parsed,
                tool_use_id,
                show_tool_call_id=show_tool_call_id,
                defer_display_to_tool_result=defer_display_to_tool_result,
            )
        if name.casefold() == PROCESS_TOOL_NAME and (self._uses_unified_process_profile()):
            try:
                parsed_process = parse_minimal_process_arguments(
                    arguments,
                    min_wait_seconds=MIN_PROCESS_POLL_WAIT_SECONDS,
                    max_wait_seconds=self._max_process_poll_seconds,
                )
            except ValueError as exc:
                return _text_result(str(exc), is_error=True)
            if parsed_process.action == "list":
                return await self.list_processes()
            if isinstance(parsed_process, MinimalProcessReadOutputArguments):
                return await self.read_process_output(parsed_process)
            if parsed_process.action == "stop":
                return await self._call_process_lifecycle_tool(
                    TERMINATE_PROCESS_TOOL_NAME,
                    {"process_id": parsed_process.process_id},
                    tool_use_id=tool_use_id,
                )
            wait_sec = (
                0
                if parsed_process.action == "status"
                else (
                    parsed_process.wait_sec
                    if parsed_process.wait_sec is not None
                    else self._minimal_process_wait_seconds()
                )
            )
            return await self._call_process_lifecycle_tool(
                POLL_PROCESS_TOOL_NAME,
                {
                    "process_id": parsed_process.process_id,
                    "wait_sec": wait_sec,
                },
                tool_use_id=tool_use_id,
                output_preview_limit=parsed_process.limit,
            )
        if name == GROK_SHELL_TOOL_NAME and self._grok_shell_profile:
            try:
                parsed = parse_grok_shell_arguments(arguments)
            except ValueError as exc:
                return self._invalid_execute_result(str(exc))
            return await self._execute_parsed(
                parsed,
                tool_use_id,
                show_tool_call_id=show_tool_call_id,
                defer_display_to_tool_result=defer_display_to_tool_result,
            )
        if name == LUNA_EXEC_TOOL_NAME and self._luna_exec_profile:
            try:
                parsed = parse_luna_exec_arguments(arguments)
            except ValueError as exc:
                return self._invalid_execute_result(str(exc))
            return await self._execute_parsed(
                parsed,
                tool_use_id,
                show_tool_call_id=show_tool_call_id,
                defer_display_to_tool_result=defer_display_to_tool_result,
            )
        if name == EXECUTE_TOOL_NAME:
            return await self.execute(
                arguments,
                tool_use_id,
                show_tool_call_id=show_tool_call_id,
                defer_display_to_tool_result=defer_display_to_tool_result,
            )
        if name == POLL_PROCESS_TOOL_NAME:
            return await self._call_process_lifecycle_tool(
                name,
                arguments,
                tool_use_id=tool_use_id,
            )
        if name == TERMINATE_PROCESS_TOOL_NAME:
            return await self._call_process_lifecycle_tool(
                name,
                arguments,
                tool_use_id=tool_use_id,
            )
        return _text_result(f"Error: unknown shell runtime tool {name!r}", is_error=True)

    async def _call_process_lifecycle_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None,
        *,
        tool_use_id: str | None,
        output_preview_limit: int | None = None,
    ) -> CallToolResult:
        process_id = (arguments or {}).get("process_id")
        payload = arguments or {}
        process_metadata = self.process_tool_metadata(name, payload)
        start_details = self._progress.process_details(process_metadata)
        if name == POLL_PROCESS_TOOL_NAME:
            operation = self.poll_process(
                arguments,
                progress_tool_use_id=tool_use_id,
                output_preview_limit=output_preview_limit,
            )
        else:
            operation = self.terminate_process(arguments)

        elapsed = process_metadata.get("elapsed_seconds")
        process_elapsed_seconds = (
            float(elapsed)
            if isinstance(elapsed, (int, float)) and not isinstance(elapsed, bool)
            else None
        )
        command = process_metadata.get("command_summary")
        wait_sec = process_metadata.get("wait_sec")
        seconds_since_last_output = process_metadata.get("seconds_since_last_output")
        has_observed_output = process_metadata.get("has_observed_output")
        total_output_bytes = process_metadata.get("total_output_bytes")
        seconds_since_last_stdout = process_metadata.get("seconds_since_last_stdout")
        seconds_since_last_stderr = process_metadata.get("seconds_since_last_stderr")
        stdout_bytes = process_metadata.get("stdout_bytes")
        stderr_bytes = process_metadata.get("stderr_bytes")
        self._progress.emit(
            action=ProgressAction.CALLING_TOOL,
            tool_use_id=tool_use_id,
            tool_name=name,
            tool_event="start",
            details=start_details,
            process_elapsed_seconds=process_elapsed_seconds,
            process_command=command if isinstance(command, str) else None,
            process_id=str(process_metadata.get("process_id") or "process"),
            process_wait_seconds=(
                wait_sec if name == POLL_PROCESS_TOOL_NAME and type(wait_sec) is int else None
            ),
            process_has_observed_output=(
                has_observed_output if isinstance(has_observed_output, bool) else None
            ),
            process_seconds_since_last_output=(
                float(seconds_since_last_output)
                if isinstance(seconds_since_last_output, (int, float))
                and not isinstance(seconds_since_last_output, bool)
                else None
            ),
            process_total_output_bytes=(
                total_output_bytes
                if type(total_output_bytes) is int and total_output_bytes >= 0
                else None
            ),
            process_seconds_since_last_stdout=(
                float(seconds_since_last_stdout)
                if isinstance(seconds_since_last_stdout, (int, float))
                and not isinstance(seconds_since_last_stdout, bool)
                else None
            ),
            process_seconds_since_last_stderr=(
                float(seconds_since_last_stderr)
                if isinstance(seconds_since_last_stderr, (int, float))
                and not isinstance(seconds_since_last_stderr, bool)
                else None
            ),
            process_stdout_bytes=(
                stdout_bytes if type(stdout_bytes) is int and stdout_bytes >= 0 else None
            ),
            process_stderr_bytes=(
                stderr_bytes if type(stderr_bytes) is int and stderr_bytes >= 0 else None
            ),
        )
        result = await operation
        metadata = process_result_metadata(result)
        status = metadata.get("process_status") if metadata is not None else None
        yield_reason = metadata.get("process_yield_reason") if metadata is not None else None
        details = f"{process_id}: {status}" if process_id and status else status
        self._progress.emit(
            action=ProgressAction.TOOL_PROGRESS,
            tool_use_id=tool_use_id,
            tool_name=name,
            details=details or ("failed" if result.is_error else "completed"),
            tool_state="failed" if result.is_error else "completed",
            tool_terminal=True,
            process_yield_reason=yield_reason,
        )
        return result

    async def close(self) -> None:
        """Terminate session processes and detach persistent processes."""
        async with self._processes_lock:
            processes = list(self._managed_processes.values())
            self._managed_processes.clear()
        running = [process for process in processes if not process.task.done()]
        durable_running: list[DurableProcessSnapshot] = []
        if self._durable_process_store is not None:
            for process_id in sorted(self._attached_durable_processes):
                try:
                    snapshot = await asyncio.to_thread(
                        self._durable_process_store.get,
                        process_id,
                    )
                except DurableProcessRecordError:
                    continue
                if self._durable_status(snapshot) == "running":
                    durable_running.append(snapshot)
        if running:
            console.console.print(
                f"Warning: {len(running)} background process"
                f"{'es are' if len(running) != 1 else ' is'} still running "
                "at fast-agent shutdown:",
                style="yellow",
            )
            for process in running:
                os_pid = process.callbacks.os_process_id
                pid_details = f", os_pid={os_pid}" if os_pid is not None else ""
                console.console.print(
                    f"  {process.process_id}{pid_details}, "
                    f"lifecycle={process.lifecycle}"
                    + (
                        f", output_spool={process.request.output_spool_path}"
                        if process.request.output_spool_path is not None
                        else ""
                    ),
                    style="yellow",
                )
        if durable_running:
            console.console.print(
                f"{len(durable_running)} durable background process"
                f"{'es remain' if len(durable_running) != 1 else ' remains'} available "
                "for a later fast-agent invocation:",
                style="yellow",
            )
            for snapshot in durable_running:
                console.console.print(
                    f"  {snapshot.spec.process_id}, os_pid={snapshot.status.child_pid}, "
                    "lifecycle=persistent",
                    style="yellow",
                )
        for process in processes:
            if not process.task.done():
                process.terminated = process.lifecycle == "session"
                process.task.cancel()
        if processes:
            await asyncio.gather(
                *(process.task for process in processes),
                return_exceptions=True,
            )
        if self._retained_output_directory is not None:
            shutil.rmtree(self._retained_output_directory, ignore_errors=True)
            self._retained_output_directory = None

    async def execute_shell(
        self,
        command: str,
        *,
        cwd: str | Path | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> ShellExecutionResult:
        execution = await execute_shell(
            self._environment,
            command,
            cwd=cwd if cwd is not None else self._working_directory,
            env=env,
            timeout=self._timeout_seconds if timeout is None else timeout,
        )
        return execution

    async def execute_direct_shell(
        self,
        command: str,
        *,
        cwd: str | Path | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> ShellExecutionResult:
        """Execute a user-entered shell command and display output live when available."""
        runtime_execution = await self._execute_shell_command(
            command,
            cwd=cwd,
            env=env,
            timeout=timeout,
            output_byte_limit=None,
            defer_display_to_tool_result=False,
            display_line_limit=None,
            respect_tool_display=False,
        )
        execution = runtime_execution.execution
        output_state = runtime_execution.output_state
        display_state = runtime_execution.display_state

        if not output_state.had_stream_output:
            if execution.result.stdout:
                self._record_stream_output(
                    execution.result.stdout,
                    style=None,
                    output_state=output_state,
                    display_state=display_state,
                    is_stderr=False,
                )
            if execution.result.stderr:
                self._record_stream_output(
                    execution.result.stderr,
                    style="red",
                    output_state=output_state,
                    display_state=display_state,
                    is_stderr=True,
                )
        self._flush_live_display_tail(display_state)
        if execution.timed_out:
            self._print_timeout_notice(
                display_state,
                timeout_seconds=execution.options.timeout_seconds,
            )
        return execution.result

    async def execute(
        self,
        arguments: dict[str, Any] | None = None,
        tool_use_id: str | None = None,
        *,
        show_tool_call_id: bool = False,
        defer_display_to_tool_result: bool = False,
    ) -> CallToolResult:
        """Execute a command until completion or yield it as a managed process."""
        try:
            parsed = parse_execute_arguments(arguments)
        except ValueError as exc:
            return self._invalid_execute_result(str(exc))
        return await self._execute_parsed(
            parsed,
            tool_use_id,
            show_tool_call_id=show_tool_call_id,
            defer_display_to_tool_result=defer_display_to_tool_result,
        )

    async def _execute_parsed(
        self,
        parsed: ShellExecuteArguments,
        tool_use_id: str | None,
        *,
        show_tool_call_id: bool,
        defer_display_to_tool_result: bool,
    ) -> CallToolResult:
        idle_yield_seconds = (
            self._idle_yield_seconds
            if parsed.yield_after_idle_sec is None
            else parsed.yield_after_idle_sec
        )
        self._logger.debug(
            "Executing command with "
            f"idle_yield={idle_yield_seconds}s, "
            f"foreground_yield={self._foreground_yield_seconds}s, "
            f"foreground_auto_await_max={self._foreground_auto_await_max_seconds}s total, "
            f"background={parsed.background}, "
            f"lifecycle={parsed.lifecycle}"
        )
        if (
            parsed.background
            and parsed.lifecycle == "persistent"
            and self._durable_process_store is not None
        ):
            self._progress.emit(
                action=ProgressAction.CALLING_TOOL,
                tool_use_id=tool_use_id,
                tool_event="start",
            )
            try:
                snapshot = await self._start_durable_process(parsed)
            except Exception as exc:
                self._logger.error(f"Durable process launch failed: {exc}")
                self._progress.emit(
                    action=ProgressAction.TOOL_PROGRESS,
                    tool_use_id=tool_use_id,
                    details=f"failed: {exc}",
                    tool_state="failed",
                    tool_terminal=True,
                )
                return _text_result(f"Command execution failed: {exc}", is_error=True)
            result = self._durable_launch_result(snapshot)
            status = self._durable_status(snapshot)
            progress_details = (
                f"running ({snapshot.spec.process_id})"
                if status == "running"
                else (
                    f"{status} (exit {snapshot.status.exit_code})"
                    if snapshot.status.exit_code is not None
                    else status
                )
            )
            self._progress.emit(
                action=ProgressAction.TOOL_PROGRESS,
                tool_use_id=tool_use_id,
                details=progress_details,
                tool_state=("failed" if status in {"failed", "unavailable"} else "completed"),
                tool_terminal=True,
                process_yield_reason="background",
            )
            return result

        progress_context = progress_display.paused() if display_tools_enabled() else nullcontext()
        process: ManagedShellProcess | None = None
        with progress_context:
            try:
                self._progress.emit(
                    action=ProgressAction.CALLING_TOOL,
                    tool_use_id=tool_use_id,
                    tool_event="start",
                )

                process = await self._start_managed_process(
                    parsed,
                    defer_display_to_tool_result=defer_display_to_tool_result,
                )
                initial_yield_reason: Literal["idle", "foreground"] | None = None
                yielded_reason: str | None
                if parsed.background:
                    started_task = asyncio.create_task(process.callbacks.started_event.wait())
                    try:
                        await asyncio.wait(
                            (process.task, started_task),
                            timeout=1,
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                    finally:
                        if not started_task.done():
                            started_task.cancel()
                            with suppress(asyncio.CancelledError):
                                await started_task
                    yielded_reason = "background"
                elif parsed.hard_timeout_seconds is not None:
                    completed, _ = await asyncio.wait(
                        (process.task,),
                        timeout=parsed.hard_timeout_seconds,
                        return_when=asyncio.ALL_COMPLETED,
                    )
                    if process.task not in completed:
                        await self._terminate_managed_process_task(process)
                        result = self._managed_process_result(process)
                        for block in result.content:
                            if isinstance(block, TextContent):
                                block.text = (
                                    f"{block.text}\noutcome: timed_out\n"
                                    f"timeout_seconds: {parsed.hard_timeout_seconds}"
                                )
                                break
                        metadata = process_result_metadata(result)
                        if metadata is not None:
                            metadata["process_status"] = "timed_out"
                        result.is_error = True
                        self._progress.emit(
                            action=ProgressAction.TOOL_PROGRESS,
                            tool_use_id=tool_use_id,
                            details=(f"timed out after {parsed.hard_timeout_seconds}s"),
                            tool_state="failed",
                            tool_terminal=True,
                        )
                        return result
                    yielded_reason = None
                else:
                    initial_yield_reason = await self._wait_for_initial_process_result(
                        process,
                        idle_yield_seconds=idle_yield_seconds,
                    )
                    yielded_reason = initial_yield_reason

                foreground_auto_await: ForegroundAutoAwaitMetadata | None = None
                if (
                    not parsed.background
                    and parsed.hard_timeout_seconds is None
                    and initial_yield_reason is not None
                ):
                    if self._foreground_auto_await_max_seconds > 0:
                        remaining_auto_await_seconds = max(
                            process.started_at
                            + self._foreground_auto_await_max_seconds
                            - time.monotonic(),
                            0.0,
                        )
                        self._progress.emit(
                            action=ProgressAction.CALLING_TOOL,
                            tool_use_id=tool_use_id,
                            tool_event="progress",
                            details=f"auto-awaiting {process.process_id}",
                            process_elapsed_seconds=max(
                                time.monotonic() - process.started_at,
                                0.0,
                            ),
                            process_command=summarize_command(process.command),
                            process_id=process.process_id,
                            process_wait_seconds=math.ceil(remaining_auto_await_seconds),
                            process_yield_reason=initial_yield_reason,
                            log_message="Foreground process auto-await",
                        )
                    foreground_auto_await = await self._auto_await_foreground_process(
                        process,
                        initial_yield_reason=initial_yield_reason,
                    )
                    if foreground_auto_await["outcome"] == "process_finished":
                        yielded_reason = None
                    elif foreground_auto_await["outcome"] == "disabled":
                        yielded_reason = initial_yield_reason
                    else:
                        yielded_reason = "auto_await_cap"

                result = self._managed_process_result(
                    process,
                    yielded_reason=yielded_reason,
                )
                if process.task.done() and not process.task.cancelled():
                    exception = process.task.exception()
                    if exception is None:
                        result = self._finalize_shell_result_display(
                            result,
                            shell_result=process.task.result().result,
                            output_state=process.output_state,
                            display_state=process.display_state,
                            tool_use_id=tool_use_id,
                            show_tool_call_id=show_tool_call_id,
                            defer_display_to_tool_result=defer_display_to_tool_result,
                        )
                else:
                    self._flush_live_display_tail(process.display_state)
                    process.display_state.use_live_shell_display = False
                    if defer_display_to_tool_result:
                        update_tool_result_display_metadata(result, {"suppress_display": False})
                    else:
                        self._display.show_managed_process_status(
                            process_id=process.process_id,
                            status="running",
                            reason=yielded_reason,
                            elapsed_seconds=time.monotonic() - process.started_at,
                            os_process_id=process.callbacks.os_process_id,
                        )

                metadata = cast("ProcessResultMetadata", process_result_metadata(result))
                process_status = metadata["process_status"]
                if process_status == "running":
                    completion_details = f"running ({process.process_id})"
                elif (
                    process.task.done()
                    and not process.task.cancelled()
                    and process.task.exception() is not None
                ):
                    completion_details = f"failed: {process.task.exception()}"
                elif (
                    process.task.done()
                    and not process.task.cancelled()
                    and process.task.exception() is None
                ):
                    completion_details = (
                        f"{process_status} (exit {process.task.result().result.exit_code})"
                    )
                else:
                    completion_details = process_status
                self._progress.emit(
                    action=ProgressAction.TOOL_PROGRESS,
                    tool_use_id=tool_use_id,
                    details=completion_details,
                    tool_state="failed" if result.is_error else "completed",
                    tool_terminal=True,
                    process_yield_reason=metadata.get("process_yield_reason"),
                )
                return result

            except asyncio.CancelledError:
                if process is not None and process.lifecycle == "session":
                    await self._terminate_managed_process_task(process)
                self._progress.emit(
                    action=ProgressAction.TOOL_PROGRESS,
                    tool_use_id=tool_use_id,
                    details="cancelled",
                    tool_state="failed",
                    tool_terminal=True,
                )
                raise
            except Exception as exc:
                self._logger.error(f"Execute tool failed: {exc}")
                self._progress.emit(
                    action=ProgressAction.TOOL_PROGRESS,
                    tool_use_id=tool_use_id,
                    details=f"failed: {exc}",
                    tool_state="failed",
                    tool_terminal=True,
                )
                return _text_result(f"Command execution failed: {exc}", is_error=True)
