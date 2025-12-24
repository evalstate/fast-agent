"""
ACP capability Protocols for type-safe isinstance checks.

These Protocols define optional capabilities that agents may implement.
Use isinstance() checks instead of hasattr() to verify capability support.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from fast_agent.acp.filesystem_runtime import ACPFilesystemRuntime
    from fast_agent.acp.terminal_runtime import ACPTerminalRuntime
    from fast_agent.tools.shell_runtime import ShellRuntime
    from fast_agent.workflow_telemetry import PlanTelemetryProvider, WorkflowTelemetryProvider


@runtime_checkable
class ShellRuntimeCapable(Protocol):
    """Agent that supports external shell runtime injection."""

    _shell_runtime: "ShellRuntime"

    @property
    def _shell_runtime_enabled(self) -> bool: ...

    def set_external_runtime(self, runtime: "ACPTerminalRuntime") -> None: ...


@runtime_checkable
class FilesystemRuntimeCapable(Protocol):
    """Agent that supports external filesystem runtime injection."""

    def set_filesystem_runtime(self, runtime: "ACPFilesystemRuntime") -> None: ...


@runtime_checkable
class InstructionContextCapable(Protocol):
    """Agent that supports dynamic instruction context updates."""

    def set_instruction_context(self, context: dict[str, str]) -> None: ...


@runtime_checkable
class WorkflowTelemetryCapable(Protocol):
    """Agent that supports workflow telemetry."""

    workflow_telemetry: "WorkflowTelemetryProvider | None"


@runtime_checkable
class PlanTelemetryCapable(Protocol):
    """Agent that supports plan telemetry."""

    plan_telemetry: "PlanTelemetryProvider | None"
