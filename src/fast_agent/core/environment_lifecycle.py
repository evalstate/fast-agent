"""Progress helpers for opening execution environments."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from fast_agent.core.logging.progress_payloads import build_progress_payload
from fast_agent.event_progress import ProgressAction

if TYPE_CHECKING:
    from fast_agent.tools.execution_environment import ShellEnvironment, ShellRuntimeInfo


class _ProgressLogger(Protocol):
    def info(self, message: str, *, data: dict[str, object]) -> None: ...


async def open_environment_with_progress(
    environment: "ShellEnvironment",
    *,
    logger: _ProgressLogger,
) -> None:
    """Open an execution environment while emitting lifecycle progress events."""

    info = environment.runtime_info()
    action = (
        ProgressAction.CONNECTING
        if info.kind in {"docker", "remote"} or info.provider in {"docker", "huggingface"}
        else ProgressAction.STARTING
    )
    target = _environment_target(info, environment)
    _emit_environment_progress(
        logger,
        action=action,
        target=target,
        details=_environment_details(info, environment),
    )

    def emit_stage(stage: str) -> None:
        _emit_environment_progress(
            logger,
            action=ProgressAction.TOOL_PROGRESS,
            target=target,
            details=stage,
        )

    _set_startup_progress_callback(environment, emit_stage)
    try:
        await environment.open()
    except Exception as exc:
        _emit_environment_progress(
            logger,
            action=ProgressAction.FATAL_ERROR,
            target=target,
            details=f"{_environment_details(info, environment)} - {exc}",
        )
        raise
    finally:
        _set_startup_progress_callback(environment, None)
    _emit_environment_progress(
        logger,
        action=ProgressAction.READY,
        target=target,
        details=_environment_details(environment.runtime_info(), environment),
    )


def _set_startup_progress_callback(
    environment: "ShellEnvironment",
    callback,
) -> None:
    from fast_agent.tools.execution_environment import EnvironmentStartupProgress

    if isinstance(environment, EnvironmentStartupProgress):
        environment.set_startup_progress_callback(callback)


def _environment_target(info: "ShellRuntimeInfo", environment: "ShellEnvironment") -> str:
    from fast_agent.tools.environment_registry import environment_name

    if info.environment_name:
        return info.environment_name
    registered_name = environment_name(environment)
    if registered_name:
        return registered_name
    if info.provider:
        return info.provider
    return info.kind or "environment"


def _environment_details(
    info: "ShellRuntimeInfo",
    environment: "ShellEnvironment",
) -> str:
    from fast_agent.tools.environment_registry import environment_name

    parts: list[str] = []
    resolved_name = info.environment_name or environment_name(environment)
    if resolved_name:
        parts.append(resolved_name)
    parts.append(info.kind)
    if info.provider and info.provider != info.kind:
        parts.append(info.provider)
    return f"{' '.join(parts)} | cwd: {environment.cwd}"


def _emit_environment_progress(
    logger: _ProgressLogger,
    *,
    action: ProgressAction,
    target: str,
    details: str,
) -> None:
    payload = build_progress_payload(
        action=action,
        details=details,
        extra={"target": target},
    )
    try:
        logger.info("Environment lifecycle", data=payload)
    except TypeError:
        return


__all__ = ["open_environment_with_progress"]
