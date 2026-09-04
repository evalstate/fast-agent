"""Restore durable process observation when a persisted session resumes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from fast_agent.tools.durable_processes import (
    DurableProcessRecordError,
    DurableProcessSnapshot,
)

_ACTIVE_STATES = frozenset({"created", "starting", "running", "stopping"})

if TYPE_CHECKING:
    from collections.abc import Mapping


@runtime_checkable
class _ShellRuntimeProvider(Protocol):
    @property
    def shell_runtime(self) -> object | None: ...


@runtime_checkable
class _DurableProcessRuntime(Protocol):
    async def discover_durable_processes(self) -> tuple[DurableProcessSnapshot, ...]: ...

    async def attach_durable_process(
        self,
        process_id: str,
        *,
        session_id: str | None = None,
    ) -> DurableProcessSnapshot: ...


@dataclass(frozen=True, slots=True)
class DurableProcessResumeResult:
    attached: tuple[DurableProcessSnapshot, ...] = ()
    unavailable: tuple[DurableProcessSnapshot, ...] = ()
    unattached: tuple[DurableProcessSnapshot, ...] = ()


async def resume_durable_processes(
    agents: Mapping[str, object],
    *,
    session_id: str,
    fallback_agent_name: str | None,
) -> DurableProcessResumeResult:
    """Attach active durable processes already associated with a resumed session."""

    runtimes: dict[str, _DurableProcessRuntime] = {}
    for agent_name, agent in agents.items():
        if not isinstance(agent, _ShellRuntimeProvider):
            continue
        runtime = agent.shell_runtime
        if isinstance(runtime, _DurableProcessRuntime):
            runtimes[agent_name] = runtime
    if not runtimes:
        return DurableProcessResumeResult()

    discovered: dict[str, DurableProcessSnapshot] = {}
    for runtime in runtimes.values():
        for snapshot in await runtime.discover_durable_processes():
            discovered.setdefault(snapshot.spec.process_id, snapshot)

    attached: list[DurableProcessSnapshot] = []
    unavailable: dict[str, DurableProcessSnapshot] = {}
    unattached: dict[str, DurableProcessSnapshot] = {}
    for snapshot in discovered.values():
        if session_id not in snapshot.session_ids:
            continue
        if snapshot.status.state == "unavailable":
            unavailable[snapshot.spec.process_id] = snapshot
            continue
        if snapshot.status.state not in _ACTIVE_STATES:
            continue

        candidates: list[_DurableProcessRuntime] = []
        for agent_name in (snapshot.spec.agent_name, fallback_agent_name):
            if agent_name is None:
                continue
            runtime = runtimes.get(agent_name)
            if runtime is not None and all(runtime is not candidate for candidate in candidates):
                candidates.append(runtime)
        if not candidates:
            unattached[snapshot.spec.process_id] = snapshot
            continue

        resumed = None
        for runtime in candidates:
            try:
                resumed = await runtime.attach_durable_process(snapshot.spec.process_id)
                break
            except (DurableProcessRecordError, OSError, ValueError):
                continue
        if resumed is None:
            unattached[snapshot.spec.process_id] = snapshot
            continue
        if resumed.status.state in _ACTIVE_STATES:
            attached.append(resumed)
        elif resumed.status.state == "unavailable":
            unavailable[resumed.spec.process_id] = resumed

    return DurableProcessResumeResult(
        attached=tuple(attached),
        unavailable=tuple(unavailable.values()),
        unattached=tuple(unattached.values()),
    )
