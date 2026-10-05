"""Immutable, owner-scoped MCP lifecycle diagnostics shared by runtime and UI."""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from time import monotonic
from typing import Literal

StartupState = Literal["pending", "ready", "auth", "error"]
StartupPhase = Literal["connect", "authentication", "discovery", "cancelled", "ready"]


@dataclass(frozen=True)
class ServerStartupStatus:
    server_name: str
    state: StartupState
    failure_detail: str | None = None
    owner: str = ""
    phase: StartupPhase = "connect"
    timestamp: datetime | None = None
    duration_seconds: float = 0.0
    transport: str | None = None
    resolved_at: datetime | None = None


class MCPStartup:
    def __init__(self) -> None:
        self._states: dict[tuple[str, str], ServerStartupStatus] = {}
        self._inactive_owners: set[str] = set()
        self._history: list[ServerStartupStatus] = []
        self._started: dict[tuple[str, str], float] = {}
        self._tasks: set[asyncio.Task[None]] = set()
        self._listeners: set[Callable[[], None]] = set()

    def subscribe(self, listener: Callable[[], None]) -> Callable[[], None]:
        """Notify on status changes (e.g. to redraw a toolbar); returns an unsubscribe."""
        self._listeners.add(listener)
        return lambda: self._listeners.discard(listener)

    def track(self, task: asyncio.Task[None]) -> None:
        """Register background startup work that the first prompt waits for."""
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    @property
    def pending(self) -> bool:
        return bool(self._tasks)

    async def wait(self) -> None:
        """Wait for tracked startup work; cancelling the waiter leaves startup running."""
        await asyncio.shield(asyncio.gather(*self._tasks, return_exceptions=True))

    def set_status(
        self,
        owner: str,
        server_name: str,
        state: StartupState,
        failure_detail: str | None = None,
        *,
        phase: StartupPhase | None = None,
        transport: str | None = None,
    ) -> None:
        key = owner, server_name
        now = datetime.now(UTC)
        previous = self._states.get(key)
        if previous is None or (state == "pending" and previous.state in {"ready", "error"}):
            self._started[key] = monotonic()
        if previous is not None and previous.state in {"auth", "error"}:
            if state != previous.state or failure_detail != previous.failure_detail:
                self._history.append(
                    replace(previous, resolved_at=now if state == "ready" else None)
                )
        if state == "ready":
            self._history = [
                replace(item, resolved_at=now)
                if item.owner == owner
                and item.server_name == server_name
                and item.resolved_at is None
                else item
                for item in self._history
            ]
        self._states[key] = ServerStartupStatus(
            server_name=server_name,
            state=state,
            failure_detail=failure_detail,
            owner=owner,
            phase=phase
            or (
                "authentication"
                if state == "auth"
                else "ready"
                if state == "ready"
                else previous.phase
                if state == "error" and previous
                else "connect"
            ),
            timestamp=now,
            duration_seconds=monotonic() - self._started[key],
            transport=transport or (previous.transport if previous else None),
        )
        for listener in tuple(self._listeners):
            listener()

    def snapshot(self, owner: str | None = None) -> tuple[ServerStartupStatus, ...]:
        return tuple(
            value
            for value in self._states.values()
            if (value.owner not in self._inactive_owners if owner is None else value.owner == owner)
        )

    def history(self, owner: str | None = None) -> tuple[ServerStartupStatus, ...]:
        return tuple(value for value in self._history if owner is None or value.owner == owner)

    def get_startup_errors(
        self,
        server_name: str | None = None,
        *,
        owner: str | None = None,
    ) -> tuple[ServerStartupStatus, ...]:
        return tuple(
            value
            for value in self.snapshot(owner)
            if value.state in {"auth", "error"}
            and (server_name is None or value.server_name == server_name)
        )

    def clear(self, owner: str) -> None:
        self._history.extend(self.get_startup_errors(owner=owner))
        self._states = {key: value for key, value in self._states.items() if key[0] != owner}
        self._started = {key: value for key, value in self._started.items() if key[0] != owner}

    def deactivate(self, owner: str) -> None:
        """Exclude closed owners from context status, retaining owner diagnostics."""
        self._inactive_owners.add(owner)

    def clear_server(self, owner: str, server_name: str) -> None:
        status = self._states.pop((owner, server_name), None)
        if status is not None and status.state in {"auth", "error"}:
            self._history.append(replace(status, resolved_at=datetime.now(UTC)))
        self._started.pop((owner, server_name), None)
