"""Generic live session ownership registry."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from typing import TYPE_CHECKING, Generic, Protocol, TypeVar

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from fast_agent.core.agent_instance_factory import AgentInstanceFactory
    from fast_agent.core.fastagent import AgentInstance


class LiveSessionRecord(Protocol):
    """Record for a live session backed by one owned ``AgentInstance``."""

    session_id: str
    instance: AgentInstance
    closed: bool


RecordT = TypeVar("RecordT")
ContextT = TypeVar("ContextT")


class InMemoryLiveSessionRegistry(Generic[RecordT, ContextT]):
    """Own live session records and their ``AgentInstance`` lifecycle."""

    def __init__(
        self,
        *,
        instance_factory: AgentInstanceFactory,
        create_record: Callable[[str, AgentInstance, ContextT], Awaitable[RecordT]],
        record_instance: Callable[[RecordT], AgentInstance],
        close_record: Callable[[RecordT], None],
    ) -> None:
        self._instance_factory = instance_factory
        self._create_record = create_record
        self._record_instance = record_instance
        self._close_record = close_record
        self._lock = asyncio.Lock()
        self._records: dict[str, RecordT] = {}

    @property
    def lock(self) -> asyncio.Lock:
        return self._lock

    async def get(self, session_id: str) -> RecordT:
        async with self._lock:
            try:
                return self._records[session_id]
            except KeyError as exc:
                raise KeyError(f"Session '{session_id}' not found") from exc

    async def create(self, session_id: str, context: ContextT) -> RecordT:
        async with self._lock:
            if session_id in self._records:
                raise ValueError(f"Session '{session_id}' already exists")

            instance = await self._instance_factory.create_instance()
            try:
                record = await self._create_record(session_id, instance, context)
            except Exception:
                await self._instance_factory.dispose_instance(instance)
                raise

            self._records[session_id] = record
            return record

    async def get_or_create(self, session_id: str, context: ContextT) -> RecordT:
        async with self._lock:
            existing = self._records.get(session_id)
        if existing is not None:
            return existing
        try:
            return await self.create(session_id, context)
        except ValueError as exc:
            try:
                return await self.get(session_id)
            except KeyError:
                raise exc

    async def delete(
        self,
        session_id: str,
        *,
        before_delete: Callable[[RecordT], None] | None = None,
    ) -> RecordT | None:
        async with self._lock:
            record = self._records.get(session_id)
            if record is None:
                return None
            if before_delete is not None:
                before_delete(record)
            del self._records[session_id]
            self._close_record(record)
            instance = self._record_instance(record)

        await self._instance_factory.dispose_instance(instance)
        return record

    async def close_all(self) -> None:
        async with self._lock:
            records = list(self._records.values())
            self._records.clear()
            for record in records:
                self._close_record(record)

        for record in records:
            with suppress(Exception):
                await self._instance_factory.dispose_instance(self._record_instance(record))


__all__ = [
    "InMemoryLiveSessionRegistry",
    "LiveSessionRecord",
]
