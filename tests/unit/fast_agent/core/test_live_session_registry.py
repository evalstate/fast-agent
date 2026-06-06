from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import pytest

from fast_agent.core.agent_instance_factory import CallableAgentInstanceFactory
from fast_agent.core.live_session_registry import InMemoryLiveSessionRegistry

if TYPE_CHECKING:
    from fast_agent.core.fastagent import AgentInstance


@dataclass(slots=True)
class FakeRecord:
    session_id: str
    instance: AgentInstance
    closed: bool = False


class FakeFactory:
    def __init__(self) -> None:
        self.instances: list[AgentInstance] = []
        self.disposed: list[AgentInstance] = []
        self.fail_dispose = False

    async def create(self) -> AgentInstance:
        instance = cast("AgentInstance", object())
        self.instances.append(instance)
        return instance

    async def dispose(self, instance: AgentInstance) -> None:
        self.disposed.append(instance)
        if self.fail_dispose:
            raise RuntimeError("dispose failed")


def _registry(
    factory: FakeFactory,
) -> InMemoryLiveSessionRegistry[FakeRecord, None]:
    async def create_record(
        session_id: str,
        instance: AgentInstance,
        context: None,
    ) -> FakeRecord:
        del context
        return FakeRecord(session_id=session_id, instance=instance)

    return InMemoryLiveSessionRegistry(
        instance_factory=CallableAgentInstanceFactory(
            create=factory.create,
            dispose=factory.dispose,
        ),
        create_record=create_record,
        record_instance=lambda record: record.instance,
        close_record=lambda record: setattr(record, "closed", True),
    )


@pytest.mark.asyncio
async def test_live_session_registry_create_get_delete() -> None:
    factory = FakeFactory()
    registry = _registry(factory)

    record = await registry.create("demo", None)

    assert await registry.get("demo") is record
    with pytest.raises(ValueError, match="already exists"):
        await registry.create("demo", None)

    deleted = await registry.delete("demo")

    assert deleted is record
    assert record.closed is True
    assert factory.disposed == [record.instance]
    with pytest.raises(KeyError, match="Session 'demo' not found"):
        await registry.get("demo")


@pytest.mark.asyncio
async def test_live_session_registry_disposes_instance_when_record_creation_fails() -> None:
    factory = FakeFactory()

    async def create_record(
        session_id: str,
        instance: AgentInstance,
        context: None,
    ) -> FakeRecord:
        del session_id, instance, context
        raise RuntimeError("record failed")

    registry: InMemoryLiveSessionRegistry[FakeRecord, None] = InMemoryLiveSessionRegistry(
        instance_factory=CallableAgentInstanceFactory(
            create=factory.create,
            dispose=factory.dispose,
        ),
        create_record=create_record,
        record_instance=lambda record: record.instance,
        close_record=lambda record: setattr(record, "closed", True),
    )

    with pytest.raises(RuntimeError, match="record failed"):
        await registry.create("demo", None)

    assert factory.disposed == factory.instances
    with pytest.raises(KeyError):
        await registry.get("demo")


@pytest.mark.asyncio
async def test_live_session_registry_delete_hook_can_reject_delete() -> None:
    factory = FakeFactory()
    registry = _registry(factory)
    record = await registry.create("demo", None)

    def reject_delete(_record: FakeRecord) -> None:
        raise RuntimeError("busy")

    with pytest.raises(RuntimeError, match="busy"):
        await registry.delete("demo", before_delete=reject_delete)

    assert await registry.get("demo") is record
    assert record.closed is False
    assert factory.disposed == []


@pytest.mark.asyncio
async def test_live_session_registry_close_all_suppresses_disposal_errors() -> None:
    factory = FakeFactory()
    registry = _registry(factory)
    first = await registry.create("one", None)
    second = await registry.create("two", None)
    factory.fail_dispose = True

    await registry.close_all()

    assert first.closed is True
    assert second.closed is True
    assert factory.disposed == [first.instance, second.instance]
    with pytest.raises(KeyError):
        await registry.get("one")
