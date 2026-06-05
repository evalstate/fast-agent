from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest

from fast_agent import AgentHarness, FastAgent, HarnessSession, HarnessSessions
from fast_agent.core.agent_app import AgentApp
from fast_agent.core.fastagent import AgentInstance
from fast_agent.types import PromptMessageExtended

if TYPE_CHECKING:
    from pathlib import Path

    from fast_agent.interfaces import AgentProtocol


def test_public_harness_exports_and_factory() -> None:
    fast = FastAgent("test", parse_cli_args=False)

    harness = fast.harness()

    assert isinstance(harness, AgentHarness)
    assert HarnessSession.__name__ == "HarnessSession"
    assert HarnessSessions.__name__ == "HarnessSessions"


@pytest.mark.asyncio
async def test_harness_loads_environment_agent_cards(tmp_path: "Path") -> None:
    cards_dir = tmp_path / "agent-cards"
    cards_dir.mkdir()
    (cards_dir / "support.md").write_text("---\ntype: agent\n---\n", encoding="utf-8")
    fast = FastAgent("test", parse_cli_args=False, environment_dir=tmp_path)

    await fast.app.initialize()
    try:
        fast.harness()._load_environment_agent_cards()
    finally:
        await fast.app.cleanup()

    assert "support" in fast.agents


@pytest.mark.asyncio
async def test_harness_session_creates_persisted_session_folder(tmp_path: "Path") -> None:
    fast = FastAgent("test", parse_cli_args=False, environment_dir=tmp_path)
    agents = {"support": cast("AgentProtocol", FakeAgent("support", default=True))}
    instance = AgentInstance(AgentApp(agents), agents)

    await fast.app.initialize()
    try:
        persisted = await fast.harness()._create_persisted_session(
            "customer-123",
            instance,
            "support",
        )
    finally:
        await fast.app.cleanup()

    assert persisted is not None
    assert (tmp_path / "sessions" / "customer-123" / "session.json").exists()


class FakeAgent:
    def __init__(
        self,
        name: str,
        *,
        default: bool = False,
        block_generate: asyncio.Event | None = None,
        release_generate: asyncio.Event | None = None,
    ) -> None:
        self.name = name
        self.config = SimpleNamespace(default=default)
        self.received: list[str] = []
        self.clears: list[bool] = []
        self.shutdown_count = 0
        self.block_generate = block_generate
        self.release_generate = release_generate

    async def send(self, message: Any, request_params: Any = None) -> str:
        self.received.append(str(message))
        return f"{self.name}:{message}"

    async def generate(self, messages: Any, request_params: Any = None) -> PromptMessageExtended:
        self.received.append(str(messages))
        if self.block_generate is not None:
            self.block_generate.set()
        if self.release_generate is not None:
            await self.release_generate.wait()
        return PromptMessageExtended(role="assistant", content=[])

    async def structured(
        self,
        messages: Any,
        model: type[Any],
        request_params: Any = None,
    ) -> tuple[Any | None, PromptMessageExtended]:
        return None, await self.generate(messages, request_params)

    async def structured_schema(
        self,
        messages: Any,
        schema: dict[str, Any],
        request_params: Any = None,
    ) -> tuple[Any | None, PromptMessageExtended]:
        return {"agent": self.name}, await self.generate(messages, request_params)

    def clear(self, *, clear_prompts: bool = False) -> None:
        self.clears.append(clear_prompts)

    async def shutdown(self) -> None:
        self.shutdown_count += 1


class InstanceFactory:
    def __init__(self) -> None:
        self.instances: list[AgentInstance] = []
        self.disposed: list[AgentInstance] = []
        self.block_generate: asyncio.Event | None = None
        self.release_generate: asyncio.Event | None = None

    async def create(self) -> AgentInstance:
        index = len(self.instances)
        main = FakeAgent(
            f"main-{index}",
            default=True,
            block_generate=self.block_generate,
            release_generate=self.release_generate,
        )
        support = FakeAgent(f"support-{index}")
        agents = {
            "main": cast("AgentProtocol", main),
            "support": cast("AgentProtocol", support),
        }
        instance = AgentInstance(AgentApp(agents), agents)
        self.instances.append(instance)
        return instance

    async def dispose(self, instance: AgentInstance) -> None:
        self.disposed.append(instance)
        await instance.shutdown()


@pytest.fixture
def sessions_factory() -> tuple[HarnessSessions, InstanceFactory]:
    factory = InstanceFactory()
    sessions = HarnessSessions(create_instance=factory.create, dispose_instance=factory.dispose)
    return sessions, factory


@pytest.mark.asyncio
async def test_session_id_normalization(sessions_factory: tuple[HarnessSessions, InstanceFactory]):
    sessions, _factory = sessions_factory

    default_session = await sessions.create()
    stripped_session = await sessions.create("  customer-1  ")

    assert default_session.id == "default"
    assert stripped_session.id == "customer-1"
    with pytest.raises(ValueError, match="must not be empty"):
        await sessions.create("  ")
    with pytest.raises(ValueError, match="letters, digits, dashes, or underscores"):
        await sessions.create("../agent-cards")
    with pytest.raises(ValueError, match="letters, digits, dashes, or underscores"):
        await sessions.create("customer 1")
    with pytest.raises(ValueError, match="1-128 characters"):
        await sessions.create("a" * 129)


@pytest.mark.asyncio
async def test_persisted_session_failure_disposes_created_instance() -> None:
    factory = InstanceFactory()

    async def fail_persistence(
        session_id: str,
        instance: AgentInstance,
        agent_name: str | None,
    ) -> tuple[Any, Any] | None:
        del session_id, instance, agent_name
        raise RuntimeError("persistence failed")

    sessions = HarnessSessions(
        create_instance=factory.create,
        dispose_instance=factory.dispose,
        create_persisted_session=fail_persistence,
    )

    with pytest.raises(RuntimeError, match="persistence failed"):
        await sessions.create("demo")

    assert factory.disposed == factory.instances
    with pytest.raises(KeyError):
        await sessions.get("demo")


@pytest.mark.asyncio
async def test_get_create_get_or_create_and_delete_behavior(
    sessions_factory: tuple[HarnessSessions, InstanceFactory],
):
    sessions, factory = sessions_factory

    with pytest.raises(KeyError):
        await sessions.get("demo")

    created = await sessions.create("demo")
    assert await sessions.get("demo") is created
    assert await sessions.get_or_create("demo") is created

    with pytest.raises(ValueError, match="already exists"):
        await sessions.create("demo")

    await sessions.delete("demo")
    await sessions.delete("demo")

    assert factory.disposed == [created._record.instance]
    with pytest.raises(RuntimeError, match="closed"):
        await created.send("hello")


@pytest.mark.asyncio
async def test_same_session_reuses_object_and_instance(
    sessions_factory: tuple[HarnessSessions, InstanceFactory],
):
    sessions, _factory = sessions_factory

    first = await sessions.get_or_create("same")
    second = await sessions.get_or_create("same")

    assert second is first
    assert second._record.instance is first._record.instance


@pytest.mark.asyncio
async def test_different_sessions_get_different_instances(
    sessions_factory: tuple[HarnessSessions, InstanceFactory],
):
    sessions, _factory = sessions_factory

    first = await sessions.create("one")
    second = await sessions.create("two")

    assert first._record.instance is not second._record.instance


@pytest.mark.asyncio
async def test_close_all_disposes_remaining_instances(
    sessions_factory: tuple[HarnessSessions, InstanceFactory],
):
    sessions, factory = sessions_factory

    first = await sessions.create("one")
    second = await sessions.create("two")

    await sessions._close_all()

    assert factory.disposed == [first._record.instance, second._record.instance]


@pytest.mark.asyncio
async def test_per_call_agent_override_beats_session_default(
    sessions_factory: tuple[HarnessSessions, InstanceFactory],
):
    sessions, _factory = sessions_factory
    session = await sessions.create("demo", agent_name="support")

    response = await session.send("hello", agent_name="main")

    assert response == "main-0:hello"


@pytest.mark.asyncio
async def test_session_default_beats_app_default(
    sessions_factory: tuple[HarnessSessions, InstanceFactory],
):
    sessions, _factory = sessions_factory
    session = await sessions.create("demo", agent_name="support")

    response = await session.send("hello")

    assert response == "support-0:hello"


@pytest.mark.asyncio
async def test_invalid_agent_raises_and_disposes_created_instance(
    sessions_factory: tuple[HarnessSessions, InstanceFactory],
):
    sessions, factory = sessions_factory

    with pytest.raises(ValueError, match="Agent 'missing' not found"):
        await sessions.create("demo", agent_name="missing")

    assert factory.disposed == factory.instances


@pytest.mark.asyncio
async def test_invalid_agent_does_not_create_persisted_session() -> None:
    factory = InstanceFactory()
    persisted_calls: list[str] = []

    async def create_persistence(
        session_id: str,
        instance: AgentInstance,
        agent_name: str | None,
    ) -> tuple[Any, Any] | None:
        del instance, agent_name
        persisted_calls.append(session_id)
        return None

    sessions = HarnessSessions(
        create_instance=factory.create,
        dispose_instance=factory.dispose,
        create_persisted_session=create_persistence,
    )

    with pytest.raises(ValueError, match="Agent 'missing' not found"):
        await sessions.create("demo", agent_name="missing")

    assert persisted_calls == []
    assert factory.disposed == factory.instances


@pytest.mark.asyncio
async def test_concurrent_same_session_operations_reject(
    sessions_factory: tuple[HarnessSessions, InstanceFactory],
):
    sessions, factory = sessions_factory
    factory.block_generate = asyncio.Event()
    factory.release_generate = asyncio.Event()
    session = await sessions.create("support-123")

    running = asyncio.create_task(session.generate("first"))
    await factory.block_generate.wait()

    with pytest.raises(RuntimeError, match="already running generate"):
        await session.generate("second")

    factory.release_generate.set()
    await running


@pytest.mark.asyncio
async def test_delete_active_session_rejects(
    sessions_factory: tuple[HarnessSessions, InstanceFactory],
):
    sessions, factory = sessions_factory
    factory.block_generate = asyncio.Event()
    factory.release_generate = asyncio.Event()
    session = await sessions.create("support-123")

    running = asyncio.create_task(session.generate("first"))
    await factory.block_generate.wait()

    with pytest.raises(RuntimeError, match="running generate; wait before deleting"):
        await session.delete()

    factory.release_generate.set()
    await running


@pytest.mark.asyncio
async def test_clear_only_resolved_target_agent(
    sessions_factory: tuple[HarnessSessions, InstanceFactory],
):
    sessions, _factory = sessions_factory
    session = await sessions.create("demo", agent_name="support")

    await session.clear(clear_prompts=True)

    agents = session._record.instance.agents
    main = cast("FakeAgent", agents["main"])
    support = cast("FakeAgent", agents["support"])
    assert main.clears == []
    assert support.clears == [True]
