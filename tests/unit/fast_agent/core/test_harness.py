from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest

from fast_agent import AgentHarness, FastAgent, HarnessSession, HarnessSessions
from fast_agent.core.agent_app import AgentApp
from fast_agent.core.agent_instance_factory import CallableAgentInstanceFactory
from fast_agent.core.fastagent import AgentInstance
from fast_agent.tools.session_environment import ShellExecutionResult
from fast_agent.types import PromptMessageExtended

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from fast_agent.core.harness_persistence import HarnessSessionPersistence
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
        self.context: Any = None
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


class FakePersistence:
    def __init__(self) -> None:
        self.created: list[tuple[str, str | None]] = []
        self.saved: list[tuple[object, str]] = []
        self.deleted: list[str] = []

    async def create_or_load(
        self,
        session_id: str,
        instance: AgentInstance,
        default_agent_name: str | None,
    ) -> object:
        del instance
        self.created.append((session_id, default_agent_name))
        return {"session_id": session_id}

    async def save(
        self,
        handle: object,
        agent: "AgentProtocol",
        agent_registry: Mapping[str, "AgentProtocol"],
    ) -> None:
        del agent_registry
        self.saved.append((handle, agent.name))

    async def delete(self, session_id: str) -> None:
        self.deleted.append(session_id)


class FakeShellExecutor:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Path | str | None, Mapping[str, str] | None, float | None]] = []

    async def execute_shell(
        self,
        command: str,
        *,
        cwd: str | Path | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> ShellExecutionResult:
        self.calls.append((command, cwd, env, timeout))
        return ShellExecutionResult(stdout="out", stderr="err", exit_code=7)


@pytest.fixture
def sessions_factory() -> tuple[HarnessSessions, InstanceFactory]:
    factory = InstanceFactory()
    sessions = HarnessSessions(create_instance=factory.create, dispose_instance=factory.dispose)
    return sessions, factory


@pytest.mark.asyncio
async def test_harness_sessions_accepts_instance_factory() -> None:
    factory = InstanceFactory()
    sessions = HarnessSessions(
        instance_factory=CallableAgentInstanceFactory(
            create=factory.create,
            dispose=factory.dispose,
        )
    )

    session = await sessions.create("demo")
    await session.delete()

    assert factory.instances == [session._record.instance]
    assert factory.disposed == [session._record.instance]


def test_harness_sessions_rejects_ambiguous_factory_configuration() -> None:
    factory = InstanceFactory()

    with pytest.raises(ValueError, match="either instance_factory or create_instance"):
        HarnessSessions(
            instance_factory=CallableAgentInstanceFactory(
                create=factory.create,
                dispose=factory.dispose,
            ),
            create_instance=factory.create,
            dispose_instance=factory.dispose,
        )


def test_harness_sessions_requires_complete_factory_configuration() -> None:
    factory = InstanceFactory()

    with pytest.raises(ValueError, match="create_instance and dispose_instance are required"):
        HarnessSessions(create_instance=factory.create)


@pytest.mark.asyncio
async def test_harness_sessions_uses_persistence_protocol() -> None:
    factory = InstanceFactory()
    persistence = FakePersistence()
    sessions = HarnessSessions(
        create_instance=factory.create,
        dispose_instance=factory.dispose,
        persistence=persistence,
    )

    session = await sessions.create("demo", agent_name="support")
    response = await session.send("hello")
    await session.delete()

    assert response == "support-0:hello"
    assert persistence.created == [("demo", "support")]
    assert persistence.saved == [({"session_id": "demo"}, "support-0")]
    assert persistence.deleted == ["demo"]


@pytest.mark.asyncio
async def test_harness_session_compact_passes_settings_and_persists(monkeypatch) -> None:
    from fast_agent.config import CompactionSettings, Settings
    from fast_agent.context import Context
    from fast_agent.history.compaction import CompactionResult

    factory = InstanceFactory()
    persistence = FakePersistence()
    sessions = HarnessSessions(
        create_instance=factory.create,
        dispose_instance=factory.dispose,
        persistence=persistence,
    )

    session = await sessions.create("demo", agent_name="support")
    # The target agent carries config that compaction should pick up.
    agent = session._resolve_agent("support")
    agent.context = Context(config=Settings(compaction=CompactionSettings(keep_turns=4)))

    captured: dict[str, Any] = {}

    async def fake_compact(target, *, settings, instructions):
        captured["agent"] = target
        captured["settings"] = settings
        captured["instructions"] = instructions
        return CompactionResult(
            agent_name=target.name,
            summary_text="summary",
            messages_before=10,
            messages_after=3,
            tokens_before=900,
            tokens_after_estimate=120,
            context_window=100_000,
            archive_file=None,
        )

    # compact() imports compact_conversation lazily from its source module.
    monkeypatch.setattr(
        "fast_agent.history.compaction.compact_conversation", fake_compact, raising=False
    )

    result = await session.compact(instructions="keep the order number")

    assert result.messages_before == 10
    assert result.messages_after == 3
    assert captured["agent"] is agent
    assert captured["settings"].keep_turns == 4
    assert captured["instructions"] == "keep the order number"
    # Persistence is invoked with the compacted agent.
    assert persistence.saved == [({"session_id": "demo"}, "support-0")]

    await session.delete()


@pytest.mark.asyncio
async def test_harness_session_compact_defaults_settings_without_config(monkeypatch) -> None:
    from fast_agent.config import CompactionSettings
    from fast_agent.history.compaction import CompactionResult

    factory = InstanceFactory()
    sessions = HarnessSessions(
        create_instance=factory.create,
        dispose_instance=factory.dispose,
    )
    session = await sessions.create("demo", agent_name="support")

    captured: dict[str, Any] = {}

    async def fake_compact(target, *, settings, instructions):
        captured["settings"] = settings
        return CompactionResult(
            agent_name=target.name,
            summary_text="s",
            messages_before=2,
            messages_after=1,
            tokens_before=None,
            tokens_after_estimate=10,
            context_window=None,
            archive_file=None,
        )

    monkeypatch.setattr(
        "fast_agent.history.compaction.compact_conversation", fake_compact, raising=False
    )

    await session.compact()

    # No agent context/config -> default settings.
    assert isinstance(captured["settings"], CompactionSettings)
    assert captured["settings"].keep_turns == CompactionSettings().keep_turns

    await session.delete()


@pytest.mark.asyncio
async def test_harness_session_shell_uses_configured_executor(tmp_path: "Path") -> None:
    factory = InstanceFactory()
    shell_executor = FakeShellExecutor()
    sessions = HarnessSessions(
        create_instance=factory.create,
        dispose_instance=factory.dispose,
        shell_executor=shell_executor,
    )
    session = await sessions.create("demo")

    result = await session.shell(
        "pwd",
        cwd=tmp_path,
        env={"FAST_AGENT_TEST": "1"},
        timeout=2.5,
    )

    assert result == ShellExecutionResult(stdout="out", stderr="err", exit_code=7)
    assert shell_executor.calls == [("pwd", tmp_path, {"FAST_AGENT_TEST": "1"}, 2.5)]


@pytest.mark.asyncio
async def test_harness_session_shell_rejects_during_active_operation(
    sessions_factory: tuple[HarnessSessions, InstanceFactory],
) -> None:
    sessions, factory = sessions_factory
    factory.block_generate = asyncio.Event()
    factory.release_generate = asyncio.Event()
    session = await sessions.create("support-123")

    running = asyncio.create_task(session.generate("first"))
    await factory.block_generate.wait()

    with pytest.raises(RuntimeError, match="already running generate"):
        await session.shell("pwd")

    factory.release_generate.set()
    await running


def test_harness_sessions_rejects_ambiguous_persistence_configuration() -> None:
    factory = InstanceFactory()
    persistence: "HarnessSessionPersistence" = FakePersistence()

    async def create_persistence(
        session_id: str,
        instance: AgentInstance,
        agent_name: str | None,
    ) -> tuple[Any, Any] | None:
        del session_id, instance, agent_name
        return None

    with pytest.raises(ValueError, match="either persistence or create_persisted_session"):
        HarnessSessions(
            create_instance=factory.create,
            dispose_instance=factory.dispose,
            persistence=persistence,
            create_persisted_session=create_persistence,
        )


def test_harness_sessions_rejects_delete_persistence_without_create() -> None:
    factory = InstanceFactory()

    async def delete_persistence(session_id: str) -> None:
        del session_id

    with pytest.raises(ValueError, match="delete_persisted_session requires"):
        HarnessSessions(
            create_instance=factory.create,
            dispose_instance=factory.dispose,
            delete_persisted_session=delete_persistence,
        )


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
