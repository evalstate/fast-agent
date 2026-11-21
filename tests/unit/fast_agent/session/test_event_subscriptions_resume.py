from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.commands.context import (
    CommandContext,
    NonInteractiveCommandIOBase,
    StaticAgentProvider,
)
from fast_agent.commands.handlers.events import handle_events
from fast_agent.config import MCPServerSettings, Settings
from fast_agent.context import Context
from fast_agent.integrations.herdr_session import _resumable_session_id
from fast_agent.llm.request_params import RequestParams
from fast_agent.mcp.events_extension import (
    EventDescriptor,
    ListEventsResult,
    PollEventsResult,
)
from fast_agent.mcp.events_subscriptions import (
    EventSubscription,
    EventSubscriptionManager,
    SubscriptionDelivery,
    SubscriptionOrigin,
)
from fast_agent.mcp_server_registry import ServerRegistry
from fast_agent.session import (
    SessionAgentSnapshot,
    SessionAttachmentRef,
    SessionContinuationSnapshot,
    SessionEventSubscriptionSnapshot,
    SessionHttpMcpServerSnapshot,
    SessionHydrator,
    SessionManager,
    SessionSaveIdentity,
    SessionSnapshot,
    capture_session_snapshot,
    load_session_snapshot,
)
from fast_agent.session.hydrator import DEFAULT_RESTORE_EVENT_MAX_AGE_MS

if TYPE_CHECKING:
    from pathlib import Path

    from fast_agent.interfaces import AgentProtocol


class _ProviderStub:
    config_name = "anthropic"


class _FakeEventsAgent:
    def __init__(
        self,
        *,
        name: str = "worker",
        attached_servers: list[str] | None = None,
        registered_servers: dict[str, MCPServerSettings] | None = None,
        subscriptions: list[EventSubscription] | None = None,
        fail_restore_events: set[str] | None = None,
    ) -> None:
        self.name = name
        self.instruction = "Base instruction"
        self.config = AgentConfig(name, instruction=self.instruction, model="passthrough")
        self.config.default_request_params = RequestParams(
            use_history=True,
            system_prompt=self.instruction,
        )
        self.llm = SimpleNamespace(
            default_request_params=self.config.default_request_params.model_copy(deep=True),
            model_name="passthrough",
            provider=_ProviderStub(),
            resolved_model=SimpleNamespace(overlay=None),
            web_search_supported=False,
            web_search_enabled=False,
            web_fetch_supported=False,
            web_fetch_enabled=False,
            x_search_supported=False,
            x_search_enabled=False,
        )
        self.usage_accumulator = None
        self.message_history: list[object] = []
        self._attached_servers = list(attached_servers or [])
        self._subscriptions = list(subscriptions or [])
        self._fail_restore_events = set(fail_restore_events or ())
        self.attach_calls: list[tuple[str, MCPServerSettings | None]] = []
        self.restore_calls: list[dict[str, Any]] = []
        self.call_order: list[str] = []

        registry = ServerRegistry(config=Settings())
        for server_name, cfg in (registered_servers or {}).items():
            registry.registry[server_name] = cfg
        self.context: Context = Context(server_registry=registry)
        self._registered_names = set(registered_servers or {})

    def clear(self, *, clear_prompts: bool = False) -> None:
        del clear_prompts
        self.message_history.clear()

    def list_attached_mcp_servers(self) -> list[str]:
        return list(self._attached_servers)

    async def attach_mcp_server(
        self,
        *,
        server_name: str,
        server_config: MCPServerSettings | None = None,
        options: object | None = None,
    ) -> None:
        del options
        self.attach_calls.append((server_name, server_config))
        self.call_order.append(f"attach:{server_name}")
        if server_config is None and server_name not in self._registered_names:
            raise ValueError(f"Server '{server_name}' not found in registry")
        self._attached_servers.append(server_name)

    def list_active_event_subscriptions(self) -> list[EventSubscription]:
        return [sub for sub in self._subscriptions if sub.status == "active"]

    async def restore_event_subscription(
        self,
        server_name: str,
        name: str,
        arguments: dict[str, Any] | None = None,
        *,
        wake: bool = False,
        once: bool = False,
        min_poll_ms: int | None = None,
        origin: SubscriptionOrigin = "user",
        cursor: str | None = None,
        max_age_ms: int | None = None,
        delivery: SubscriptionDelivery | None = None,
    ) -> EventSubscription:
        self.call_order.append(f"restore:{server_name}/{name}")
        if name in self._fail_restore_events:
            raise ValueError(f"Server '{server_name}' does not offer event '{name}'")
        self.restore_calls.append(
            {
                "server_name": server_name,
                "name": name,
                "arguments": dict(arguments or {}),
                "wake": wake,
                "once": once,
                "min_poll_ms": min_poll_ms,
                "origin": origin,
                "cursor": cursor,
                "max_age_ms": max_age_ms,
            }
        )
        subscription = EventSubscription(
            server_name=server_name,
            name=name,
            arguments=dict(arguments or {}),
            cursor=cursor,
            consumed_cursor=cursor,
            wake=wake,
            once=once,
            min_poll_ms=min_poll_ms,
            origin=origin,
        )
        self._subscriptions.append(subscription)
        return subscription


def _write_snapshot(session_dir: Path, snapshot: SessionSnapshot) -> None:
    (session_dir / "session.json").write_text(
        json.dumps(snapshot.model_dump(mode="json"), indent=2),
        encoding="utf-8",
    )


def test_capture_persists_active_event_subscriptions_with_consumed_cursor(tmp_path: Path) -> None:
    manager = SessionManager(cwd=tmp_path, home_override=tmp_path / ".fast-agent")
    session = manager.create_session(name="2611260000-evcap1")

    active_sub = EventSubscription(
        server_name="jobs",
        name="job.status_changed",
        arguments={"job_id": "42"},
        cursor="c2",
        consumed_cursor="c1",
        status="active",
        wake=True,
        once=False,
        min_poll_ms=15_000,
        origin="user",
    )
    completed_sub = EventSubscription(
        server_name="jobs",
        name="job.finished",
        arguments={"job_id": "99"},
        cursor="c9",
        consumed_cursor="c9",
        status="completed",
        wake=False,
        once=True,
        origin="model",
    )
    agent = _FakeEventsAgent(
        attached_servers=["jobs"],
        subscriptions=[active_sub, completed_sub],
    )

    snapshot = capture_session_snapshot(
        session=session,
        active_agent=cast("AgentProtocol", agent),
        agent_registry={"worker": cast("AgentProtocol", agent)},
        identity=SessionSaveIdentity(
            manager=manager,
            session=session,
            created=False,
            acp_session_id=None,
            session_cwd=tmp_path,
            session_store_scope="workspace",
            session_store_cwd=tmp_path,
        ),
    )

    persisted = snapshot.continuation.agents["worker"].event_subscriptions
    assert persisted == [
        SessionEventSubscriptionSnapshot(
            server="jobs",
            name="job.status_changed",
            arguments={"job_id": "42"},
            wake=True,
            once=False,
            min_poll_ms=15_000,
            origin="user",
            cursor="c1",
            delivery="poll",
        )
    ]


@pytest.mark.asyncio
async def test_hydrate_restores_event_subscriptions_after_mcp_attachments_and_emits_notice(
    tmp_path: Path,
) -> None:
    manager = SessionManager(cwd=tmp_path, home_override=tmp_path / ".fast-agent")
    session = manager.create_session(name="2611260000-evres1")

    snapshot = SessionSnapshot(
        session_id=session.info.name,
        created_at=session.info.created_at,
        last_activity=session.info.last_activity,
        continuation=SessionContinuationSnapshot(
            active_agent="worker",
            agents={
                "worker": SessionAgentSnapshot(
                    attachment_refs=[SessionAttachmentRef(ref="mcp_server:jobs")],
                    http_mcp_servers=[
                        SessionHttpMcpServerSnapshot(
                            name="jobs",
                            url="https://jobs.example.com/mcp",
                            transport="http",
                        )
                    ],
                    event_subscriptions=[
                        SessionEventSubscriptionSnapshot(
                            server="jobs",
                            name="job.status_changed",
                            arguments={"job_id": "1"},
                            wake=True,
                            once=False,
                            min_poll_ms=10_000,
                            origin="user",
                            cursor="c1",
                        ),
                        SessionEventSubscriptionSnapshot(
                            server="jobs",
                            name="job.created",
                            arguments={},
                            wake=False,
                            once=True,
                            min_poll_ms=None,
                            origin="model",
                            cursor=None,
                        ),
                    ],
                )
            },
        ),
    )
    _write_snapshot(session.directory, snapshot)

    agent = _FakeEventsAgent(attached_servers=[])
    hydrator = SessionHydrator()
    result = await hydrator.hydrate_session(
        session=session,
        agents={"worker": cast("AgentProtocol", agent)},
        fallback_agent_name="worker",
    )

    assert result.warnings == []
    assert agent.call_order == [
        "attach:jobs",
        "restore:jobs/job.status_changed",
        "restore:jobs/job.created",
    ]
    assert agent.restore_calls == [
        {
            "server_name": "jobs",
            "name": "job.status_changed",
            "arguments": {"job_id": "1"},
            "wake": True,
            "once": False,
            "min_poll_ms": 10_000,
            "origin": "user",
            "cursor": "c1",
            "max_age_ms": DEFAULT_RESTORE_EVENT_MAX_AGE_MS,
        },
        {
            "server_name": "jobs",
            "name": "job.created",
            "arguments": {},
            "wake": False,
            "once": True,
            "min_poll_ms": None,
            "origin": "model",
            "cursor": None,
            "max_age_ms": None,
        },
    ]
    assert result.usage_notices == ["Restored 2 event subscriptions (1 wakes)"]


@pytest.mark.asyncio
async def test_hydrate_singular_notice_without_wakes(tmp_path: Path) -> None:
    manager = SessionManager(cwd=tmp_path, home_override=tmp_path / ".fast-agent")
    session = manager.create_session(name="2611260000-evres2")

    snapshot = SessionSnapshot(
        session_id=session.info.name,
        created_at=session.info.created_at,
        last_activity=session.info.last_activity,
        continuation=SessionContinuationSnapshot(
            active_agent="worker",
            agents={
                "worker": SessionAgentSnapshot(
                    event_subscriptions=[
                        SessionEventSubscriptionSnapshot(
                            server="jobs",
                            name="job.created",
                            wake=False,
                            cursor="c1",
                        )
                    ]
                )
            },
        ),
    )
    _write_snapshot(session.directory, snapshot)

    agent = _FakeEventsAgent(attached_servers=["jobs"])
    result = await SessionHydrator().hydrate_session(
        session=session,
        agents={"worker": cast("AgentProtocol", agent)},
        fallback_agent_name="worker",
    )

    assert result.warnings == []
    assert result.usage_notices == ["Restored 1 event subscription"]


@pytest.mark.asyncio
async def test_hydrate_warns_when_event_server_missing_or_restore_fails(tmp_path: Path) -> None:
    manager = SessionManager(cwd=tmp_path, home_override=tmp_path / ".fast-agent")
    session = manager.create_session(name="2611260000-evwarn1")

    snapshot = SessionSnapshot(
        session_id=session.info.name,
        created_at=session.info.created_at,
        last_activity=session.info.last_activity,
        continuation=SessionContinuationSnapshot(
            active_agent="worker",
            agents={
                "worker": SessionAgentSnapshot(
                    event_subscriptions=[
                        SessionEventSubscriptionSnapshot(
                            server="missing-server",
                            name="job.status_changed",
                            cursor="c1",
                        ),
                        SessionEventSubscriptionSnapshot(
                            server="jobs",
                            name="job.removed",
                            cursor="c2",
                        ),
                        SessionEventSubscriptionSnapshot(
                            server="jobs",
                            name="job.ok",
                            wake=True,
                            cursor="c3",
                        ),
                    ]
                )
            },
        ),
    )
    _write_snapshot(session.directory, snapshot)

    agent = _FakeEventsAgent(
        attached_servers=["jobs"],
        fail_restore_events={"job.removed"},
    )
    result = await SessionHydrator().hydrate_session(
        session=session,
        agents={"worker": cast("AgentProtocol", agent)},
        fallback_agent_name="worker",
    )

    assert len(result.warnings) == 2
    assert all(w.code == "event-subscription-restore-failed" for w in result.warnings)
    assert result.warnings[0].ref == "missing-server/job.status_changed"
    assert result.warnings[1].ref == "jobs/job.removed"
    assert result.usage_notices == ["Restored 1 event subscription (1 wakes)"]


def test_legacy_snapshot_without_event_subscriptions_loads_cleanly() -> None:
    raw_snapshot = {
        "schema_version": 8,
        "session_id": "2611260000-legacy",
        "created_at": "2026-11-26T00:00:00",
        "last_activity": "2026-11-26T00:00:00",
        "continuation": {
            "active_agent": "worker",
            "agents": {
                "worker": {
                    "attachment_refs": [],
                }
            },
        },
    }

    snapshot = load_session_snapshot(raw_snapshot)
    assert snapshot.continuation.agents["worker"].event_subscriptions == []


def test_resumable_session_id_accepts_session_with_persisted_event_subscriptions(
    tmp_path: Path,
) -> None:
    manager = SessionManager(cwd=tmp_path, home_override=tmp_path / ".fast-agent")
    empty_session = manager.create_session(name="2611260000-empty1")
    assert _resumable_session_id(empty_session) is None

    events_session = manager.create_session(name="2611260000-evonly1")
    snapshot = SessionSnapshot(
        session_id=events_session.info.name,
        created_at=events_session.info.created_at,
        last_activity=events_session.info.last_activity,
        continuation=SessionContinuationSnapshot(
            active_agent="worker",
            agents={
                "worker": SessionAgentSnapshot(
                    event_subscriptions=[
                        SessionEventSubscriptionSnapshot(
                            server="jobs",
                            name="job.status_changed",
                            cursor="c1",
                        )
                    ]
                )
            },
        ),
    )
    _write_snapshot(events_session.directory, snapshot)
    assert events_session.info.history_files == []
    assert _resumable_session_id(events_session) == events_session.info.name


class _CommandEventsAggregator:
    def __init__(self) -> None:
        self.server_names = ["jobs"]
        self.events = EventSubscriptionManager(self, sleep=self._sleep)

    @staticmethod
    async def _sleep(seconds: float) -> None:
        del seconds
        await asyncio.Event().wait()

    async def supports_events(self, server_name: str) -> bool:
        return server_name == "jobs"

    async def list_events(self, server_name: str, cursor: str | None = None) -> ListEventsResult:
        del server_name, cursor
        return ListEventsResult(
            events=[EventDescriptor(name="job.status_changed", delivery=["poll"])]
        )

    def can_hold_event_stream(self, server_name: str) -> bool:
        return True

    async def poll_events(
        self,
        server_name: str,
        name: str,
        arguments: dict[str, Any] | None = None,
        *,
        cursor: str | None = None,
        max_age_ms: int | None = None,
        max_events: int | None = None,
    ) -> PollEventsResult:
        del server_name, name, arguments, cursor, max_age_ms, max_events
        return PollEventsResult(events=[], cursor="c0", next_poll_ms=30_000)


class _CommandEventsAgent(_FakeEventsAgent):
    def __init__(self) -> None:
        super().__init__(attached_servers=["jobs"])
        self.aggregator = _CommandEventsAggregator()

    def list_active_event_subscriptions(self) -> list[EventSubscription]:
        return self.aggregator.events.persistable_subscriptions()


@pytest.mark.asyncio
async def test_events_command_persists_session_on_subscribe_and_unsubscribe(
    tmp_path: Path,
) -> None:
    manager = SessionManager(cwd=tmp_path, home_override=tmp_path / ".fast-agent")
    session = manager.create_session(name="2611260000-cmdsv1")
    agent = _CommandEventsAgent()
    ctx = CommandContext(
        agent_provider=StaticAgentProvider({"worker": agent}),
        current_agent_name="worker",
        io=NonInteractiveCommandIOBase(),
        session_manager=manager,
    )

    try:
        sub_outcome = await handle_events(
            ctx,
            agent_name="worker",
            argument="subscribe jobs job.status_changed job_id=7 --wake",
        )
        assert not any(m.channel == "error" for m in sub_outcome.messages)

        saved = session.load_snapshot()
        assert saved.continuation.agents["worker"].event_subscriptions == [
            SessionEventSubscriptionSnapshot(
                server="jobs",
                name="job.status_changed",
                arguments={"job_id": 7},
                wake=True,
                once=False,
                min_poll_ms=None,
                origin="user",
                cursor="c0",
                delivery="poll",
            )
        ]

        sub_key = agent.list_active_event_subscriptions()[0].key
        unsub_outcome = await handle_events(
            ctx,
            agent_name="worker",
            argument=f"unsubscribe {sub_key}",
        )
        assert not any(m.channel == "error" for m in unsub_outcome.messages)

        saved_after = session.load_snapshot()
        assert saved_after.continuation.agents["worker"].event_subscriptions == []
    finally:
        await agent.aggregator.events.aclose()


@pytest.mark.asyncio
async def test_events_subscribe_creates_session_when_none_exists_and_reports_osc(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fast_agent.integrations import herdr_session

    reported: list[tuple[object, object]] = []
    monkeypatch.setattr(
        herdr_session,
        "report_current_session",
        lambda mgr, *, agent=None, agent_lookup=None: reported.append((mgr, agent)),
    )

    manager = SessionManager(cwd=tmp_path, home_override=tmp_path / ".fast-agent")
    assert manager.current_session is None
    agent = _CommandEventsAgent()
    ctx = CommandContext(
        agent_provider=StaticAgentProvider({"worker": agent}),
        current_agent_name="worker",
        io=NonInteractiveCommandIOBase(),
        session_manager=manager,
    )

    try:
        outcome = await handle_events(
            ctx,
            agent_name="worker",
            argument="subscribe jobs job.status_changed job_id=7 --wake",
        )
        assert not any(m.channel == "error" for m in outcome.messages)
        assert manager.current_session is not None

        saved = manager.current_session.load_snapshot()
        assert saved.continuation.agents["worker"].event_subscriptions == [
            SessionEventSubscriptionSnapshot(
                server="jobs",
                name="job.status_changed",
                arguments={"job_id": 7},
                wake=True,
                once=False,
                min_poll_ms=None,
                origin="user",
                cursor="c0",
                delivery="poll",
            )
        ]
        assert reported == [(manager, agent)]
    finally:
        await agent.aggregator.events.aclose()


def test_capture_persists_completed_once_subscription_with_pending_inbox(tmp_path: Path) -> None:
    manager = SessionManager(cwd=tmp_path, home_override=tmp_path / ".fast-agent")
    session = manager.create_session(name="2611260000-evonce")

    completed_pending_once = EventSubscription(
        server_name="jobs",
        name="job.finished",
        arguments={"job_id": "99"},
        cursor="c2",
        consumed_cursor="c1",
        status="completed",
        wake=True,
        once=True,
        origin="user",
    )

    class _PendingOnceAgent(_FakeEventsAgent):
        def list_active_event_subscriptions(self) -> list[EventSubscription]:
            return [completed_pending_once]

    agent = _PendingOnceAgent(attached_servers=["jobs"])
    snapshot = capture_session_snapshot(
        session=session,
        active_agent=cast("AgentProtocol", agent),
        agent_registry={"worker": cast("AgentProtocol", agent)},
        identity=SessionSaveIdentity(
            manager=manager,
            session=session,
            created=False,
            acp_session_id=None,
            session_cwd=tmp_path,
            session_store_scope="workspace",
            session_store_cwd=tmp_path,
        ),
    )

    assert snapshot.continuation.agents["worker"].event_subscriptions == [
        SessionEventSubscriptionSnapshot(
            server="jobs",
            name="job.finished",
            arguments={"job_id": "99"},
            wake=True,
            once=True,
            min_poll_ms=None,
            origin="user",
            cursor="c1",
            delivery="poll",
        )
    ]


@pytest.mark.asyncio
async def test_handle_resume_session_preserves_restored_subscriptions_and_replayed_events(
    tmp_path: Path,
) -> None:
    from fast_agent.commands.handlers.sessions import handle_resume_session
    from fast_agent.mcp.events_extension import EventOccurrence

    class _ReplayEventsAggregator(_CommandEventsAggregator):
        async def poll_events(
            self,
            server_name: str,
            name: str,
            arguments: dict[str, Any] | None = None,
            *,
            cursor: str | None = None,
            max_age_ms: int | None = None,
            max_events: int | None = None,
        ) -> PollEventsResult:
            del server_name, arguments, max_age_ms, max_events
            if cursor == "c0":
                return PollEventsResult(
                    events=[
                        EventOccurrence(
                            event_id="ev-missed",
                            name=name,
                            timestamp="2026-11-27T10:00:00Z",
                            data={"stage": "COMPLETED"},
                        )
                    ],
                    cursor="c1",
                    next_poll_ms=30_000,
                )
            return PollEventsResult(events=[], cursor="c0", next_poll_ms=30_000)

    class _RealResumeEventsAgent(_FakeEventsAgent):
        def __init__(self) -> None:
            super().__init__(attached_servers=["jobs"])
            self.aggregator = _ReplayEventsAggregator()

        def list_active_event_subscriptions(self) -> list[EventSubscription]:
            return self.aggregator.events.persistable_subscriptions()

        async def reset_events(self) -> None:
            await self.aggregator.events.reset()

        async def restore_event_subscription(
            self,
            server_name: str,
            name: str,
            arguments: dict[str, Any] | None = None,
            *,
            wake: bool = False,
            once: bool = False,
            min_poll_ms: int | None = None,
            origin: SubscriptionOrigin = "user",
            cursor: str | None = None,
            max_age_ms: int | None = None,
        ) -> EventSubscription:
            return await self.aggregator.events.subscribe(
                server_name,
                name,
                arguments,
                wake=wake,
                once=once,
                min_poll_ms=min_poll_ms,
                origin=origin,
                cursor=cursor,
                max_age_ms=max_age_ms,
            )

    manager = SessionManager(cwd=tmp_path, home_override=tmp_path / ".fast-agent")
    saved_session = manager.create_session(name="2611270000-rsm1")
    _write_snapshot(
        saved_session.directory,
        SessionSnapshot(
            session_id=saved_session.info.name,
            created_at=saved_session.info.created_at,
            last_activity=saved_session.info.last_activity,
            continuation=SessionContinuationSnapshot(
                active_agent="worker",
                agents={
                    "worker": SessionAgentSnapshot(
                        event_subscriptions=[
                            SessionEventSubscriptionSnapshot(
                                server="jobs",
                                name="job.status_changed",
                                arguments={"job_id": "42"},
                                wake=True,
                                origin="user",
                                cursor="c0",
                            )
                        ]
                    )
                },
            ),
        ),
    )

    agent = _RealResumeEventsAgent()
    ctx = CommandContext(
        agent_provider=StaticAgentProvider({"worker": agent}),
        current_agent_name="worker",
        io=NonInteractiveCommandIOBase(),
        session_manager=manager,
    )

    try:
        # Start a different subscription in the current session.
        previous = await agent.aggregator.events.subscribe(
            "jobs",
            "job.status_changed",
            {"job_id": "old"},
            origin="user",
        )
        assert len(agent.aggregator.events.subscriptions()) == 1

        # Missing session leaves existing subscriptions untouched.
        missing_outcome = await handle_resume_session(
            ctx, agent_name="worker", session_id="2611270000-missing"
        )
        assert any(m.channel == "error" for m in missing_outcome.messages)
        assert [s.key for s in agent.aggregator.events.subscriptions()] == [previous.key]

        # Valid resume replaces old subscription with restored one and keeps replayed event.
        outcome = await handle_resume_session(
            ctx, agent_name="worker", session_id=saved_session.info.name
        )
        assert not any(m.channel == "error" for m in outcome.messages)
        subs = agent.aggregator.events.subscriptions()
        assert len(subs) == 1
        assert subs[0].arguments == {"job_id": "42"}
        assert subs[0].wake is True
        items = agent.aggregator.events.take()
        assert len(items) == 1
        assert items[0].occurrence is not None
        assert items[0].occurrence.event_id == "ev-missed"
    finally:
        await agent.aggregator.events.aclose()


@pytest.mark.asyncio
async def test_hydrator_downgrades_removed_declared_event_origin_to_user(
    tmp_path: Path,
) -> None:
    class _ConfigCheckingAgent(_FakeEventsAgent):
        def __init__(self, *, declared: set[tuple[str, str]]) -> None:
            super().__init__(attached_servers=["jobs"])
            self._declared = declared
            self.reset_calls = 0

        async def reset_events(self) -> None:
            self.reset_calls += 1

        def is_declared_event(
            self,
            server_name: str,
            name: str,
            arguments: dict[str, Any] | None = None,
        ) -> bool:
            del arguments
            return (server_name, name) in self._declared

    manager = SessionManager(cwd=tmp_path, home_override=tmp_path / ".fast-agent")
    session = manager.create_session(name="2611270000-decl1")
    snapshot = SessionSnapshot(
        session_id=session.info.name,
        created_at=session.info.created_at,
        last_activity=session.info.last_activity,
        continuation=SessionContinuationSnapshot(
            active_agent="worker",
            agents={
                "worker": SessionAgentSnapshot(
                    event_subscriptions=[
                        SessionEventSubscriptionSnapshot(
                            server="jobs",
                            name="job.removed",
                            arguments={},
                            origin="declared",
                            cursor="c1",
                        ),
                        SessionEventSubscriptionSnapshot(
                            server="jobs",
                            name="job.still_declared",
                            arguments={},
                            origin="declared",
                            cursor="c2",
                        ),
                        SessionEventSubscriptionSnapshot(
                            server="jobs",
                            name="job.newly_declared",
                            arguments={},
                            origin="model",
                            cursor="c3",
                        ),
                    ]
                )
            },
        ),
    )
    _write_snapshot(session.directory, snapshot)

    agent = _ConfigCheckingAgent(
        declared={("jobs", "job.still_declared"), ("jobs", "job.newly_declared")}
    )
    omitted_agent = _ConfigCheckingAgent(declared=set())

    hydrator = SessionHydrator()
    await hydrator.hydrate_session(
        session=session,
        agents={
            "worker": cast("AgentProtocol", agent),
            "other": cast("AgentProtocol", omitted_agent),
        },
        fallback_agent_name="worker",
    )

    by_name = {call["name"]: call for call in agent.restore_calls}
    assert by_name["job.removed"]["origin"] == "user"
    assert by_name["job.still_declared"]["origin"] == "declared"
    assert by_name["job.newly_declared"]["origin"] == "declared"
    assert omitted_agent.reset_calls == 1
