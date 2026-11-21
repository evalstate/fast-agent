from __future__ import annotations

import pytest

from fast_agent.mcp.events_extension import EventDescriptor
from fast_agent.mcp.events_subscriptions import EventSubscription
from fast_agent.ui.events_picker import (
    SUBSCRIBED_GROUP,
    EventsPickerState,
)


def _sample_descriptors() -> dict[str, list[EventDescriptor]]:
    return {
        "jobs": [
            EventDescriptor(
                name="job.status_changed",
                description="Fires when a job transitions stage",
                delivery=["poll"],
                input_schema={
                    "type": "object",
                    "properties": {
                        "job_id": {"type": "string", "description": "Job identifier"},
                        "stage": {
                            "type": "string",
                            "enum": ["QUEUED", "RUNNING", "COMPLETED"],
                            "default": "COMPLETED",
                        },
                    },
                    "required": ["job_id"],
                },
                payload_schema={
                    "type": "object",
                    "properties": {
                        "stage": {"type": "string"},
                        "progress": {"type": "number"},
                    },
                },
            ),
            EventDescriptor(
                name="job.heartbeat",
                description="Periodic worker heartbeat",
                delivery=["poll"],
            ),
            EventDescriptor(
                name="job.log_line",
                description="Live log line (push only)",
                delivery=["push"],
            ),
        ],
        "ci": [
            EventDescriptor(
                name="build.finished",
                description="CI build completed",
                delivery=["poll"],
                input_schema={
                    "type": "object",
                    "properties": {"repo": {"type": "string"}},
                    "required": ["repo"],
                },
            ),
        ],
    }


def _sample_subscriptions() -> list[EventSubscription]:
    return [
        EventSubscription(
            server_name="jobs",
            name="job.status_changed",
            arguments={"job_id": "abc123"},
            status="active",
            delivered=3,
            wake=True,
        ),
        EventSubscription(
            server_name="ci",
            name="build.finished",
            arguments={"repo": "fast-agent"},
            status="completed",
            delivered=1,
            wake=False,
            once=True,
        ),
    ]


def test_events_picker_state_groups_rows_and_formatting() -> None:
    state = EventsPickerState.from_data(
        subscriptions=_sample_subscriptions(),
        descriptors_by_server=_sample_descriptors(),
    )

    assert state.groups == (SUBSCRIBED_GROUP, "jobs", "ci")
    assert state.group == SUBSCRIBED_GROUP
    assert len(state.visible) == 2

    active_row, completed_row = state.visible
    assert active_row.format_row() == (
        "● jobs job.status_changed job_id=abc123 · wake · 3 delivered · active"
    )
    assert not active_row.is_dimmed
    assert completed_row.format_row() == (
        "● ci build.finished repo=fast-agent · passive · 1 delivered · completed"
    )
    assert completed_row.is_dimmed

    state.cycle_group(1)
    assert state.group == "jobs"
    assert [item.format_row() for item in state.visible] == [
        "○ jobs job.status_changed (job_id*) poll",
        "○ jobs job.heartbeat poll",
        "○ jobs job.log_line push",
    ]
    assert not state.visible[0].is_dimmed
    assert state.visible[0].is_selectable
    assert state.visible[2].is_dimmed
    assert not state.visible[2].is_selectable


def test_events_picker_starts_on_first_server_when_no_subscriptions() -> None:
    state = EventsPickerState.from_data(
        subscriptions=[],
        descriptors_by_server=_sample_descriptors(),
    )

    assert state.group == "jobs"
    assert len(state.visible) == 3


def test_events_picker_filtering_and_detail_pane() -> None:
    state = EventsPickerState.from_data(
        subscriptions=_sample_subscriptions(),
        descriptors_by_server=_sample_descriptors(),
    )

    state.cycle_group(1)  # jobs
    state.type_text("status")
    assert [item.name for item in state.visible] == ["job.status_changed"]

    detail = state.detail_text()
    assert "jobs / job.status_changed" in detail
    assert "Fires when a job transitions stage" in detail
    assert "job_id*" in detail
    assert "enum: QUEUED, RUNNING, COMPLETED" in detail
    assert "default='COMPLETED'" in detail
    assert "stage, progress" in detail

    state.clear_query()
    assert len(state.visible) == 3


def test_events_picker_toggle_wake_and_enter_actions() -> None:
    subs = _sample_subscriptions()
    state = EventsPickerState.from_data(
        subscriptions=subs,
        descriptors_by_server=_sample_descriptors(),
    )

    # On active subscribed row: toggle wake flips wake flag; Enter closes.
    toggled = state.toggle_wake_current()
    assert toggled is subs[0]
    assert subs[0].wake is False
    assert state.enter_action() == ("close", None)

    # On completed subscribed row: toggle wake is a no-op.
    state.move(1)
    assert state.toggle_wake_current() is None

    # Switch to jobs group:
    state.cycle_group(1)
    # 1. Required args -> prefill draft
    assert state.enter_action() == (
        "prefill",
        "/events subscribe jobs job.status_changed job_id=",
    )
    # 2. No required args -> immediate subscribe
    state.move(1)
    assert state.enter_action() == ("subscribe", None)
    # 3. Non-pollable -> noop
    state.move(1)
    assert state.enter_action() == ("noop", None)


def test_events_picker_refresh_subscriptions_clamps_cursor() -> None:
    subs = _sample_subscriptions()
    state = EventsPickerState.from_data(
        subscriptions=subs,
        descriptors_by_server=_sample_descriptors(),
    )
    state.move(1)
    assert state.cursor == 1

    state.refresh_subscriptions(subs[:1])
    assert state.cursor == 0
    assert len(state.visible) == 1


def test_events_picker_prefill_quotes_server_with_spaces() -> None:
    desc = EventDescriptor(
        name="job.status_changed",
        delivery=["poll"],
        input_schema={
            "type": "object",
            "properties": {"job_id": {"type": "string"}},
            "required": ["job_id"],
        },
    )
    state = EventsPickerState.from_data(
        subscriptions=[],
        descriptors_by_server={"my jobs": [desc]},
    )
    assert state.enter_action() == (
        "prefill",
        "/events subscribe 'my jobs' job.status_changed job_id=",
    )


@pytest.mark.asyncio
async def test_events_picker_toggle_wake_takes_user_ownership_and_removes_ended() -> None:
    import asyncio

    from fast_agent.mcp.events_extension import ListEventsResult, PollEventsResult
    from fast_agent.mcp.events_subscriptions import EventSubscriptionManager
    from fast_agent.ui.events_picker import _EventsPicker

    class _Poller:
        async def list_events(
            self, server_name: str, cursor: str | None = None
        ) -> ListEventsResult:
            del server_name, cursor
            return ListEventsResult(events=[EventDescriptor(name="job.done", delivery=["poll"])])

        def can_hold_event_stream(self, server_name: str) -> bool:
            return True

        async def poll_events(
            self,
            server_name: str,
            name: str,
            arguments: dict[str, object] | None = None,
            *,
            cursor: str | None = None,
            max_age_ms: int | None = None,
            max_events: int | None = None,
        ) -> PollEventsResult:
            del server_name, name, arguments, cursor, max_age_ms, max_events
            return PollEventsResult(events=[], cursor="c1")

    manager = EventSubscriptionManager(_Poller())
    try:
        sub = await manager.subscribe("jobs", "job.done", {}, wake=False, origin="model")
        ended = EventSubscription(
            server_name="jobs", name="job.done", arguments={"id": "old"}, status="completed"
        )
        manager._terminated[ended.key] = ended

        picker = _EventsPicker(
            manager=manager,
            descriptors_by_server={"jobs": [EventDescriptor(name="job.done", delivery=["poll"])]},
        )

        # Toggling wake via picker goes through manager.subscribe(..., origin="user").
        await picker._toggle_wake_item(sub)
        assert sub.wake is True
        assert sub.origin == "user"

        # Removing a completed row via unsubscribe works too.
        await picker._unsubscribe_item(ended.key)
        assert ended.key not in {s.key for s in manager.subscriptions()}

        # Duplicate submits while one mutation is in flight are ignored, and _cancel_tasks cancels pending work.
        gate = asyncio.Event()
        started = asyncio.Event()

        async def _slow() -> None:
            started.set()
            await gate.wait()

        picker._spawn(_slow())
        await started.wait()
        second_ran = False

        async def _duplicate() -> None:
            nonlocal second_ran
            second_ran = True

        picker._spawn(_duplicate())
        assert second_ran is False
        assert len(picker._background_tasks) == 1

        await picker._cancel_tasks()
        assert len(picker._background_tasks) == 0
        assert picker._mutation_in_flight is False
    finally:
        await manager.aclose()
