import asyncio

import pytest
from mcp.shared.exceptions import MCPError

from fast_agent.config import MCPEventSubscriptionSettings, MCPServerSettings
from fast_agent.mcp.helpers.content_helpers import get_text

JOB_EVENTS_SERVER = "../../../examples/mcp/events/job_events_server.py"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_polled_event_reaches_the_next_model_call(fast_agent) -> None:
    @fast_agent.agent(name="probe", model="passthrough", servers=["job_events"])
    async def run_probe() -> None:
        async with fast_agent.run() as app:
            aggregator = app.probe.aggregator
            assert await aggregator.supports_events("job_events")
            manager = aggregator.events

            event_types = await manager.list_event_types("job_events")
            assert [event.name for event in event_types] == ["job.status_changed"]

            with pytest.raises(ValueError, match="does not offer event"):
                await manager.subscribe("job_events", "job.deleted")

            subscription = await manager.subscribe(
                "job_events", "job.status_changed", {"job_id": "job-1"}, delivery="poll"
            )
            assert subscription.delivery == "poll"

            await app.probe.call_tool("set_job_stage", {"job_id": "job-2", "stage": "RUNNING"})
            await app.probe.call_tool("set_job_stage", {"job_id": "job-1", "stage": "COMPLETED"})
            assert manager.poll_now(subscription.key)
            for _ in range(100):
                if manager.pending_count:
                    break
                await asyncio.sleep(0.05)
            else:
                pytest.fail("the poll did not deliver the event")

            await app.probe.send("any news?")

            user_texts = [
                get_text(content) or ""
                for message in app.probe.message_history
                if message.role == "user"
                for content in message.content
            ]
            event_texts = [text for text in user_texts if "<mcp-event " in text]
            assert len(event_texts) == 1
            assert 'name="job.status_changed"' in event_texts[0]
            assert '"stage": "COMPLETED"' in event_texts[0]
            assert "job-2" not in event_texts[0]
            assert manager.pending_count == 0

    await run_probe()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_pushed_event_arrives_on_the_stream_and_wakes_the_agent(fast_agent) -> None:
    @fast_agent.agent(name="probe", model="passthrough", servers=["job_events"])
    async def run_probe() -> None:
        async with fast_agent.run() as app:
            manager = app.probe.aggregator.events
            with pytest.raises(MCPError):
                await manager.subscribe("job_events", "job.status_changed", {"job_id": 1})

            subscription = await manager.subscribe(
                "job_events", "job.status_changed", {"job_id": "job-1"}, wake=True
            )
            assert subscription.delivery == "push"
            other = await manager.subscribe("job_events", "job.status_changed", {"job_id": "job-2"})

            await app.probe.call_tool("set_job_stage", {"job_id": "job-2", "stage": "RUNNING"})
            await app.probe.call_tool("set_job_stage", {"job_id": "job-1", "stage": "COMPLETED"})
            await asyncio.wait_for(app.probe.wait_for_event_wake(), timeout=10)
            # Each subscription has its own stream, so job-2's event can land after the wake.
            for _ in range(100):
                if manager.pending_count == 2:
                    break
                await asyncio.sleep(0.05)
            else:
                pytest.fail("the job-2 event did not arrive")

            [text] = app.probe.take_event_messages()
            assert text.count("<mcp-event ") == 2
            assert '"stage": "COMPLETED"' in text
            assert '"stage": "RUNNING"' in text
            assert (subscription.delivered, other.delivered) == (1, 1)
            assert subscription.cursor == "1"

            assert await manager.unsubscribe(subscription.key)
            assert await manager.unsubscribe(other.key)

    await run_probe()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_configured_subscriptions_follow_server_attach_and_detach(fast_agent) -> None:
    @fast_agent.agent(name="probe", model="passthrough", servers=["job_events_configured"])
    async def run_probe() -> None:
        async with fast_agent.run() as app:
            manager = app.probe.aggregator.events
            assert manager.subscriptions() == []
            await app.probe.start_configured_events()
            [configured] = manager.subscriptions()
            assert configured.arguments == {"job_id": "job-1"}
            assert configured.wake

            await app.attach_mcp_server(
                "probe",
                "runtime_events",
                MCPServerSettings(
                    command="uv",
                    args=["run", JOB_EVENTS_SERVER],
                    protocol_mode="modern",
                    events=[
                        MCPEventSubscriptionSettings(
                            name="job.status_changed", arguments={"job_id": "job-2"}
                        ),
                        MCPEventSubscriptionSettings(name="job.deleted"),
                    ],
                ),
            )
            # The event type the server does not offer is skipped.
            assert [subscription.server_name for subscription in manager.subscriptions()] == [
                "job_events_configured",
                "runtime_events",
            ]

            await app.probe.call_tool(
                "job_events_configured__set_job_stage", {"job_id": "job-1", "stage": "RUNNING"}
            )
            assert manager.poll_now(configured.key)
            await asyncio.wait_for(app.probe.wait_for_event_wake(), timeout=10)
            [text] = app.probe.take_event_messages()
            assert 'server="job_events_configured"' in text
            assert '"stage": "RUNNING"' in text

            await app.detach_mcp_server("probe", "runtime_events")
            assert manager.subscriptions() == [configured]

    await run_probe()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_configured_subscriptions_start_only_for_the_agent_in_use(fast_agent) -> None:
    @fast_agent.agent(name="probe", model="passthrough", servers=["job_events_configured"])
    @fast_agent.agent(name="unused", model="passthrough", servers=["job_events_configured"])
    async def run_probe() -> None:
        async with fast_agent.run() as app:
            await app.probe.send("hello")

            [configured] = app.probe.aggregator.events.subscriptions()
            assert configured.name == "job.status_changed"
            assert app.unused.aggregator.events.subscriptions() == []

    await run_probe()
