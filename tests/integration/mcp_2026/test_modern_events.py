import asyncio
import os
import subprocess

import pytest

from fast_agent.mcp.events_subscriptions import EventSubscriptionManager
from fast_agent.mcp.helpers.content_helpers import get_text


@pytest.mark.integration
@pytest.mark.asyncio
async def test_polled_event_reaches_the_next_model_call(
    fast_agent,
    mcp_test_ports,
    wait_for_port,
) -> None:
    process = subprocess.Popen(
        ["uv", "run", "events_server.py"],
        cwd=os.path.dirname(__file__),
        env=os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        await wait_for_port("127.0.0.1", mcp_test_ports["http"], process=process, timeout=15.0)

        @fast_agent.agent(name="probe", model="passthrough", servers=["modern_http"])
        async def run_probe() -> None:
            async with fast_agent.run() as app:
                aggregator = app.probe._aggregator
                assert await aggregator.supports_events("modern_http")

                # The server asks for 50 ms polls; lower the client floor to match.
                aggregator._events = EventSubscriptionManager(aggregator, min_poll_ms=50)
                manager = aggregator.events

                event_types = await manager.list_event_types("modern_http")
                assert [event.name for event in event_types] == ["job.status_changed"]

                with pytest.raises(ValueError, match="does not offer event"):
                    await manager.subscribe("modern_http", "job.deleted")

                subscription = await manager.subscribe(
                    "modern_http", "job.status_changed", {"job_id": "job-1"}
                )
                assert subscription.cursor == "0"

                await app.probe.call_tool("set_job_stage", {"job_id": "job-2", "stage": "RUNNING"})
                await app.probe.call_tool(
                    "set_job_stage", {"job_id": "job-1", "stage": "COMPLETED"}
                )
                for _ in range(100):
                    if manager.pending_count:
                        break
                    await asyncio.sleep(0.05)
                else:
                    pytest.fail("the poll loop did not deliver the event")

                # Delivered but not yet consumed.
                assert subscription.cursor == "0"

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
                assert subscription.cursor == "1"

                assert await manager.unsubscribe(subscription.key)

        await run_probe()
    finally:
        process.terminate()
        process.communicate(timeout=5)
