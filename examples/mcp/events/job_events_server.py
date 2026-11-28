"""An MCP Events (SEP-3415) server with one event type, job.status_changed, by push or poll."""

import asyncio
from datetime import UTC, datetime
from typing import Any

from mcp.server.extension import Extension, MethodBinding
from mcp.server.mcpserver import MCPServer
from mcp.shared.exceptions import MCPError
from mcp.shared.subscriptions import SUBSCRIPTION_ID_META_KEY
from mcp_types import INVALID_PARAMS, Notification, PaginatedRequestParams
from pydantic import BaseModel

from fast_agent.mcp.events_extension import (
    EVENTS_EXTENSION,
    EVENTS_NOT_FOUND,
    EventDescriptor,
    EventOccurrence,
    EventStreamActive,
    EventStreamHeartbeat,
    ListEventsResult,
    PollEventsRequestParams,
    PollEventsResult,
    StreamEventsRequestParams,
    StreamEventsResult,
)

EVENT_NAME = "job.status_changed"
POLL_INTERVAL_MS = 1_000
HEARTBEAT_MS = 15_000
STREAM_CHECK_SECONDS = 0.05

# job_id -> [(stage, timestamp)], oldest first. The cursor is an index into this list.
job_history: dict[str, list[tuple[str, str]]] = {}


async def list_events(ctx: Any, params: PaginatedRequestParams) -> ListEventsResult:
    del ctx, params
    return ListEventsResult(
        events=[
            EventDescriptor(
                name=EVENT_NAME,
                description="Fires when a job changes stage",
                delivery=["push", "poll"],
                input_schema={
                    "type": "object",
                    "properties": {"job_id": {"type": "string"}},
                    "required": ["job_id"],
                },
                payload_schema={
                    "type": "object",
                    "properties": {"job_id": {"type": "string"}, "stage": {"type": "string"}},
                },
            )
        ]
    )


def _subscription(params: PollEventsRequestParams | StreamEventsRequestParams) -> tuple[str, int]:
    """Check a subscription and return its job and the position it resumes from."""
    if params.name != EVENT_NAME:
        raise MCPError(EVENTS_NOT_FOUND, "NotFound", {"kind": "event"})
    job_id = params.arguments.get("job_id")
    if not isinstance(job_id, str):
        raise MCPError(INVALID_PARAMS, "job_id is required")
    if params.cursor is None:
        return job_id, len(job_history.get(job_id, []))
    return job_id, int(params.cursor)


def _occurrences(job_id: str, start: int) -> list[EventOccurrence]:
    return [
        EventOccurrence(
            event_id=f"{job_id}:{index}:{stage}",
            name=EVENT_NAME,
            timestamp=timestamp,
            data={"job_id": job_id, "stage": stage},
            cursor=str(index + 1),
        )
        for index, (stage, timestamp) in enumerate(job_history.get(job_id, [])[start:], start=start)
    ]


async def poll_events(ctx: Any, params: PollEventsRequestParams) -> PollEventsResult:
    del ctx
    job_id, start = _subscription(params)
    return PollEventsResult(
        events=_occurrences(job_id, start),
        cursor=str(len(job_history.get(job_id, []))),
        next_poll_ms=POLL_INTERVAL_MS,
    )


async def stream_events(ctx: Any, params: StreamEventsRequestParams) -> StreamEventsResult:
    job_id, position = _subscription(params)
    heartbeat_ms = params.heartbeat_ms or HEARTBEAT_MS
    meta = {SUBSCRIPTION_ID_META_KEY: ctx.request_id}
    loop = asyncio.get_running_loop()

    async def send(kind: str, frame: BaseModel) -> None:
        notification = Notification[dict[str, Any], str](
            method=f"notifications/events/{kind}",
            params=frame.model_dump(by_alias=True, exclude_none=True) | {"_meta": meta},
        )
        await ctx.session.send_notification(notification, related_request_id=ctx.request_id)

    await send("active", EventStreamActive(cursor=str(position), heartbeat_ms=heartbeat_ms))
    last_sent = loop.time()
    # The stream runs until the client cancels the request.
    while True:
        for occurrence in _occurrences(job_id, position):
            await send("event", occurrence)
            position += 1
            last_sent = loop.time()
        if loop.time() - last_sent >= heartbeat_ms / 1000:
            await send("heartbeat", EventStreamHeartbeat(cursor=str(position)))
            last_sent = loop.time()
        await asyncio.sleep(STREAM_CHECK_SECONDS)


class JobEvents(Extension):
    identifier = EVENTS_EXTENSION

    def methods(self) -> list[MethodBinding]:
        return [
            MethodBinding("events/list", PaginatedRequestParams, list_events),
            MethodBinding("events/poll", PollEventsRequestParams, poll_events),
            MethodBinding("events/stream", StreamEventsRequestParams, stream_events),
        ]


server = MCPServer("events", extensions=[JobEvents()])


@server.tool()
def set_job_stage(job_id: str, stage: str) -> str:
    """Record a new stage for a job, which fires job.status_changed."""
    timestamp = datetime.now(UTC).isoformat()
    job_history.setdefault(job_id, []).append((stage, timestamp))
    return f"{job_id} is now {stage}"


if __name__ == "__main__":
    server.run(transport="stdio")
