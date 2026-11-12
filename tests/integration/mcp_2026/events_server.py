"""A poll-only MCP Events (SEP-3415) server for integration tests."""

import os
from datetime import UTC, datetime
from typing import Any

from mcp.server.extension import Extension, MethodBinding
from mcp.server.mcpserver import MCPServer
from mcp.shared.exceptions import MCPError
from mcp_types import INVALID_PARAMS, PaginatedRequestParams

from fast_agent.mcp.events_extension import (
    EVENTS_EXTENSION,
    EVENTS_NOT_FOUND,
    EventDescriptor,
    EventOccurrence,
    ListEventsResult,
    PollEventsRequestParams,
    PollEventsResult,
)

EVENT_NAME = "job.status_changed"
POLL_INTERVAL_MS = 50

# job_id -> [(stage, timestamp)], oldest first. The cursor is an index into this list.
job_history: dict[str, list[tuple[str, str]]] = {}


async def list_events(ctx: Any, params: PaginatedRequestParams) -> ListEventsResult:
    del ctx, params
    return ListEventsResult(
        events=[
            EventDescriptor(
                name=EVENT_NAME,
                description="Fires when a job changes stage",
                delivery=["poll"],
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


async def poll_events(ctx: Any, params: PollEventsRequestParams) -> PollEventsResult:
    del ctx
    if params.name != EVENT_NAME:
        raise MCPError(EVENTS_NOT_FOUND, "NotFound", {"kind": "event"})
    job_id = params.arguments.get("job_id")
    if not isinstance(job_id, str):
        raise MCPError(INVALID_PARAMS, "job_id is required")

    history = job_history.get(job_id, [])
    start = len(history) if params.cursor is None else int(params.cursor)
    return PollEventsResult(
        events=[
            EventOccurrence(
                event_id=f"{job_id}:{index}:{stage}",
                name=EVENT_NAME,
                timestamp=timestamp,
                data={"job_id": job_id, "stage": stage},
            )
            for index, (stage, timestamp) in enumerate(history[start:], start=start)
        ],
        cursor=str(len(history)),
        next_poll_ms=POLL_INTERVAL_MS,
    )


class JobEvents(Extension):
    identifier = EVENTS_EXTENSION

    def methods(self) -> list[MethodBinding]:
        return [
            MethodBinding("events/list", PaginatedRequestParams, list_events),
            MethodBinding("events/poll", PollEventsRequestParams, poll_events),
        ]


server = MCPServer("events", extensions=[JobEvents()])


@server.tool()
def set_job_stage(job_id: str, stage: str) -> str:
    """Record a new stage for a job, which fires job.status_changed."""
    timestamp = datetime.now(UTC).isoformat()
    job_history.setdefault(job_id, []).append((stage, timestamp))
    return f"{job_id} is now {stage}"


if __name__ == "__main__":
    server.run(
        transport="streamable-http",
        host="127.0.0.1",
        port=int(os.environ["FAST_AGENT_TEST_HTTP_PORT"]),
    )
