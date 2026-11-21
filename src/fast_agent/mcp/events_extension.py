"""Local wire models for the MCP Events extension (SEP-3415).

The pinned MCP SDK does not yet provide Events Extension request/result types.
Discovery, poll delivery and push delivery are modelled.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from mcp_types import (
    ErrorData,
    PaginatedRequestParams,
    PaginatedResult,
    Request,
    RequestParams,
    Result,
    ServerCapabilities,
)
from pydantic import BaseModel, ConfigDict, Field

EVENTS_EXTENSION = "io.modelcontextprotocol/events"

EVENTS_NOT_FOUND = -32023
EVENTS_FORBIDDEN = -32024
EVENTS_RESOURCE_EXHAUSTED = -32025
EVENTS_UNSUPPORTED = -32026
EVENTS_CALLBACK_ENDPOINT_ERROR = -32027

type DeliveryMode = Literal["poll", "push", "webhook"]


def server_supports_mcp_events(capabilities: ServerCapabilities | None) -> bool:
    """Whether the server declared the Events extension."""
    if capabilities is None or not isinstance(capabilities.extensions, Mapping):
        return False
    return isinstance(capabilities.extensions.get(EVENTS_EXTENSION), Mapping)


class EventDescriptor(BaseModel):
    """An event type a server offers for subscription."""

    model_config = ConfigDict(populate_by_name=True)

    name: str
    description: str | None = None
    delivery: list[str]
    input_schema: dict[str, Any] | None = Field(default=None, alias="inputSchema")
    payload_schema: dict[str, Any] | None = Field(default=None, alias="payloadSchema")
    meta: dict[str, Any] | None = Field(default=None, alias="_meta")

    def supports(self, mode: DeliveryMode) -> bool:
        return mode in self.delivery


class EventOccurrence(BaseModel):
    """One delivered event."""

    model_config = ConfigDict(populate_by_name=True)

    event_id: str = Field(alias="eventId")
    name: str
    timestamp: str
    data: dict[str, Any]
    cursor: str | None = None
    meta: dict[str, Any] | None = Field(default=None, alias="_meta")


class ListEventsRequestParams(PaginatedRequestParams):
    """Parameters for the ``events/list`` request."""


class ListEventsRequest(Request[ListEventsRequestParams, Literal["events/list"]]):
    """Request for the SEP-3415 event type listing."""

    method: Literal["events/list"] = "events/list"
    params: ListEventsRequestParams


class ListEventsResult(PaginatedResult):
    """The paginated response to ``events/list``."""

    events: list[EventDescriptor]
    result_type: Literal["complete"] = Field(default="complete", alias="resultType")


class _SubscriptionRequestParams(RequestParams):
    """The fields ``events/poll`` and ``events/stream`` share.

    ``cursor`` of ``None`` means "start from now" and is omitted on the wire.
    """

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    cursor: str | None = None
    max_age_ms: int | None = Field(default=None, alias="maxAgeMs")


class PollEventsRequestParams(_SubscriptionRequestParams):
    """Parameters for the ``events/poll`` request."""

    max_events: int | None = Field(default=None, alias="maxEvents")


class PollEventsRequest(Request[PollEventsRequestParams, Literal["events/poll"]]):
    """Request for one subscription's events since a cursor."""

    method: Literal["events/poll"] = "events/poll"
    params: PollEventsRequestParams
    name_param = "name"


class PollEventsResult(Result):
    """The response to ``events/poll``."""

    events: list[EventOccurrence] = Field(default_factory=list)
    cursor: str | None = None
    truncated: bool = False
    has_more: bool = Field(default=False, alias="hasMore")
    next_poll_ms: int | None = Field(default=None, alias="nextPollMs")
    result_type: Literal["complete"] = Field(default="complete", alias="resultType")


class StreamEventsRequestParams(_SubscriptionRequestParams):
    """Parameters for the ``events/stream`` request."""

    heartbeat_ms: int | None = Field(default=None, alias="heartbeatMs")


class StreamEventsRequest(Request[StreamEventsRequestParams, Literal["events/stream"]]):
    """Request that holds one push subscription open."""

    method: Literal["events/stream"] = "events/stream"
    params: StreamEventsRequestParams
    name_param = "name"


class StreamEventsResult(Result):
    """The empty result a server sends when it closes an ``events/stream`` request."""

    result_type: Literal["complete"] = Field(default="complete", alias="resultType")


class _StreamFrame(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    meta: dict[str, Any] | None = Field(default=None, alias="_meta")


class EventStreamActive(_StreamFrame):
    """``notifications/events/active``: the stream is confirmed, or it skipped events."""

    cursor: str | None = None
    truncated: bool = False
    heartbeat_ms: int | None = Field(default=None, alias="heartbeatMs")


class EventStreamHeartbeat(_StreamFrame):
    """``notifications/events/heartbeat``: the stream is alive at this cursor."""

    cursor: str | None = None


class EventStreamError(_StreamFrame):
    """``notifications/events/error``: one occurrence failed and the stream stays open."""

    error: ErrorData


class EventStreamTerminated(_StreamFrame):
    """``notifications/events/terminated``: the server ended the subscription."""

    error: ErrorData


type EventStreamFrame = (
    EventStreamActive
    | EventOccurrence
    | EventStreamHeartbeat
    | EventStreamError
    | EventStreamTerminated
)

EVENT_STREAM_NOTIFICATIONS: dict[str, type[EventStreamFrame]] = {
    "notifications/events/active": EventStreamActive,
    "notifications/events/event": EventOccurrence,
    "notifications/events/heartbeat": EventStreamHeartbeat,
    "notifications/events/error": EventStreamError,
    "notifications/events/terminated": EventStreamTerminated,
}
