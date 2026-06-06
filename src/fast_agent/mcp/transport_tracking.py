from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from threading import Lock
from typing import TYPE_CHECKING, Literal, cast

from mcp.types import (
    JSONRPCError,
    JSONRPCMessage,
    JSONRPCNotification,
    JSONRPCRequest,
    JSONRPCResponse,
    RequestId,
)
from pydantic import BaseModel, ConfigDict

from fast_agent.utils.text import strip_casefold

if TYPE_CHECKING:
    from collections.abc import Callable

ChannelName = Literal["post-json", "post-sse", "get", "resumption", "stdio"]
EventType = Literal["message", "connect", "disconnect", "keepalive", "error"]
PostChannelName = Literal["post-json", "post-sse"]
PostMode = Literal["json", "sse"]
POST_CHANNEL_MODE_BY_NAME: dict[PostChannelName, PostMode] = {
    "post-json": "json",
    "post-sse": "sse",
}
POST_CHANNEL_NAME_BY_MODE: dict[PostMode, PostChannelName] = {
    mode: channel for channel, mode in POST_CHANNEL_MODE_BY_NAME.items()
}
POST_CHANNEL_NAMES: tuple[PostChannelName, ...] = tuple(POST_CHANNEL_MODE_BY_NAME)
TRACKED_CHANNEL_NAMES: tuple[ChannelName, ...] = (
    *POST_CHANNEL_NAMES,
    "get",
    "resumption",
    "stdio",
)


class ActivityState(StrEnum):
    ERROR = "error"
    DISABLED = "disabled"
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    PING = "ping"
    NONE = "none"


@dataclass(slots=True)
class ChannelEvent:
    """Event emitted by the tracking transport indicating channel activity."""

    channel: ChannelName
    event_type: EventType
    message: JSONRPCMessage | None = None
    raw_event: str | None = None
    detail: str | None = None
    status_code: int | None = None


@dataclass
class MessageCounts:
    request: int = 0
    notification: int = 0
    response: int = 0

    def increment(self, classification: ActivityState) -> None:
        if classification is ActivityState.REQUEST:
            self.request += 1
        elif classification is ActivityState.NOTIFICATION:
            self.notification += 1
        elif classification is ActivityState.RESPONSE:
            self.response += 1


@dataclass
class ModeStats:
    messages: int = 0
    counts: MessageCounts = field(default_factory=MessageCounts)
    last_summary: str | None = None
    last_at: datetime | None = None
    last_error: str | None = None
    last_event: str | None = None
    last_event_at: datetime | None = None

    @property
    def has_activity(self) -> bool:
        return bool(self.messages or self.last_error)


def _summarise_message(message: JSONRPCMessage) -> str:
    root = message.root
    if isinstance(root, JSONRPCRequest):
        method = root.method or ""
        return f"request {method}"
    if isinstance(root, JSONRPCNotification):
        method = root.method or ""
        return f"notify {method}"
    if isinstance(root, JSONRPCResponse):
        return "response"
    if isinstance(root, JSONRPCError):
        code = getattr(root.error, "code", None)
        return f"error {code}" if code is not None else "error"
    return "message"


def _summarise_classified_message(
    classification: ActivityState,
    message: JSONRPCMessage,
) -> str:
    if classification is ActivityState.PING:
        return "ping"
    return _summarise_message(message)


class ChannelSnapshot(BaseModel):
    """Snapshot of aggregated activity for a single transport channel."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    message_count: int = 0
    mode: str | None = None
    mode_counts: dict[str, int] | None = None
    last_message_summary: str | None = None
    last_message_at: datetime | None = None
    connected: bool | None = None
    state: str | None = None
    last_event: str | None = None
    last_event_at: datetime | None = None
    ping_count: int | None = None
    ping_last_at: datetime | None = None
    last_error: str | None = None
    connect_at: datetime | None = None
    disconnect_at: datetime | None = None
    last_status_code: int | None = None
    request_count: int = 0
    response_count: int = 0
    notification_count: int = 0
    activity_buckets: list[str] | None = None
    activity_bucket_seconds: int | None = None
    activity_bucket_count: int | None = None


class TransportSnapshot(BaseModel):
    """Collection of channel snapshots for a transport."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    post: ChannelSnapshot | None = None
    post_json: ChannelSnapshot | None = None
    post_sse: ChannelSnapshot | None = None
    get: ChannelSnapshot | None = None
    resumption: ChannelSnapshot | None = None
    stdio: ChannelSnapshot | None = None
    activity_bucket_seconds: int | None = None
    activity_bucket_count: int | None = None


class TransportChannelMetrics:
    """Aggregates low-level channel events into user-visible metrics."""

    def __init__(
        self,
        bucket_seconds: int | None = None,
        bucket_count: int | None = None,
    ) -> None:
        self._lock = Lock()

        self._init_post_metrics()
        self._init_get_metrics()
        self._init_resumption_metrics()
        self._init_stdio_metrics()
        self._channel_event_handlers: dict[
            ChannelName,
            Callable[[ChannelEvent, datetime], None],
        ] = {
            "post-json": self._handle_post_event,
            "post-sse": self._handle_post_event,
            "get": self._handle_get_event,
            "resumption": self._handle_resumption_event,
            "stdio": self._handle_stdio_event,
        }

        self._response_channel_by_id: dict[RequestId, ChannelName] = {}
        self._ping_request_ids: set[RequestId] = set()

        self._init_history(bucket_seconds, bucket_count)

    def _init_post_metrics(self) -> None:
        self._post_modes: set[str] = set()
        self._post_count = 0
        self._post_counts = MessageCounts()
        self._post_last_summary: str | None = None
        self._post_last_at: datetime | None = None
        self._post_last_error: str | None = None
        self._post_last_event: str | None = None
        self._post_last_event_at: datetime | None = None
        self._post_mode_stats: dict[PostMode, ModeStats] = {
            "json": ModeStats(),
            "sse": ModeStats(),
        }

    def _init_get_metrics(self) -> None:
        self._get_connected = False
        self._get_had_connection = False
        self._get_connect_at: datetime | None = None
        self._get_disconnect_at: datetime | None = None
        self._get_last_summary: str | None = None
        self._get_last_at: datetime | None = None
        self._get_last_event: str | None = None
        self._get_last_event_at: datetime | None = None
        self._get_last_error: str | None = None
        self._get_last_status_code: int | None = None
        self._get_message_count = 0
        self._get_counts = MessageCounts()
        self._get_ping_count = 0
        self._get_last_ping_at: datetime | None = None

    def _init_resumption_metrics(self) -> None:
        self._resumption_count = 0
        self._resumption_last_summary: str | None = None
        self._resumption_last_at: datetime | None = None
        self._resumption_last_error: str | None = None
        self._resumption_last_event: str | None = None
        self._resumption_last_event_at: datetime | None = None
        self._resumption_counts = MessageCounts()

    def _init_stdio_metrics(self) -> None:
        self._stdio_connected = False
        self._stdio_had_connection = False
        self._stdio_connect_at: datetime | None = None
        self._stdio_disconnect_at: datetime | None = None
        self._stdio_count = 0
        self._stdio_last_summary: str | None = None
        self._stdio_last_at: datetime | None = None
        self._stdio_last_event: str | None = None
        self._stdio_last_event_at: datetime | None = None
        self._stdio_last_error: str | None = None
        self._stdio_counts = MessageCounts()

    def _init_history(self, bucket_seconds: int | None, bucket_count: int | None) -> None:
        try:
            seconds = 30 if bucket_seconds is None else int(bucket_seconds)
        except (TypeError, ValueError):
            seconds = 30
        if seconds <= 0:
            seconds = 30

        try:
            count = 20 if bucket_count is None else int(bucket_count)
        except (TypeError, ValueError):
            count = 20
        if count <= 0:
            count = 20

        self._history_bucket_seconds = seconds
        self._history_bucket_count = count
        self._history_priority = {
            ActivityState.ERROR: 5,
            ActivityState.DISABLED: 4,
            ActivityState.REQUEST: 4,
            ActivityState.RESPONSE: 3,
            ActivityState.NOTIFICATION: 2,
            ActivityState.PING: 2,
            ActivityState.NONE: 1,
        }
        self._history: dict[ChannelName, deque[tuple[int, ActivityState]]] = {
            channel: deque(maxlen=self._history_bucket_count) for channel in TRACKED_CHANNEL_NAMES
        }

    def record_event(self, event: ChannelEvent) -> None:
        now = datetime.now(timezone.utc)
        with self._lock:
            self._channel_event_handlers[event.channel](event, now)

    def register_ping_request(self, request_id: RequestId) -> None:
        with self._lock:
            self._ping_request_ids.add(request_id)

    def discard_ping_request(self, request_id: RequestId) -> None:
        with self._lock:
            self._ping_request_ids.discard(request_id)

    def _handle_post_event(self, event: ChannelEvent, now: datetime) -> None:
        mode = POST_CHANNEL_MODE_BY_NAME[cast("PostChannelName", event.channel)]
        if event.event_type == "message" and event.message is not None:
            self._post_modes.add(mode)
            self._post_count += 1

            mode_stats = self._post_mode_stats[mode]
            mode_stats.messages += 1

            classification = self._tally_message_counts(
                "post",
                event.message,
                now,
                sub_mode=mode,
            )

            summary = _summarise_classified_message(classification, event.message)
            mode_stats.last_summary = summary
            mode_stats.last_at = now
            self._post_last_summary = summary
            self._post_last_at = now

            self._record_response_channel(event)
            if classification is not ActivityState.PING:
                self._record_history(event.channel, classification, now)
        elif event.event_type == "error":
            self._post_last_error = event.detail
            self._post_last_event = "error"
            self._post_last_event_at = now
            mode_stats = self._post_mode_stats[mode]
            mode_stats.last_error = event.detail
            mode_stats.last_event = "error"
            mode_stats.last_event_at = now
            self._record_history(event.channel, ActivityState.ERROR, now)

    def _handle_get_event(self, event: ChannelEvent, now: datetime) -> None:
        if event.event_type == "connect":
            self._get_connected = True
            self._get_had_connection = True
            self._get_connect_at = now
            self._get_last_event = "connect"
            self._get_last_event_at = now
            self._get_last_error = None
            self._get_last_status_code = None
        elif event.event_type == "disconnect":
            self._get_connected = False
            self._get_disconnect_at = now
            self._get_last_event = "disconnect"
            self._get_last_event_at = now
        elif event.event_type == "keepalive":
            self._register_ping(now)
            self._get_last_event = event.raw_event or "keepalive"
            self._get_last_event_at = now
            self._record_history("get", ActivityState.PING, now)
        elif event.event_type == "message" and event.message is not None:
            self._get_message_count += 1
            classification = self._tally_message_counts("get", event.message, now)
            summary = _summarise_classified_message(classification, event.message)
            self._get_last_summary = summary
            self._get_last_at = now
            self._get_last_event = "ping" if classification is ActivityState.PING else "message"
            self._get_last_event_at = now

            self._record_response_channel(event)
            self._record_history("get", classification, now)
        elif event.event_type == "error":
            self._get_last_status_code = event.status_code
            self._get_last_error = event.detail
            self._get_last_event = "error"
            self._get_last_event_at = now
            # Record 405 as "disabled" in timeline, not "error"
            timeline_state = (
                ActivityState.DISABLED if event.status_code == 405 else ActivityState.ERROR
            )
            self._record_history("get", timeline_state, now)

    def _handle_resumption_event(self, event: ChannelEvent, now: datetime) -> None:
        if event.event_type == "message" and event.message is not None:
            self._resumption_count += 1
            classification = self._tally_message_counts("resumption", event.message, now)
            summary = _summarise_classified_message(classification, event.message)
            self._resumption_last_summary = summary
            self._resumption_last_at = now

            self._record_response_channel(event)
            self._record_history("resumption", classification, now)
        elif event.event_type == "error":
            self._resumption_last_error = event.detail
            self._resumption_last_event = "error"
            self._resumption_last_event_at = now
            self._record_history("resumption", ActivityState.ERROR, now)

    def _handle_stdio_event(self, event: ChannelEvent, now: datetime) -> None:
        if event.event_type == "connect":
            self._stdio_connected = True
            self._stdio_had_connection = True
            self._stdio_connect_at = now
            self._stdio_last_event = "connect"
            self._stdio_last_event_at = now
            self._stdio_last_error = None
        elif event.event_type == "disconnect":
            self._stdio_connected = False
            self._stdio_disconnect_at = now
            self._stdio_last_event = "disconnect"
            self._stdio_last_event_at = now
        elif event.event_type == "message":
            self._stdio_count += 1

            # Handle synthetic events (from ServerStats) vs real message events
            if event.message is not None:
                # Real message event with JSON-RPC content
                classification = self._tally_message_counts("stdio", event.message, now)
                summary = _summarise_classified_message(classification, event.message)
                self._record_response_channel(event)
            else:
                # Synthetic event from MCP operation activity
                classification = ActivityState.REQUEST
                self._stdio_counts.increment(classification)
                summary = event.detail or "request"

            self._stdio_last_summary = summary
            self._stdio_last_at = now
            self._stdio_last_event = "message"
            self._stdio_last_event_at = now
            self._record_history("stdio", classification, now)
        elif event.event_type == "error":
            self._stdio_last_error = event.detail
            self._stdio_last_event = "error"
            self._stdio_last_event_at = now
            self._record_history("stdio", ActivityState.ERROR, now)

    def _record_response_channel(self, event: ChannelEvent) -> None:
        if event.message is None:
            return
        root = event.message.root
        request_id: RequestId | None = None
        if isinstance(root, (JSONRPCResponse, JSONRPCError)):
            request_id = root.id
        if request_id is None:
            return
        self._response_channel_by_id[request_id] = event.channel

    def consume_response_channel(self, request_id: RequestId | None) -> ChannelName | None:
        if request_id is None:
            return None
        with self._lock:
            return self._response_channel_by_id.pop(request_id, None)

    def _tally_message_counts(
        self,
        channel_key: str,
        message: JSONRPCMessage,
        timestamp: datetime,
        *,
        sub_mode: PostMode | None = None,
    ) -> ActivityState:
        classification = self._classify_message(message)
        root = message.root
        request_id = self._message_request_id(root)
        classification = self._classify_ping_exchange(classification, root, request_id)
        self._tally_classification(channel_key, classification, timestamp, sub_mode=sub_mode)
        return classification

    @staticmethod
    def _message_request_id(
        root: JSONRPCRequest | JSONRPCResponse | JSONRPCError | JSONRPCNotification,
    ) -> RequestId | None:
        if isinstance(root, (JSONRPCRequest, JSONRPCResponse, JSONRPCError)):
            return root.id
        return None

    def _classify_ping_exchange(
        self,
        classification: ActivityState,
        root: JSONRPCRequest | JSONRPCResponse | JSONRPCError | JSONRPCNotification,
        request_id: RequestId | None,
    ) -> ActivityState:
        if request_id is None:
            return classification
        if classification is ActivityState.PING and isinstance(root, JSONRPCRequest):
            self._ping_request_ids.add(request_id)
            return classification
        if classification is ActivityState.RESPONSE and request_id in self._ping_request_ids:
            self._ping_request_ids.discard(request_id)
            return ActivityState.PING
        return classification

    def _tally_classification(
        self,
        channel_key: str,
        classification: ActivityState,
        timestamp: datetime,
        *,
        sub_mode: PostMode | None = None,
    ) -> None:
        if channel_key == "post":
            self._tally_post_classification(classification, sub_mode)
        elif channel_key == "get":
            self._tally_get_classification(classification, timestamp)
        elif channel_key == "resumption":
            self._tally_resumption_classification(classification)
        elif channel_key == "stdio":
            self._tally_stdio_classification(classification)

    def _tally_post_classification(
        self,
        classification: ActivityState,
        sub_mode: PostMode | None,
    ) -> None:
        self._post_counts.increment(classification)

        if sub_mode:
            self._post_mode_stats[sub_mode].counts.increment(classification)

    def _tally_get_classification(
        self,
        classification: ActivityState,
        timestamp: datetime,
    ) -> None:
        if classification is ActivityState.PING:
            self._register_ping(timestamp)
        else:
            self._get_counts.increment(classification)

    def _tally_resumption_classification(self, classification: ActivityState) -> None:
        self._resumption_counts.increment(classification)

    def _tally_stdio_classification(self, classification: ActivityState) -> None:
        self._stdio_counts.increment(classification)

    def _register_ping(self, timestamp: datetime) -> None:
        self._get_ping_count += 1
        self._get_last_ping_at = timestamp

    def _classify_message(self, message: JSONRPCMessage | None) -> ActivityState:
        if message is None:
            return ActivityState.NONE
        root = message.root
        method = getattr(root, "method", "")
        normalized_method = strip_casefold(method) if isinstance(method, str) else ""

        if isinstance(root, JSONRPCRequest):
            return self._classify_method_message(normalized_method, ActivityState.REQUEST)
        if isinstance(root, JSONRPCNotification):
            return self._classify_method_message(normalized_method, ActivityState.NOTIFICATION)
        if isinstance(root, (JSONRPCResponse, JSONRPCError)):
            return ActivityState.RESPONSE
        return ActivityState.NONE

    def _classify_method_message(
        self,
        method_lower: str,
        fallback: ActivityState,
    ) -> ActivityState:
        if self._is_ping_method(method_lower):
            return ActivityState.PING
        return fallback

    @staticmethod
    def _is_ping_method(method: str) -> bool:
        if not method:
            return False
        return method == "ping" or method.endswith(("/ping", ".ping"))

    def _record_history(
        self,
        channel: ChannelName,
        state: ActivityState,
        timestamp: datetime,
    ) -> None:
        if state is ActivityState.NONE:
            return
        history = self._history.get(channel)
        if history is None:
            return

        bucket = int(timestamp.timestamp() // self._history_bucket_seconds)
        if history and history[-1][0] == bucket:
            existing = history[-1][1]
            if self._history_priority.get(state, 0) >= self._history_priority.get(existing, 0):
                history[-1] = (bucket, state)
            return

        while history and bucket - history[0][0] >= self._history_bucket_count:
            history.popleft()

        history.append((bucket, state))

    def _build_activity_buckets(self, key: ChannelName, now: datetime) -> list[str]:
        history = self._history.get(key)
        if not history:
            return ["none"] * self._history_bucket_count

        history_map = dict(history)
        current_bucket = int(now.timestamp() // self._history_bucket_seconds)
        buckets: list[str] = []
        for offset in range(self._history_bucket_count - 1, -1, -1):
            bucket_index = current_bucket - offset
            buckets.append(history_map.get(bucket_index, ActivityState.NONE).value)
        return buckets

    def _merge_activity_buckets(
        self,
        keys: tuple[ChannelName, ...],
        now: datetime,
    ) -> list[str] | None:
        sequences = [self._build_activity_buckets(key, now) for key in keys if key in self._history]
        if not sequences:
            return None

        merged: list[str] = []
        for idx in range(self._history_bucket_count):
            best_state = ActivityState.NONE
            best_priority = 0
            for seq in sequences:
                state = ActivityState(seq[idx])
                priority = self._history_priority.get(state, 0)
                if priority > best_priority:
                    best_state = state
                    best_priority = priority
            merged.append(best_state.value)

        if all(state == "none" for state in merged):
            return None
        return merged

    def _build_post_mode_snapshot(self, mode: PostMode, now: datetime) -> ChannelSnapshot | None:
        stats = self._post_mode_stats[mode]
        if not stats.has_activity:
            return None
        return ChannelSnapshot(
            message_count=stats.messages,
            mode=mode,
            state="error" if stats.last_error else None,
            request_count=stats.counts.request,
            response_count=stats.counts.response,
            notification_count=stats.counts.notification,
            last_message_summary=stats.last_summary,
            last_message_at=stats.last_at,
            last_error=stats.last_error,
            last_event=stats.last_event,
            last_event_at=stats.last_event_at,
            activity_buckets=self._build_activity_buckets(POST_CHANNEL_NAME_BY_MODE[mode], now),
            activity_bucket_seconds=self._history_bucket_seconds,
            activity_bucket_count=self._history_bucket_count,
        )

    def snapshot(self) -> TransportSnapshot:
        with self._lock:
            if not self._has_snapshot_activity():
                return TransportSnapshot()

            now = datetime.now(timezone.utc)
            return TransportSnapshot(
                post=self._build_post_snapshot(now),
                post_json=self._build_post_mode_snapshot("json", now),
                post_sse=self._build_post_mode_snapshot("sse", now),
                get=self._build_get_snapshot(now),
                resumption=self._build_resumption_snapshot(now),
                stdio=self._build_stdio_snapshot(now),
                activity_bucket_seconds=self._history_bucket_seconds,
                activity_bucket_count=self._history_bucket_count,
            )

    def _has_snapshot_activity(self) -> bool:
        return bool(
            self._has_post_snapshot_activity()
            or self._has_get_snapshot_activity()
            or self._has_resumption_snapshot_activity()
            or self._has_stdio_snapshot_activity()
        )

    def _has_post_snapshot_activity(self) -> bool:
        return bool(self._post_count or self._post_last_error)

    def _build_post_snapshot(self, now: datetime) -> ChannelSnapshot | None:
        if not self._has_post_snapshot_activity():
            return None

        post_mode_counts: dict[str, int] = {
            mode: stats.messages for mode, stats in self._post_mode_stats.items() if stats.messages
        }
        return ChannelSnapshot(
            message_count=self._post_count,
            mode=self._post_mode(),
            state="error" if self._post_last_error else None,
            mode_counts=post_mode_counts or None,
            last_message_summary=self._post_last_summary,
            last_message_at=self._post_last_at,
            last_error=self._post_last_error,
            last_event=self._post_last_event,
            last_event_at=self._post_last_event_at,
            request_count=self._post_counts.request,
            response_count=self._post_counts.response,
            notification_count=self._post_counts.notification,
            activity_buckets=self._merge_activity_buckets(POST_CHANNEL_NAMES, now),
            activity_bucket_seconds=self._history_bucket_seconds,
            activity_bucket_count=self._history_bucket_count,
        )

    def _post_mode(self) -> str | None:
        if not self._post_modes:
            return None
        if len(self._post_modes) == 1:
            return next(iter(self._post_modes))
        return "mixed"

    def _build_get_snapshot(self, now: datetime) -> ChannelSnapshot | None:
        if not self._has_get_snapshot_activity():
            return None
        return ChannelSnapshot(
            connected=self._get_connected,
            state=self._get_state(),
            connect_at=self._get_connect_at,
            disconnect_at=self._get_disconnect_at,
            message_count=self._get_message_count,
            last_message_summary=self._get_last_summary,
            last_message_at=self._get_last_at,
            ping_count=self._get_ping_count,
            ping_last_at=self._get_last_ping_at,
            last_error=self._get_last_error,
            last_event=self._get_last_event,
            last_event_at=self._get_last_event_at,
            last_status_code=self._get_last_status_code,
            request_count=self._get_counts.request,
            response_count=self._get_counts.response,
            notification_count=self._get_counts.notification,
            activity_buckets=self._build_activity_buckets("get", now),
            activity_bucket_seconds=self._history_bucket_seconds,
            activity_bucket_count=self._history_bucket_count,
        )

    def _has_get_snapshot_activity(self) -> bool:
        return bool(
            self._get_message_count
            or self._get_ping_count
            or self._get_connected
            or self._get_disconnect_at
            or self._get_last_error
        )

    def _get_state(self) -> str:
        if self._get_connected:
            return "open"
        if self._get_last_error is not None:
            return "disabled" if self._get_last_status_code == 405 else "error"
        if self._get_had_connection:
            return "off"
        return "idle"

    def _build_resumption_snapshot(self, now: datetime) -> ChannelSnapshot | None:
        if not self._has_resumption_snapshot_activity():
            return None
        return ChannelSnapshot(
            message_count=self._resumption_count,
            state="error" if self._resumption_last_error else None,
            last_message_summary=self._resumption_last_summary,
            last_message_at=self._resumption_last_at,
            last_error=self._resumption_last_error,
            last_event=self._resumption_last_event,
            last_event_at=self._resumption_last_event_at,
            request_count=self._resumption_counts.request,
            response_count=self._resumption_counts.response,
            notification_count=self._resumption_counts.notification,
            activity_buckets=self._build_activity_buckets("resumption", now),
            activity_bucket_seconds=self._history_bucket_seconds,
            activity_bucket_count=self._history_bucket_count,
        )

    def _has_resumption_snapshot_activity(self) -> bool:
        return bool(self._resumption_count or self._resumption_last_error)

    def _build_stdio_snapshot(self, now: datetime) -> ChannelSnapshot | None:
        if not self._has_stdio_snapshot_activity():
            return None
        return ChannelSnapshot(
            connected=self._stdio_connected,
            state=self._stdio_state(),
            connect_at=self._stdio_connect_at,
            disconnect_at=self._stdio_disconnect_at,
            message_count=self._stdio_count,
            last_message_summary=self._stdio_last_summary,
            last_message_at=self._stdio_last_at,
            last_error=self._stdio_last_error,
            last_event=self._stdio_last_event,
            last_event_at=self._stdio_last_event_at,
            request_count=self._stdio_counts.request,
            response_count=self._stdio_counts.response,
            notification_count=self._stdio_counts.notification,
            activity_buckets=self._build_activity_buckets("stdio", now),
            activity_bucket_seconds=self._history_bucket_seconds,
            activity_bucket_count=self._history_bucket_count,
        )

    def _has_stdio_snapshot_activity(self) -> bool:
        return bool(
            self._stdio_count
            or self._stdio_connected
            or self._stdio_disconnect_at
            or self._stdio_last_error
        )

    def _stdio_state(self) -> str:
        if self._stdio_connected:
            return "open"
        if self._stdio_last_error is not None:
            return "error"
        if self._stdio_had_connection:
            return "off"
        return "idle"
