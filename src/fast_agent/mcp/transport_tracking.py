from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Lock
from typing import Literal

from mcp.types import (
    JSONRPCError,
    JSONRPCMessage,
    JSONRPCNotification,
    JSONRPCRequest,
    JSONRPCResponse,
    RequestId,
)
from pydantic import BaseModel, ConfigDict

ChannelName = Literal["post-json", "post-sse", "get", "resumption", "stdio"]
EventType = Literal["message", "connect", "disconnect", "keepalive", "error"]


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
class ModeStats:
    messages: int = 0
    request: int = 0
    notification: int = 0
    response: int = 0
    last_summary: str | None = None
    last_at: datetime | None = None


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

        self._post_modes: set[str] = set()
        self._post_count = 0
        self._post_request_count = 0
        self._post_response_count = 0
        self._post_notification_count = 0
        self._post_last_summary: str | None = None
        self._post_last_at: datetime | None = None
        self._post_mode_stats: dict[str, ModeStats] = {
            "json": ModeStats(),
            "sse": ModeStats(),
        }

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
        self._get_request_count = 0
        self._get_response_count = 0
        self._get_notification_count = 0
        self._get_ping_count = 0
        self._get_last_ping_at: datetime | None = None

        self._resumption_count = 0
        self._resumption_last_summary: str | None = None
        self._resumption_last_at: datetime | None = None
        self._resumption_request_count = 0
        self._resumption_response_count = 0
        self._resumption_notification_count = 0

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
        self._stdio_request_count = 0
        self._stdio_response_count = 0
        self._stdio_notification_count = 0

        self._response_channel_by_id: dict[RequestId, ChannelName] = {}

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
            "error": 5,
            "disabled": 4,
            "request": 4,
            "response": 3,
            "notification": 2,
            "ping": 2,
            "none": 1,
        }
        self._history: dict[str, deque[tuple[int, str]]] = {
            "post-json": deque(maxlen=self._history_bucket_count),
            "post-sse": deque(maxlen=self._history_bucket_count),
            "get": deque(maxlen=self._history_bucket_count),
            "resumption": deque(maxlen=self._history_bucket_count),
            "stdio": deque(maxlen=self._history_bucket_count),
        }

    def record_event(self, event: ChannelEvent) -> None:
        now = datetime.now(timezone.utc)
        with self._lock:
            if event.channel in ("post-json", "post-sse"):
                self._handle_post_event(event, now)
            elif event.channel == "get":
                self._handle_get_event(event, now)
            elif event.channel == "resumption":
                self._handle_resumption_event(event, now)
            elif event.channel == "stdio":
                self._handle_stdio_event(event, now)

    def _handle_post_event(self, event: ChannelEvent, now: datetime) -> None:
        mode = "json" if event.channel == "post-json" else "sse"
        if event.event_type == "message" and event.message is not None:
            self._post_modes.add(mode)
            self._post_count += 1

            mode_stats = self._post_mode_stats[mode]
            mode_stats.messages += 1

            classification = self._tally_message_counts("post", event.message, now, sub_mode=mode)

            summary = "ping" if classification == "ping" else _summarise_message(event.message)
            mode_stats.last_summary = summary
            mode_stats.last_at = now
            self._post_last_summary = summary
            self._post_last_at = now

            self._record_response_channel(event)
            self._record_history(event.channel, classification, now)
        elif event.event_type == "error":
            self._record_history(event.channel, "error", now)

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
            self._record_history("get", "ping", now)
        elif event.event_type == "message" and event.message is not None:
            self._get_message_count += 1
            classification = self._tally_message_counts("get", event.message, now)
            summary = "ping" if classification == "ping" else _summarise_message(event.message)
            self._get_last_summary = summary
            self._get_last_at = now
            self._get_last_event = "ping" if classification == "ping" else "message"
            self._get_last_event_at = now

            self._record_response_channel(event)
            self._record_history("get", classification, now)
        elif event.event_type == "error":
            self._get_last_status_code = event.status_code
            self._get_last_error = event.detail
            self._get_last_event = "error"
            self._get_last_event_at = now
            # Record 405 as "disabled" in timeline, not "error"
            timeline_state = "disabled" if event.status_code == 405 else "error"
            self._record_history("get", timeline_state, now)

    def _handle_resumption_event(self, event: ChannelEvent, now: datetime) -> None:
        if event.event_type == "message" and event.message is not None:
            self._resumption_count += 1
            classification = self._tally_message_counts("resumption", event.message, now)
            summary = "ping" if classification == "ping" else _summarise_message(event.message)
            self._resumption_last_summary = summary
            self._resumption_last_at = now

            self._record_response_channel(event)
            self._record_history("resumption", classification, now)
        elif event.event_type == "error":
            self._record_history("resumption", "error", now)

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
                summary = "ping" if classification == "ping" else _summarise_message(event.message)
                self._record_response_channel(event)
            else:
                # Synthetic event from MCP operation activity
                classification = "request"  # MCP operations are always requests from client perspective
                self._stdio_request_count += 1
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
            self._record_history("stdio", "error", now)

    def _record_response_channel(self, event: ChannelEvent) -> None:
        if event.message is None:
            return
        root = event.message.root
        request_id: RequestId | None = None
        if isinstance(root, (JSONRPCResponse, JSONRPCError, JSONRPCRequest)):
            request_id = getattr(root, "id", None)
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
        sub_mode: str | None = None,
    ) -> str:
        classification = self._classify_message(message)

        if channel_key == "post":
            if classification == "request":
                self._post_request_count += 1
            elif classification == "notification":
                self._post_notification_count += 1
            elif classification == "response":
                self._post_response_count += 1

            if sub_mode:
                stats = self._post_mode_stats[sub_mode]
                if classification in {"request", "notification", "response"}:
                    setattr(stats, classification, getattr(stats, classification) + 1)
        elif channel_key == "get":
            if classification == "ping":
                self._register_ping(timestamp)
            elif classification == "request":
                self._get_request_count += 1
            elif classification == "notification":
                self._get_notification_count += 1
            elif classification == "response":
                self._get_response_count += 1
        elif channel_key == "resumption":
            if classification == "request":
                self._resumption_request_count += 1
            elif classification == "notification":
                self._resumption_notification_count += 1
            elif classification == "response":
                self._resumption_response_count += 1
        elif channel_key == "stdio":
            if classification == "request":
                self._stdio_request_count += 1
            elif classification == "notification":
                self._stdio_notification_count += 1
            elif classification == "response":
                self._stdio_response_count += 1

        return classification

    def _register_ping(self, timestamp: datetime) -> None:
        self._get_ping_count += 1
        self._get_last_ping_at = timestamp

    def _classify_message(self, message: JSONRPCMessage | None) -> str:
        if message is None:
            return "none"
        root = message.root
        method = getattr(root, "method", "")
        method_lower = method.lower() if isinstance(method, str) else ""

        if isinstance(root, JSONRPCRequest):
            if self._is_ping_method(method_lower):
                return "ping"
            return "request"
        if isinstance(root, JSONRPCNotification):
            if self._is_ping_method(method_lower):
                return "ping"
            return "notification"
        if isinstance(root, (JSONRPCResponse, JSONRPCError)):
            return "response"
        return "none"

    @staticmethod
    def _is_ping_method(method: str) -> bool:
        if not method:
            return False
        return (
            method == "ping"
            or method.endswith("/ping")
            or method.endswith(".ping")
        )

    def _record_history(self, channel: str, state: str, timestamp: datetime) -> None:
        if state in {"none", ""}:
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

    def _build_activity_buckets(self, key: str, now: datetime) -> list[str]:
        history = self._history.get(key)
        if not history:
            return ["none"] * self._history_bucket_count

        history_map = {bucket: state for bucket, state in history}
        current_bucket = int(now.timestamp() // self._history_bucket_seconds)
        buckets: list[str] = []
        for offset in range(self._history_bucket_count - 1, -1, -1):
            bucket_index = current_bucket - offset
            buckets.append(history_map.get(bucket_index, "none"))
        return buckets

    def _merge_activity_buckets(self, keys: list[str], now: datetime) -> list[str] | None:
        sequences = [self._build_activity_buckets(key, now) for key in keys if key in self._history]
        if not sequences:
            return None

        merged: list[str] = []
        for idx in range(self._history_bucket_count):
            best_state = "none"
            best_priority = 0
            for seq in sequences:
                state = seq[idx]
                priority = self._history_priority.get(state, 0)
                if priority > best_priority:
                    best_state = state
                    best_priority = priority
            merged.append(best_state)

        if all(state == "none" for state in merged):
            return None
        return merged

    def _build_post_mode_snapshot(self, mode: str, now: datetime) -> ChannelSnapshot | None:
        stats = self._post_mode_stats[mode]
        if stats.messages == 0:
            return None
        return ChannelSnapshot(
            message_count=stats.messages,
            mode=mode,
            request_count=stats.request,
            response_count=stats.response,
            notification_count=stats.notification,
            last_message_summary=stats.last_summary,
            last_message_at=stats.last_at,
            activity_buckets=self._build_activity_buckets(f"post-{mode}", now),
            activity_bucket_seconds=self._history_bucket_seconds,
            activity_bucket_count=self._history_bucket_count,
        )

    def snapshot(self) -> TransportSnapshot:
        with self._lock:
            if (
                not self._post_count
                and not self._get_message_count
                and not self._get_ping_count
                and not self._resumption_count
                and not self._stdio_count
                and not self._get_connected
                and not self._stdio_connected
            ):
                return TransportSnapshot()

            now = datetime.now(timezone.utc)

            post_mode_counts = {
                mode: stats.messages
                for mode, stats in self._post_mode_stats.items()
                if stats.messages
            }
            post_snapshot = None
            if self._post_count:
                if len(self._post_modes) == 0:
                    mode = None
                elif len(self._post_modes) == 1:
                    mode = next(iter(self._post_modes))
                else:
                    mode = "mixed"
                post_snapshot = ChannelSnapshot(
                    message_count=self._post_count,
                    mode=mode,
                    mode_counts=post_mode_counts or None,
                    last_message_summary=self._post_last_summary,
                    last_message_at=self._post_last_at,
                    request_count=self._post_request_count,
                    response_count=self._post_response_count,
                    notification_count=self._post_notification_count,
                    activity_buckets=self._merge_activity_buckets(["post-json", "post-sse"], now),
                    activity_bucket_seconds=self._history_bucket_seconds,
                    activity_bucket_count=self._history_bucket_count,
                )

            post_json_snapshot = self._build_post_mode_snapshot("json", now)
            post_sse_snapshot = self._build_post_mode_snapshot("sse", now)

            get_snapshot = None
            if (
                self._get_message_count
                or self._get_ping_count
                or self._get_connected
                or self._get_disconnect_at
                or self._get_last_error
            ):
                if self._get_connected:
                    state = "open"
                elif self._get_last_error is not None:
                    state = "disabled" if self._get_last_status_code == 405 else "error"
                elif self._get_had_connection:
                    state = "off"
                else:
                    state = "idle"

                get_snapshot = ChannelSnapshot(
                    connected=self._get_connected,
                    state=state,
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
                    request_count=self._get_request_count,
                    response_count=self._get_response_count,
                    notification_count=self._get_notification_count,
                    activity_buckets=self._build_activity_buckets("get", now),
                    activity_bucket_seconds=self._history_bucket_seconds,
                    activity_bucket_count=self._history_bucket_count,
                )

            resumption_snapshot = None
            if self._resumption_count:
                resumption_snapshot = ChannelSnapshot(
                    message_count=self._resumption_count,
                    last_message_summary=self._resumption_last_summary,
                    last_message_at=self._resumption_last_at,
                    request_count=self._resumption_request_count,
                    response_count=self._resumption_response_count,
                    notification_count=self._resumption_notification_count,
                    activity_buckets=self._build_activity_buckets("resumption", now),
                    activity_bucket_seconds=self._history_bucket_seconds,
                    activity_bucket_count=self._history_bucket_count,
                )

            stdio_snapshot = None
            if (
                self._stdio_count
                or self._stdio_connected
                or self._stdio_disconnect_at
                or self._stdio_last_error
            ):
                if self._stdio_connected:
                    state = "open"
                elif self._stdio_last_error is not None:
                    state = "error"
                elif self._stdio_had_connection:
                    state = "off"
                else:
                    state = "idle"

                stdio_snapshot = ChannelSnapshot(
                    connected=self._stdio_connected,
                    state=state,
                    connect_at=self._stdio_connect_at,
                    disconnect_at=self._stdio_disconnect_at,
                    message_count=self._stdio_count,
                    last_message_summary=self._stdio_last_summary,
                    last_message_at=self._stdio_last_at,
                    last_error=self._stdio_last_error,
                    last_event=self._stdio_last_event,
                    last_event_at=self._stdio_last_event_at,
                    request_count=self._stdio_request_count,
                    response_count=self._stdio_response_count,
                    notification_count=self._stdio_notification_count,
                    activity_buckets=self._build_activity_buckets("stdio", now),
                    activity_bucket_seconds=self._history_bucket_seconds,
                    activity_bucket_count=self._history_bucket_count,
                )

            return TransportSnapshot(
                post=post_snapshot,
                post_json=post_json_snapshot,
                post_sse=post_sse_snapshot,
                get=get_snapshot,
                resumption=resumption_snapshot,
                stdio=stdio_snapshot,
                activity_bucket_seconds=self._history_bucket_seconds,
                activity_bucket_count=self._history_bucket_count,
            )
