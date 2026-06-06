import pytest
from mcp.types import ErrorData, JSONRPCError, JSONRPCMessage, JSONRPCRequest, JSONRPCResponse

from fast_agent.mcp.transport_tracking import ChannelEvent, ChannelName, TransportChannelMetrics


def test_ping_response_not_counted_as_post_response():
    metrics = TransportChannelMetrics()
    metrics.register_ping_request(1)

    message = JSONRPCMessage(JSONRPCResponse(jsonrpc="2.0", id=1, result={}))
    metrics.record_event(
        ChannelEvent(
            channel="post-json",
            event_type="message",
            message=message,
        )
    )

    snapshot = metrics.snapshot()
    assert snapshot.post_json is not None
    assert snapshot.post_json.response_count == 0
    assert snapshot.post_json.request_count == 0
    assert snapshot.post_json.notification_count == 0


def test_transport_message_counts_are_tallied_by_channel() -> None:
    metrics = TransportChannelMetrics()

    metrics.record_event(
        ChannelEvent(
            channel="post-json",
            event_type="message",
            message=JSONRPCMessage(JSONRPCRequest(jsonrpc="2.0", id=1, method="tools/call")),
        )
    )
    metrics.record_event(
        ChannelEvent(
            channel="get",
            event_type="message",
            message=JSONRPCMessage(JSONRPCResponse(jsonrpc="2.0", id=1, result={})),
        )
    )
    metrics.record_event(
        ChannelEvent(
            channel="resumption",
            event_type="message",
            message=JSONRPCMessage(JSONRPCRequest(jsonrpc="2.0", id=2, method="resources/read")),
        )
    )
    metrics.record_event(
        ChannelEvent(
            channel="stdio",
            event_type="message",
            message=JSONRPCMessage(JSONRPCResponse(jsonrpc="2.0", id=2, result={})),
        )
    )

    snapshot = metrics.snapshot()
    assert snapshot.post is not None
    assert snapshot.post.request_count == 1
    assert snapshot.post_json is not None
    assert snapshot.post_json.request_count == 1
    assert snapshot.get is not None
    assert snapshot.get.response_count == 1
    assert snapshot.resumption is not None
    assert snapshot.resumption.request_count == 1
    assert snapshot.stdio is not None
    assert snapshot.stdio.response_count == 1


def test_activity_bucket_priority_emits_public_state_strings():
    metrics = TransportChannelMetrics(bucket_seconds=60, bucket_count=2)

    metrics.record_event(ChannelEvent(channel="get", event_type="keepalive"))
    metrics.record_event(
        ChannelEvent(
            channel="get",
            event_type="error",
            status_code=405,
            detail="method not allowed",
        )
    )

    snapshot = metrics.snapshot()
    assert snapshot.get is not None
    assert snapshot.get.activity_buckets == ["none", "disabled"]


@pytest.mark.parametrize(
    ("channel", "snapshot_attr"),
    [
        ("post-json", "post_json"),
        ("post-sse", "post_sse"),
    ],
)
def test_post_error_only_events_are_visible(
    channel: ChannelName,
    snapshot_attr: str,
) -> None:
    metrics = TransportChannelMetrics(bucket_seconds=60, bucket_count=2)

    metrics.record_event(
        ChannelEvent(
            channel=channel,
            event_type="error",
            detail="parse failed",
        )
    )

    snapshot = metrics.snapshot()
    post_snapshot = snapshot.post
    mode_snapshot = getattr(snapshot, snapshot_attr)
    assert post_snapshot is not None
    assert post_snapshot.state == "error"
    assert post_snapshot.last_error == "parse failed"
    assert post_snapshot.activity_buckets == ["none", "error"]
    assert mode_snapshot is not None
    assert mode_snapshot.state == "error"
    assert mode_snapshot.last_error == "parse failed"
    assert mode_snapshot.activity_buckets == ["none", "error"]


def test_resumption_error_only_event_is_visible() -> None:
    metrics = TransportChannelMetrics(bucket_seconds=60, bucket_count=2)

    metrics.record_event(
        ChannelEvent(
            channel="resumption",
            event_type="error",
            detail="resume failed",
        )
    )

    snapshot = metrics.snapshot()
    assert snapshot.resumption is not None
    assert snapshot.resumption.state == "error"
    assert snapshot.resumption.last_error == "resume failed"
    assert snapshot.resumption.activity_buckets == ["none", "error"]


@pytest.mark.parametrize("channel", ["get", "stdio"])
def test_connect_then_disconnect_only_events_are_visible(channel: ChannelName) -> None:
    metrics = TransportChannelMetrics()

    metrics.record_event(ChannelEvent(channel=channel, event_type="connect"))
    metrics.record_event(ChannelEvent(channel=channel, event_type="disconnect"))

    channel_snapshot = getattr(metrics.snapshot(), channel)
    assert channel_snapshot is not None
    assert channel_snapshot.connected is False
    assert channel_snapshot.state == "off"
    assert channel_snapshot.disconnect_at is not None


def test_stdio_error_only_event_is_visible() -> None:
    metrics = TransportChannelMetrics(bucket_seconds=60, bucket_count=2)

    metrics.record_event(
        ChannelEvent(
            channel="stdio",
            event_type="error",
            detail="process exited",
        )
    )

    snapshot = metrics.snapshot()
    assert snapshot.stdio is not None
    assert snapshot.stdio.state == "error"
    assert snapshot.stdio.last_error == "process exited"
    assert snapshot.stdio.activity_buckets == ["none", "error"]


def test_response_channel_ignores_requests_until_response_arrives() -> None:
    metrics = TransportChannelMetrics()

    metrics.record_event(
        ChannelEvent(
            channel="post-json",
            event_type="message",
            message=JSONRPCMessage(JSONRPCRequest(jsonrpc="2.0", id=42, method="tools/call")),
        )
    )

    assert metrics.consume_response_channel(42) is None

    metrics.record_event(
        ChannelEvent(
            channel="get",
            event_type="message",
            message=JSONRPCMessage(JSONRPCResponse(jsonrpc="2.0", id=42, result={})),
        )
    )

    assert metrics.consume_response_channel(42) == "get"


def test_response_channel_records_error_response() -> None:
    metrics = TransportChannelMetrics()

    metrics.record_event(
        ChannelEvent(
            channel="get",
            event_type="message",
            message=JSONRPCMessage(
                JSONRPCError(
                    jsonrpc="2.0",
                    id=7,
                    error=ErrorData(code=-32000, message="failed"),
                )
            ),
        )
    )

    assert metrics.consume_response_channel(7) == "get"


@pytest.mark.parametrize(
    "method",
    ["ping", "PING", " notifications/PING ", "mcp.ping"],
)
def test_ping_request_variants_are_classified_as_ping(method: str) -> None:
    metrics = TransportChannelMetrics()
    message = JSONRPCMessage(
        JSONRPCRequest(
            jsonrpc="2.0",
            id=1,
            method=method,
        )
    )

    metrics.record_event(
        ChannelEvent(
            channel="stdio",
            event_type="message",
            message=message,
        )
    )

    snapshot = metrics.snapshot()
    assert snapshot.stdio is not None
    assert snapshot.stdio.activity_buckets is not None
    assert snapshot.stdio.last_message_summary == "ping"
    assert snapshot.stdio.activity_buckets[-1] == "ping"
    assert snapshot.stdio.request_count == 0
