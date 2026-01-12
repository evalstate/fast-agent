from mcp.types import JSONRPCMessage, JSONRPCResponse

from fast_agent.mcp.transport_tracking import ChannelEvent, TransportChannelMetrics


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
