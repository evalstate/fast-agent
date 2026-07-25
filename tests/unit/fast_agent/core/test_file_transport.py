from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from fast_agent.core.logging.events import Event
from fast_agent.core.logging.transport import FileTransport, flatten_event_data

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.asyncio
async def test_file_transport_flattens_structured_data_payload(tmp_path: Path) -> None:
    log_path = tmp_path / "fast-agent-log.jsonl"
    transport = FileTransport(log_path)
    event = Event(
        type="error",
        namespace="fast_agent.test",
        message="Streaming idle timeout",
        data={
            "data": {
                "model": "gpt-test",
                "stream_timing": {"events_received": 4},
            },
            "error_type": "StreamIdleTimeoutError",
        },
    )

    await transport.send_matched_event(event)

    record = json.loads(log_path.read_text())
    assert record["data"] == {
        "model": "gpt-test",
        "stream_timing": {"events_received": 4},
        "error_type": "StreamIdleTimeoutError",
    }


def test_flatten_lifts_nested_data_and_prefers_direct_keys() -> None:
    flattened = flatten_event_data(
        {"data": {"model": "nested", "stream_timing": {"events_received": 4}}, "model": "direct"}
    )

    assert flattened == {"model": "direct", "stream_timing": {"events_received": 4}}


def test_flatten_leaves_non_mapping_data_alone() -> None:
    payload = {"data": "not-a-mapping", "model": "gpt-test"}

    assert flatten_event_data(payload) == payload


def test_flatten_leaves_already_flat_data_alone() -> None:
    payload = {"model": "gpt-test", "stream_timing": {"events_received": 4}}

    assert flatten_event_data(payload) == payload
