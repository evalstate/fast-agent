from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from fast_agent.core.logging.events import Event
from fast_agent.core.logging.transport import FileTransport

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
