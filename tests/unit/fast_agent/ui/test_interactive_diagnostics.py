from __future__ import annotations

import json
import os

import pytest

from fast_agent.ui.interactive_diagnostics import write_interactive_trace


@pytest.mark.asyncio
async def test_write_interactive_trace_appends_jsonl(tmp_path) -> None:
    trace_path = tmp_path / "interactive-trace.jsonl"
    previous = os.environ.get("FAST_AGENT_INTERACTIVE_DEBUG_TRACE")
    os.environ["FAST_AGENT_INTERACTIVE_DEBUG_TRACE"] = str(trace_path)
    try:
        write_interactive_trace("prompt.send.start", agent="dev")
    finally:
        if previous is None:
            os.environ.pop("FAST_AGENT_INTERACTIVE_DEBUG_TRACE", None)
        else:
            os.environ["FAST_AGENT_INTERACTIVE_DEBUG_TRACE"] = previous

    records = [json.loads(line) for line in trace_path.read_text().splitlines() if line]
    assert len(records) == 1
    assert records[0]["event"] == "prompt.send.start"
    assert records[0]["agent"] == "dev"
    assert records[0]["task"] is not None
