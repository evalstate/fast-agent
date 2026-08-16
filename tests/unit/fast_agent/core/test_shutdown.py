from __future__ import annotations

import asyncio
import json
import time

import pytest

from fast_agent.core.shutdown import ShutdownBudget


@pytest.mark.asyncio
async def test_shutdown_budget_bounds_all_phases_to_one_deadline() -> None:
    budget = ShutdownBudget(0.04, reason="test")

    async def first_phase() -> None:
        await asyncio.sleep(0.02)

    async def blocked_phase() -> None:
        await asyncio.Event().wait()

    started_at = time.monotonic()
    assert await budget.run("first", first_phase)
    assert not await budget.run("blocked", blocked_phase)

    assert time.monotonic() - started_at < 0.15


@pytest.mark.asyncio
async def test_shutdown_budget_writes_phase_timings(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trace_path = tmp_path / "shutdown.jsonl"
    monkeypatch.setenv("FAST_AGENT_SHUTDOWN_DEBUG_TRACE", str(trace_path))
    budget = ShutdownBudget(0.1, reason="test")

    async def cleanup() -> None:
        await asyncio.sleep(0)

    assert await budget.run("cleanup", cleanup)
    budget.complete()

    records = [json.loads(line) for line in trace_path.read_text().splitlines()]
    assert [record["event"] for record in records] == [
        "shutdown.begin",
        "shutdown.phase.start",
        "shutdown.phase.end",
        "shutdown.complete",
    ]
    assert records[2]["phase"] == "cleanup"
    assert records[2]["outcome"] == "completed"
    assert records[2]["elapsed_ms"] >= 0
