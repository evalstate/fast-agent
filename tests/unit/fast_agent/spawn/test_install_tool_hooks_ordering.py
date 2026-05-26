"""Pin the pause-FIRST ordering invariant in
``isolated_runner._install_tool_hooks``.

Why this file exists
--------------------
The reorder that moved ``pause_before_llm`` / ``pause_before_tool`` to
the head of their merged chains is load-bearing for UX: it bounds the
user-perceived pause latency at signal-delivery + one hook call. The
"chạy chán chê rồi mới pause" regression was the symptom when pause
was last; the reorder is the fix. Without a test pinning the order,
the next "sort alphabetically for readability" pass re-introduces the
regression silently — provider streams still complete, no exceptions,
just a 5-30s perceived latency on pause clicks.

Two branches must be covered: the agent's AgentCard ALREADY supplied
``tool_runner_hooks`` (``existing is not None``) vs the fresh-build
path. Both go through pause-first builders now — see commit message
for the helper extraction rationale.
"""
from __future__ import annotations

import pytest

from fast_agent.spawn.isolated_runner import (
    _build_merged_before_llm_existing,
    _build_merged_before_llm_fresh,
    _build_merged_before_tool,
)


async def _record(name: str, sink: list[str]):
    async def hook(_r, _m):
        sink.append(name)
    return hook


# ─── before_llm — agent with pre-existing AgentCard hooks ─────────────────────


@pytest.mark.asyncio
async def test_before_llm_existing_runs_pause_first_then_spawn_orig_rtac():
    """Order is the load-bearing invariant — assert exactly, not 'pause
    comes before spawn'. Future hooks insert SOMEWHERE in this chain,
    and the reviewer should have to update the test on purpose."""
    order: list[str] = []
    pause = await _record("pause", order)
    spawn = await _record("spawn", order)
    orig = await _record("orig", order)
    rtac = await _record("rtac", order)

    merged = _build_merged_before_llm_existing(pause, spawn, orig, rtac)
    await merged(None, None)

    assert order == ["pause", "spawn", "orig", "rtac"]


@pytest.mark.asyncio
async def test_before_llm_existing_skips_missing_pause_and_orig_and_rtac():
    """``pause_before_llm`` is None when the optional ``pause_signal_handler``
    import failed; ``orig_before_llm`` is None when the card has hooks
    but didn't set this slot; ``rtac_before_llm`` is None when RTAC is
    disabled. Each may independently be absent."""
    order: list[str] = []
    spawn = await _record("spawn", order)

    merged = _build_merged_before_llm_existing(None, spawn, None, None)
    await merged(None, None)

    assert order == ["spawn"]


@pytest.mark.asyncio
async def test_before_llm_existing_propagates_pause_exception():
    """If pause hook raises (e.g. ``PauseProtected``), the rest of the
    chain MUST NOT run — otherwise spawn-event hooks or RTAC would
    burn one more turn before the pause settled, which is exactly the
    UX regression this test is here to prevent."""
    order: list[str] = []

    async def pause(_r, _m):
        order.append("pause")
        raise RuntimeError("pause-protected")

    spawn = await _record("spawn", order)
    orig = await _record("orig", order)
    rtac = await _record("rtac", order)

    merged = _build_merged_before_llm_existing(pause, spawn, orig, rtac)
    with pytest.raises(RuntimeError, match="pause-protected"):
        await merged(None, None)
    assert order == ["pause"]  # spawn/orig/rtac never reached


# ─── before_llm — fresh build (no existing AgentCard hooks) ────────────────────


@pytest.mark.asyncio
async def test_before_llm_fresh_runs_pause_first_then_spawn_then_rtac():
    order: list[str] = []
    pause = await _record("pause", order)
    spawn = await _record("spawn", order)
    rtac = await _record("rtac", order)

    merged = _build_merged_before_llm_fresh(pause, spawn, rtac)
    await merged(None, None)

    assert order == ["pause", "spawn", "rtac"]


@pytest.mark.asyncio
async def test_before_llm_fresh_skips_missing_pause_and_rtac():
    order: list[str] = []
    spawn = await _record("spawn", order)

    merged = _build_merged_before_llm_fresh(None, spawn, None)
    await merged(None, None)

    assert order == ["spawn"]


@pytest.mark.asyncio
async def test_before_llm_fresh_propagates_pause_exception():
    order: list[str] = []

    async def pause(_r, _m):
        order.append("pause")
        raise RuntimeError("pause-protected")

    spawn = await _record("spawn", order)
    rtac = await _record("rtac", order)

    merged = _build_merged_before_llm_fresh(pause, spawn, rtac)
    with pytest.raises(RuntimeError, match="pause-protected"):
        await merged(None, None)
    assert order == ["pause"]


# ─── before_tool — covers both branches ───────────────────────────────────────


@pytest.mark.asyncio
async def test_before_tool_runs_pause_first_then_orig_then_before_tool():
    """The before_tool chain is the same shape regardless of whether
    the agent had pre-existing hooks — both branches in
    ``_install_tool_hooks`` now route through this builder."""
    order: list[str] = []
    pause = await _record("pause", order)
    orig = await _record("orig", order)
    before_tool = await _record("before_tool", order)

    merged = _build_merged_before_tool(pause, orig, before_tool)
    await merged(None, None)

    assert order == ["pause", "orig", "before_tool"]


@pytest.mark.asyncio
async def test_before_tool_skips_missing_pause_and_orig():
    order: list[str] = []
    before_tool = await _record("before_tool", order)

    merged = _build_merged_before_tool(None, None, before_tool)
    await merged(None, None)

    assert order == ["before_tool"]


@pytest.mark.asyncio
async def test_before_tool_propagates_pause_exception():
    order: list[str] = []

    async def pause(_r, _req):
        order.append("pause")
        raise RuntimeError("pause-protected")

    orig = await _record("orig", order)
    before_tool = await _record("before_tool", order)

    merged = _build_merged_before_tool(pause, orig, before_tool)
    with pytest.raises(RuntimeError, match="pause-protected"):
        await merged(None, None)
    assert order == ["pause"]
