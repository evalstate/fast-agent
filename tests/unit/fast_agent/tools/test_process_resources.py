import time
from pathlib import Path

import pytest

import fast_agent.tools.process_resources as process_resources
from fast_agent.tools.process_resources import (
    ProcessResourceObservationState,
    ProcessResourceSnapshot,
    observe_resource_changes,
)

GIB = 1024**3


def _snapshot(
    *,
    sampled_at: float = 1.0,
    disk_total: int = 10 * GIB,
    disk_free: int = 8 * GIB,
    memory_used: int = 2 * GIB,
    memory_limit: int = 10 * GIB,
    rss: int = GIB,
    cpu: float = 1.0,
    process_count: int = 1,
    cpu_capacity: float = 2.0,
) -> ProcessResourceSnapshot:
    return ProcessResourceSnapshot(
        sampled_at=sampled_at,
        disk_total_bytes=disk_total,
        disk_free_bytes=disk_free,
        memory_used_bytes=memory_used,
        memory_limit_bytes=memory_limit,
        process_rss_bytes=rss,
        process_cpu_seconds=cpu,
        process_count=process_count,
        cpu_capacity=cpu_capacity,
    )


def test_resource_observations_are_silent_for_small_changes() -> None:
    state = ProcessResourceObservationState()

    assert observe_resource_changes(state, _snapshot()) is None
    assert (
        observe_resource_changes(
            state,
            _snapshot(
                sampled_at=2.0,
                disk_free=15 * GIB // 2,
                memory_used=3 * GIB,
                rss=GIB + 100,
                cpu=1.5,
            ),
        )
        is None
    )


def test_resource_observation_reports_large_disk_decline_once() -> None:
    state = ProcessResourceObservationState()
    observe_resource_changes(state, _snapshot())

    observation = observe_resource_changes(
        state,
        _snapshot(sampled_at=2.0, disk_free=6 * GIB, cpu=1.2),
    )
    repeated = observe_resource_changes(
        state,
        _snapshot(sampled_at=3.0, disk_free=6 * GIB, cpu=1.4),
    )

    assert observation is not None
    assert "disk free 6.0 GiB/10.0 GiB" in observation
    assert "down 2.0 GiB" in observation
    assert repeated is None


def test_resource_observation_reports_pressure_and_high_cpu() -> None:
    state = ProcessResourceObservationState()
    observe_resource_changes(state, _snapshot())

    observation = observe_resource_changes(
        state,
        _snapshot(
            sampled_at=2.0,
            disk_free=GIB,
            memory_used=9 * GIB,
            cpu=3.0,
            process_count=5,
        ),
    )

    assert observation is not None
    assert "disk free 1.0 GiB/10.0 GiB (10%" in observation
    assert "memory used 9.0 GiB/10.0 GiB (90%)" in observation
    assert "process tree CPU 100% of 2.0-core capacity, 5 processes" in observation


@pytest.mark.asyncio
async def test_sampler_timeout_never_waits_for_blocked_collector() -> None:
    def blocked_collector(
        working_directory: str,
        pid: int | None,
    ) -> ProcessResourceSnapshot:
        del working_directory, pid
        while True:
            time.sleep(1)

    sampler = process_resources._ProcessResourceSampler(blocked_collector)
    started = time.monotonic()

    snapshot = await sampler.sample("/workspace", 1, timeout_seconds=0.01)

    assert snapshot is None
    assert time.monotonic() - started < 0.2


@pytest.mark.asyncio
async def test_sampler_swallows_collector_exception() -> None:
    def failed_collector(
        working_directory: str,
        pid: int | None,
    ) -> ProcessResourceSnapshot:
        del working_directory, pid
        raise OSError("unavailable")

    sampler = process_resources._ProcessResourceSampler(failed_collector)

    assert await sampler.sample("/workspace", 1, timeout_seconds=0.1) is None


def test_windows_collection_has_disk_only_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        process_resources.shutil,
        "disk_usage",
        lambda path: process_resources.shutil._ntuple_diskusage(
            10 * GIB,
            4 * GIB,
            6 * GIB,
        ),
    )

    snapshot = process_resources._collect_process_resource_snapshot(
        str(tmp_path),
        1234,
        platform_name="win32",
    )

    assert snapshot.disk_total_bytes == 10 * GIB
    assert snapshot.disk_free_bytes == 6 * GIB
    assert snapshot.process_rss_bytes is None
    assert snapshot.process_cpu_seconds is None
    assert snapshot.memory_limit_bytes is None
