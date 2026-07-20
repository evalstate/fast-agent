"""Best-effort, bounded resource observations for managed shell processes."""

from __future__ import annotations

import asyncio
import os
import queue
import shutil
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from collections.abc import Callable

_GIB = 1024**3
_SAMPLE_TIMEOUT_SECONDS = 0.05
_DISK_LOW_RATIO = 0.20
_DISK_RECOVERED_RATIO = 0.25
_DISK_LOW_BYTES = 2 * _GIB
_DISK_RECOVERED_BYTES = 3 * _GIB
_MEMORY_HIGH_RATIO = 0.80
_MEMORY_RECOVERED_RATIO = 0.75
_CPU_HIGH_RATIO = 0.90
_CPU_RECOVERED_RATIO = 0.75
_LARGE_BYTE_CHANGE = _GIB
_LARGE_DISK_RATIO_CHANGE = 0.10


class ProcessResourceSnapshotMetadata(TypedDict, total=False):
    """Serializable resource fields attached to managed-process metadata."""

    sampled_at: float
    disk_total_bytes: int
    disk_free_bytes: int
    memory_used_bytes: int
    memory_limit_bytes: int
    process_rss_bytes: int
    process_cpu_seconds: float
    process_count: int
    cpu_capacity: float


@dataclass(frozen=True, slots=True)
class ProcessResourceSnapshot:
    sampled_at: float
    disk_total_bytes: int | None = None
    disk_free_bytes: int | None = None
    memory_used_bytes: int | None = None
    memory_limit_bytes: int | None = None
    process_rss_bytes: int | None = None
    process_cpu_seconds: float | None = None
    process_count: int | None = None
    cpu_capacity: float | None = None

    def metadata(self) -> ProcessResourceSnapshotMetadata:
        metadata: ProcessResourceSnapshotMetadata = {
            "sampled_at": self.sampled_at,
        }
        if self.disk_total_bytes is not None:
            metadata["disk_total_bytes"] = self.disk_total_bytes
        if self.disk_free_bytes is not None:
            metadata["disk_free_bytes"] = self.disk_free_bytes
        if self.memory_used_bytes is not None:
            metadata["memory_used_bytes"] = self.memory_used_bytes
        if self.memory_limit_bytes is not None:
            metadata["memory_limit_bytes"] = self.memory_limit_bytes
        if self.process_rss_bytes is not None:
            metadata["process_rss_bytes"] = self.process_rss_bytes
        if self.process_cpu_seconds is not None:
            metadata["process_cpu_seconds"] = self.process_cpu_seconds
        if self.process_count is not None:
            metadata["process_count"] = self.process_count
        if self.cpu_capacity is not None:
            metadata["cpu_capacity"] = self.cpu_capacity
        return metadata


@dataclass(slots=True)
class ProcessResourceObservationState:
    baseline: ProcessResourceSnapshot | None = None
    previous: ProcessResourceSnapshot | None = None
    last_reported: ProcessResourceSnapshot | None = None
    active_warnings: set[str] = field(default_factory=set)


def _read_int(path: Path) -> int | None:
    try:
        value = path.read_text(encoding="utf-8").strip()
        return int(value)
    except (OSError, ValueError):
        return None


def _process_cgroup_path(pid: int) -> Path | None:
    try:
        lines = Path(f"/proc/{pid}/cgroup").read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in lines:
        fields = line.split(":", 2)
        if len(fields) == 3 and fields[0] == "0" and fields[1] == "":
            return Path("/sys/fs/cgroup") / fields[2].lstrip("/")
    return None


def _memory_metrics(pid: int) -> tuple[int | None, int | None]:
    cgroup_path = _process_cgroup_path(pid)
    if cgroup_path is not None:
        used = _read_int(cgroup_path / "memory.current")
        try:
            raw_limit = (cgroup_path / "memory.max").read_text(encoding="utf-8").strip()
            limit = None if raw_limit == "max" else int(raw_limit)
        except (OSError, ValueError):
            limit = None
        if used is not None and limit is not None and limit > 0:
            return used, limit

    try:
        lines = Path("/proc/meminfo").read_text(encoding="utf-8").splitlines()
    except OSError:
        return None, None
    values: dict[str, int] = {}
    for line in lines:
        name, separator, raw_value = line.partition(":")
        if not separator:
            continue
        number = raw_value.strip().split(maxsplit=1)[0]
        try:
            values[name] = int(number) * 1024
        except ValueError:
            continue
    total = values.get("MemTotal")
    available = values.get("MemAvailable")
    if total is None or available is None:
        return None, None
    return max(total - available, 0), total


def _cpu_capacity(pid: int) -> float:
    cgroup_path = _process_cgroup_path(pid)
    if cgroup_path is not None:
        try:
            quota_text, period_text = (
                cgroup_path / "cpu.max"
            ).read_text(encoding="utf-8").split(maxsplit=1)
            if quota_text != "max":
                quota = int(quota_text)
                period = int(period_text)
                if quota > 0 and period > 0:
                    return quota / period
        except (OSError, ValueError):
            pass
    return float(os.cpu_count() or 1)


def _process_children(pid: int) -> tuple[int, ...]:
    try:
        raw = Path(f"/proc/{pid}/task/{pid}/children").read_text(encoding="utf-8")
    except OSError:
        return ()
    children: list[int] = []
    for token in raw.split():
        try:
            children.append(int(token))
        except ValueError:
            continue
    return tuple(children)


def _process_tree(pid: int) -> tuple[int, ...]:
    pending = [pid]
    observed: set[int] = set()
    while pending:
        current = pending.pop()
        if current in observed:
            continue
        observed.add(current)
        pending.extend(_process_children(current))
    return tuple(observed)


def _process_stat(pid: int) -> tuple[float, int] | None:
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except OSError:
        return None
    command_end = raw.rfind(")")
    if command_end < 0:
        return None
    fields = raw[command_end + 2 :].split()
    if len(fields) < 22:
        return None
    try:
        ticks = int(fields[11]) + int(fields[12])
        rss_pages = int(fields[21])
        clock_ticks = os.sysconf("SC_CLK_TCK")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError):
        return None
    return ticks / clock_ticks, max(rss_pages, 0) * page_size


def _linux_process_metrics(
    pid: int,
) -> tuple[
    int | None,
    int | None,
    float | None,
    int | None,
    int | None,
    float | None,
]:
    memory_used, memory_limit = _memory_metrics(pid)
    total_cpu = 0.0
    total_rss = 0
    process_count = 0
    for process_id in _process_tree(pid):
        stat = _process_stat(process_id)
        if stat is None:
            continue
        cpu_seconds, rss_bytes = stat
        total_cpu += cpu_seconds
        total_rss += rss_bytes
        process_count += 1
    return (
        memory_used,
        memory_limit,
        total_cpu if process_count else None,
        total_rss if process_count else None,
        process_count if process_count else None,
        _cpu_capacity(pid),
    )


def _collect_process_resource_snapshot(
    working_directory: str,
    pid: int | None,
    *,
    platform_name: str = sys.platform,
) -> ProcessResourceSnapshot:
    disk_total: int | None = None
    disk_free: int | None = None
    try:
        disk = shutil.disk_usage(working_directory)
        disk_total = disk.total
        disk_free = disk.free
    except OSError:
        pass

    memory_used: int | None = None
    memory_limit: int | None = None
    process_cpu: float | None = None
    process_rss: int | None = None
    process_count: int | None = None
    cpu_capacity: float | None = None
    if pid is not None and platform_name.startswith("linux"):
        (
            memory_used,
            memory_limit,
            process_cpu,
            process_rss,
            process_count,
            cpu_capacity,
        ) = _linux_process_metrics(pid)

    return ProcessResourceSnapshot(
        sampled_at=time.monotonic(),
        disk_total_bytes=disk_total,
        disk_free_bytes=disk_free,
        memory_used_bytes=memory_used,
        memory_limit_bytes=memory_limit,
        process_rss_bytes=process_rss,
        process_cpu_seconds=process_cpu,
        process_count=process_count,
        cpu_capacity=cpu_capacity,
    )


@dataclass(frozen=True, slots=True)
class _SampleRequest:
    working_directory: str
    pid: int | None
    loop: asyncio.AbstractEventLoop
    future: asyncio.Future[ProcessResourceSnapshot | None]


class _ProcessResourceSampler:
    """Run observations on one daemon worker so a blocked syscall cannot block exit."""

    def __init__(
        self,
        collector: Callable[[str, int | None], ProcessResourceSnapshot] | None = None,
    ) -> None:
        self._collector = collector or _collect_process_resource_snapshot
        self._requests: queue.Queue[_SampleRequest] = queue.Queue(maxsize=1)
        self._thread: threading.Thread | None = None
        self._thread_lock = threading.Lock()

    def _ensure_started(self) -> None:
        with self._thread_lock:
            if self._thread is not None:
                return
            self._thread = threading.Thread(
                target=self._run,
                name="fast-agent-resource-observer",
                daemon=True,
            )
            self._thread.start()

    @staticmethod
    def _deliver(
        future: asyncio.Future[ProcessResourceSnapshot | None],
        snapshot: ProcessResourceSnapshot | None,
    ) -> None:
        if not future.done():
            future.set_result(snapshot)

    def _run(self) -> None:
        while True:
            request = self._requests.get()
            try:
                snapshot = self._collector(request.working_directory, request.pid)
            except Exception:
                snapshot = None
            try:
                request.loop.call_soon_threadsafe(
                    self._deliver,
                    request.future,
                    snapshot,
                )
            except RuntimeError:
                continue

    async def sample(
        self,
        working_directory: str,
        pid: int | None,
        *,
        timeout_seconds: float = _SAMPLE_TIMEOUT_SECONDS,
    ) -> ProcessResourceSnapshot | None:
        self._ensure_started()
        loop = asyncio.get_running_loop()
        future: asyncio.Future[ProcessResourceSnapshot | None] = loop.create_future()
        try:
            self._requests.put_nowait(
                _SampleRequest(
                    working_directory=working_directory,
                    pid=pid,
                    loop=loop,
                    future=future,
                )
            )
        except queue.Full:
            return None
        try:
            async with asyncio.timeout(timeout_seconds):
                return await future
        except TimeoutError:
            return None


_SAMPLER = _ProcessResourceSampler()


async def sample_process_resources(
    working_directory: str,
    pid: int | None,
) -> ProcessResourceSnapshot | None:
    """Return a resource snapshot or None within a small fixed time budget."""
    return await _SAMPLER.sample(working_directory, pid)


def _gib(value: int) -> str:
    return f"{value / _GIB:.1f} GiB"


def _disk_observation(
    state: ProcessResourceObservationState,
    snapshot: ProcessResourceSnapshot,
    comparison: ProcessResourceSnapshot,
) -> str | None:
    total = snapshot.disk_total_bytes
    free = snapshot.disk_free_bytes
    if total is None or free is None or total <= 0:
        return None
    ratio = free / total
    pressure = ratio <= _DISK_LOW_RATIO or free <= _DISK_LOW_BYTES
    newly_low = pressure and "disk" not in state.active_warnings
    if pressure:
        state.active_warnings.add("disk")
    elif ratio >= _DISK_RECOVERED_RATIO and free >= _DISK_RECOVERED_BYTES:
        state.active_warnings.discard("disk")

    prior_free = comparison.disk_free_bytes
    decline = max(prior_free - free, 0) if prior_free is not None else 0
    large_decline = decline >= _LARGE_BYTE_CHANGE or decline / total >= _LARGE_DISK_RATIO_CHANGE
    if not newly_low and not large_decline:
        return None
    detail = f"disk free {_gib(free)}/{_gib(total)} ({ratio:.0%}"
    if decline:
        detail += f", down {_gib(decline)}"
    return f"{detail})"


def _memory_observation(
    state: ProcessResourceObservationState,
    snapshot: ProcessResourceSnapshot,
) -> str | None:
    used = snapshot.memory_used_bytes
    limit = snapshot.memory_limit_bytes
    if used is None or limit is None or limit <= 0:
        return None
    ratio = used / limit
    pressure = ratio >= _MEMORY_HIGH_RATIO
    newly_high = pressure and "memory" not in state.active_warnings
    if pressure:
        state.active_warnings.add("memory")
    elif ratio <= _MEMORY_RECOVERED_RATIO:
        state.active_warnings.discard("memory")
    if not newly_high:
        return None
    return f"memory used {_gib(used)}/{_gib(limit)} ({ratio:.0%})"


def _rss_observation(
    snapshot: ProcessResourceSnapshot,
    comparison: ProcessResourceSnapshot,
) -> str | None:
    rss = snapshot.process_rss_bytes
    prior_rss = comparison.process_rss_bytes
    if rss is None or prior_rss is None:
        return None
    growth = rss - prior_rss
    if growth < _LARGE_BYTE_CHANGE:
        return None
    process_label = (
        f", {snapshot.process_count} processes"
        if snapshot.process_count is not None
        else ""
    )
    return f"process tree RSS {_gib(rss)} (up {_gib(growth)}{process_label})"


def _cpu_observation(
    state: ProcessResourceObservationState,
    snapshot: ProcessResourceSnapshot,
) -> str | None:
    previous = state.previous
    if (
        previous is None
        or previous.process_cpu_seconds is None
        or snapshot.process_cpu_seconds is None
        or snapshot.cpu_capacity is None
        or snapshot.cpu_capacity <= 0
    ):
        return None
    elapsed = snapshot.sampled_at - previous.sampled_at
    if elapsed <= 0:
        return None
    cpu_ratio = max(snapshot.process_cpu_seconds - previous.process_cpu_seconds, 0) / (
        elapsed * snapshot.cpu_capacity
    )
    newly_high = cpu_ratio >= _CPU_HIGH_RATIO and "cpu" not in state.active_warnings
    if cpu_ratio >= _CPU_HIGH_RATIO:
        state.active_warnings.add("cpu")
    elif cpu_ratio <= _CPU_RECOVERED_RATIO:
        state.active_warnings.discard("cpu")
    if not newly_high:
        return None
    process_label = (
        f", {snapshot.process_count} processes"
        if snapshot.process_count is not None
        else ""
    )
    return (
        f"process tree CPU {cpu_ratio:.0%} of {snapshot.cpu_capacity:.1f}-core capacity"
        f"{process_label}"
    )


def observe_resource_changes(
    state: ProcessResourceObservationState,
    snapshot: ProcessResourceSnapshot,
) -> str | None:
    """Return one compact observation only for warnings or large changes."""
    if state.baseline is None:
        state.baseline = snapshot
        state.previous = snapshot
        return None

    comparison = state.last_reported or state.baseline
    observations = [
        observation
        for observation in (
            _disk_observation(state, snapshot, comparison),
            _memory_observation(state, snapshot),
            _rss_observation(snapshot, comparison),
            _cpu_observation(state, snapshot),
        )
        if observation is not None
    ]
    state.previous = snapshot
    if not observations:
        return None
    state.last_reported = snapshot
    return "; ".join(observations)
