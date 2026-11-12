"""Count pre-request attachment uploads so the progress display can show ``n/total``.

Providers normalize attachments by walking a request and uploading on cache misses.
:func:`run_with_upload_progress` runs that walk once as a planning pass: each upload
site calls :func:`plan_upload` after its cache check, which records the key and tells
the site to skip the network call. With nothing to upload (the common case), the
planning result is the real result. Otherwise the walk runs again, uploading and
reporting progress, so totals come from the same code that performs the uploads.
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


@dataclass
class _UploadTracker:
    report: Callable[[int, int], None]
    planning: bool = True
    pending: set[str] = field(default_factory=set)
    uploaded: int = 0


_active_tracker: ContextVar[_UploadTracker | None] = ContextVar(
    "fast_agent_upload_tracker", default=None
)


def plan_upload(key: str) -> bool:
    """Call at an upload site after a cache miss; skip the upload when this returns True."""
    tracker = _active_tracker.get()
    if tracker is None:
        return False
    if tracker.planning:
        tracker.pending.add(key)
        return True
    tracker.uploaded += 1
    tracker.report(tracker.uploaded, max(len(tracker.pending), tracker.uploaded))
    return False


async def run_with_upload_progress[T](
    walk: Callable[[], Awaitable[T]],
    report: Callable[[int, int], None],
) -> tuple[T, bool]:
    """Run ``walk`` with upload counting; return its result and whether it uploaded."""
    tracker = _UploadTracker(report)
    token = _active_tracker.set(tracker)
    try:
        planned = await walk()
        if not tracker.pending:
            return planned, False
        tracker.planning = False
        return await walk(), True
    finally:
        _active_tracker.reset(token)
