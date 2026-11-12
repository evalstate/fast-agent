"""Contract for counting pre-request attachment uploads."""

import pytest

from fast_agent.llm.upload_progress import plan_upload, run_with_upload_progress


class _Uploader:
    def __init__(self) -> None:
        self.cache: dict[str, str] = {}
        self.uploads: list[str] = []
        self.walks = 0

    async def walk(self, keys: list[str]) -> list[str]:
        self.walks += 1
        results = []
        for key in keys:
            if key in self.cache:
                results.append(self.cache[key])
                continue
            if plan_upload(key):
                results.append("")
                continue
            self.uploads.append(key)
            self.cache[key] = f"id-{key}"
            results.append(self.cache[key])
        return results


@pytest.mark.asyncio
async def test_uploads_report_distinct_progress_and_return_real_results() -> None:
    uploader = _Uploader()
    progress: list[tuple[int, int]] = []

    result, uploaded = await run_with_upload_progress(
        lambda: uploader.walk(["a", "b", "a"]), lambda n, total: progress.append((n, total))
    )

    assert (result, uploaded) == (["id-a", "id-b", "id-a"], True)
    assert uploader.uploads == ["a", "b"]
    assert progress == [(1, 2), (2, 2)]


@pytest.mark.asyncio
async def test_cached_requests_walk_once_without_progress() -> None:
    uploader = _Uploader()
    uploader.cache["a"] = "id-a"
    progress: list[tuple[int, int]] = []

    result, uploaded = await run_with_upload_progress(
        lambda: uploader.walk(["a"]), lambda n, total: progress.append((n, total))
    )

    assert (result, uploaded) == (["id-a"], False)
    assert uploader.walks == 1
    assert progress == []


def test_upload_sites_proceed_outside_tracking() -> None:
    assert plan_upload("key") is False
