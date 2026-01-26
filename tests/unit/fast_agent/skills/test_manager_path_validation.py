from pathlib import Path

import pytest

from fast_agent.skills import manager


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("skills/example", "skills/example"),
        ("skills/example/", "skills/example"),
        ("skills\\example", "skills/example"),
        ("/absolute/path", None),
        ("../escape", None),
        ("skills/../escape", None),
        ("", None),
        ("   ", None),
        (".", None),
    ],
)
def test_normalize_repo_path(value: str, expected: str | None) -> None:
    assert manager._normalize_repo_path(value) == expected


def test_resolve_repo_subdir_rejects_escape(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    with pytest.raises(ValueError, match="escapes repository root"):
        manager._resolve_repo_subdir(repo_root, "../outside")


def test_candidate_marketplace_urls_for_github_repo() -> None:
    urls = manager._candidate_marketplace_urls("https://github.com/anthropics/skills")
    assert urls == [
        "https://raw.githubusercontent.com/anthropics/skills/main/.claude-plugin/marketplace.json",
        "https://raw.githubusercontent.com/anthropics/skills/main/marketplace.json",
        "https://raw.githubusercontent.com/anthropics/skills/master/.claude-plugin/marketplace.json",
        "https://raw.githubusercontent.com/anthropics/skills/master/marketplace.json",
    ]


def test_candidate_marketplace_urls_for_github_blob_marketplace() -> None:
    urls = manager._candidate_marketplace_urls(
        "https://github.com/fast-agent-ai/skills/blob/main/marketplace.json"
    )
    assert urls == [
        "https://raw.githubusercontent.com/fast-agent-ai/skills/main/marketplace.json"
    ]
