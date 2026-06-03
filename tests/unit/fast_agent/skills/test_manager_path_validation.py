from pathlib import Path

import pytest

from fast_agent.skills.marketplace_parsing import normalize_repo_path
from fast_agent.skills.models import InstalledSkillSource
from fast_agent.skills.operations import (
    _resolve_repo_subdir,
    _validate_source_path_exists,
    candidate_marketplace_urls,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("skills/example", "skills/example"),
        ("skills/example/", "skills/example"),
        ("skills\\example", "skills/example"),
        ("/absolute/path", None),
        ("C:\\skills\\example", None),
        ("../escape", None),
        ("skills/../escape", None),
        ("", None),
        ("   ", None),
        (".", "."),
    ],
)
def test_normalize_repo_path(value: str, expected: str | None) -> None:
    assert normalize_repo_path(value) == expected


def test_resolve_repo_subdir_rejects_escape(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    with pytest.raises(ValueError, match="escapes repository root"):
        _resolve_repo_subdir(repo_root, "../outside")


def test_validate_source_path_rejects_manifest_directory(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    skill_dir = repo_root / "skills" / "alpha"
    (skill_dir / "SKILL.md").mkdir(parents=True)

    source = InstalledSkillSource(
        schema_version=1,
        installed_via="marketplace",
        source_origin="local",
        repo_url=repo_root.as_posix(),
        repo_ref=None,
        repo_path="skills/alpha",
        source_url=None,
        installed_commit=None,
        installed_path_oid=None,
        installed_revision="local",
        installed_at="2026-01-01T00:00:00Z",
        content_fingerprint="sha256:test",
    )

    assert (
        _validate_source_path_exists(source, "alpha")
        == "SKILL.md not found in repository path: skills/alpha"
    )


def test_candidate_marketplace_urls_for_github_repo() -> None:
    urls = candidate_marketplace_urls("https://github.com/anthropics/skills")
    assert urls == [
        "https://raw.githubusercontent.com/anthropics/skills/main/.claude-plugin/marketplace.json",
        "https://raw.githubusercontent.com/anthropics/skills/main/marketplace.json",
        "https://raw.githubusercontent.com/anthropics/skills/master/.claude-plugin/marketplace.json",
        "https://raw.githubusercontent.com/anthropics/skills/master/marketplace.json",
    ]


def test_candidate_marketplace_urls_for_github_blob_marketplace() -> None:
    urls = candidate_marketplace_urls(
        "https://github.com/fast-agent-ai/skills/blob/main/marketplace.json"
    )
    assert urls == [
        "https://raw.githubusercontent.com/fast-agent-ai/skills/main/marketplace.json"
    ]
