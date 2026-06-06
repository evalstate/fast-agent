"""Direct skill source coverage.

These tests exercise the new source-adapter and service boundary only; broader
marketplace install/update behavior remains covered by existing skills tests.
"""

from __future__ import annotations

import pytest

from fast_agent.skills.direct_sources import (
    DirectSkillSourceError,
    is_direct_skill_source,
    resolve_direct_skill_source_sync,
)
from fast_agent.skills.service import install_direct_skill_sync


def test_direct_local_source_uses_manifest_name_and_copies_assets(tmp_path):
    source_dir = tmp_path / "repo-name"
    source_dir.mkdir()
    source_dir.joinpath("SKILL.md").write_text(
        "---\nname: canonical-name\ndescription: Test skill\n---\n\nBody.\n",
        encoding="utf-8",
    )
    source_dir.joinpath("references").mkdir()
    source_dir.joinpath("references", "notes.md").write_text("notes", encoding="utf-8")

    direct = resolve_direct_skill_source_sync(source_dir.as_posix())

    assert direct.skill.name == "canonical-name"
    assert direct.skill.description == "Test skill"
    assert direct.skill.install_dir_name == "canonical-name"
    assert direct.skill.repo_path == "."


def test_direct_local_source_accepts_mixed_case_manifest_file(tmp_path):
    manifest_path = tmp_path / "Skill.MD"
    manifest_path.write_text(
        "---\nname: canonical-name\ndescription: Test skill\n---\n\nBody.\n",
        encoding="utf-8",
    )

    direct = resolve_direct_skill_source_sync(manifest_path.as_posix())

    assert direct.skill.name == "canonical-name"
    assert direct.skill.repo_path == "."


@pytest.mark.parametrize(
    "name",
    [
        "Uppercase",
        "-leading",
        "trailing-",
        "has_underscore",
        "a" * 65,
    ],
)
def test_direct_source_rejects_skill_names_outside_agent_skills_spec(tmp_path, name):
    source_dir = tmp_path / "skill"
    source_dir.mkdir()
    source_dir.joinpath("SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Test skill\n---\n\nBody.\n",
        encoding="utf-8",
    )

    with pytest.raises(DirectSkillSourceError, match="Does not meet Agent Skills specification"):
        resolve_direct_skill_source_sync(source_dir.as_posix())


def test_direct_source_rejects_missing_frontmatter(tmp_path):
    source_dir = tmp_path / "skill"
    source_dir.mkdir()
    source_dir.joinpath("SKILL.md").write_text("No frontmatter.\n", encoding="utf-8")

    with pytest.raises(DirectSkillSourceError, match="Invalid SKILL.md frontmatter"):
        resolve_direct_skill_source_sync(source_dir.as_posix())


def test_direct_source_wraps_git_root_mismatch(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_dir = tmp_path / "skill"
    source_dir.mkdir()
    source_dir.joinpath("SKILL.md").write_text(
        "---\nname: example\ndescription: Test skill\n---\n\nBody.\n",
        encoding="utf-8",
    )
    unrelated_root = tmp_path / "other"
    unrelated_root.mkdir()

    monkeypatch.setattr(
        "fast_agent.skills.direct_sources._find_git_root",
        lambda _path: unrelated_root,
    )

    with pytest.raises(DirectSkillSourceError, match="inside its detected git root"):
        resolve_direct_skill_source_sync(source_dir.as_posix())


def test_github_skill_url_is_direct_source():
    assert is_direct_skill_source("https://GitHub.com/org/repo/blob/main/skills/example/Skill.MD")
    assert is_direct_skill_source("https://github.com/org/repo/tree/main/skills/example")
    assert is_direct_skill_source(
        "https://RAW.githubusercontent.com/org/repo/feature/demo/skills/example/Skill.MD"
    )


def test_scp_style_git_url_is_not_direct_skill_source() -> None:
    assert not is_direct_skill_source("git@github.com:org/repo.git")


@pytest.mark.parametrize(
    "url",
    [
        "https://github.com/org/repo/blob/main/README.md",
        "https://raw.githubusercontent.com/org/repo/main/README.md",
        "https://raw.githubusercontent.com/org/repo/feature/demo/README.md",
    ],
)
def test_non_skill_github_file_url_is_not_direct_source(url: str):
    assert not is_direct_skill_source(url)


def test_github_skill_url_with_slash_branch_resolves_direct_source(monkeypatch):
    skill_md = "---\nname: example\ndescription: Demo\n---\n\nBody.\n"
    requested_urls: list[str] = []

    class _Response:
        text = skill_md

        def raise_for_status(self) -> None:
            return None

    class _AsyncClient:
        def __init__(self, **_kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args) -> None:
            return None

        async def get(self, url: str):
            requested_urls.append(url)
            return _Response()

    monkeypatch.setattr("fast_agent.skills.direct_sources.httpx.AsyncClient", _AsyncClient)

    direct = resolve_direct_skill_source_sync(
        "https://github.com/org/repo/blob/feature/demo/skills/example/SKILL.md"
    )

    assert direct.skill.repo_ref == "feature/demo"
    assert direct.skill.repo_path == "skills/example"
    assert requested_urls == [
        "https://raw.githubusercontent.com/org/repo/feature/demo/skills/example/SKILL.md"
    ]


def test_install_direct_skill_installs_under_manifest_name_with_assets(tmp_path):
    """Service smoke: direct source -> existing installer -> canonical managed dir."""
    source_dir = tmp_path / "source-name"
    source_dir.mkdir()
    source_dir.joinpath("SKILL.md").write_text(
        "---\nname: canonical-name\ndescription: Test skill\n---\n\nBody.\n",
        encoding="utf-8",
    )
    source_dir.joinpath("assets").mkdir()
    source_dir.joinpath("assets", "example.txt").write_text("asset", encoding="utf-8")

    managed_dir = tmp_path / "managed"
    record = install_direct_skill_sync(source_dir.as_posix(), destination_root=managed_dir)

    assert record.name == "canonical-name"
    assert record.skill_dir == managed_dir / "canonical-name"
    assert (managed_dir / "canonical-name" / "SKILL.md").exists()
    assert (managed_dir / "canonical-name" / "assets" / "example.txt").read_text(
        encoding="utf-8"
    ) == "asset"
    assert not (managed_dir / "source-name").exists()
