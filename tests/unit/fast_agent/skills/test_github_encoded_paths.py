from __future__ import annotations

import shutil
import subprocess
from typing import TYPE_CHECKING
from urllib.parse import quote

import pytest

from fast_agent.marketplace import source_urls
from fast_agent.skills import operations
from fast_agent.skills.marketplace_parsing import parse_marketplace_payload
from fast_agent.skills.provenance import read_installed_skill_source

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("folder", ["hello world", "literal%20name", "hash#name", "snow 雪"])
@pytest.mark.parametrize("kind", ["tree", "raw"])
def test_install_from_encoded_github_skill_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, folder: str, kind: str
) -> None:
    repo = tmp_path / "repo"
    skill_dir = repo / "skills" / folder
    skill_dir.mkdir(parents=True)
    content = "---\nname: hello\ndescription: Example skill\n---\n\nHello.\n"
    (skill_dir / "SKILL.md").write_text(content, encoding="utf-8")
    for args in (
        ["init", "-b", "main"],
        ["config", "user.email", "tests@example.com"],
        ["config", "user.name", "Test User"],
        ["add", "."],
        ["commit", "-m", "initial"],
    ):
        subprocess.run(
            ["git", "-c", "core.autocrlf=true", "-C", str(repo), *args],
            check=True,
            capture_output=True,
        )

    def clone_source(
        *,
        repo_url: str,
        repo_ref: str | None,
        repo_subdir: str,
        destination_dir: Path,
        checkout_ref: str | None,
    ) -> None:
        assert repo_url == "https://github.com/example/skills"
        assert repo_ref == "main"
        assert repo_subdir == f"skills/{folder}"
        shutil.copytree(repo, destination_dir, dirs_exist_ok=True)

    monkeypatch.setattr(operations.marketplace_git_sources, "clone_sparse_checkout", clone_source)
    path = quote(f"skills/{folder}", safe="/")
    url = (
        f"https://github.com/example/skills/tree/main/{path}"
        if kind == "tree"
        else f"https://raw.githubusercontent.com/example/skills/main/{path}/SKILL.md"
    )
    skills = parse_marketplace_payload([{"name": "hello", "repo_url": url}])
    installed = operations.install_marketplace_skill_sync(skills[0], tmp_path / "installed")
    assert (installed / "SKILL.md").read_text(encoding="utf-8") == content
    source = read_installed_skill_source(installed).source
    assert source is not None
    assert source.repo_path == f"skills/{folder}"


@pytest.mark.parametrize("ref", ["main", "release/v1", "release%2Fv1"])
@pytest.mark.parametrize(
    "path",
    ["skills/hello world/SKILL.md", "skills/hash#name/SKILL.md", "skills/literal%20/SKILL.md"],
)
def test_github_raw_url_round_trip(ref: str, path: str) -> None:
    raw = source_urls.github_raw_file_url(
        repo_url="https://github.com/example/skills", repo_ref=ref, repo_path=path
    )
    assert raw == f"https://raw.githubusercontent.com/example/skills/{ref}/{quote(path, safe='/')}"
    parsed = source_urls.parse_github_url(raw)
    assert parsed is not None
    assert parsed.repo_ref == ref
    assert parsed.repo_path == path
    assert source_urls.normalize_marketplace_url(raw) == raw


def test_explicit_repository_path_is_not_url_decoded() -> None:
    skill = parse_marketplace_payload(
        [
            {
                "name": "hello",
                "repo_url": "https://github.com/example/skills",
                "repo_path": "skills/literal%20",
            }
        ]
    )[0]
    assert skill.repo_path == "skills/literal%20"


@pytest.mark.parametrize("path", ["skills/trailing ", " leading/skill", "skills/back\\slash"])
def test_marketplace_rejects_url_paths_that_normalization_would_change(path: str) -> None:
    url = f"https://github.com/example/skills/tree/main/{quote(path, safe='/')}"
    assert source_urls.parse_github_url(url) is None
    assert parse_marketplace_payload([{"name": "hello", "repo_url": url}]) == []


@pytest.mark.parametrize("encoded_path", ["skills/%FF", "skills/%C3%28"])
def test_marketplace_rejects_invalid_utf8_url_paths(encoded_path: str) -> None:
    url = f"https://github.com/example/skills/tree/main/{encoded_path}"
    assert source_urls.parse_github_url(url) is None
    assert parse_marketplace_payload([{"name": "hello", "repo_url": url}]) == []


@pytest.mark.parametrize("kind", ["tree", "raw"])
def test_marketplace_candidate_path_encoding_is_stable(kind: str) -> None:
    encoded = "skills/hello%20world"
    url = (
        f"https://github.com/example/skills/tree/main/{encoded}"
        if kind == "tree"
        else f"https://raw.githubusercontent.com/example/skills/main/{encoded}/marketplace.json"
    )
    if kind == "tree":
        assert source_urls.candidate_marketplace_urls(url) == [
            f"https://raw.githubusercontent.com/example/skills/main/{encoded}/.claude-plugin/marketplace.json",
            f"https://raw.githubusercontent.com/example/skills/main/{encoded}/marketplace.json",
        ]
    else:
        assert source_urls.normalize_marketplace_url(url) == url
