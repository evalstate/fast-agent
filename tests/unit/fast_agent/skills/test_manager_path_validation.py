import ntpath
import posixpath
from pathlib import Path
from typing import Any

import pytest

from fast_agent.marketplace.provenance_io import safe_install_dir_name
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


def _is_direct_child(module: Any, root: str, name: str) -> bool:
    """Whether joining ``name`` onto ``root`` yields the direct child of ``root`` named ``name``.

    ``module`` is ``posixpath`` or ``ntpath``, so this asks the question of a *chosen* path
    flavour rather than of the host the test happens to run on. That matters because the
    guard is deliberately host-independent: a marketplace payload written for one platform is
    installed on whichever platform runs it, and ``"nested\\name"`` is a traversal on Windows
    while being one legal filename on POSIX.
    """
    joined = module.normpath(module.join(root, name))
    return module.dirname(joined) == root and module.basename(joined) == name


_POSIX_ROOT = "/managed/root"
_WINDOWS_ROOT = "C:\\managed\\root"


@pytest.mark.parametrize(
    "name",
    [
        "example",
        "example-skill",
        "example_skill.v2",
        "a..b",
        "~",
    ],
)
def test_safe_install_dir_name_accepts_contained_component(name: str, tmp_path: Path) -> None:
    assert safe_install_dir_name(name, label="Skill") == name
    # Assert the contract itself rather than the shape: joining an accepted name onto a root
    # yields the direct child of that root that carries the name. An accepted name has to
    # satisfy that under *both* flavours, since the name may be installed on either.
    assert _is_direct_child(posixpath, _POSIX_ROOT, name)
    assert _is_direct_child(ntpath, _WINDOWS_ROOT, name)
    # And on the running host, for the flavour that will actually create the directory.
    resolved = (tmp_path / name).resolve()
    assert resolved.parent == tmp_path.resolve()
    assert resolved.name == name


@pytest.mark.parametrize(
    "name",
    [
        "..",
        ".",
        "",
        "../escape",
        "..\\escape",
        "nested/name",
        "nested\\name",
        "/absolute",
        "C:/absolute",
        "C:relative",
        "//server/share",
    ],
)
def test_safe_install_dir_name_rejects_non_component_names(name: str) -> None:
    # Every rejected name breaks the same contract from the accepting test: joining it does
    # not produce the direct child of the root that carries the name. It is enough for that
    # to hold under *one* flavour - the guard rejects a name that is a traversal anywhere,
    # because the payload that carries it is installed on whichever platform runs it. Whether
    # a name also leaves the root is incidental: "C:relative" only escapes when the root sits
    # on another drive, so the rejection is anchored on the contract, not on escaping.
    if name:
        assert not (
            _is_direct_child(posixpath, _POSIX_ROOT, name)
            and _is_direct_child(ntpath, _WINDOWS_ROOT, name)
        )

    with pytest.raises(ValueError, match="not a single path component"):
        safe_install_dir_name(name, label="Skill")


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
    assert urls == ["https://raw.githubusercontent.com/fast-agent-ai/skills/main/marketplace.json"]
