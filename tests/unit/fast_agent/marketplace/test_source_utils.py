from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from fast_agent.marketplace import source_utils

if TYPE_CHECKING:
    from pathlib import Path


def test_candidate_marketplace_urls_for_github_repo() -> None:
    urls = source_utils.candidate_marketplace_urls("https://github.com/example/skills")
    assert urls == [
        "https://raw.githubusercontent.com/example/skills/main/.claude-plugin/marketplace.json",
        "https://raw.githubusercontent.com/example/skills/main/marketplace.json",
        "https://raw.githubusercontent.com/example/skills/master/.claude-plugin/marketplace.json",
        "https://raw.githubusercontent.com/example/skills/master/marketplace.json",
    ]


def test_parse_installed_source_fields_validates_and_normalizes() -> None:
    payload = {
        "schema_version": 1,
        "installed_via": "marketplace",
        "source_origin": "remote",
        "repo_url": "  https://github.com/example/skills  ",
        "repo_ref": " main ",
        "repo_path": "skills/example",
        "source_url": "",
        "installed_commit": None,
        "installed_path_oid": None,
        "installed_revision": "abc123",
        "installed_at": "2026-02-25T00:00:00Z",
        "content_fingerprint": "sha256:deadbeef",
    }

    parsed = source_utils.parse_installed_source_fields(
        payload,
        expected_schema_version=1,
        normalize_repo_path=lambda value: value.strip("/"),
    )

    assert parsed.repo_url == "https://github.com/example/skills"
    assert parsed.repo_ref == "main"
    assert parsed.source_url is None


def test_parse_installed_source_fields_rejects_invalid_repo_path() -> None:
    payload = {
        "schema_version": 1,
        "installed_via": "marketplace",
        "source_origin": "remote",
        "repo_url": "https://github.com/example/skills",
        "repo_ref": "main",
        "repo_path": "../escape",
        "source_url": None,
        "installed_commit": None,
        "installed_path_oid": None,
        "installed_revision": "abc123",
        "installed_at": "2026-02-25T00:00:00Z",
        "content_fingerprint": "sha256:deadbeef",
    }

    with pytest.raises(ValueError, match="repo_path is invalid"):
        source_utils.parse_installed_source_fields(
            payload,
            expected_schema_version=1,
            normalize_repo_path=lambda _: None,
        )


def test_derive_local_repo_root_from_marketplace_file(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    marketplace = repo_root / ".claude-plugin" / "marketplace.json"
    marketplace.parent.mkdir(parents=True)
    marketplace.write_text("{}", encoding="utf-8")

    resolved = source_utils.derive_local_repo_root(str(marketplace))

    assert resolved == str(repo_root)
