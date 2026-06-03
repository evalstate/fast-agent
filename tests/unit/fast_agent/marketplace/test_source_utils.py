from __future__ import annotations

import asyncio
import io
import json
import subprocess
import tarfile
from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, Literal

import pytest

from fast_agent.marketplace import source_utils

if TYPE_CHECKING:
    from pathlib import Path

_VALID_SHA256_FINGERPRINT = "sha256:" + ("0" * 64)


@dataclass(frozen=True)
class _InstalledSourcePayload:
    schema_version: int = 1
    installed_via: str = "marketplace"
    source_origin: Literal["remote", "local"] = "remote"
    repo_url: str = "https://github.com/example/skills"
    repo_ref: str | None = "main"
    repo_path: str = "skills/alpha"
    source_url: str | None = None
    installed_commit: str | None = "abc123"
    installed_path_oid: str | None = "def456"
    installed_revision: str = "abc123"
    installed_at: str = "2026-02-25T00:00:00Z"
    content_fingerprint: str = _VALID_SHA256_FINGERPRINT


def test_candidate_marketplace_urls_for_github_repo() -> None:
    urls = source_utils.candidate_marketplace_urls("https://github.com/example/skills")
    assert urls == [
        "https://raw.githubusercontent.com/example/skills/main/.claude-plugin/marketplace.json",
        "https://raw.githubusercontent.com/example/skills/main/marketplace.json",
        "https://raw.githubusercontent.com/example/skills/master/.claude-plugin/marketplace.json",
        "https://raw.githubusercontent.com/example/skills/master/marketplace.json",
    ]


def test_candidate_marketplace_urls_normalize_github_host_case() -> None:
    urls = source_utils.candidate_marketplace_urls("https://GitHub.com/example/skills")

    assert urls == [
        "https://raw.githubusercontent.com/example/skills/main/.claude-plugin/marketplace.json",
        "https://raw.githubusercontent.com/example/skills/main/marketplace.json",
        "https://raw.githubusercontent.com/example/skills/master/.claude-plugin/marketplace.json",
        "https://raw.githubusercontent.com/example/skills/master/marketplace.json",
    ]


def test_github_host_helpers_normalize_case_and_padding() -> None:
    assert source_utils.is_github_web_host(" GitHub.com ")
    assert source_utils.is_github_web_host(" WWW.GITHUB.COM ")
    assert source_utils.is_github_raw_host(" RAW.GITHUBUSERCONTENT.COM ")
    assert not source_utils.is_github_web_host("raw.githubusercontent.com")


def test_normalize_marketplace_url_strips_outer_whitespace_for_github_blob() -> None:
    normalized = source_utils.normalize_marketplace_url(
        "  https://github.com/example/skills/blob/main/skills/alpha/marketplace.json  "
    )

    assert normalized == (
        "https://raw.githubusercontent.com/example/skills/"
        "main/skills/alpha/marketplace.json"
    )


def test_parse_github_url_strips_outer_whitespace() -> None:
    parsed = source_utils.parse_github_url(
        "  https://github.com/example/skills/tree/main/skills/alpha  "
    )

    assert parsed is not None
    assert parsed.repo_url == "https://github.com/example/skills"
    assert parsed.repo_ref == "main"
    assert parsed.repo_path == "skills/alpha"


def test_candidate_marketplace_urls_keep_default_branch_before_nested_skill_path() -> None:
    urls = source_utils.candidate_marketplace_urls(
        "https://github.com/org/repo/tree/main/examples/skills/foo"
    )

    assert urls == [
        "https://raw.githubusercontent.com/org/repo/main/examples/skills/foo/.claude-plugin/marketplace.json",
        "https://raw.githubusercontent.com/org/repo/main/examples/skills/foo/marketplace.json",
    ]


def test_candidate_marketplace_urls_preserve_slash_branch_for_skill_path() -> None:
    urls = source_utils.candidate_marketplace_urls(
        "https://github.com/org/repo/tree/feature/demo/skills/foo"
    )

    assert urls == [
        "https://raw.githubusercontent.com/org/repo/feature/demo/skills/foo/.claude-plugin/marketplace.json",
        "https://raw.githubusercontent.com/org/repo/feature/demo/skills/foo/marketplace.json",
    ]


def test_candidate_marketplace_urls_preserve_anchor_named_slash_branch() -> None:
    urls = source_utils.candidate_marketplace_urls(
        "https://github.com/org/repo/tree/feature/skills/foo"
    )

    assert urls == [
        "https://raw.githubusercontent.com/org/repo/feature/skills/foo/.claude-plugin/marketplace.json",
        "https://raw.githubusercontent.com/org/repo/feature/skills/foo/marketplace.json",
    ]


def test_candidate_marketplace_urls_detects_uppercase_marketplace_filename() -> None:
    urls = source_utils.candidate_marketplace_urls(
        "https://github.com/org/repo/tree/main/catalog/MARKETPLACE.JSON"
    )

    assert urls == [
        "https://raw.githubusercontent.com/org/repo/main/catalog/MARKETPLACE.JSON",
    ]


def test_is_probable_url_requires_scheme_and_host() -> None:
    assert source_utils.is_probable_url("https://github.com/example/skills") is True
    assert source_utils.is_probable_url("file:///tmp/marketplace.json") is False
    assert source_utils.is_probable_url("git@github.com:example/skills.git") is False
    assert source_utils.is_probable_url("plugins/finder") is False
    assert source_utils.is_probable_url("") is False


def test_is_git_source_url_accepts_scp_style_git_sources() -> None:
    assert source_utils.is_git_source_url("https://github.com/example/skills") is True
    assert source_utils.is_git_source_url("git@github.com:example/skills.git") is True
    assert source_utils.is_git_source_url("plugins/finder") is False
    assert source_utils.is_git_source_url("") is False


def test_github_raw_file_url_builds_raw_url_for_github_repo() -> None:
    assert (
        source_utils.github_raw_file_url(
            repo_url="https://github.com/example/skills",
            repo_ref="release/v1",
            repo_path="skills/alpha/SKILL.md",
        )
        == "https://raw.githubusercontent.com/example/skills/release/v1/skills/alpha/SKILL.md"
    )


def test_github_raw_file_url_rejects_non_github_repo_url() -> None:
    assert (
        source_utils.github_raw_file_url(
            repo_url="https://gitlab.com/example/skills",
            repo_ref="main",
            repo_path="skills/alpha/SKILL.md",
        )
        is None
    )


def test_normalize_marketplace_payload_does_not_mutate_extracted_entries() -> None:
    entry = {"name": "alpha"}
    info = SimpleNamespace(
        context={
            "source_url": "https://example.com/marketplace.json",
            "repo_url": "https://github.com/example/skills",
            "repo_ref": "main",
        }
    )

    normalized = source_utils.normalize_marketplace_payload(
        {"entries": [entry]},
        info,
        extract_entries=lambda _payload: [entry],
    )

    assert entry == {"name": "alpha"}
    assert normalized == {
        "entries": [
            {
                "name": "alpha",
                "source_url": "https://example.com/marketplace.json",
                "repo_url": "https://github.com/example/skills",
                "repo_ref": "main",
            }
        ]
    }


def test_subprocess_failure_detail_prefers_stderr_then_stdout() -> None:
    assert (
        source_utils.subprocess_failure_detail(
            subprocess.CompletedProcess(["git"], 1, stdout=" out \n", stderr=" err \n"),
            "fallback",
        )
        == "err"
    )
    assert (
        source_utils.subprocess_failure_detail(
            subprocess.CompletedProcess(["git"], 1, stdout=" out \n", stderr=" \n"),
            "fallback",
        )
        == "out"
    )
    assert (
        source_utils.subprocess_failure_detail(
            subprocess.CompletedProcess(["git"], 1, stdout="", stderr=""),
            "fallback",
        )
        == "fallback"
    )


def test_parse_ls_remote_commit_returns_first_commit() -> None:
    output = "\n abc123\trefs/heads/main\n def456\trefs/heads/dev\n"

    assert source_utils.parse_ls_remote_commit(output) == "abc123"


def test_parse_ls_remote_commit_prefers_peeled_annotated_tag_commit() -> None:
    output = "\n tag-object\trefs/tags/v1\n commit-object\trefs/tags/v1^{}\n"

    assert source_utils.parse_ls_remote_commit(output) == "commit-object"


def test_parse_ls_remote_commit_ignores_non_ref_lines() -> None:
    output = "\n warning: ignored text\n abc123\trefs/heads/main\n"

    assert source_utils.parse_ls_remote_commit(output) == "abc123"


def test_first_nonempty_str_returns_first_trimmed_string() -> None:
    payload: dict[str, object] = {
        "name": "   ",
        "slug": " finder ",
        "title": "Ignored",
        "count": 3,
    }

    assert source_utils.first_nonempty_str(payload, "name", "slug", "title") == "finder"
    assert source_utils.first_nonempty_str(payload, "count", "missing") is None


def test_explicit_entry_source_url_excludes_context_source_url() -> None:
    payload = {
        "source_url": "https://example.com/marketplace.json",
        "url": "https://github.com/example/skills",
    }

    assert source_utils.explicit_entry_source_url(
        {"source_url": "https://example.com/marketplace.json"},
        "source_url",
        default_source_url="https://example.com/marketplace.json",
    ) is None
    assert source_utils.explicit_entry_source_url(
        payload,
        "source_url",
        "url",
        default_source_url="https://example.com/marketplace.json",
    ) == "https://github.com/example/skills"


def test_marketplace_repo_fields_normalizes_github_tree_repo_url() -> None:
    fields = source_utils.marketplace_repo_fields(
        {
            "repo_url": "https://github.com/org/repo/tree/main/examples/skills/foo",
        },
        repo_path_keys=("path", "repo_path"),
    )

    assert fields.repo_url == "https://github.com/org/repo"
    assert fields.repo_ref == "main"
    assert fields.repo_path == "examples/skills/foo"


def test_marketplace_repo_fields_preserves_slash_branch_before_common_path_prefix() -> None:
    fields = source_utils.marketplace_repo_fields(
        {
            "repo_url": "https://github.com/org/repo/tree/feature/demo/examples/skills/foo",
        },
        repo_path_keys=("path", "repo_path"),
    )

    assert fields.repo_url == "https://github.com/org/repo"
    assert fields.repo_ref == "feature/demo"
    assert fields.repo_path == "examples/skills/foo"


def test_marketplace_repo_fields_preserves_explicit_ref_and_path() -> None:
    fields = source_utils.marketplace_repo_fields(
        {
            "repo": "https://github.com/org/repo/tree/main/examples/skills/foo",
            "ref": "release/v1",
            "repo_path": "skills/bar",
        },
        repo_path_keys=("path", "repo_path"),
    )

    assert fields.repo_url == "https://github.com/org/repo"
    assert fields.repo_ref == "release/v1"
    assert fields.repo_path == "skills/bar"


def test_extract_dict_entries_reads_first_matching_list_key() -> None:
    payload: dict[str, object] = {
        "ignored": [{"name": "ignored"}],
        "entries": [{"name": "alpha"}, "skip", {"name": "beta"}],
    }

    entries = source_utils.extract_dict_entries(payload, ("missing", "entries", "ignored"))

    assert entries == [{"name": "alpha"}, {"name": "beta"}]


def test_extract_dict_entries_can_read_mapping_values() -> None:
    payload = {
        "alpha": {"name": "alpha"},
        "beta": {"name": "beta"},
    }

    entries = source_utils.extract_dict_entries(
        payload,
        ("entries",),
        allow_mapping_values=True,
    )

    assert entries == [{"name": "alpha"}, {"name": "beta"}]


def test_extract_dict_entries_rejects_mixed_mapping_values() -> None:
    payload = {
        "alpha": {"name": "alpha"},
        "beta": "skip",
    }

    with pytest.raises(ValueError, match="Unsupported marketplace payload format"):
        source_utils.extract_dict_entries(
            payload,
            ("entries",),
            allow_mapping_values=True,
        )


def test_clone_sparse_checkout_builds_expected_commands(tmp_path: Path) -> None:
    commands: list[list[str]] = []

    source_utils.clone_sparse_checkout(
        repo_url="https://github.com/example/skills",
        repo_ref="main",
        repo_subdir="skills/alpha",
        destination_dir=tmp_path / "repo",
        checkout_ref="abc123",
        run_git_fn=commands.append,
    )

    assert commands == [
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--filter=blob:none",
            "--sparse",
            "--branch",
            "main",
            "https://github.com/example/skills",
            str(tmp_path / "repo"),
        ],
        [
            "git",
            "-C",
            str(tmp_path / "repo"),
            "sparse-checkout",
            "set",
            "skills/alpha",
        ],
        ["git", "-C", str(tmp_path / "repo"), "checkout", "abc123"],
    ]


def test_clone_sparse_checkout_uses_default_checkout_without_ref(tmp_path: Path) -> None:
    commands: list[list[str]] = []

    source_utils.clone_sparse_checkout(
        repo_url="https://github.com/example/skills",
        repo_ref=None,
        repo_subdir="skills/alpha",
        destination_dir=tmp_path / "repo",
        run_git_fn=commands.append,
    )

    assert commands[-1] == ["git", "-C", str(tmp_path / "repo"), "checkout"]


@pytest.mark.parametrize(
    ("pinned_revision", "expected"),
    [
        (None, None),
        ("local", None),
        ("abc123", "abc123"),
    ],
)
def test_pinned_checkout_ref_ignores_local_revision(
    pinned_revision: str | None,
    expected: str | None,
) -> None:
    assert (
        source_utils.pinned_checkout_ref(pinned_revision, local_revision="local")
        == expected
    )


def test_resolve_git_path_oid_if_commit_skips_missing_commit(tmp_path: Path) -> None:
    calls: list[tuple[Path, str, str]] = []

    assert (
        source_utils.resolve_git_path_oid_if_commit(
            tmp_path,
            None,
            "skills/alpha",
            resolve_git_path_oid_fn=lambda *args: calls.append(args) or "oid",
        )
        is None
    )
    assert calls == []


def test_resolve_git_path_oid_if_commit_delegates_for_commit(tmp_path: Path) -> None:
    assert (
        source_utils.resolve_git_path_oid_if_commit(
            tmp_path,
            "abc123",
            "skills/alpha",
            resolve_git_path_oid_fn=lambda repo, commit, path: f"{repo.name}:{commit}:{path}",
        )
        == f"{tmp_path.name}:abc123:skills/alpha"
    )


def test_parse_github_url_preserves_slash_branch_for_skill_path() -> None:
    parsed = source_utils.parse_github_url(
        "https://github.com/example/skills/blob/feature/demo/skills/alpha/SKILL.md"
    )

    assert parsed is not None
    assert parsed.repo_url == "https://github.com/example/skills"
    assert parsed.repo_ref == "feature/demo"
    assert parsed.repo_path == "skills/alpha/SKILL.md"


def test_parse_raw_github_url_keeps_main_ref_for_nested_skill_path() -> None:
    parsed = source_utils.parse_github_url(
        "https://raw.githubusercontent.com/org/repo/main/examples/skills/foo/SKILL.md"
    )

    assert parsed is not None
    assert parsed.repo_url == "https://github.com/org/repo"
    assert parsed.repo_ref == "main"
    assert parsed.repo_path == "examples/skills/foo/SKILL.md"


def test_parse_raw_github_url_preserves_slash_branch_before_common_path_prefix() -> None:
    parsed = source_utils.parse_github_url(
        "https://raw.githubusercontent.com/org/repo/feature/demo/examples/skills/foo/SKILL.md"
    )

    assert parsed is not None
    assert parsed.repo_url == "https://github.com/org/repo"
    assert parsed.repo_ref == "feature/demo"
    assert parsed.repo_path == "examples/skills/foo/SKILL.md"


def test_parse_raw_github_url_normalizes_host_case() -> None:
    parsed = source_utils.parse_github_url(
        "https://RAW.GitHubUserContent.com/org/repo/main/skills/foo/SKILL.md"
    )

    assert parsed is not None
    assert parsed.repo_url == "https://github.com/org/repo"
    assert parsed.repo_ref == "main"
    assert parsed.repo_path == "skills/foo/SKILL.md"


def test_normalize_marketplace_url_preserves_slash_branch_for_github_blob() -> None:
    normalized = source_utils.normalize_marketplace_url(
        "https://github.com/example/skills/blob/feature/demo/skills/alpha/marketplace.json"
    )

    assert normalized == (
        "https://raw.githubusercontent.com/example/skills/"
        "feature/demo/skills/alpha/marketplace.json"
    )


def test_normalize_marketplace_url_keeps_raw_github_url_stable() -> None:
    raw_url = (
        "https://raw.githubusercontent.com/example/skills/"
        "release/candidate/.claude-plugin/marketplace.json"
    )

    assert source_utils.normalize_marketplace_url(raw_url) == raw_url


def test_normalize_marketplace_url_leaves_github_tree_url_for_candidate_expansion() -> None:
    tree_url = "https://github.com/example/skills/tree/main/plugins/example"

    assert source_utils.normalize_marketplace_url(tree_url) == tree_url


@pytest.mark.parametrize("kind", ["tree", "blob"])
def test_parse_github_url_keeps_default_branch_before_nested_skill_path(
    kind: str,
) -> None:
    parsed = source_utils.parse_github_url(
        f"https://github.com/org/repo/{kind}/main/examples/skills/foo"
    )

    assert parsed is not None
    assert parsed.repo_url == "https://github.com/org/repo"
    assert parsed.repo_ref == "main"
    assert parsed.repo_path == "examples/skills/foo"


@pytest.mark.parametrize("repo_ref", ["dev", "gh-pages"])
def test_parse_github_url_keeps_common_branch_before_nested_skill_path(
    repo_ref: str,
) -> None:
    parsed = source_utils.parse_github_url(
        f"https://github.com/example/skills/tree/{repo_ref}/examples/skills/alpha"
    )

    assert parsed is not None
    assert parsed.repo_url == "https://github.com/example/skills"
    assert parsed.repo_ref == repo_ref
    assert parsed.repo_path == "examples/skills/alpha"


def test_parse_github_url_keeps_single_segment_branch_before_common_path_prefix() -> None:
    parsed = source_utils.parse_github_url(
        "https://github.com/example/skills/tree/candidate/examples/skills/alpha"
    )

    assert parsed is not None
    assert parsed.repo_url == "https://github.com/example/skills"
    assert parsed.repo_ref == "candidate"
    assert parsed.repo_path == "examples/skills/alpha"


def test_parse_github_url_preserves_namespaced_branch_before_common_path_prefix() -> None:
    parsed = source_utils.parse_github_url(
        "https://github.com/org/repo/tree/feature/examples/skills/foo"
    )

    assert parsed is not None
    assert parsed.repo_url == "https://github.com/org/repo"
    assert parsed.repo_ref == "feature/examples"
    assert parsed.repo_path == "skills/foo"


def test_parse_github_url_preserves_slash_branch_before_common_path_prefix() -> None:
    parsed = source_utils.parse_github_url(
        "https://github.com/org/repo/tree/feature/demo/examples/skills/foo"
    )

    assert parsed is not None
    assert parsed.repo_url == "https://github.com/org/repo"
    assert parsed.repo_ref == "feature/demo"
    assert parsed.repo_path == "examples/skills/foo"


def test_parse_github_url_preserves_anchor_named_slash_branch() -> None:
    parsed = source_utils.parse_github_url(
        "https://github.com/org/repo/tree/feature/skills/foo"
    )

    assert parsed is not None
    assert parsed.repo_url == "https://github.com/org/repo"
    assert parsed.repo_ref == "feature/skills"
    assert parsed.repo_path == "foo"


def test_parse_raw_github_url_preserves_slash_branch_for_marketplace_path() -> None:
    parsed = source_utils.parse_github_url(
        "https://raw.githubusercontent.com/example/skills/release/candidate/.claude-plugin/marketplace.json"
    )

    assert parsed is not None
    assert parsed.repo_url == "https://github.com/example/skills"
    assert parsed.repo_ref == "release/candidate"
    assert parsed.repo_path == ".claude-plugin/marketplace.json"


def test_read_installed_source_file_returns_empty_result_when_missing(tmp_path: Path) -> None:
    result = source_utils.read_installed_source_file(
        tmp_path / ".source.json",
        parse_payload=lambda payload: payload["name"],
    )

    assert result.source is None
    assert result.error is None


def test_read_installed_source_file_validates_root_object(tmp_path: Path) -> None:
    sidecar = tmp_path / ".source.json"
    sidecar.write_text("[]", encoding="utf-8")

    result = source_utils.read_installed_source_file(
        sidecar,
        parse_payload=lambda payload: payload["name"],
    )

    assert result.source is None
    assert result.error == "metadata root must be an object"


def test_read_installed_source_file_returns_parse_errors(tmp_path: Path) -> None:
    sidecar = tmp_path / ".source.json"
    sidecar.write_text('{"name": ""}', encoding="utf-8")

    def parse_payload(payload: dict[str, object]) -> str:
        name = payload.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("name is required")
        return name

    result = source_utils.read_installed_source_file(sidecar, parse_payload=parse_payload)

    assert result.source is None
    assert result.error == "name is required"


def test_installed_source_payload_serializes_common_sidecar_fields() -> None:
    source = _InstalledSourcePayload()

    assert source_utils.installed_source_payload(source) == {
        "schema_version": 1,
        "installed_via": "marketplace",
        "source_origin": "remote",
        "repo_url": "https://github.com/example/skills",
        "repo_ref": "main",
        "repo_path": "skills/alpha",
        "source_url": None,
        "installed_commit": "abc123",
        "installed_path_oid": "def456",
        "installed_revision": "abc123",
        "installed_at": "2026-02-25T00:00:00Z",
        "content_fingerprint": _VALID_SHA256_FINGERPRINT,
    }


def test_write_installed_source_file_merges_extra_payload(tmp_path: Path) -> None:
    sidecar = tmp_path / ".source.json"

    source_utils.write_installed_source_file(
        sidecar,
        _InstalledSourcePayload(),
        extra_payload={"name": "alpha", "installed_files": ["agent.yaml"]},
    )

    payload = json.loads(sidecar.read_text(encoding="utf-8"))

    assert payload["repo_url"] == "https://github.com/example/skills"
    assert payload["name"] == "alpha"
    assert payload["installed_files"] == ["agent.yaml"]


def test_compute_directory_content_fingerprint_skips_sidecar_and_ignored_paths(
    tmp_path: Path,
) -> None:
    root = tmp_path / "source"
    root.mkdir()
    (root / "manifest.yaml").write_text("name: alpha\n", encoding="utf-8")
    sidecar = root / ".source.json"
    sidecar.write_text('{"generated": true}', encoding="utf-8")

    before = source_utils.compute_directory_content_fingerprint(
        root,
        sidecar_path=sidecar,
        ignore_path=lambda path: "__pycache__" in path.parts,
    )

    sidecar.write_text('{"generated": false}', encoding="utf-8")
    pycache = root / "__pycache__"
    pycache.mkdir()
    (pycache / "manifest.cpython-314.pyc").write_bytes(b"generated bytecode")

    assert (
        source_utils.compute_directory_content_fingerprint(
            root,
            sidecar_path=sidecar,
            ignore_path=lambda path: "__pycache__" in path.parts,
        )
        == before
    )


@pytest.mark.parametrize(
    (
        "available_path_oid",
        "current_path_oid",
        "available_revision",
        "current_revision",
        "expected_status",
        "expected_detail",
    ),
    [
        ("new-tree", "old-tree", "new-rev", "old-rev", "update_available", "content changed"),
        ("same-tree", "same-tree", "new-rev", "old-rev", "up_to_date", "already up to date"),
        (None, "old-tree", "new-rev", "old-rev", "update_available", "new revision available"),
        (None, "old-tree", "same-rev", "same-rev", "up_to_date", "already up to date"),
    ],
)
def test_decide_source_update_status(
    available_path_oid: str | None,
    current_path_oid: str | None,
    available_revision: str,
    current_revision: str,
    expected_status: source_utils.SourceUpdateStatus,
    expected_detail: str,
) -> None:
    decision = source_utils.decide_source_update_status(
        available_path_oid=available_path_oid,
        current_path_oid=current_path_oid,
        available_revision=available_revision,
        current_revision=current_revision,
        content_changed_detail="content changed",
    )

    assert decision.status == expected_status
    assert decision.detail == expected_detail


def test_parse_installed_source_fields_validates_and_normalizes() -> None:
    payload: dict[str, object] = {
        "schema_version": 1,
        "installed_via": "marketplace",
        "source_origin": "remote",
        "repo_url": "  https://github.com/example/skills  ",
        "repo_ref": " main ",
        "repo_path": "skills/example",
        "source_url": "",
        "installed_commit": " commit-sha ",
        "installed_path_oid": " path-oid ",
        "installed_revision": " abc123 ",
        "installed_at": " 2026-02-25T00:00:00Z ",
        "content_fingerprint": _VALID_SHA256_FINGERPRINT,
    }

    parsed = source_utils.parse_installed_source_fields(
        payload,
        expected_schema_version=1,
        normalize_repo_path=lambda value: value.strip("/"),
    )

    assert parsed.repo_url == "https://github.com/example/skills"
    assert parsed.repo_ref == "main"
    assert parsed.source_url is None
    assert parsed.installed_commit == "commit-sha"
    assert parsed.installed_path_oid == "path-oid"
    assert parsed.installed_revision == "abc123"
    assert parsed.installed_at == "2026-02-25T00:00:00Z"


def test_parse_installed_source_fields_rejects_invalid_repo_path() -> None:
    payload: dict[str, object] = {
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
        "content_fingerprint": _VALID_SHA256_FINGERPRINT,
    }

    with pytest.raises(ValueError, match="repo_path is invalid"):
        source_utils.parse_installed_source_fields(
            payload,
            expected_schema_version=1,
            normalize_repo_path=lambda _: None,
        )


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    [
        ("repo_url", "   ", "repo_url is required"),
        ("repo_ref", 123, "repo_ref must be a string or null"),
        ("installed_commit", "", "installed_commit must be a non-empty string or null"),
        (
            "installed_path_oid",
            "   ",
            "installed_path_oid must be a non-empty string or null",
        ),
        ("installed_revision", "", "installed_revision is required"),
        ("installed_at", None, "installed_at is required"),
    ],
)
def test_parse_installed_source_fields_rejects_invalid_string_fields(
    field_name: str,
    value: object,
    message: str,
) -> None:
    payload: dict[str, object] = {
        "schema_version": 1,
        "installed_via": "marketplace",
        "source_origin": "remote",
        "repo_url": "https://github.com/example/skills",
        "repo_ref": "main",
        "repo_path": "skills/example",
        "source_url": None,
        "installed_commit": None,
        "installed_path_oid": None,
        "installed_revision": "abc123",
        "installed_at": "2026-02-25T00:00:00Z",
        "content_fingerprint": _VALID_SHA256_FINGERPRINT,
    }
    payload[field_name] = value

    with pytest.raises(ValueError, match=message):
        source_utils.parse_installed_source_fields(
            payload,
            expected_schema_version=1,
            normalize_repo_path=lambda value: value.strip("/"),
        )


@pytest.mark.parametrize(
    "value",
    [
        None,
        "",
        "sha256:",
        "sha256:deadbeef",
        "sha256:" + ("g" * 64),
        "SHA256:" + ("0" * 64),
    ],
)
def test_parse_installed_source_fields_rejects_invalid_content_fingerprint(
    value: object,
) -> None:
    payload: dict[str, object] = {
        "schema_version": 1,
        "installed_via": "marketplace",
        "source_origin": "remote",
        "repo_url": "https://github.com/example/skills",
        "repo_ref": "main",
        "repo_path": "skills/example",
        "source_url": None,
        "installed_commit": None,
        "installed_path_oid": None,
        "installed_revision": "abc123",
        "installed_at": "2026-02-25T00:00:00Z",
        "content_fingerprint": value,
    }

    with pytest.raises(ValueError, match="content_fingerprint must be a sha256 fingerprint"):
        source_utils.parse_installed_source_fields(
            payload,
            expected_schema_version=1,
            normalize_repo_path=lambda value: value.strip("/"),
        )


@pytest.mark.parametrize(
    ("value", "allow_current_dir", "expected"),
    [
        ("skills/example", False, "skills/example"),
        ("skills/example/", False, "skills/example"),
        ("skills\\example", False, "skills/example"),
        ("/absolute/path", False, None),
        ("C:\\skills\\example", False, None),
        ("C:/skills/example", False, None),
        ("../escape", False, None),
        ("skills/../escape", False, None),
        ("", False, None),
        ("   ", False, None),
        (".", False, None),
        (".", True, "."),
    ],
)
def test_normalize_relative_repo_path(
    value: str,
    allow_current_dir: bool,
    expected: str | None,
) -> None:
    assert (
        source_utils.normalize_relative_repo_path(
            value,
            allow_current_dir=allow_current_dir,
        )
        == expected
    )


@pytest.mark.parametrize(
    ("repo_path", "manifest_filename", "expected"),
    [
        ("skills/alpha/SKILL.md", "SKILL.md", "skills/alpha"),
        ("skills/alpha/skill.md", "SKILL.md", "skills/alpha"),
        (r"skills\alpha\SKILL.md", "SKILL.md", "skills/alpha"),
        ("plugins/finder/plugin.yaml", "plugin.yaml", "plugins/finder"),
        ("plugins/finder/PLUGIN.YAML", "plugin.yaml", "plugins/finder"),
        ("packs/demo/card-pack.yaml", "card-pack.yaml", "packs/demo"),
        ("plugin.yaml", "plugin.yaml", "."),
        ("skills/alpha", "SKILL.md", "skills/alpha"),
    ],
)
def test_repo_subdir_for_manifest_path(
    repo_path: str,
    manifest_filename: str,
    expected: str,
) -> None:
    assert (
        source_utils.repo_subdir_for_manifest_path(repo_path, manifest_filename)
        == expected
    )


@pytest.mark.parametrize(
    ("repo_path", "manifest_filename", "expected"),
    [
        ("skills/alpha/SKILL.md", "SKILL.md", "alpha"),
        ("skills/alpha/skill.md", "SKILL.md", "alpha"),
        (r"plugins\finder\plugin.yaml", "plugin.yaml", "finder"),
        (r"plugins\finder\PLUGIN.YAML", "plugin.yaml", "finder"),
        ("packs/demo/card-pack.yaml", "card-pack.yaml", "demo"),
        ("skills/alpha", "SKILL.md", "alpha"),
    ],
)
def test_repo_name_for_manifest_path(
    repo_path: str,
    manifest_filename: str,
    expected: str,
) -> None:
    assert (
        source_utils.repo_name_for_manifest_path(repo_path, manifest_filename)
        == expected
    )


def test_derive_local_repo_root_from_marketplace_file(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    marketplace = repo_root / ".claude-plugin" / "marketplace.json"
    marketplace.parent.mkdir(parents=True)
    marketplace.write_text("{}", encoding="utf-8")

    resolved = source_utils.derive_local_repo_root(str(marketplace))

    assert resolved == str(repo_root)


def test_derive_local_repo_root_ignores_remote_git_sources() -> None:
    assert source_utils.derive_local_repo_root("https://github.com/example/skills") is None
    assert source_utils.derive_local_repo_root("git@github.com:example/skills.git") is None


def test_resolve_local_repo_handles_paths_and_file_uris(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    assert source_utils.resolve_local_repo(repo_root.as_posix()) == repo_root
    assert source_utils.resolve_local_repo(repo_root.as_uri()) == repo_root


def test_resolve_local_repo_ignores_remote_git_sources() -> None:
    assert source_utils.resolve_local_repo("https://github.com/example/skills") is None
    assert source_utils.resolve_local_repo("git@github.com:example/skills.git") is None


def test_marketplace_source_context_from_github_marketplace_url() -> None:
    context = source_utils.marketplace_source_context(
        "https://github.com/example/skills/blob/main/.claude-plugin/marketplace.json"
    )

    assert context.as_validation_context() == {
        "source_url": "https://github.com/example/skills/blob/main/.claude-plugin/marketplace.json",
        "repo_url": "https://github.com/example/skills",
        "repo_ref": "main",
    }


def test_marketplace_source_context_from_local_marketplace_file(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    marketplace = repo_root / "marketplace.json"
    marketplace.parent.mkdir(parents=True)
    marketplace.write_text("{}", encoding="utf-8")

    context = source_utils.marketplace_source_context(str(marketplace))

    assert context.as_validation_context() == {
        "source_url": str(marketplace),
        "repo_url": str(repo_root),
        "repo_ref": None,
    }


def test_load_local_marketplace_payload_rejects_directory_without_manifest(
    tmp_path: Path,
) -> None:
    marketplace_dir = tmp_path / "repo"
    marketplace_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="marketplace.json not found"):
        source_utils.load_local_marketplace_payload(marketplace_dir.as_posix())


def test_load_local_marketplace_payload_rejects_file_uri_directory_without_manifest(
    tmp_path: Path,
) -> None:
    marketplace_dir = tmp_path / "repo"
    marketplace_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="marketplace.json not found"):
        source_utils.load_local_marketplace_payload(marketplace_dir.as_uri())


def test_load_local_marketplace_payload_ignores_remote_git_sources() -> None:
    assert source_utils.load_local_marketplace_payload("https://github.com/example/skills") is None
    assert source_utils.load_local_marketplace_payload("git@github.com:example/skills.git") is None


def test_fetch_marketplace_entries_continues_after_bad_local_candidate(
    tmp_path: Path,
) -> None:
    bad_candidate = tmp_path / "repo"
    bad_candidate.mkdir()
    good_candidate = tmp_path / "marketplace.json"
    good_candidate.write_text('{"entries": [{"name": "alpha"}]}', encoding="utf-8")

    entries, source = asyncio.run(
        source_utils.fetch_marketplace_entries_with_source(
            "ignored",
            candidate_urls=lambda _url: [bad_candidate.as_posix(), good_candidate.as_posix()],
            normalize_url=lambda value: value,
            load_local_payload=source_utils.load_local_marketplace_payload,
            parse_payload=lambda payload, _source: source_utils.extract_dict_entries(
                payload,
                ("entries",),
            ),
        )
    )

    assert entries == [{"name": "alpha"}]
    assert source == good_candidate.as_posix()


def test_resolve_repo_subdir_rejects_escape_with_label(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    with pytest.raises(ValueError, match="Plugin path escapes repository root"):
        source_utils.resolve_repo_subdir(repo_root, "../outside", label="Plugin")


def test_resolve_repo_subdir_accepts_nested_path(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    nested = repo_root / "plugins" / "demo"
    nested.mkdir(parents=True)

    assert source_utils.resolve_repo_subdir(
        repo_root,
        "plugins/demo",
        label="Plugin",
    ) == nested.resolve()


def test_extract_tar_safely_rejects_links_outside_destination(tmp_path: Path) -> None:
    archive_file = io.BytesIO()
    with tarfile.open(fileobj=archive_file, mode="w") as archive:
        content = b"schema_version: 1\nname: finder\ncommands: {}\n"
        manifest = tarfile.TarInfo("plugin.yaml")
        manifest.size = len(content)
        archive.addfile(manifest, io.BytesIO(content))

        link = tarfile.TarInfo("safe_name")
        link.type = tarfile.SYMTYPE
        link.linkname = "../outside"
        archive.addfile(link)

    archive_file.seek(0)

    with pytest.raises(tarfile.TarError):
        source_utils.extract_tar_safely(archive_file, tmp_path / "plugin")

    assert not (tmp_path / "outside").exists()


def test_extract_tar_safely_rejects_path_traversal_member(tmp_path: Path) -> None:
    archive_file = io.BytesIO()
    with tarfile.open(fileobj=archive_file, mode="w") as archive:
        content = b"outside"
        escaped = tarfile.TarInfo("../outside.txt")
        escaped.size = len(content)
        archive.addfile(escaped, io.BytesIO(content))

    archive_file.seek(0)

    with pytest.raises(tarfile.TarError, match="escapes destination"):
        source_utils.extract_tar_safely(archive_file, tmp_path / "plugin")

    assert not (tmp_path / "outside.txt").exists()
