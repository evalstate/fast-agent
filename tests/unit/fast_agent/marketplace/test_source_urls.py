from __future__ import annotations

import pytest

from fast_agent.marketplace import source_urls


def test_candidate_marketplace_urls_for_github_repo() -> None:
    urls = source_urls.candidate_marketplace_urls("https://github.com/example/skills")
    assert urls == [
        "https://raw.githubusercontent.com/example/skills/main/.claude-plugin/marketplace.json",
        "https://raw.githubusercontent.com/example/skills/main/marketplace.json",
        "https://raw.githubusercontent.com/example/skills/master/.claude-plugin/marketplace.json",
        "https://raw.githubusercontent.com/example/skills/master/marketplace.json",
    ]


def test_candidate_marketplace_urls_normalize_github_host_case() -> None:
    urls = source_urls.candidate_marketplace_urls("https://GitHub.com/example/skills")

    assert urls == [
        "https://raw.githubusercontent.com/example/skills/main/.claude-plugin/marketplace.json",
        "https://raw.githubusercontent.com/example/skills/main/marketplace.json",
        "https://raw.githubusercontent.com/example/skills/master/.claude-plugin/marketplace.json",
        "https://raw.githubusercontent.com/example/skills/master/marketplace.json",
    ]


def test_github_host_helpers_normalize_case_and_padding() -> None:
    assert source_urls.is_github_web_host(" GitHub.com ")
    assert source_urls.is_github_web_host(" WWW.GITHUB.COM ")
    assert source_urls.is_github_raw_host(" RAW.GITHUBUSERCONTENT.COM ")
    assert not source_urls.is_github_web_host("raw.githubusercontent.com")


def test_normalize_marketplace_url_strips_outer_whitespace_for_github_blob() -> None:
    normalized = source_urls.normalize_marketplace_url(
        "  https://github.com/example/skills/blob/main/skills/alpha/marketplace.json  "
    )

    assert normalized == (
        "https://raw.githubusercontent.com/example/skills/"
        "main/skills/alpha/marketplace.json"
    )


def test_parse_github_url_strips_outer_whitespace() -> None:
    parsed = source_urls.parse_github_url(
        "  https://github.com/example/skills/tree/main/skills/alpha  "
    )

    assert parsed is not None
    assert parsed.repo_url == "https://github.com/example/skills"
    assert parsed.repo_ref == "main"
    assert parsed.repo_path == "skills/alpha"


def test_candidate_marketplace_urls_keep_default_branch_before_nested_skill_path() -> None:
    urls = source_urls.candidate_marketplace_urls(
        "https://github.com/org/repo/tree/main/examples/skills/foo"
    )

    assert urls == [
        "https://raw.githubusercontent.com/org/repo/main/examples/skills/foo/.claude-plugin/marketplace.json",
        "https://raw.githubusercontent.com/org/repo/main/examples/skills/foo/marketplace.json",
    ]


def test_candidate_marketplace_urls_preserve_slash_branch_for_skill_path() -> None:
    urls = source_urls.candidate_marketplace_urls(
        "https://github.com/org/repo/tree/feature/demo/skills/foo"
    )

    assert urls == [
        "https://raw.githubusercontent.com/org/repo/feature/demo/skills/foo/.claude-plugin/marketplace.json",
        "https://raw.githubusercontent.com/org/repo/feature/demo/skills/foo/marketplace.json",
    ]


def test_candidate_marketplace_urls_preserve_anchor_named_slash_branch() -> None:
    urls = source_urls.candidate_marketplace_urls(
        "https://github.com/org/repo/tree/feature/skills/foo"
    )

    assert urls == [
        "https://raw.githubusercontent.com/org/repo/feature/skills/foo/.claude-plugin/marketplace.json",
        "https://raw.githubusercontent.com/org/repo/feature/skills/foo/marketplace.json",
    ]


def test_candidate_marketplace_urls_detects_uppercase_marketplace_filename() -> None:
    urls = source_urls.candidate_marketplace_urls(
        "https://github.com/org/repo/tree/main/catalog/MARKETPLACE.JSON"
    )

    assert urls == [
        "https://raw.githubusercontent.com/org/repo/main/catalog/MARKETPLACE.JSON",
    ]


def test_is_probable_url_requires_scheme_and_host() -> None:
    assert source_urls.is_probable_url("https://github.com/example/skills") is True
    assert source_urls.is_probable_url("file:///tmp/marketplace.json") is False
    assert source_urls.is_probable_url("git@github.com:example/skills.git") is False
    assert source_urls.is_probable_url("plugins/finder") is False
    assert source_urls.is_probable_url("") is False


def test_is_git_source_url_accepts_scp_style_git_sources() -> None:
    assert source_urls.is_git_source_url("https://github.com/example/skills") is True
    assert source_urls.is_git_source_url("git@github.com:example/skills.git") is True
    assert source_urls.is_git_source_url("plugins/finder") is False
    assert source_urls.is_git_source_url("") is False


def test_github_raw_file_url_builds_raw_url_for_github_repo() -> None:
    assert (
        source_urls.github_raw_file_url(
            repo_url="https://github.com/example/skills",
            repo_ref="release/v1",
            repo_path="skills/alpha/SKILL.md",
        )
        == "https://raw.githubusercontent.com/example/skills/release/v1/skills/alpha/SKILL.md"
    )


def test_github_raw_file_url_rejects_non_github_repo_url() -> None:
    assert (
        source_urls.github_raw_file_url(
            repo_url="https://gitlab.com/example/skills",
            repo_ref="main",
            repo_path="skills/alpha/SKILL.md",
        )
        is None
    )


def test_first_nonempty_str_returns_first_trimmed_string() -> None:
    payload: dict[str, object] = {
        "name": "   ",
        "slug": " finder ",
        "title": "Ignored",
        "count": 3,
    }

    assert source_urls.first_nonempty_str(payload, "name", "slug", "title") == "finder"
    assert source_urls.first_nonempty_str(payload, "count", "missing") is None


def test_explicit_entry_source_url_excludes_context_source_url() -> None:
    payload = {
        "source_url": "https://example.com/marketplace.json",
        "url": "https://github.com/example/skills",
    }

    assert source_urls.explicit_entry_source_url(
        {"source_url": "https://example.com/marketplace.json"},
        "source_url",
        default_source_url="https://example.com/marketplace.json",
    ) is None
    assert source_urls.explicit_entry_source_url(
        payload,
        "source_url",
        "url",
        default_source_url="https://example.com/marketplace.json",
    ) == "https://github.com/example/skills"


def test_marketplace_repo_fields_normalizes_github_tree_repo_url() -> None:
    fields = source_urls.marketplace_repo_fields(
        {
            "repo_url": "https://github.com/org/repo/tree/main/examples/skills/foo",
        },
        repo_path_keys=("path", "repo_path"),
    )

    assert fields.repo_url == "https://github.com/org/repo"
    assert fields.repo_ref == "main"
    assert fields.repo_path == "examples/skills/foo"


def test_marketplace_repo_fields_preserves_slash_branch_before_common_path_prefix() -> None:
    fields = source_urls.marketplace_repo_fields(
        {
            "repo_url": "https://github.com/org/repo/tree/feature/demo/examples/skills/foo",
        },
        repo_path_keys=("path", "repo_path"),
    )

    assert fields.repo_url == "https://github.com/org/repo"
    assert fields.repo_ref == "feature/demo"
    assert fields.repo_path == "examples/skills/foo"


def test_marketplace_repo_fields_preserves_explicit_ref_and_path() -> None:
    fields = source_urls.marketplace_repo_fields(
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


def test_parse_github_url_preserves_slash_branch_for_skill_path() -> None:
    parsed = source_urls.parse_github_url(
        "https://github.com/example/skills/blob/feature/demo/skills/alpha/SKILL.md"
    )

    assert parsed is not None
    assert parsed.repo_url == "https://github.com/example/skills"
    assert parsed.repo_ref == "feature/demo"
    assert parsed.repo_path == "skills/alpha/SKILL.md"


def test_parse_raw_github_url_keeps_main_ref_for_nested_skill_path() -> None:
    parsed = source_urls.parse_github_url(
        "https://raw.githubusercontent.com/org/repo/main/examples/skills/foo/SKILL.md"
    )

    assert parsed is not None
    assert parsed.repo_url == "https://github.com/org/repo"
    assert parsed.repo_ref == "main"
    assert parsed.repo_path == "examples/skills/foo/SKILL.md"


def test_parse_raw_github_url_preserves_slash_branch_before_common_path_prefix() -> None:
    parsed = source_urls.parse_github_url(
        "https://raw.githubusercontent.com/org/repo/feature/demo/examples/skills/foo/SKILL.md"
    )

    assert parsed is not None
    assert parsed.repo_url == "https://github.com/org/repo"
    assert parsed.repo_ref == "feature/demo"
    assert parsed.repo_path == "examples/skills/foo/SKILL.md"


def test_parse_raw_github_url_normalizes_host_case() -> None:
    parsed = source_urls.parse_github_url(
        "https://RAW.GitHubUserContent.com/org/repo/main/skills/foo/SKILL.md"
    )

    assert parsed is not None
    assert parsed.repo_url == "https://github.com/org/repo"
    assert parsed.repo_ref == "main"
    assert parsed.repo_path == "skills/foo/SKILL.md"


def test_normalize_marketplace_url_preserves_slash_branch_for_github_blob() -> None:
    normalized = source_urls.normalize_marketplace_url(
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

    assert source_urls.normalize_marketplace_url(raw_url) == raw_url


def test_normalize_marketplace_url_leaves_github_tree_url_for_candidate_expansion() -> None:
    tree_url = "https://github.com/example/skills/tree/main/plugins/example"

    assert source_urls.normalize_marketplace_url(tree_url) == tree_url


@pytest.mark.parametrize("kind", ["tree", "blob"])
def test_parse_github_url_keeps_default_branch_before_nested_skill_path(
    kind: str,
) -> None:
    parsed = source_urls.parse_github_url(
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
    parsed = source_urls.parse_github_url(
        f"https://github.com/example/skills/tree/{repo_ref}/examples/skills/alpha"
    )

    assert parsed is not None
    assert parsed.repo_url == "https://github.com/example/skills"
    assert parsed.repo_ref == repo_ref
    assert parsed.repo_path == "examples/skills/alpha"


def test_parse_github_url_keeps_single_segment_branch_before_common_path_prefix() -> None:
    parsed = source_urls.parse_github_url(
        "https://github.com/example/skills/tree/candidate/examples/skills/alpha"
    )

    assert parsed is not None
    assert parsed.repo_url == "https://github.com/example/skills"
    assert parsed.repo_ref == "candidate"
    assert parsed.repo_path == "examples/skills/alpha"


def test_parse_github_url_preserves_namespaced_branch_before_common_path_prefix() -> None:
    parsed = source_urls.parse_github_url(
        "https://github.com/org/repo/tree/feature/examples/skills/foo"
    )

    assert parsed is not None
    assert parsed.repo_url == "https://github.com/org/repo"
    assert parsed.repo_ref == "feature/examples"
    assert parsed.repo_path == "skills/foo"


def test_parse_github_url_preserves_slash_branch_before_common_path_prefix() -> None:
    parsed = source_urls.parse_github_url(
        "https://github.com/org/repo/tree/feature/demo/examples/skills/foo"
    )

    assert parsed is not None
    assert parsed.repo_url == "https://github.com/org/repo"
    assert parsed.repo_ref == "feature/demo"
    assert parsed.repo_path == "examples/skills/foo"


def test_parse_github_url_preserves_anchor_named_slash_branch() -> None:
    parsed = source_urls.parse_github_url(
        "https://github.com/org/repo/tree/feature/skills/foo"
    )

    assert parsed is not None
    assert parsed.repo_url == "https://github.com/org/repo"
    assert parsed.repo_ref == "feature/skills"
    assert parsed.repo_path == "foo"


def test_parse_raw_github_url_preserves_slash_branch_for_marketplace_path() -> None:
    parsed = source_urls.parse_github_url(
        "https://raw.githubusercontent.com/example/skills/release/candidate/.claude-plugin/marketplace.json"
    )

    assert parsed is not None
    assert parsed.repo_url == "https://github.com/example/skills"
    assert parsed.repo_ref == "release/candidate"
    assert parsed.repo_path == ".claude-plugin/marketplace.json"


