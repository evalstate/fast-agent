"""Marketplace source URL and repository field helpers."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from fast_agent.io.path_uri import file_uri_to_path
from fast_agent.marketplace.source_models import MarketplaceRepoFields, ParsedGitHubUrl
from fast_agent.utils.text import strip_casefold, strip_str_to_none

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from urllib.parse import ParseResult

_GITHUB_WEB_HOSTS = frozenset({"github.com", "www.github.com"})
GITHUB_RAW_HOST = "raw.githubusercontent.com"
_GITHUB_SOURCE_PATH_ANCHORS = frozenset(
    {
        ".claude-plugin",
        "cards",
        "marketplace.json",
        "packs",
        "plugins",
        "skills",
        "SKILL.md",
    }
)
_GITHUB_COMMON_SINGLE_SEGMENT_REFS = frozenset(
    {
        "dev",
        "develop",
        "development",
        "gh-pages",
        "main",
        "master",
        "trunk",
    }
)
_GITHUB_COMMON_REPO_PATH_PREFIXES = frozenset(
    {
        ".github",
        "docs",
        "examples",
        "resources",
        "src",
        "tests",
    }
)
_GITHUB_COMMON_SLASH_BRANCH_PREFIXES = frozenset(
    {
        "bugfix",
        "feature",
        "fix",
        "hotfix",
        "release",
    }
)
SCP_LIKE_GIT_SOURCE_RE = r"^[^@\s]+@[^:\s]+:[^\s]+$"
MARKETPLACE_REPO_URL_KEYS = ("repo", "repository", "git", "repo_url")
MARKETPLACE_REPO_REF_KEYS = ("repo_ref", "ref", "branch", "tag", "revision", "commit")


def normalize_marketplace_url(url: str) -> str:
    normalized = url.strip()
    parsed = urlparse(normalized)
    if is_github_web_host(parsed.netloc) or is_github_raw_host(parsed.netloc):
        parts = parsed.path.strip("/").split("/")
        is_github_blob = is_github_web_host(parsed.netloc) and (
            len(parts) >= 5 and parts[2] == "blob"
        )
        is_raw_github = is_github_raw_host(parsed.netloc)
        if is_github_blob or is_raw_github:
            parsed_source = parse_github_url(normalized)
            return _github_raw_content_url(parsed_source) or normalized
    return normalized


def is_github_web_host(host: str) -> bool:
    return strip_casefold(host) in _GITHUB_WEB_HOSTS


def is_github_raw_host(host: str) -> bool:
    return strip_casefold(host) == GITHUB_RAW_HOST


def path_name_matches(path: PurePosixPath, filename: str) -> bool:
    return strip_casefold(path.name) == strip_casefold(filename)


def github_raw_file_url(
    *,
    repo_url: str,
    repo_ref: str | None,
    repo_path: str,
) -> str | None:
    parsed_repo = urlparse(repo_url)
    repo_parts = parsed_repo.path.strip("/").split("/")
    if not is_github_web_host(parsed_repo.netloc) or len(repo_parts) != 2:
        return None
    org, repo = repo_parts
    ref = repo_ref or "main"
    return f"https://{GITHUB_RAW_HOST}/{org}/{repo}/{ref}/{repo_path}"


def _github_raw_content_url(parsed_source: ParsedGitHubUrl | None) -> str | None:
    if parsed_source is None or parsed_source.repo_ref is None:
        return None
    return github_raw_file_url(
        repo_url=parsed_source.repo_url,
        repo_ref=parsed_source.repo_ref,
        repo_path=parsed_source.repo_path,
    )


def _is_local_marketplace_candidate(parsed: "ParseResult") -> bool:
    return parsed.scheme in {"file", ""} and parsed.netloc == ""


def _local_marketplace_path(parsed: "ParseResult") -> Path:
    if parsed.scheme == "file":
        return file_uri_to_path(parsed)
    return Path(parsed.path).expanduser()


def _candidate_local_marketplace_urls(normalized: str, parsed: "ParseResult") -> list[str]:
    path = _local_marketplace_path(parsed)
    if path.exists() and path.is_dir():
        claude_plugin = path / ".claude-plugin" / "marketplace.json"
        if claude_plugin.exists():
            return [claude_plugin.as_posix()]
        fallback = path / "marketplace.json"
        if fallback.exists():
            return [fallback.as_posix()]
    return [normalized]


def _candidate_github_marketplace_urls(normalized: str, parsed: "ParseResult") -> list[str]:
    if not is_github_web_host(parsed.netloc):
        return []

    parts = parsed.path.strip("/").split("/")
    if len(parts) < 2:
        return []

    org, repo = parts[:2]
    if len(parts) >= 4 and parts[2] in {"tree", "blob"}:
        parsed_source = parse_github_url(normalized)
        if parsed_source is not None and parsed_source.repo_ref is not None:
            return _github_marketplace_candidates(
                org,
                repo,
                parsed_source.repo_ref,
                parsed_source.repo_path,
            )

    if len(parts) == 2:
        return [
            *_github_marketplace_candidates(org, repo, "main", ""),
            *_github_marketplace_candidates(org, repo, "master", ""),
        ]

    return []


def candidate_marketplace_urls(url: str) -> list[str]:
    normalized = strip_str_to_none(url)
    if normalized is None:
        return []

    parsed = urlparse(normalized)
    if _is_local_marketplace_candidate(parsed):
        return _candidate_local_marketplace_urls(normalized, parsed)

    github_candidates = _candidate_github_marketplace_urls(normalized, parsed)
    if github_candidates:
        return github_candidates

    return [normalized]


def _github_marketplace_candidates(org: str, repo: str, ref: str, base_path: str) -> list[str]:
    suffixes = _marketplace_path_candidates(base_path)
    return [
        f"https://{GITHUB_RAW_HOST}/{org}/{repo}/{ref}/{suffix}"
        for suffix in suffixes
    ]


def _marketplace_path_candidates(base_path: str) -> list[str]:
    cleaned = base_path.strip().strip("/")
    if not cleaned:
        return [".claude-plugin/marketplace.json", "marketplace.json"]

    path = PurePosixPath(cleaned)
    if path_name_matches(path, "marketplace.json"):
        return [str(path)]
    if path.name == ".claude-plugin":
        return [str(path / "marketplace.json")]

    return [
        str(path / ".claude-plugin" / "marketplace.json"),
        str(path / "marketplace.json"),
    ]


def _split_github_ref_and_path(parts: "Sequence[str]") -> tuple[str, str] | None:
    if len(parts) < 2:
        return None

    if parts[0] in _GITHUB_COMMON_SINGLE_SEGMENT_REFS:
        return _github_ref_path_split(parts, repo_path_index=1)

    if parts[0] in _GITHUB_COMMON_SLASH_BRANCH_PREFIXES:
        prefixed_ref_split = _split_github_prefixed_ref(parts)
        if prefixed_ref_split is not None:
            return prefixed_ref_split

    if len(parts) >= 3 and parts[1] in _GITHUB_COMMON_REPO_PATH_PREFIXES:
        return _github_ref_path_split(parts, repo_path_index=1)

    anchor_split = _split_github_ref_at_source_path_anchor(parts)
    if anchor_split is not None:
        return anchor_split

    return _github_ref_path_split(parts, repo_path_index=1)


def _github_ref_path_split(
    parts: "Sequence[str]",
    *,
    repo_path_index: int,
) -> tuple[str, str] | None:
    ref = "/".join(parts[:repo_path_index])
    repo_path = "/".join(parts[repo_path_index:])
    if ref and repo_path:
        return ref, repo_path
    return None


def _split_github_prefixed_ref(parts: "Sequence[str]") -> tuple[str, str] | None:
    if len(parts) >= 3 and parts[1] in _GITHUB_SOURCE_PATH_ANCHORS:
        return _github_ref_path_split(parts, repo_path_index=2)
    if len(parts) >= 4 and parts[2] in _GITHUB_COMMON_REPO_PATH_PREFIXES:
        return _github_ref_path_split(parts, repo_path_index=2)
    return _split_github_ref_at_source_path_anchor(parts)


def _split_github_ref_at_source_path_anchor(
    parts: "Sequence[str]",
) -> tuple[str, str] | None:
    for index, part in enumerate(parts[1:], start=1):
        if part in _GITHUB_SOURCE_PATH_ANCHORS:
            ref = "/".join(parts[:index])
            repo_path = "/".join(parts[index:])
            if ref and repo_path:
                return ref, repo_path
    return None


def parse_github_url(url: str | None) -> ParsedGitHubUrl | None:
    normalized = strip_str_to_none(url)
    if normalized is None:
        return None
    parsed = urlparse(normalized)
    if is_github_web_host(parsed.netloc):
        parts = parsed.path.strip("/").split("/")
        if len(parts) >= 5 and parts[2] in {"blob", "tree"}:
            return _parsed_github_source(parts[:2], parts[3:])
    if is_github_raw_host(parsed.netloc):
        parts = parsed.path.strip("/").split("/")
        if len(parts) >= 4:
            return _parsed_github_source(parts[:2], parts[2:])
    return None


def _parsed_github_source(
    repo_parts: "Sequence[str]",
    ref_and_path_parts: "Sequence[str]",
) -> ParsedGitHubUrl | None:
    if len(repo_parts) < 2:
        return None
    split_ref = _split_github_ref_and_path(ref_and_path_parts)
    if split_ref is None:
        return None
    org, repo = repo_parts[:2]
    ref, repo_path = split_ref
    return ParsedGitHubUrl(
        repo_url=f"https://github.com/{org}/{repo}",
        repo_ref=ref,
        repo_path=repo_path,
    )


def is_probable_url(value: str | None) -> bool:
    if not value:
        return False
    parsed = urlparse(value)
    return bool(parsed.scheme and parsed.netloc)


def is_git_source_url(value: str | None) -> bool:
    stripped = strip_str_to_none(value)
    if stripped is None:
        return False
    if is_probable_url(stripped):
        return True
    return re_match_scp_like_git_source(stripped)


def re_match_scp_like_git_source(value: str) -> bool:
    import re

    return re.match(SCP_LIKE_GIT_SOURCE_RE, value) is not None


def first_nonempty_str(data: Mapping[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = data.get(key)
        if (normalized := strip_str_to_none(value)) is not None:
            return normalized
    return None


def explicit_entry_source_url(
    data: Mapping[str, Any],
    *keys: str,
    default_source_url: str | None,
) -> str | None:
    """Return an entry-provided source URL, excluding registry/context provenance."""
    for key in keys:
        value = data.get(key)
        if (source_url := strip_str_to_none(value)) is not None:
            if source_url != default_source_url:
                return source_url
    return None


def marketplace_repo_fields(
    data: Mapping[str, Any],
    *,
    repo_path_keys: "Sequence[str]",
    repo_url_keys: "Sequence[str]" = MARKETPLACE_REPO_URL_KEYS,
    repo_ref_keys: "Sequence[str]" = MARKETPLACE_REPO_REF_KEYS,
) -> MarketplaceRepoFields:
    """Extract and normalize common repository fields from a marketplace entry."""
    repo_url = first_nonempty_str(data, *repo_url_keys)
    repo_ref = first_nonempty_str(data, *repo_ref_keys)
    repo_path = first_nonempty_str(data, *repo_path_keys)

    parsed = parse_github_url(repo_url) if repo_url else None
    if parsed is None:
        return MarketplaceRepoFields(
            repo_url=repo_url,
            repo_ref=repo_ref,
            repo_path=repo_path,
        )

    return MarketplaceRepoFields(
        repo_url=parsed.repo_url,
        repo_ref=repo_ref or parsed.repo_ref,
        repo_path=repo_path or parsed.repo_path,
    )


def repo_fields_from_source_url(
    *,
    repo_url: str | None,
    repo_ref: str | None,
    repo_path: str | None,
    source_url: str | None,
    default_repo_url: str | None,
) -> MarketplaceRepoFields:
    """Apply a marketplace source URL to repo fields when it is the best source."""
    if not source_url or (repo_url and repo_url != default_repo_url and repo_path):
        return MarketplaceRepoFields(
            repo_url=repo_url,
            repo_ref=repo_ref,
            repo_path=repo_path,
        )

    parsed = parse_github_url(source_url)
    if parsed:
        return MarketplaceRepoFields(
            repo_url=parsed.repo_url,
            repo_ref=parsed.repo_ref,
            repo_path=parsed.repo_path,
        )

    return MarketplaceRepoFields(
        repo_url=source_url if not repo_url or repo_url == default_repo_url else repo_url,
        repo_ref=repo_ref,
        repo_path=repo_path,
    )
