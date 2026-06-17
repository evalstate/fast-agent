"""Helpers for formatting and resolving marketplace registry URL lists."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal
from urllib.parse import urlparse

from fast_agent.utils.collections import unique_preserve_order
from fast_agent.utils.text import strip_casefold, strip_to_none

if TYPE_CHECKING:
    from collections.abc import Sequence
    from urllib.parse import ParseResult


_DEFAULT_MARKETPLACE_PATHS = frozenset(
    {".claude-plugin", ".claude-plugin/marketplace.json", "marketplace.json"}
)
_DEFAULT_GITHUB_REFS = frozenset({"main", "master"})
_GITHUB_HOSTS = {"github.com", "www.github.com"}
GitHubPathKind = Literal["repo", "tree", "blob"]


@dataclass(frozen=True)
class GitHubRegistryPath:
    org: str
    repo: str
    kind: GitHubPathKind
    ref: str = ""
    file_path: str = ""


def _url_host(parsed: "ParseResult") -> str:
    return strip_casefold(parsed.netloc)


def format_marketplace_display_url(url: str) -> str:
    """Normalize a registry URL for concise display in UI lists."""
    normalized = strip_to_none(url)
    if normalized is None:
        return ""

    parsed = urlparse(normalized)
    host = _url_host(parsed)
    if host == "raw.githubusercontent.com":
        return _display_raw_github_registry_url(parsed, fallback=normalized)
    if host in _GITHUB_HOSTS:
        return _display_github_registry_url(parsed, fallback=normalized)
    return normalized


def _display_github_registry_url(parsed: "ParseResult", *, fallback: str) -> str:
    github_path = _parse_github_registry_path(parsed)
    if github_path is None:
        return fallback

    repo_url = _github_repo_url(github_path.org, github_path.repo)
    if github_path.kind == "repo":
        return repo_url

    return repo_url if _is_default_github_registry_path(github_path) else fallback


def _display_raw_github_registry_url(parsed: "ParseResult", *, fallback: str) -> str:
    parts = parsed.path.strip("/").split("/")
    if len(parts) < 4:
        return fallback

    org, repo, ref = parts[:3]
    default_url = _canonical_default_marketplace_url(
        org=org,
        repo=repo,
        ref=ref,
        file_path="/".join(parts[3:]),
        default_ref_equivalence=True,
    )
    return _github_repo_url(org, repo) if default_url is not None else fallback


def _canonical_registry_url(
    url: str,
    *,
    default_ref_equivalence: bool = False,
) -> str:
    """Return a canonical source URL key for de-duplicating equivalent entries."""
    normalized = url.strip()
    parsed = urlparse(normalized)
    host = _url_host(parsed)

    if host in _GITHUB_HOSTS:
        canonical = _canonical_github_registry_url(
            parsed,
            default_ref_equivalence=default_ref_equivalence,
        )
        if canonical is not None:
            return canonical

    if host == "raw.githubusercontent.com":
        canonical = _canonical_raw_github_registry_url(
            parsed,
            default_ref_equivalence=default_ref_equivalence,
        )
        if canonical is not None:
            return canonical
    return normalized


def _github_repo_url(org: str, repo: str) -> str:
    return f"https://github.com/{org}/{repo}"


def _raw_github_url(org: str, repo: str, ref: str, file_path: str) -> str:
    return f"https://raw.githubusercontent.com/{org}/{repo}/{ref}/{file_path}"


def _canonical_default_marketplace_url(
    *,
    org: str,
    repo: str,
    ref: str,
    file_path: str,
    default_ref_equivalence: bool,
) -> str | None:
    if ref in _DEFAULT_GITHUB_REFS and _is_default_marketplace_path(file_path):
        if default_ref_equivalence:
            return _github_repo_url(org, repo)
        return f"{_github_repo_url(org, repo)}/tree/{ref}/.claude-plugin"
    return None


def _canonical_github_registry_url(
    parsed: "ParseResult",
    *,
    default_ref_equivalence: bool,
) -> str | None:
    github_path = _parse_github_registry_path(parsed)
    if github_path is None:
        return None

    if github_path.kind == "repo":
        return _github_repo_url(github_path.org, github_path.repo)

    if _is_default_github_registry_path(github_path):
        return _canonical_default_marketplace_url(
            org=github_path.org,
            repo=github_path.repo,
            ref=github_path.ref,
            file_path=github_path.file_path,
            default_ref_equivalence=default_ref_equivalence,
        )
    if github_path.kind == "blob":
        return _raw_github_url(
            github_path.org,
            github_path.repo,
            github_path.ref,
            github_path.file_path,
        )

    return None


def _is_default_github_registry_path(github_path: GitHubRegistryPath) -> bool:
    return (
        github_path.kind in {"tree", "blob"}
        and github_path.ref in _DEFAULT_GITHUB_REFS
        and _is_default_marketplace_path(github_path.file_path)
    )


def _parse_github_registry_path(parsed: "ParseResult") -> GitHubRegistryPath | None:
    parts = parsed.path.strip("/").split("/")
    if len(parts) < 2:
        return None

    org, repo = parts[:2]
    if len(parts) == 2:
        return GitHubRegistryPath(org=org, repo=repo, kind="repo")

    if len(parts) >= 4 and parts[2] == "tree":
        return GitHubRegistryPath(
            org=org,
            repo=repo,
            kind="tree",
            ref=parts[3],
            file_path="/".join(parts[4:]),
        )

    if len(parts) >= 5 and parts[2] == "blob":
        return GitHubRegistryPath(
            org=org,
            repo=repo,
            kind="blob",
            ref=parts[3],
            file_path="/".join(parts[4:]),
        )

    return None


def _canonical_raw_github_registry_url(
    parsed: "ParseResult",
    *,
    default_ref_equivalence: bool,
) -> str | None:
    parts = parsed.path.strip("/").split("/")
    if len(parts) < 4:
        return None

    org, repo, ref = parts[:3]
    return _canonical_default_marketplace_url(
        org=org,
        repo=repo,
        ref=ref,
        file_path="/".join(parts[3:]),
        default_ref_equivalence=default_ref_equivalence,
    )


def _is_default_marketplace_path(file_path: str) -> bool:
    return file_path in {"", *_DEFAULT_MARKETPLACE_PATHS}


def resolve_registry_urls(
    configured_urls: Sequence[str] | None,
    *,
    default_urls: Sequence[str],
    active_url: str | None = None,
) -> list[str]:
    """Build a stable registry list with canonical source-level de-duplication."""
    registry_entries: list[tuple[str, bool]] = [
        (url, False) for url in (list(configured_urls) if configured_urls else list(default_urls))
    ]
    if active_url:
        registry_entries.append((active_url, True))

    normalized_entries = [
        (normalized_url, is_active_url)
        for url, is_active_url in registry_entries
        if (normalized_url := strip_to_none(url)) is not None
    ]
    return [
        url
        for url, _is_active_url in unique_preserve_order(
            normalized_entries,
            key=lambda entry: _canonical_registry_url(
                entry[0],
                default_ref_equivalence=entry[1],
            ),
        )
    ]
