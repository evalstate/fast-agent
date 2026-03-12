"""Helpers for formatting and resolving marketplace registry URL lists."""

from __future__ import annotations

from typing import TYPE_CHECKING
from urllib.parse import urlparse

if TYPE_CHECKING:
    from collections.abc import Sequence


def format_marketplace_display_url(url: str) -> str:
    """Normalize a registry URL for concise display in UI lists."""
    parsed = urlparse(url)
    if parsed.netloc == "raw.githubusercontent.com":
        parts = parsed.path.strip("/").split("/")
        if len(parts) >= 4:
            org, repo = parts[:2]
            return f"https://github.com/{org}/{repo}"
    if parsed.netloc in {"github.com", "www.github.com"}:
        parts = parsed.path.strip("/").split("/")
        if len(parts) >= 2:
            org, repo = parts[:2]
            return f"https://github.com/{org}/{repo}"
    return url


def _canonical_registry_url(url: str) -> str:
    """Return a canonical source URL key for de-duplicating equivalent entries."""
    normalized = url.strip()
    parsed = urlparse(normalized)

    def _is_default_marketplace_path(file_path: str) -> bool:
        return file_path in {".claude-plugin/marketplace.json", "marketplace.json"}

    if parsed.netloc in {"github.com", "www.github.com"}:
        parts = parsed.path.strip("/").split("/")
        if len(parts) >= 2:
            org, repo = parts[:2]
            if len(parts) == 2:
                return f"https://github.com/{org}/{repo}"
            if len(parts) >= 4 and parts[2] == "tree":
                ref = parts[3]
                file_path = "/".join(parts[4:])
                if ref in {"main", "master"} and not file_path:
                    return f"https://github.com/{org}/{repo}"
        if len(parts) >= 5 and parts[2] == "blob":
            org, repo, _, ref = parts[:4]
            file_path = "/".join(parts[4:])
            if ref in {"main", "master"} and _is_default_marketplace_path(file_path):
                return f"https://github.com/{org}/{repo}"
            return f"https://raw.githubusercontent.com/{org}/{repo}/{ref}/{file_path}"
    if parsed.netloc == "raw.githubusercontent.com":
        parts = parsed.path.strip("/").split("/")
        if len(parts) >= 4:
            org, repo, ref = parts[:3]
            file_path = "/".join(parts[3:])
            if ref in {"main", "master"} and _is_default_marketplace_path(file_path):
                return f"https://github.com/{org}/{repo}"
    return normalized


def resolve_registry_urls(
    configured_urls: Sequence[str] | None,
    *,
    default_urls: Sequence[str],
    active_url: str | None = None,
) -> list[str]:
    """Build a stable registry list with canonical source-level de-duplication."""
    registry_urls = list(configured_urls) if configured_urls else list(default_urls)
    if active_url:
        registry_urls.append(active_url)

    deduped: list[str] = []
    seen_source_urls: set[str] = set()
    for url in registry_urls:
        source_url = _canonical_registry_url(url)
        if source_url in seen_source_urls:
            continue
        seen_source_urls.add(source_url)
        deduped.append(url)

    return deduped
