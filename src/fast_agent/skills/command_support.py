"""Shared helpers for skills command parsing and marketplace presentation."""

from __future__ import annotations

import shlex
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from fast_agent.skills.models import MarketplaceSkill


def skills_usage_lines() -> list[str]:
    """Return the shared usage/help text for skills management commands."""
    return [
        "Usage: /skills [list|available|search|add|remove|update|registry|help] [args]",
        "",
        "Examples:",
        "- /skills available",
        "- /skills search docker",
        "- /skills add <number|name>",
        "- /skills registry",
    ]


def marketplace_search_tokens(query: str) -> list[str]:
    """Split a marketplace search query into normalized search tokens."""
    try:
        tokens = shlex.split(query)
    except ValueError:
        tokens = query.split()
    return [token.lower() for token in tokens if token.strip()]


def filter_marketplace_skills(
    marketplace: "Sequence[MarketplaceSkill]",
    query: str,
) -> list["MarketplaceSkill"]:
    """Filter marketplace skills by query tokens across key descriptive fields."""
    tokens = marketplace_search_tokens(query)
    if not tokens:
        return list(marketplace)

    filtered: list[MarketplaceSkill] = []
    for entry in marketplace:
        haystack = " ".join(
            value
            for value in (
                entry.name,
                entry.description or "",
                entry.bundle_name or "",
                entry.bundle_description or "",
            )
            if value
        ).lower()
        if all(token in haystack for token in tokens):
            filtered.append(entry)
    return filtered


def marketplace_repository_hint(marketplace: "Sequence[MarketplaceSkill]") -> str | None:
    """Return a concise repository hint for a marketplace listing."""
    if not marketplace:
        return None
    repo_url = marketplace[0].repo_url
    repo_ref = marketplace[0].repo_ref
    return f"{repo_url}@{repo_ref}" if repo_ref else repo_url
