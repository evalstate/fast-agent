"""Formatting helpers shared across marketplace managers."""

from __future__ import annotations

import re
from datetime import UTC, datetime

from fast_agent.utils.text import strip_to_none

_COMMIT_LIKE_REVISION_RE = re.compile(r"[0-9a-f]{8,}", re.IGNORECASE)


def format_revision_short(revision: str | None) -> str:
    trimmed = strip_to_none(revision)
    if trimmed is None:
        return "?"
    if _COMMIT_LIKE_REVISION_RE.fullmatch(trimmed):
        return trimmed[:7]
    return trimmed


def format_installed_at_display(installed_at: str | None) -> str:
    normalized = strip_to_none(installed_at)
    if normalized is None:
        return "unknown"
    if normalized.endswith("Z"):
        normalized = f"{normalized[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return normalized
    return parsed.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S")


def format_installed_revision_display(
    installed_at: str | None,
    revision: str | None,
    *,
    separator: str = " ",
    revision_label: str = "revision: ",
) -> str:
    return (
        f"{format_installed_at_display(installed_at)}{separator}"
        f"{revision_label}{format_revision_short(revision)}"
    )


def iso_utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def format_source_location(repo_url: str, repo_ref: str | None) -> str:
    normalized_url = repo_url.strip()
    normalized_ref = strip_to_none(repo_ref)
    ref_label = f"@{normalized_ref}" if normalized_ref else ""
    return f"{normalized_url}{ref_label}"


def format_source_provenance(repo_url: str, repo_ref: str | None, repo_path: str) -> str:
    return f"{format_source_location(repo_url, repo_ref)} ({repo_path.strip()})"
