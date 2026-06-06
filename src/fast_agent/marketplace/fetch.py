"""Marketplace payload fetching, local source context, and payload normalization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar
from urllib.parse import urlparse

import httpx

from fast_agent.io.path_uri import file_uri_to_path
from fast_agent.marketplace.provenance_io import read_json_file
from fast_agent.marketplace.source_models import MarketplaceSourceContext
from fast_agent.marketplace.source_urls import is_git_source_url, parse_github_url

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from pydantic import ValidationInfo

EntryT = TypeVar("EntryT")


@dataclass(frozen=True, slots=True)
class _SourceLocation:
    path: Path | None
    is_remote: bool = False


def load_local_marketplace_payload(url: str) -> Any | None:
    source = _source_location(url)
    if source.is_remote or source.path is None:
        return None
    candidate = source.path.expanduser()
    if candidate.exists():
        if candidate.is_dir():
            raise FileNotFoundError(f"marketplace.json not found in directory: {candidate}")
        return read_json_file(candidate)
    return None


def _source_location(value: str) -> _SourceLocation:
    parsed = urlparse(value)
    if parsed.scheme == "file":
        return _SourceLocation(path=file_uri_to_path(parsed))
    if is_git_source_url(value):
        return _SourceLocation(path=None, is_remote=True)
    return _SourceLocation(path=Path(value))


def derive_local_repo_root(source_url: str) -> str | None:
    source = _source_location(source_url)
    if source.is_remote or source.path is None:
        return None

    path = source.path.expanduser()
    if not path.is_absolute():
        path = path.resolve()

    if not path.exists():
        return None

    if path.is_file() and path.name == "marketplace.json":
        repo_root = path.parent.parent if path.parent.name == ".claude-plugin" else path.parent
        if repo_root.exists():
            return str(repo_root)

    if path.is_dir():
        return str(path)

    return None


def marketplace_source_context(source_url: str | None) -> MarketplaceSourceContext:
    if not source_url:
        return MarketplaceSourceContext()

    parsed = parse_github_url(source_url)
    if parsed:
        return MarketplaceSourceContext(
            source_url=source_url,
            repo_url=parsed.repo_url,
            repo_ref=parsed.repo_ref,
        )

    return MarketplaceSourceContext(
        source_url=source_url,
        repo_url=derive_local_repo_root(source_url),
        repo_ref=None,
    )


def normalize_marketplace_payload(
    data: Any,
    info: "ValidationInfo",
    *,
    extract_entries: "Callable[[Any], list[dict[str, Any]]]",
) -> dict[str, list[dict[str, Any]]]:
    entries = extract_entries(data)
    context = info.context or {}
    source_url = context.get("source_url")
    repo_url = context.get("repo_url")
    repo_ref = context.get("repo_ref")
    normalized_entries: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        entry = dict(entry)
        if source_url and "source_url" not in entry:
            entry["source_url"] = source_url
        if repo_url and "repo_url" not in entry and "repo" not in entry:
            entry["repo_url"] = repo_url
        if repo_ref and "repo_ref" not in entry and "ref" not in entry:
            entry["repo_ref"] = repo_ref
        normalized_entries.append(entry)
    return {"entries": normalized_entries}


def _dict_entries_from_sequence(values: "Sequence[Any]") -> list[dict[str, Any]]:
    return [entry for entry in values if isinstance(entry, dict)]


def extract_dict_entries(
    payload: Any,
    keys: "Sequence[str]",
    *,
    allow_mapping_values: bool = False,
) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return _dict_entries_from_sequence(payload)

    if isinstance(payload, dict):
        for key in keys:
            value = payload.get(key)
            if isinstance(value, list):
                return _dict_entries_from_sequence(value)

        if allow_mapping_values and all(isinstance(value, dict) for value in payload.values()):
            return _dict_entries_from_sequence(list(payload.values()))

    raise ValueError("Unsupported marketplace payload format.")


async def fetch_marketplace_entries_with_source(
    url: str,
    *,
    candidate_urls: "Callable[[str], Sequence[str]]",
    normalize_url: "Callable[[str], str]",
    load_local_payload: "Callable[[str], Any | None]",
    parse_payload: "Callable[[Any, str | None], list[EntryT]]",
) -> tuple[list[EntryT], str]:
    candidates = candidate_urls(url)
    last_error: Exception | None = None
    for candidate in candidates:
        normalized = normalize_url(candidate)
        try:
            local_payload = load_local_payload(normalized)
            if local_payload is not None:
                return parse_payload(local_payload, normalized), normalized
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(normalized)
                response.raise_for_status()
                data = response.json()
            return parse_payload(data, normalized), normalized
        except Exception as exc:
            last_error = exc
            continue

    if last_error is not None:
        raise last_error

    return [], normalize_url(url)
