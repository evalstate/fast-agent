from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING, Sequence, cast

if TYPE_CHECKING:
    from importlib.abc import Traversable

from fast_agent.core.exceptions import AgentConfigError

_SHARED_DIR = Path("resources") / "shared"
_MANIFEST_FILENAME = "internal_resources_manifest.json"


@dataclass(frozen=True)
class InternalResource:
    """Definition for a bundled read-only internal resource."""

    uri: str
    title: str
    description: str
    why: str
    source: str
    mime_type: str = "text/markdown"
    tags: tuple[str, ...] = ()


@lru_cache(maxsize=1)
def list_internal_resources() -> tuple[InternalResource, ...]:
    """Load and validate the bundled internal resource catalog manifest."""
    raw_manifest = _load_manifest_payload()

    if not isinstance(raw_manifest, list):
        raise AgentConfigError(
            "Invalid internal resources manifest",
            "Manifest must be a JSON array of resource entries.",
        )

    resources: list[InternalResource] = []
    seen_uris: set[str] = set()

    for index, entry in enumerate(raw_manifest):
        resource = _parse_manifest_entry(index, entry)
        if resource.uri in seen_uris:
            raise AgentConfigError(
                "Duplicate internal resource URI",
                f"URI '{resource.uri}' appears more than once in {_MANIFEST_FILENAME}.",
            )
        seen_uris.add(resource.uri)
        resources.append(resource)

    return tuple(resources)


@lru_cache(maxsize=1)
def _resources_by_uri() -> dict[str, InternalResource]:
    return {resource.uri: resource for resource in list_internal_resources()}


def get_internal_resource(uri: str) -> InternalResource:
    """Resolve an internal resource by URI."""
    normalized = uri.strip()
    if not normalized:
        raise AgentConfigError(
            "Invalid internal resource URI",
            "URI must not be empty.",
        )

    resource = _resources_by_uri().get(normalized)
    if resource is None:
        raise AgentConfigError(
            "Unknown internal resource URI",
            f"URI: {normalized}",
        )
    return resource


def read_internal_resource(uri: str) -> str:
    """Read internal resource text by allow-listed URI."""
    resource = get_internal_resource(uri)
    return _read_shared_resource_text(resource.source)


def format_internal_resources_for_prompt(resources: Sequence[InternalResource]) -> str:
    """Format internal resources as a compact XML-ish list for prompts."""
    if not resources:
        return ""

    lines: list[str] = ["<available_resources>"]
    for resource in resources:
        lines.append("  <resource>")
        lines.append(f"    <uri>{resource.uri}</uri>")
        lines.append(f"    <description>{resource.description}</description>")
        lines.append(f"    <why>{resource.why}</why>")
        lines.append("  </resource>")
    lines.append("</available_resources>")
    return "\n".join(lines)


def _parse_manifest_entry(index: int, entry: object) -> InternalResource:
    if not isinstance(entry, dict):
        raise AgentConfigError(
            "Invalid internal resources manifest entry",
            f"Entry at index {index} must be an object.",
        )

    typed_entry = cast("dict[str, object]", entry)

    uri = _require_str(typed_entry, "uri", index)
    title = _require_str(typed_entry, "title", index)
    description = _require_str(typed_entry, "description", index)
    why = _require_str(typed_entry, "why", index)
    source = _require_source_path(typed_entry, index)

    mime_type = typed_entry.get("mime_type", "text/markdown")
    if not isinstance(mime_type, str) or not mime_type.strip():
        raise AgentConfigError(
            "Invalid internal resources manifest entry",
            f"Entry at index {index} has invalid 'mime_type'.",
        )

    tags_raw = typed_entry.get("tags", [])
    tags: tuple[str, ...]
    if tags_raw is None:
        tags = ()
    elif isinstance(tags_raw, list) and all(isinstance(tag, str) for tag in tags_raw):
        tag_values = cast("list[str]", tags_raw)
        tags = tuple(tag.strip() for tag in tag_values if tag.strip())
    else:
        raise AgentConfigError(
            "Invalid internal resources manifest entry",
            f"Entry at index {index} has invalid 'tags' (must be a list of strings).",
        )

    if not uri.startswith("internal://"):
        raise AgentConfigError(
            "Invalid internal resource URI",
            f"Entry at index {index} URI must start with 'internal://'.",
        )

    return InternalResource(
        uri=uri,
        title=title,
        description=description,
        why=why,
        source=source,
        mime_type=mime_type.strip(),
        tags=tags,
    )


def _require_str(entry: dict[str, object], key: str, index: int) -> str:
    value = entry.get(key)
    if not isinstance(value, str) or not value.strip():
        raise AgentConfigError(
            "Invalid internal resources manifest entry",
            f"Entry at index {index} is missing non-empty '{key}'.",
        )
    return value.strip()


def _require_source_path(entry: dict[str, object], index: int) -> str:
    source = _require_str(entry, "source", index)
    source_path = Path(source)
    if source_path.is_absolute() or ".." in source_path.parts:
        raise AgentConfigError(
            "Invalid internal resource source path",
            f"Entry at index {index} has unsafe source path '{source}'.",
        )
    return source


def _load_manifest_payload() -> object:
    source_manifest = Path(__file__).resolve().parents[3] / _SHARED_DIR / _MANIFEST_FILENAME
    if source_manifest.is_file():
        return json.loads(source_manifest.read_text(encoding="utf-8"))

    packaged_manifest = files("fast_agent").joinpath("resources").joinpath("shared").joinpath(
        _MANIFEST_FILENAME
    )
    if packaged_manifest.is_file():
        return json.loads(packaged_manifest.read_text(encoding="utf-8"))

    return []


def _read_shared_resource_text(relative_source: str) -> str:
    relative_path = Path(relative_source)

    source_resource = Path(__file__).resolve().parents[3] / _SHARED_DIR / relative_path
    if source_resource.is_file():
        return source_resource.read_text(encoding="utf-8")

    packaged_shared = files("fast_agent").joinpath("resources").joinpath("shared")
    packaged_resource = _join_traversable(packaged_shared, relative_path)
    if packaged_resource.is_file():
        return packaged_resource.read_text(encoding="utf-8")

    raise AgentConfigError(
        "Internal resource source file not found",
        f"Source: {relative_source}",
    )


def _join_traversable(base: "Traversable", relative_path: Path) -> "Traversable":
    target = base
    for part in relative_path.parts:
        target = target.joinpath(part)
    return target
