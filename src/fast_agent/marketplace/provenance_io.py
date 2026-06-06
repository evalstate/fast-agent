"""Installed marketplace source provenance IO and path helpers."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, TypeVar

from fast_agent.marketplace.source_models import (
    InstalledSourcePayloadFields,
    InstalledSourceReadResult,
    ParsedInstalledSourceFields,
)
from fast_agent.marketplace.source_urls import path_name_matches
from fast_agent.utils.text import strip_str_to_none

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

SourceT = TypeVar("SourceT")
_SHA256_FINGERPRINT_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def read_installed_source_file(
    sidecar_path: Path,
    *,
    parse_payload: "Callable[[dict[str, Any]], SourceT]",
) -> InstalledSourceReadResult[SourceT]:
    if not sidecar_path.exists():
        return InstalledSourceReadResult()

    try:
        payload = read_json_file(sidecar_path)
    except Exception as exc:
        return InstalledSourceReadResult(error=f"invalid json: {exc}")

    if not isinstance(payload, dict):
        return InstalledSourceReadResult(error="metadata root must be an object")

    try:
        source = parse_payload(payload)
    except ValueError as exc:
        return InstalledSourceReadResult(error=str(exc))
    return InstalledSourceReadResult(source=source)


def installed_source_payload(source: InstalledSourcePayloadFields) -> dict[str, Any]:
    return {
        "schema_version": source.schema_version,
        "installed_via": source.installed_via,
        "source_origin": source.source_origin,
        "repo_url": source.repo_url,
        "repo_ref": source.repo_ref,
        "repo_path": source.repo_path,
        "source_url": source.source_url,
        "installed_commit": source.installed_commit,
        "installed_path_oid": source.installed_path_oid,
        "installed_revision": source.installed_revision,
        "installed_at": source.installed_at,
        "content_fingerprint": source.content_fingerprint,
    }


def write_installed_source_file(
    sidecar_path: Path,
    source: InstalledSourcePayloadFields,
    *,
    extra_payload: "Mapping[str, Any] | None" = None,
) -> None:
    payload = installed_source_payload(source)
    if extra_payload:
        payload.update(extra_payload)
    sidecar_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def compute_directory_content_fingerprint(
    directory: Path,
    *,
    sidecar_path: Path,
    ignore_path: "Callable[[Path], bool] | None" = None,
) -> str:
    digest = hashlib.sha256()
    root = directory.resolve()
    resolved_sidecar = sidecar_path.resolve()

    for path in sorted(root.rglob("*")):
        if path == resolved_sidecar:
            continue
        if not path.is_file():
            continue
        if ignore_path is not None and ignore_path(path):
            continue
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")

    return f"sha256:{digest.hexdigest()}"


def read_json_file(path: Path) -> Any:
    content = path.read_text(encoding="utf-8")
    return json.loads(content)


def parse_installed_source_fields(
    payload: "Mapping[str, Any]",
    *,
    expected_schema_version: int,
    normalize_repo_path: "Callable[[str], str | None]",
) -> ParsedInstalledSourceFields:
    schema_version = payload.get("schema_version")
    if schema_version != expected_schema_version:
        raise ValueError(f"unsupported schema_version: {schema_version}")

    installed_via = payload.get("installed_via")
    if not isinstance(installed_via, str) or installed_via.strip() != "marketplace":
        raise ValueError("installed_via must be 'marketplace'")

    source_origin_raw = payload.get("source_origin")
    if source_origin_raw not in {"remote", "local"}:
        raise ValueError("source_origin must be 'remote' or 'local'")

    repo_url = _required_installed_source_string(payload, "repo_url")
    repo_ref = _optional_installed_source_string(payload, "repo_ref")

    repo_path_raw = payload.get("repo_path")
    if not isinstance(repo_path_raw, str):
        raise ValueError("repo_path is required")
    repo_path = normalize_repo_path(repo_path_raw)
    if not repo_path:
        raise ValueError("repo_path is invalid")

    source_url = _optional_installed_source_string(payload, "source_url")
    installed_commit = _optional_installed_source_string(
        payload,
        "installed_commit",
        allow_blank=False,
    )
    installed_path_oid = _optional_installed_source_string(
        payload,
        "installed_path_oid",
        allow_blank=False,
    )
    installed_revision = _required_installed_source_string(payload, "installed_revision")
    installed_at = _required_installed_source_string(payload, "installed_at")

    content_fingerprint = _required_sha256_fingerprint(payload, "content_fingerprint")

    return ParsedInstalledSourceFields(
        source_origin=source_origin_raw,
        repo_url=repo_url,
        repo_ref=repo_ref,
        repo_path=repo_path,
        source_url=source_url,
        installed_commit=installed_commit,
        installed_path_oid=installed_path_oid,
        installed_revision=installed_revision,
        installed_at=installed_at,
        content_fingerprint=content_fingerprint,
    )


def _required_installed_source_string(payload: "Mapping[str, Any]", field_name: str) -> str:
    value = payload.get(field_name)
    normalized = strip_str_to_none(value)
    if normalized is None:
        raise ValueError(f"{field_name} is required")
    return normalized


def _optional_installed_source_string(
    payload: "Mapping[str, Any]",
    field_name: str,
    *,
    allow_blank: bool = True,
) -> str | None:
    value = payload.get(field_name)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string or null")
    normalized = strip_str_to_none(value)
    if normalized is not None:
        return normalized
    if allow_blank:
        return None
    raise ValueError(f"{field_name} must be a non-empty string or null")


def _required_sha256_fingerprint(payload: "Mapping[str, Any]", field_name: str) -> str:
    value = payload.get(field_name)
    if isinstance(value, str) and _SHA256_FINGERPRINT_RE.fullmatch(value):
        return value
    raise ValueError(f"{field_name} must be a sha256 fingerprint")


def normalize_relative_repo_path(path: str, *, allow_current_dir: bool = False) -> str | None:
    raw = path.strip().replace("\\", "/")
    if not raw:
        return None
    if re.match(r"^[A-Za-z]:($|/)", raw):
        return None
    posix_path = PurePosixPath(raw)
    if posix_path.is_absolute() or ".." in posix_path.parts:
        return None
    normalized = str(posix_path).lstrip("/")
    if normalized == "" or (normalized == "." and not allow_current_dir):
        return None
    return normalized


def repo_subdir_for_manifest_path(repo_path: str, manifest_filename: str) -> str:
    path_value = (
        normalize_relative_repo_path(repo_path, allow_current_dir=True) or repo_path.strip()
    )
    path = PurePosixPath(path_value)
    if path_name_matches(path, manifest_filename):
        return str(path.parent)
    return str(path)


def repo_name_for_manifest_path(
    repo_path: str,
    manifest_filename: str,
    *,
    allow_current_dir: bool = False,
) -> str:
    path_value = (
        normalize_relative_repo_path(repo_path, allow_current_dir=allow_current_dir)
        or repo_path.strip()
    )
    path = PurePosixPath(path_value)
    if path_name_matches(path, manifest_filename) and path.parent.name:
        return path.parent.name
    return path.name or path_value
