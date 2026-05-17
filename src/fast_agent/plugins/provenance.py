"""Plugin sidecar metadata helpers."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from fast_agent.marketplace import formatting as marketplace_formatting
from fast_agent.marketplace import source_utils as marketplace_source_utils
from fast_agent.plugins.models import (
    LOCAL_REVISION,
    PLUGIN_SOURCE_FILENAME,
    PLUGIN_SOURCE_SCHEMA_VERSION,
    InstalledPluginSource,
    MarketplacePlugin,
    PluginSourceOrigin,
)


def get_plugin_source_sidecar_path(plugin_dir: Path) -> Path:
    return plugin_dir / PLUGIN_SOURCE_FILENAME


def compute_plugin_content_fingerprint(plugin_dir: Path) -> str:
    digest = hashlib.sha256()
    root = plugin_dir.resolve()
    sidecar_path = get_plugin_source_sidecar_path(root)
    for path in sorted(root.rglob("*")):
        if path == sidecar_path or not path.is_file() or _is_generated_runtime_artifact(path):
            continue
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}"


def _is_generated_runtime_artifact(path: Path) -> bool:
    if path.suffix == ".pyc":
        return True
    return "__pycache__" in path.parts


def read_installed_plugin_source(
    plugin_dir: Path,
) -> tuple[InstalledPluginSource | None, str | None]:
    sidecar_path = get_plugin_source_sidecar_path(plugin_dir)
    if not sidecar_path.exists():
        return None, None
    try:
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return None, f"invalid json: {exc}"
    if not isinstance(payload, dict):
        return None, "metadata root must be an object"
    try:
        source = parse_installed_plugin_source_payload(payload)
    except ValueError as exc:
        return None, str(exc)
    return source, None


def write_installed_plugin_source(plugin_dir: Path, source: InstalledPluginSource) -> None:
    payload = {
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
    get_plugin_source_sidecar_path(plugin_dir).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_installed_plugin_source_payload(payload: dict[str, Any]) -> InstalledPluginSource:
    parsed = marketplace_source_utils.parse_installed_source_fields(
        payload,
        expected_schema_version=PLUGIN_SOURCE_SCHEMA_VERSION,
        normalize_repo_path=normalize_repo_path,
    )
    return InstalledPluginSource(
        schema_version=PLUGIN_SOURCE_SCHEMA_VERSION,
        installed_via="marketplace",
        source_origin=parsed.source_origin,
        repo_url=parsed.repo_url,
        repo_ref=parsed.repo_ref,
        repo_path=parsed.repo_path,
        source_url=parsed.source_url,
        installed_commit=parsed.installed_commit,
        installed_path_oid=parsed.installed_path_oid,
        installed_revision=parsed.installed_revision,
        installed_at=parsed.installed_at,
        content_fingerprint=parsed.content_fingerprint,
    )


def build_installed_plugin_source(
    *,
    plugin: MarketplacePlugin,
    source_origin: PluginSourceOrigin,
    installed_commit: str | None,
    installed_path_oid: str | None,
    fingerprint: str,
) -> InstalledPluginSource:
    installed_revision = installed_commit or LOCAL_REVISION
    return InstalledPluginSource(
        schema_version=PLUGIN_SOURCE_SCHEMA_VERSION,
        installed_via="marketplace",
        source_origin=source_origin,
        repo_url=plugin.repo_url,
        repo_ref=plugin.repo_ref,
        repo_path=plugin.repo_path,
        source_url=plugin.source_url,
        installed_commit=installed_commit,
        installed_path_oid=installed_path_oid,
        installed_revision=installed_revision,
        installed_at=datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        content_fingerprint=fingerprint,
    )


def normalize_repo_path(path: str) -> str | None:
    raw = path.strip().replace("\\", "/")
    if not raw:
        return None
    posix_path = PurePosixPath(raw)
    if posix_path.is_absolute() or ".." in posix_path.parts:
        return None
    normalized = str(posix_path).lstrip("/")
    if normalized in {"", "."}:
        return None
    return normalized


def format_revision_short(revision: str | None) -> str:
    return marketplace_formatting.format_revision_short(revision)


def format_installed_at_display(installed_at: str | None) -> str:
    return marketplace_formatting.format_installed_at_display(installed_at)
