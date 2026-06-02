"""Skills provenance and sidecar metadata helpers."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from fast_agent.marketplace import formatting as marketplace_formatting
from fast_agent.marketplace import source_utils as marketplace_source_utils
from fast_agent.skills.marketplace_parsing import normalize_repo_path
from fast_agent.skills.models import (
    LOCAL_REVISION,
    SKILL_SOURCE_FILENAME,
    SKILL_SOURCE_SCHEMA_VERSION,
    InstalledSkillSource,
    MarketplaceSkill,
    SkillProvenance,
    SkillSourceOrigin,
)

if TYPE_CHECKING:
    from pathlib import Path


def get_skill_source_sidecar_path(skill_dir: Path) -> Path:
    return skill_dir / SKILL_SOURCE_FILENAME


def compute_skill_content_fingerprint(skill_dir: Path) -> str:
    digest = hashlib.sha256()
    root = skill_dir.resolve()
    sidecar_path = get_skill_source_sidecar_path(root)

    for path in sorted(root.rglob("*")):
        if path == sidecar_path:
            continue
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")

    return f"sha256:{digest.hexdigest()}"


def read_installed_skill_source(skill_dir: Path) -> tuple[InstalledSkillSource | None, str | None]:
    sidecar_path = get_skill_source_sidecar_path(skill_dir)
    if not sidecar_path.exists():
        return None, None
    try:
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return None, f"invalid json: {exc}"

    if not isinstance(payload, dict):
        return None, "metadata root must be an object"

    try:
        source = parse_installed_skill_source_payload(payload)
    except ValueError as exc:
        return None, str(exc)
    return source, None


def write_installed_skill_source(skill_dir: Path, source: InstalledSkillSource) -> None:
    sidecar_path = get_skill_source_sidecar_path(skill_dir)
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
    if source.mcp_server_name is not None:
        payload["mcp_server_name"] = source.mcp_server_name
    if source.mcp_server_version is not None:
        payload["mcp_server_version"] = source.mcp_server_version
    if source.artifact_digest is not None:
        payload["artifact_digest"] = source.artifact_digest
    if source.artifact_type is not None:
        payload["artifact_type"] = source.artifact_type
    sidecar_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def get_skill_provenance(skill_dir: Path) -> SkillProvenance:
    source, error = read_installed_skill_source(skill_dir)
    if source is None:
        if error is None:
            return SkillProvenance(
                status="unmanaged",
                summary="unmanaged (no sidecar)",
            )
        return SkillProvenance(
            status="invalid_metadata",
            summary=f"invalid metadata ({error})",
            error=error,
        )

    ref_label = f"@{source.repo_ref}" if source.repo_ref else ""
    if source.source_origin == "mcp":
        version = f"@{source.mcp_server_version}" if source.mcp_server_version else ""
        summary = (
            "managed (mcp)"
            f" • {source.mcp_server_name or source.repo_url}{version}"
            f" • {source.repo_path}"
        )
    elif source.source_origin == "remote":
        summary = (
            "managed (marketplace)"
            f" • {source.repo_url}{ref_label}"
            f" • {source.repo_path}"
        )
    else:
        summary = (
            "managed (local source)"
            f" • {source.repo_url}{ref_label}"
            f" • {source.repo_path}"
        )
    return SkillProvenance(status="managed", summary=summary, source=source)


def format_skill_provenance(skill_dir: Path) -> str:
    return get_skill_provenance(skill_dir).summary


def format_revision_short(revision: str | None) -> str:
    return marketplace_formatting.format_revision_short(revision)


def format_installed_at_display(installed_at: str | None) -> str:
    return marketplace_formatting.format_installed_at_display(installed_at)


def format_skill_provenance_details(skill_dir: Path) -> tuple[str, str | None]:
    provenance = get_skill_provenance(skill_dir)
    if provenance.status == "unmanaged":
        return "unmanaged.", None
    if provenance.status != "managed" or provenance.source is None:
        return provenance.summary, None

    source = provenance.source
    if source.source_origin == "mcp":
        version = f"@{source.mcp_server_version}" if source.mcp_server_version else ""
        integrity = " • integrity: sha256" if source.artifact_digest else ""
        provenance_value = (
            f"mcp-server {source.mcp_server_name or source.repo_url}{version} "
            f"({source.repo_path}){integrity}"
        )
        installed_value = (
            f"{format_installed_at_display(source.installed_at)} "
            f"revision: {format_revision_short(source.installed_revision)}"
        )
        return provenance_value, installed_value

    ref_label = f"@{source.repo_ref}" if source.repo_ref else ""
    provenance_value = f"{source.repo_url}{ref_label} ({source.repo_path})"
    installed_value = (
        f"{format_installed_at_display(source.installed_at)} "
        f"revision: {format_revision_short(source.installed_revision)}"
    )
    return provenance_value, installed_value


def parse_installed_skill_source_payload(payload: dict[str, Any]) -> InstalledSkillSource:
    installed_via = payload.get("installed_via")
    if installed_via == "mcp":
        return _parse_mcp_installed_skill_source_payload(payload)

    parsed = marketplace_source_utils.parse_installed_source_fields(
        payload,
        expected_schema_version=SKILL_SOURCE_SCHEMA_VERSION,
        normalize_repo_path=normalize_repo_path,
    )

    return InstalledSkillSource(
        schema_version=SKILL_SOURCE_SCHEMA_VERSION,
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


def _parse_mcp_installed_skill_source_payload(payload: dict[str, Any]) -> InstalledSkillSource:
    schema_version = payload.get("schema_version")
    if schema_version != SKILL_SOURCE_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version: {schema_version}")

    source_origin = payload.get("source_origin")
    if source_origin != "mcp":
        raise ValueError("source_origin must be 'mcp'")

    repo_url = payload.get("repo_url")
    if not isinstance(repo_url, str) or not repo_url.strip():
        raise ValueError("repo_url is required")

    repo_path = payload.get("repo_path")
    if not isinstance(repo_path, str) or not repo_path.strip():
        raise ValueError("repo_path is required")

    source_url = payload.get("source_url")
    if source_url is not None and not isinstance(source_url, str):
        raise ValueError("source_url must be a string or null")

    installed_revision = payload.get("installed_revision")
    if not isinstance(installed_revision, str) or not installed_revision.strip():
        raise ValueError("installed_revision is required")

    installed_at = payload.get("installed_at")
    if not isinstance(installed_at, str) or not installed_at.strip():
        raise ValueError("installed_at is required")

    content_fingerprint = payload.get("content_fingerprint")
    if not isinstance(content_fingerprint, str) or not content_fingerprint.startswith("sha256:"):
        raise ValueError("content_fingerprint must be a sha256 fingerprint")

    artifact_digest = payload.get("artifact_digest")
    if artifact_digest is not None and not isinstance(artifact_digest, str):
        raise ValueError("artifact_digest must be a string or null")

    artifact_type = payload.get("artifact_type")
    if artifact_type is not None and not isinstance(artifact_type, str):
        raise ValueError("artifact_type must be a string or null")

    server_name = payload.get("mcp_server_name")
    if server_name is not None and not isinstance(server_name, str):
        raise ValueError("mcp_server_name must be a string or null")

    server_version = payload.get("mcp_server_version")
    if server_version is not None and not isinstance(server_version, str):
        raise ValueError("mcp_server_version must be a string or null")

    return InstalledSkillSource(
        schema_version=SKILL_SOURCE_SCHEMA_VERSION,
        installed_via="mcp",
        source_origin="mcp",
        repo_url=repo_url.strip(),
        repo_ref=None,
        repo_path=repo_path.strip(),
        source_url=source_url.strip() if isinstance(source_url, str) else None,
        installed_commit=None,
        installed_path_oid=None,
        installed_revision=installed_revision.strip(),
        installed_at=installed_at.strip(),
        content_fingerprint=content_fingerprint,
        mcp_server_name=server_name.strip() if isinstance(server_name, str) else None,
        mcp_server_version=server_version.strip() if isinstance(server_version, str) else None,
        artifact_digest=artifact_digest.strip() if isinstance(artifact_digest, str) else None,
        artifact_type=artifact_type.strip() if isinstance(artifact_type, str) else None,
    )


def build_installed_skill_source(
    *,
    skill: MarketplaceSkill,
    source_origin: SkillSourceOrigin,
    installed_commit: str | None,
    installed_path_oid: str | None,
    fingerprint: str,
) -> InstalledSkillSource:
    installed_revision = installed_commit or LOCAL_REVISION
    return InstalledSkillSource(
        schema_version=SKILL_SOURCE_SCHEMA_VERSION,
        installed_via="marketplace",
        source_origin=source_origin,
        repo_url=skill.repo_url,
        repo_ref=skill.repo_ref,
        repo_path=skill.repo_path,
        source_url=skill.source_url,
        installed_commit=installed_commit,
        installed_path_oid=installed_path_oid,
        installed_revision=installed_revision,
        installed_at=_iso_utc_now(),
        content_fingerprint=fingerprint,
    )


def build_mcp_installed_skill_source(
    *,
    server_name: str,
    server_version: str | None,
    skill_uri: str,
    fingerprint: str,
    artifact_digest: str,
    artifact_type: str,
) -> InstalledSkillSource:
    return InstalledSkillSource(
        schema_version=SKILL_SOURCE_SCHEMA_VERSION,
        installed_via="mcp",
        source_origin="mcp",
        repo_url=f"mcp://{server_name}",
        repo_ref=None,
        repo_path=skill_uri,
        source_url=skill_uri,
        installed_commit=None,
        installed_path_oid=None,
        installed_revision=artifact_digest,
        installed_at=_iso_utc_now(),
        content_fingerprint=fingerprint,
        mcp_server_name=server_name,
        mcp_server_version=server_version,
        artifact_digest=artifact_digest,
        artifact_type=artifact_type,
    )


def _iso_utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
