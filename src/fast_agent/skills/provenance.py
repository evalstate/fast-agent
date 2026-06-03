"""Skills provenance and sidecar metadata helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

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
    root = skill_dir.resolve()
    return marketplace_source_utils.compute_directory_content_fingerprint(
        root,
        sidecar_path=get_skill_source_sidecar_path(root),
    )


def read_installed_skill_source(
    skill_dir: Path,
) -> marketplace_source_utils.InstalledSourceReadResult[InstalledSkillSource]:
    return marketplace_source_utils.read_installed_source_file(
        get_skill_source_sidecar_path(skill_dir),
        parse_payload=parse_installed_skill_source_payload,
    )


def write_installed_skill_source(skill_dir: Path, source: InstalledSkillSource) -> None:
    extra_payload: dict[str, str] = {}
    if source.mcp_server_name is not None:
        extra_payload["mcp_server_name"] = source.mcp_server_name
    if source.mcp_server_version is not None:
        extra_payload["mcp_server_version"] = source.mcp_server_version
    if source.artifact_digest is not None:
        extra_payload["artifact_digest"] = source.artifact_digest
    if source.artifact_type is not None:
        extra_payload["artifact_type"] = source.artifact_type

    marketplace_source_utils.write_installed_source_file(
        get_skill_source_sidecar_path(skill_dir),
        cast("Any", source),
        extra_payload=extra_payload or None,
    )


def get_skill_provenance(skill_dir: Path) -> SkillProvenance:
    read_result = read_installed_skill_source(skill_dir)
    if read_result.source is None:
        if read_result.error is None:
            return SkillProvenance(
                status="unmanaged",
                summary="unmanaged (no sidecar)",
            )
        return SkillProvenance(
            status="invalid_metadata",
            summary=f"invalid metadata ({read_result.error})",
            error=read_result.error,
        )

    source = read_result.source
    return SkillProvenance(
        status="managed",
        summary=_source_summary(source),
        source=source,
    )


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
    provenance_value = f"{_source_location(source)} ({source.repo_path})"
    installed_value = marketplace_formatting.format_installed_revision_display(
        source.installed_at,
        source.installed_revision,
    )
    if source.source_origin == "mcp":
        version = f"@{source.mcp_server_version}" if source.mcp_server_version else ""
        integrity = " • integrity: sha256" if source.artifact_digest else ""
        provenance_value = (
            f"mcp-server {source.mcp_server_name or source.repo_url}{version} "
            f"({source.repo_path}){integrity}"
        )
        installed_value = marketplace_formatting.format_installed_revision_display(
            source.installed_at,
            source.installed_revision,
        )
    return provenance_value, installed_value


def _source_location(source: InstalledSkillSource) -> str:
    if source.source_origin == "mcp":
        version = f"@{source.mcp_server_version}" if source.mcp_server_version else ""
        return f"mcp-server {source.mcp_server_name or source.repo_url}{version}"
    return marketplace_formatting.format_source_location(source.repo_url, source.repo_ref)


def _source_summary(source: InstalledSkillSource) -> str:
    if source.source_origin == "mcp":
        source_label = "mcp"
    elif source.source_origin == "remote":
        source_label = "marketplace"
    else:
        source_label = "local source"
    return f"managed ({source_label}) • {_source_location(source)} • {source.repo_path}"


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
    repo_path: str | None = None,
    fingerprint: str,
) -> InstalledSkillSource:
    installed_revision = installed_commit or LOCAL_REVISION
    return InstalledSkillSource(
        schema_version=SKILL_SOURCE_SCHEMA_VERSION,
        installed_via="marketplace",
        source_origin=source_origin,
        repo_url=skill.repo_url,
        repo_ref=skill.repo_ref,
        repo_path=repo_path or skill.repo_subdir,
        source_url=skill.source_url,
        installed_commit=installed_commit,
        installed_path_oid=installed_path_oid,
        installed_revision=installed_revision,
        installed_at=marketplace_formatting.iso_utc_now(),
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
        installed_at=marketplace_formatting.iso_utc_now(),
        content_fingerprint=fingerprint,
        mcp_server_name=server_name,
        mcp_server_version=server_version,
        artifact_digest=artifact_digest,
        artifact_type=artifact_type,
    )
