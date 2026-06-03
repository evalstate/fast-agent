"""Skills provenance and sidecar metadata helpers."""

from __future__ import annotations

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
    marketplace_source_utils.write_installed_source_file(
        get_skill_source_sidecar_path(skill_dir),
        source,
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
    return provenance_value, installed_value


def _source_location(source: InstalledSkillSource) -> str:
    return marketplace_formatting.format_source_location(source.repo_url, source.repo_ref)


def _source_summary(source: InstalledSkillSource) -> str:
    source_label = "marketplace" if source.source_origin == "remote" else "local source"
    return f"managed ({source_label}) • {_source_location(source)} • {source.repo_path}"


def parse_installed_skill_source_payload(payload: dict[str, Any]) -> InstalledSkillSource:
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
