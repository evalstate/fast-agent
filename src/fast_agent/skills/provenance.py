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
    if source.source_origin == "remote":
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
    ref_label = f"@{source.repo_ref}" if source.repo_ref else ""
    provenance_value = f"{source.repo_url}{ref_label} ({source.repo_path})"
    installed_value = (
        f"{format_installed_at_display(source.installed_at)} "
        f"revision: {format_revision_short(source.installed_revision)}"
    )
    return provenance_value, installed_value


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


def _iso_utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
