"""Compatibility facade for skills management.

Phase 2 extracts reusable skills-management logic into smaller internal modules
while preserving the historical ``fast_agent.skills.manager`` import surface.
"""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING

from fast_agent.config import Settings, get_settings
from fast_agent.marketplace import registry_urls as marketplace_registry_urls
from fast_agent.skills import operations as skill_operations
from fast_agent.skills.marketplace_parsing import normalize_repo_path
from fast_agent.skills.models import (
    DEFAULT_MARKETPLACE_URL,
    DEFAULT_SKILL_REGISTRIES,
    LOCAL_REVISION,
    SKILL_SOURCE_FILENAME,
    SKILL_SOURCE_SCHEMA_VERSION,
    InstalledSkillSource,
    MarketplaceSkill,
    SkillsManagementScope,
    SkillSourceOrigin,
    SkillUpdateInfo,
    SkillUpdateStatus,
)
from fast_agent.skills.provenance import (
    compute_skill_content_fingerprint,
    format_installed_at_display,
    format_revision_short,
    format_skill_provenance,
    format_skill_provenance_details,
    get_skill_provenance,
    get_skill_source_sidecar_path,
    read_installed_skill_source,
    write_installed_skill_source,
)
from fast_agent.skills.registry import SkillManifest, SkillRegistry
from fast_agent.skills.scope import (
    get_manager_directory,
    order_skill_directories_for_display,
    resolve_skill_directories,
    resolve_skills_management_scope,
)

if TYPE_CHECKING:
    from pathlib import Path

__all__ = [
    "DEFAULT_MARKETPLACE_URL",
    "DEFAULT_SKILL_REGISTRIES",
    "SKILL_SOURCE_FILENAME",
    "SKILL_SOURCE_SCHEMA_VERSION",
    "LOCAL_REVISION",
    "InstalledSkillSource",
    "MarketplaceSkill",
    "SkillSourceOrigin",
    "SkillsManagementScope",
    "SkillUpdateInfo",
    "SkillUpdateStatus",
    "apply_skill_updates",
    "candidate_marketplace_urls",
    "check_skill_updates",
    "compute_skill_content_fingerprint",
    "fetch_marketplace_skills",
    "fetch_marketplace_skills_with_source",
    "format_installed_at_display",
    "format_marketplace_display_url",
    "format_revision_short",
    "format_skill_provenance",
    "format_skill_provenance_details",
    "get_manager_directory",
    "get_marketplace_url",
    "get_skill_provenance",
    "get_skill_source_sidecar_path",
    "install_marketplace_skill",
    "list_local_skills",
    "order_skill_directories_for_display",
    "read_installed_skill_source",
    "reload_skill_manifests",
    "remove_local_skill",
    "resolve_skill_directories",
    "resolve_skill_registries",
    "resolve_skills_management_scope",
    "select_manifest_by_name_or_index",
    "select_skill_by_name_or_index",
    "select_skill_updates",
    "write_installed_skill_source",
    "_normalize_repo_path",
]


def get_marketplace_url(settings: Settings | None = None) -> str:
    resolved_settings = settings or get_settings()
    skills_settings = getattr(resolved_settings, "skills", None)
    url = None
    if skills_settings is not None:
        url = getattr(skills_settings, "marketplace_url", None)
        if not url:
            urls = getattr(skills_settings, "marketplace_urls", None)
            if urls:
                url = urls[0]
    return skill_operations.normalize_marketplace_url(url or DEFAULT_MARKETPLACE_URL)


def format_marketplace_display_url(url: str) -> str:
    return marketplace_registry_urls.format_marketplace_display_url(url)


def resolve_skill_registries(settings: Settings | None = None) -> list[str]:
    resolved_settings = settings or get_settings()
    skills_settings = getattr(resolved_settings, "skills", None)
    configured = getattr(skills_settings, "marketplace_urls", None) if skills_settings else None
    active = getattr(skills_settings, "marketplace_url", None) if skills_settings else None
    return marketplace_registry_urls.resolve_registry_urls(
        configured,
        default_urls=DEFAULT_SKILL_REGISTRIES,
        active_url=active,
    )


def list_local_skills(directory: Path) -> list[SkillManifest]:
    return SkillRegistry.load_directory(directory)


apply_skill_updates = skill_operations.apply_skill_updates
candidate_marketplace_urls = skill_operations.candidate_marketplace_urls
check_skill_updates = skill_operations.check_skill_updates
fetch_marketplace_skills = skill_operations.fetch_marketplace_skills
fetch_marketplace_skills_with_source = skill_operations.fetch_marketplace_skills_with_source
install_marketplace_skill = skill_operations.install_marketplace_skill
reload_skill_manifests = skill_operations.reload_skill_manifests
remove_local_skill = skill_operations.remove_local_skill
select_manifest_by_name_or_index = skill_operations.select_manifest_by_name_or_index
select_skill_by_name_or_index = skill_operations.select_skill_by_name_or_index
select_skill_updates = skill_operations.select_skill_updates

# Compatibility aliases for historical tests/internal imports.
_normalize_marketplace_url = skill_operations.normalize_marketplace_url
_candidate_marketplace_urls = skill_operations.candidate_marketplace_urls
_normalize_repo_path = normalize_repo_path
_install_marketplace_skill_sync = skill_operations.install_marketplace_skill_sync
_check_skill_updates = skill_operations._check_skill_updates
_evaluate_skill_update = skill_operations._evaluate_skill_update
_parse_ls_remote_commit = skill_operations.parse_ls_remote_commit
_resolve_source_path_oid = skill_operations._resolve_source_path_oid
_copy_skill_from_marketplace_source = skill_operations._copy_skill_from_marketplace_source
_atomic_replace_directory = skill_operations._atomic_replace_directory
_resolve_git_commit = skill_operations._resolve_git_commit
_resolve_git_path_oid = skill_operations._resolve_git_path_oid
_run_git = skill_operations._run_git
_load_local_marketplace_payload = skill_operations._load_local_marketplace_payload
_resolve_local_repo = skill_operations._resolve_local_repo
_resolve_repo_subdir = skill_operations._resolve_repo_subdir
_copy_skill_source = skill_operations._copy_skill_source
_resolve_skill_source_dir = skill_operations._resolve_skill_source_dir


def _resolve_source_revision(
    source: InstalledSkillSource,
    head_cache: skill_operations.HeadCache,
) -> skill_operations.HeadResolution:
    return skill_operations.resolve_source_revision(
        source,
        head_cache,
        resolve_local_repo_fn=_resolve_local_repo,
        run_subprocess_fn=subprocess.run,
    )
