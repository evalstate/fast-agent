"""Skills directory resolution helpers.

This module isolates discovery/management scope decisions from CLI and runtime
presentation code so the behavior can later move behind a smaller reusable API.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from fast_agent.config import Settings, get_settings
from fast_agent.paths import default_skill_paths
from fast_agent.skills.models import SkillsManagementScope

if TYPE_CHECKING:
    from collections.abc import Sequence


def _resolve_skill_directory_entry(raw_path: str | Path, *, base: Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = (base / path).resolve()
    return path


def resolve_skills_management_scope(
    settings: Settings | None = None,
    *,
    cwd: Path | None = None,
    managed_directory_override: str | Path | None = None,
) -> SkillsManagementScope:
    """Resolve both the managed skills directory and all discovered skill directories."""
    base = cwd or Path.cwd()
    resolved_settings = settings or get_settings()
    skills_settings = getattr(resolved_settings, "skills", None)

    configured_directories: list[Path] | None = None
    if skills_settings and getattr(skills_settings, "directories", None):
        directory_entries = skills_settings.directories
        if directory_entries:
            configured_directories = [
                _resolve_skill_directory_entry(entry, base=base) for entry in directory_entries
            ]

    if managed_directory_override is not None:
        managed_directory = _resolve_skill_directory_entry(managed_directory_override, base=base)
        management_source = "override"
    elif configured_directories:
        managed_directory = configured_directories[0]
        management_source = "settings"
    else:
        managed_directory = default_skill_paths(resolved_settings, cwd=base)[0]
        management_source = "default"

    discovered_directories = (
        list(configured_directories)
        if configured_directories is not None
        else default_skill_paths(resolved_settings, cwd=base)
    )
    if managed_directory not in discovered_directories:
        discovered_directories.append(managed_directory)

    return SkillsManagementScope(
        managed_directory=managed_directory,
        discovered_directories=discovered_directories,
        management_source=management_source,
    )


def get_manager_directory(
    settings: Settings | None = None,
    *,
    cwd: Path | None = None,
    managed_directory_override: str | Path | None = None,
) -> Path:
    """Resolve the local skills directory that management operations target."""
    return resolve_skills_management_scope(
        settings,
        cwd=cwd,
        managed_directory_override=managed_directory_override,
    ).managed_directory


def resolve_skill_directories(
    settings: Settings | None = None,
    *,
    cwd: Path | None = None,
    managed_directory_override: str | Path | None = None,
) -> list[Path]:
    return resolve_skills_management_scope(
        settings,
        cwd=cwd,
        managed_directory_override=managed_directory_override,
    ).discovered_directories


def order_skill_directories_for_display(
    directories: Sequence[Path],
    *,
    settings: Settings | None = None,
    cwd: Path | None = None,
    managed_directory_override: str | Path | None = None,
) -> list[Path]:
    managed_directory = get_manager_directory(
        settings,
        cwd=cwd,
        managed_directory_override=managed_directory_override,
    )
    ordered = [directory for directory in directories if directory != managed_directory]
    managed_entries = [directory for directory in directories if directory == managed_directory]
    return [*ordered, *managed_entries]
