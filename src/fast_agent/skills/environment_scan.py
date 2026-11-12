"""Discover Agent Skills from an environment filesystem.

Skills are part of the environment the agent works in: discovery scans the
active environment's filesystem, so the paths rendered into the prompt are
readable by the environment ``read_text_file`` tool and skill scripts are
executable by the environment shell. Mount or copy a skills tree (or the
fast-agent home) into the environment to make skills available there.
"""

from __future__ import annotations

import asyncio
import posixpath
from pathlib import Path
from typing import TYPE_CHECKING

from fast_agent.constants import DEFAULT_SKILLS_PATHS
from fast_agent.core.logging.logger import get_logger
from fast_agent.skills.models import SKILL_MANIFEST_FILENAME
from fast_agent.skills.registry import SkillManifest, SkillRegistry, merge_skill_manifests

if TYPE_CHECKING:
    from collections.abc import Sequence

    from fast_agent.tools.execution_environment import EnvironmentFilesystem

logger = get_logger(__name__)


async def scan_environment_skills(
    filesystem: "EnvironmentFilesystem",
    *,
    directories: "Sequence[str] | None" = None,
) -> tuple[list[SkillManifest], list[str]]:
    """Scan skill directories inside an environment filesystem.

    Relative directories resolve against the environment cwd. Missing
    directories are skipped silently (they are optional defaults); manifest
    parse failures and duplicate overrides are reported as warnings.
    """
    warnings: list[str] = []
    collected: list[SkillManifest] = []
    entries = list(directories) if directories is not None else list(DEFAULT_SKILLS_PATHS)
    for directory in entries:
        resolved = filesystem.resolve_path(str(directory))
        collected.extend(await _scan_directory(filesystem, resolved, warnings))
    manifests, duplicate_warnings = merge_skill_manifests(collected)
    warnings.extend(duplicate_warnings)
    return manifests, warnings


async def _scan_directory(
    filesystem: "EnvironmentFilesystem",
    directory: str,
    warnings: list[str],
) -> list[SkillManifest]:
    try:
        entries = await filesystem.list_dir(directory)
    except Exception:
        # Missing/non-directory paths use provider-specific errors; optional
        # skill directories are expected to be absent.
        logger.debug("Environment skills directory not found", data={"directory": directory})
        return []

    manifest_paths = [
        posixpath.join(entry.path, SKILL_MANIFEST_FILENAME)
        for entry in sorted(entries, key=lambda entry: entry.path)
        if entry.kind == "directory"
    ]
    manifests: list[SkillManifest] = []
    # Bound both remote I/O and task creation; gather preserves directory order.
    for start in range(0, len(manifest_paths), 8):
        results = await asyncio.gather(
            *(_read_manifest(filesystem, path) for path in manifest_paths[start : start + 8])
        )
        for manifest, warning in results:
            if manifest is not None:
                manifests.append(manifest)
            if warning is not None:
                warnings.append(warning)
    return manifests


async def _read_manifest(
    filesystem: "EnvironmentFilesystem", manifest_path: str
) -> tuple[SkillManifest | None, str | None]:
    try:
        if not await filesystem.exists(manifest_path):
            return None, None
        manifest_text = await filesystem.read_text(manifest_path)
    except Exception as exc:
        return None, f"Failed to read skill manifest {manifest_path}: {exc}"
    manifest, error = SkillRegistry.parse_manifest_text(manifest_text, path=Path(manifest_path))
    if manifest is not None:
        return manifest, None
    return None, f"Failed to parse skill manifest {manifest_path}: {error or 'invalid manifest'}"


__all__ = ["scan_environment_skills"]
