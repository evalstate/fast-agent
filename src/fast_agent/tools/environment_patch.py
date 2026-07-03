"""Apply ``apply_patch`` hunks through an environment filesystem.

Adapter authors should not reimplement patch semantics. If an environment owns
files, implement ``EnvironmentFilesystem`` and let this module stage, apply, and
sync patches through that filesystem.
"""

from __future__ import annotations

import io
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from fast_agent.patch.engine import AffectedPaths, apply_hunks_to_files, print_summary

if TYPE_CHECKING:
    from fast_agent.patch.parser import Hunk
    from fast_agent.tools.execution_environment import EnvironmentFilesystem


async def apply_patch_to_environment_filesystem(
    filesystem: "EnvironmentFilesystem",
    hunks: list["Hunk"],
) -> str:
    """Apply parsed patch hunks to an environment filesystem and return patch output."""
    with tempfile.TemporaryDirectory(prefix="fast-agent-environment-patch-") as temp_dir:
        base = Path(temp_dir)
        path_map = _PatchPathMap()
        transformed_hunks = [_transform_hunk(hunk, path_map) for hunk in hunks]
        await _stage_patch_inputs(filesystem, base, hunks, path_map)
        affected = apply_hunks_to_files(transformed_hunks, base_directory=base)
        await _sync_patch_outputs(
            filesystem,
            base,
            affected.added + affected.modified,
            affected.deleted,
            path_map,
        )

    stdout = io.StringIO()
    print_summary(path_map.restore_affected(affected), stdout)
    return stdout.getvalue().strip()


class _PatchPathMap:
    """Map possibly-absolute environment paths onto a temporary relative tree."""

    def __init__(self) -> None:
        self._remote_by_local: dict[Path, Path] = {}

    def to_local(self, remote: Path) -> Path:
        local = _local_patch_path(remote)
        self._remote_by_local[local] = remote
        return local

    def to_remote(self, local: Path) -> Path:
        return self._remote_by_local[local]

    def restore_affected(self, affected: AffectedPaths) -> AffectedPaths:
        return replace(
            affected,
            added=[self.to_remote(path) for path in affected.added],
            modified=[self.to_remote(path) for path in affected.modified],
            deleted=[self.to_remote(path) for path in affected.deleted],
        )


async def _stage_patch_inputs(
    filesystem: "EnvironmentFilesystem",
    base: Path,
    hunks: list["Hunk"],
    path_map: _PatchPathMap,
) -> None:
    for path in _input_paths(hunks):
        local_path = base / path_map.to_local(path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        content = await filesystem.read_text(str(path))
        local_path.write_text(content, encoding="utf-8", newline="")


async def _sync_patch_outputs(
    filesystem: "EnvironmentFilesystem",
    base: Path,
    changed: list[Path],
    deleted: list[Path],
    path_map: _PatchPathMap,
) -> None:
    for local_path in changed:
        remote_path = path_map.to_remote(local_path)
        content = (base / local_path).read_text(encoding="utf-8")
        await filesystem.write_text(str(remote_path), content)
    for local_path in deleted:
        remote_path = path_map.to_remote(local_path)
        await filesystem.remove(str(remote_path))


def _local_patch_path(path: Path) -> Path:
    if not path.is_absolute():
        return path
    return Path(*path.parts[1:])


def _transform_hunk(hunk: "Hunk", path_map: _PatchPathMap) -> "Hunk":
    if hunk.kind == "add":
        return replace(hunk, path=path_map.to_local(hunk.path))
    if hunk.kind == "delete":
        return replace(hunk, path=path_map.to_local(hunk.path))
    move_path = path_map.to_local(hunk.move_path) if hunk.move_path is not None else None
    return replace(hunk, path=path_map.to_local(hunk.path), move_path=move_path)


def _input_paths(hunks: list["Hunk"]) -> list[Path]:
    paths: list[Path] = []
    for hunk in hunks:
        if hunk.kind in {"delete", "update"}:
            paths.append(hunk.path)
    return paths
