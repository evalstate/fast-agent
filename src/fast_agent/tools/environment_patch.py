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
from fast_agent.patch.errors import ApplyPatchError

if TYPE_CHECKING:
    from fast_agent.patch.parser import Hunk
    from fast_agent.tools.execution_environment import EnvironmentFilesystem


async def apply_patch_to_environment_filesystem(
    filesystem: "EnvironmentFilesystem",
    hunks: list["Hunk"],
) -> str:
    """Apply parsed patch hunks to an environment filesystem and return patch output."""
    _reject_path_aliases(filesystem, hunks)
    with tempfile.TemporaryDirectory(prefix="fast-agent-environment-patch-") as temp_dir:
        base = Path(temp_dir)
        path_map = _PatchPathMap()
        transformed_hunks = [_transform_hunk(hunk, path_map) for hunk in hunks]
        await _stage_patch_inputs(filesystem, base, hunks, path_map)
        affected = apply_hunks_to_files(transformed_hunks, base_directory=base)
        await _sync_patch_outputs(
            filesystem,
            base,
            list(dict.fromkeys(_hunk_paths(transformed_hunks))),
            path_map,
        )

    stdout = io.StringIO()
    print_summary(path_map.restore_affected(affected), stdout)
    return stdout.getvalue().strip()


class _PatchPathMap:
    """Map environment paths onto opaque temporary path components."""

    def __init__(self) -> None:
        self._local_by_remote: dict[Path, Path] = {}
        self._remote_by_local: dict[Path, Path] = {}

    def to_local(self, remote: Path) -> Path:
        local = self._local_by_remote.get(remote)
        if local is not None:
            return local
        local = Path(str(len(self._local_by_remote)))
        self._local_by_remote[remote] = local
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
    touched: list[Path],
    path_map: _PatchPathMap,
) -> None:
    # A path can be created, moved, deleted, and recreated in one patch.
    # The engine's final state, not its operation summary, owns the outcome.
    deleted: list[Path] = []
    for local_path in touched:
        remote_path = path_map.to_remote(local_path)
        staged_path = base / local_path
        if staged_path.exists():
            content = staged_path.read_text(encoding="utf-8")
            await filesystem.write_text(str(remote_path), content)
        else:
            deleted.append(remote_path)
    # Preserve move sources until all destination writes have succeeded.
    for remote_path in deleted:
        # Intermediate patch files may never have existed in the environment.
        if await filesystem.exists(str(remote_path)):
            await filesystem.remove(str(remote_path))


def _transform_hunk(hunk: "Hunk", path_map: _PatchPathMap) -> "Hunk":
    if hunk.kind == "add":
        return replace(hunk, path=path_map.to_local(hunk.path))
    if hunk.kind == "delete":
        return replace(hunk, path=path_map.to_local(hunk.path))
    move_path = path_map.to_local(hunk.move_path) if hunk.move_path is not None else None
    return replace(hunk, path=path_map.to_local(hunk.path), move_path=move_path)


def _input_paths(hunks: list["Hunk"]) -> list[Path]:
    paths: list[Path] = []
    touched: set[Path] = set()
    for hunk in hunks:
        if hunk.kind in {"delete", "update"} and hunk.path not in touched:
            paths.append(hunk.path)
        # Never reload a path whose state is already determined by the patch,
        # including deleted paths: the engine must reject invalid later reads.
        touched.add(hunk.path)
        if hunk.kind == "update" and hunk.move_path is not None:
            touched.add(hunk.move_path)
    return paths


def _reject_path_aliases(filesystem: "EnvironmentFilesystem", hunks: list["Hunk"]) -> None:
    """Reject distinct patch paths that address one environment file."""
    raw_by_resolved: dict[str, Path] = {}
    for path in _hunk_paths(hunks):
        resolved = filesystem.resolve_path(str(path))
        previous = raw_by_resolved.get(resolved)
        if previous is not None and previous != path:
            raise ApplyPatchError(
                f"Patch paths {previous} and {path} resolve to the same environment path {resolved}."
            )
        raw_by_resolved[resolved] = path


def _hunk_paths(hunks: list["Hunk"]) -> list[Path]:
    paths: list[Path] = []
    for hunk in hunks:
        paths.append(hunk.path)
        if hunk.kind == "update" and hunk.move_path is not None:
            paths.append(hunk.move_path)
    return paths
