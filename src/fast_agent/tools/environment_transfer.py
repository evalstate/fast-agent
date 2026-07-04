"""Copy files between environment-backed filesystems."""

from __future__ import annotations

import posixpath
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fast_agent.tools.execution_environment import EnvironmentFilesystemWithBytes


@dataclass(frozen=True, slots=True)
class TransferReport:
    """Summary of an environment filesystem transfer."""

    files: int = 0
    directories: int = 0
    bytes: int = 0

    def __add__(self, other: TransferReport) -> TransferReport:
        return TransferReport(
            files=self.files + other.files,
            directories=self.directories + other.directories,
            bytes=self.bytes + other.bytes,
        )


async def copy_file(
    source: EnvironmentFilesystemWithBytes,
    source_path: str,
    target: EnvironmentFilesystemWithBytes,
    target_path: str,
) -> TransferReport:
    """Copy one file between environment filesystems."""

    content = await source.read_bytes(source_path)
    target_parent = posixpath.dirname(target_path)
    if target_parent:
        await target.mkdir(target_parent)
    await target.write_bytes(target_path, content)
    return TransferReport(files=1, bytes=len(content))


async def copy_tree(
    source: EnvironmentFilesystemWithBytes,
    source_path: str,
    target: EnvironmentFilesystemWithBytes,
    target_path: str,
) -> TransferReport:
    """Recursively copy a directory tree between environment filesystems."""

    await target.mkdir(target_path)
    report = TransferReport(directories=1)
    for entry in await source.list_dir(source_path):
        child_source_path = entry.path
        child_target_path = posixpath.join(target_path, entry.name)
        if entry.kind == "directory":
            report += await copy_tree(source, child_source_path, target, child_target_path)
        elif entry.kind == "file":
            report += await copy_file(source, child_source_path, target, child_target_path)
    return report


__all__ = [
    "TransferReport",
    "copy_file",
    "copy_tree",
]
