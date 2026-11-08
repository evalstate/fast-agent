"""Recoverable publication of a session's history, store and metadata files.

Callers hold the existing session checkpoint lock for staging and publication.
A journal marker follows snapshot publication, distinguishing an interrupted
commit from interrupted cleanup without changing the persisted snapshot schema.
"""

from __future__ import annotations

import asyncio
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

import anyio
from pydantic import BaseModel, ConfigDict

from fast_agent.core.logging.logger import get_logger

if TYPE_CHECKING:
    from collections.abc import Awaitable

logger = get_logger(__name__)
_DIRECTORY = ".checkpoint-transaction"
_MANIFEST = "manifest.json"
_SNAPSHOT = "session.json"
_T = TypeVar("_T")


class _Entry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    filename: str
    existed: bool
    delete: bool


class _Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    entries: list[_Entry]


def _relative_path(directory: Path, filename: str) -> Path:
    relative = Path(filename)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError(f"Checkpoint filename must remain inside the session: {filename!r}")
    path = directory / relative
    if not path.resolve().is_relative_to(directory.resolve()) or relative.parts[0] == _DIRECTORY:
        raise ValueError(f"Checkpoint filename must remain inside the session: {filename!r}")
    return path


def _write_bytes(path: Path, content: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


async def finish_checkpoint_io(awaitable: Awaitable[_T]) -> _T:
    """Do not release the checkpoint lock while its worker still accesses files."""
    task = asyncio.ensure_future(awaitable)
    cancelled = False
    with anyio.CancelScope(shield=True):
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                cancelled = True
    result = task.result()
    if cancelled:
        raise asyncio.CancelledError
    # A surrounding AnyIO scope cannot deliver cancellation inside the shield.
    # Observe it after the worker has joined, before the caller can continue.
    await anyio.lowlevel.checkpoint_if_cancelled()
    return result


class CheckpointTransaction:
    """Stage all changes, retaining originals until the commit marker is visible."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.workspace = directory / _DIRECTORY
        self.entries: list[_Entry] = []
        self.recover(directory)
        self.workspace.mkdir(mode=0o700)
        (self.workspace / "new").mkdir(mode=0o700)
        (self.workspace / "old").mkdir(mode=0o700)

    def staged_path(self, filename: str) -> Path:
        target = _relative_path(self.directory, filename)
        filename = target.relative_to(self.directory).as_posix()
        if filename == _SNAPSHOT:
            raise ValueError("session.json is reserved for the session snapshot")
        if any(entry.filename == filename for entry in self.entries):
            raise ValueError(f"Duplicate checkpoint filename: {filename!r}")
        index = len(self.entries)
        self.entries.append(_Entry(filename=filename, existed=target.exists(), delete=False))
        # Keep the suffix: HistoryExporter chooses JSON or text from it.
        staged = self.workspace / "new" / f"{index}{target.suffix}"
        staged.touch(mode=0o600, exist_ok=False)
        return staged

    def stage_bytes(self, filename: str, content: bytes) -> None:
        path = self.staged_path(filename)
        _write_bytes(path, content)

    def stage_delete(self, filename: str) -> None:
        target = _relative_path(self.directory, filename)
        filename = target.relative_to(self.directory).as_posix()
        if any(entry.filename == filename for entry in self.entries) or filename == _SNAPSHOT:
            raise ValueError(f"Duplicate or reserved checkpoint filename: {filename!r}")
        self.entries.append(_Entry(filename=filename, existed=target.exists(), delete=True))

    def prepare(self, snapshot: bytes) -> None:
        """Complete backups before making the recovery manifest visible."""
        self.entries.append(
            _Entry(filename=_SNAPSHOT, existed=(self.directory / _SNAPSHOT).exists(), delete=False)
        )
        for index, entry in enumerate(self.entries):
            if entry.existed:
                shutil.copy2(
                    _relative_path(self.directory, entry.filename),
                    self.workspace / "old" / str(index),
                )
        _write_bytes(self._new_path(len(self.entries) - 1, _SNAPSHOT), snapshot)
        manifest = _Manifest(entries=self.entries)
        _write_bytes(self.workspace / "commit-ready", b"committed\n")
        temporary = self.workspace / "manifest.tmp"
        _write_bytes(temporary, manifest.model_dump_json(indent=2).encode("utf-8"))
        temporary.replace(self.workspace / _MANIFEST)

    def _new_path(self, index: int, filename: str) -> Path:
        return self.workspace / "new" / f"{index}{Path(filename).suffix}"

    def publish(self) -> None:
        """Publish under the checkpoint lock; restore originals on any failure."""
        try:
            for index, entry in enumerate(self.entries):
                target = _relative_path(self.directory, entry.filename)
                if entry.delete:
                    target.unlink(missing_ok=True)
                else:
                    self._new_path(index, entry.filename).replace(target)
            (self.workspace / "commit-ready").replace(self.workspace / "committed")
        except BaseException:
            self.recover(self.directory)
            raise
        self._discard_committed_workspace()

    def _discard_committed_workspace(self) -> None:
        try:
            # Retire recovery instructions before deleting their marker or
            # backups. A partial rmtree must never make a completed commit
            # look like a transaction that needs to be rolled back.
            (self.workspace / _MANIFEST).unlink()
            shutil.rmtree(self.workspace)
        except OSError as exc:
            # The snapshot is already committed. A later reader can discard
            # this journal; reporting the save as failed would be misleading.
            logger.warning(
                "Failed to remove committed checkpoint journal", data={"error": str(exc)}
            )

    def abort(self) -> None:
        """Discard unpublished staging or recover a prepared transaction."""
        self.recover(self.directory)

    @staticmethod
    def pending(directory: Path) -> bool:
        return (directory / _DIRECTORY).exists()

    @classmethod
    def recover(cls, directory: Path) -> bool:
        """Recover an interrupted publication while holding its checkpoint lock."""
        workspace = directory / _DIRECTORY
        if not workspace.exists():
            return False
        manifest_path = workspace / _MANIFEST
        if not manifest_path.exists():
            shutil.rmtree(workspace)
            return True
        manifest = _Manifest.model_validate_json(manifest_path.read_bytes())
        cls._validate_manifest(directory, manifest)
        committed = (workspace / "committed").is_file()
        if not committed:
            cls._restore(directory, workspace, manifest)
        manifest_path.unlink()
        shutil.rmtree(workspace)
        return True

    @staticmethod
    def _validate_manifest(directory: Path, manifest: _Manifest) -> None:
        names = [Path(entry.filename).as_posix() for entry in manifest.entries]
        if not names or names[-1] != _SNAPSHOT or len(names) != len(set(names)):
            raise ValueError("Invalid session checkpoint recovery manifest")
        for name in names:
            _relative_path(directory, name)

    @staticmethod
    def _restore(directory: Path, workspace: Path, manifest: _Manifest) -> None:
        for index, entry in enumerate(manifest.entries):
            target = _relative_path(directory, entry.filename)
            if entry.existed:
                # Copy backups rather than consuming them: recovery itself
                # may be interrupted and must be repeatable.
                temporary = workspace / f"restore-{index}.tmp"
                shutil.copy2(workspace / "old" / str(index), temporary)
                temporary.replace(target)
            else:
                target.unlink(missing_ok=True)
