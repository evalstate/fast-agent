"""Filesystem failure and process interruption contracts for checkpoint journals."""

from __future__ import annotations

import errno
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from fast_agent.session.checkpoint import CheckpointTransaction

_OLD = {
    "history_main_previous.json": b"older turn",
    "history_main.json": b"committed turn",
    "ptc_store_main.json": b"committed store",
    "session.json": b'{"snapshot":"committed"}',
}
_NEW_SNAPSHOT = b'{"snapshot":"next"}'


def _seed(directory: Path) -> None:
    for filename, data in _OLD.items():
        (directory / filename).write_bytes(data)


def _transaction(directory: Path, *, delete_store: bool = False) -> CheckpointTransaction:
    transaction = CheckpointTransaction(directory)
    transaction.stage_bytes("history_main_previous.json", _OLD["history_main.json"])
    transaction.stage_bytes("history_main.json", b"next turn")
    if delete_store:
        transaction.stage_delete("ptc_store_main.json")
    else:
        transaction.stage_bytes("ptc_store_main.json", b"next store")
    transaction.prepare(_NEW_SNAPSHOT)
    return transaction


def _contents(directory: Path) -> dict[str, bytes]:
    return {path.name: path.read_bytes() for path in directory.iterdir() if path.is_file()}


@pytest.mark.parametrize("fail_at", range(5))
@pytest.mark.parametrize("delete_store", [False, True])
def test_publication_failure_restores_every_committed_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail_at: int, delete_store: bool
) -> None:
    _seed(tmp_path)
    transaction = _transaction(tmp_path, delete_store=delete_store)
    replace = os.replace
    unlink = Path.unlink
    published = 0
    failed = False

    def fail(path: Path) -> None:
        nonlocal published, failed
        if (path.parent == tmp_path or path.name == "committed") and not failed:
            if published == fail_at:
                failed = True
                raise OSError(errno.EIO, "publication failure")
            published += 1

    def replace_file(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
        fail(Path(dst))
        replace(src, dst)

    def remove_file(path: Path, *, missing_ok: bool = False) -> None:
        fail(path)
        unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(os, "replace", replace_file)
    monkeypatch.setattr(Path, "unlink", remove_file)

    with pytest.raises(OSError, match="publication failure"):
        transaction.publish()

    assert _contents(tmp_path) == _OLD
    assert list(tmp_path.iterdir()) == list(path for path in tmp_path.iterdir() if path.is_file())
    CheckpointTransaction.recover(tmp_path)
    assert _contents(tmp_path) == _OLD


@pytest.mark.parametrize("crash_after", [1, 2, 3, 4, 5])
def test_process_interruption_recovers_at_every_publication_boundary(
    tmp_path: Path, crash_after: int
) -> None:
    _seed(tmp_path)
    program = r"""
import os
import sys
from pathlib import Path
from fast_agent.session.checkpoint import CheckpointTransaction
root = Path(sys.argv[1])
transaction = CheckpointTransaction(root)
transaction.stage_bytes("history_main_previous.json", b"committed turn")
transaction.stage_bytes("history_main.json", b"next turn")
transaction.stage_bytes("ptc_store_main.json", b"next store")
transaction.prepare(b'{"snapshot":"next"}')
original = os.replace
count = 0
def replace(source, destination):
    global count
    original(source, destination)
    if Path(destination).parent == root or Path(destination).name == "committed":
        count += 1
        if count == int(sys.argv[2]):
            os._exit(17)
os.replace = replace
transaction.publish()
"""
    result = subprocess.run(
        [sys.executable, "-c", program, str(tmp_path), str(crash_after)],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert result.returncode == 17, result.stderr

    CheckpointTransaction.recover(tmp_path)

    expected = (
        _OLD
        if crash_after < 5
        else {
            "history_main_previous.json": b"committed turn",
            "history_main.json": b"next turn",
            "ptc_store_main.json": b"next store",
            "session.json": _NEW_SNAPSHOT,
        }
    )
    assert _contents(tmp_path) == expected
    assert not (tmp_path / ".checkpoint-transaction").exists()
    CheckpointTransaction.recover(tmp_path)
    assert _contents(tmp_path) == expected


def test_rollback_can_be_retried_after_rollback_itself_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed(tmp_path)
    transaction = _transaction(tmp_path)
    replace = os.replace
    calls = 0

    def replace_file(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
        nonlocal calls
        if Path(dst).parent == tmp_path:
            calls += 1
            if calls in {3, 5}:
                raise OSError(errno.EIO, "interrupted rollback")
        replace(src, dst)

    monkeypatch.setattr(os, "replace", replace_file)
    with pytest.raises(OSError, match="interrupted rollback"):
        transaction.publish()
    assert (tmp_path / ".checkpoint-transaction" / "manifest.json").is_file()
    monkeypatch.setattr(os, "replace", replace)

    CheckpointTransaction.recover(tmp_path)

    assert _contents(tmp_path) == _OLD


def test_unprepared_staging_is_removed_without_touching_committed_files(tmp_path: Path) -> None:
    _seed(tmp_path)
    transaction = CheckpointTransaction(tmp_path)
    transaction.stage_bytes("history_main.json", b"never prepared")

    CheckpointTransaction.recover(tmp_path)

    assert _contents(tmp_path) == _OLD
    assert not transaction.workspace.exists()


def test_committed_checkpoint_survives_journal_cleanup_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import shutil

    _seed(tmp_path)
    transaction = _transaction(tmp_path)
    original = shutil.rmtree

    def fail_cleanup(path: Path) -> None:
        raise PermissionError("journal cleanup busy")

    monkeypatch.setattr(shutil, "rmtree", fail_cleanup)
    transaction.publish()
    assert (tmp_path / "history_main.json").read_bytes() == b"next turn"
    assert (tmp_path / "session.json").read_bytes() == _NEW_SNAPSHOT
    monkeypatch.setattr(shutil, "rmtree", original)

    CheckpointTransaction.recover(tmp_path)

    assert (tmp_path / "history_main.json").read_bytes() == b"next turn"
    assert not transaction.workspace.exists()


@pytest.mark.parametrize("filename", ["../outside", "/outside", ".checkpoint-transaction/new"])
def test_invalid_staged_filename_cannot_change_other_files(tmp_path: Path, filename: str) -> None:
    _seed(tmp_path)
    transaction = CheckpointTransaction(tmp_path)

    with pytest.raises(ValueError, match="inside the session"):
        transaction.stage_bytes(filename, b"invalid")
    transaction.abort()

    assert _contents(tmp_path) == _OLD


def test_recovery_rejects_manifest_path_traversal_without_changing_files(tmp_path: Path) -> None:
    _seed(tmp_path)
    transaction = _transaction(tmp_path)
    manifest_path = transaction.workspace / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["entries"][0]["filename"] = "../outside"
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="inside the session"):
        CheckpointTransaction.recover(tmp_path)

    assert _contents(tmp_path) == _OLD
    assert manifest_path.exists()


def test_unchanged_snapshot_bytes_do_not_misidentify_failed_publication_as_committed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed(tmp_path)
    transaction = CheckpointTransaction(tmp_path)
    transaction.stage_bytes("history_main.json", b"uncommitted turn")
    transaction.prepare(_OLD["session.json"])
    replace = os.replace
    failed = False

    def replace_file(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
        nonlocal failed
        if Path(dst) == tmp_path / "session.json" and not failed:
            failed = True
            raise OSError(errno.EIO, "metadata publication failure")
        replace(src, dst)

    monkeypatch.setattr(os, "replace", replace_file)
    with pytest.raises(OSError, match="metadata publication"):
        transaction.publish()

    assert _contents(tmp_path) == _OLD


@pytest.mark.parametrize("filename", ["session.json", "./session.json", "session.json/"])
def test_snapshot_filename_aliases_are_reserved(tmp_path: Path, filename: str) -> None:
    _seed(tmp_path)
    transaction = CheckpointTransaction(tmp_path)

    with pytest.raises(ValueError, match="reserved"):
        transaction.stage_bytes(filename, b"not metadata")
    transaction.abort()

    assert _contents(tmp_path) == _OLD


def test_filename_aliases_cannot_publish_same_target_twice(tmp_path: Path) -> None:
    _seed(tmp_path)
    transaction = CheckpointTransaction(tmp_path)
    transaction.stage_bytes("history_main.json", b"staged")

    with pytest.raises(ValueError, match="Duplicate"):
        transaction.stage_bytes("./history_main.json", b"conflicting")
    transaction.abort()

    assert _contents(tmp_path) == _OLD


def test_partially_removed_committed_journal_cannot_roll_back_committed_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import shutil

    _seed(tmp_path)
    transaction = _transaction(tmp_path)
    original = shutil.rmtree

    def partial_cleanup(path: Path) -> None:
        (path / "committed").unlink(missing_ok=True)
        raise OSError(errno.EIO, "cleanup interrupted after marker removal")

    monkeypatch.setattr(shutil, "rmtree", partial_cleanup)
    transaction.publish()
    assert (tmp_path / "session.json").read_bytes() == _NEW_SNAPSHOT
    monkeypatch.setattr(shutil, "rmtree", original)

    CheckpointTransaction.recover(tmp_path)

    assert (tmp_path / "session.json").read_bytes() == _NEW_SNAPSHOT
    assert (tmp_path / "history_main.json").read_bytes() == b"next turn"


def test_partially_removed_rollback_journal_does_not_require_deleted_backups(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import shutil

    _seed(tmp_path)
    transaction = _transaction(tmp_path)
    original = shutil.rmtree

    def partial_cleanup(path: Path) -> None:
        (path / "old" / "0").unlink(missing_ok=True)
        raise OSError(errno.EIO, "cleanup interrupted after backup removal")

    monkeypatch.setattr(shutil, "rmtree", partial_cleanup)
    with pytest.raises(OSError, match="cleanup interrupted"):
        transaction.abort()
    monkeypatch.setattr(shutil, "rmtree", original)

    CheckpointTransaction.recover(tmp_path)

    assert _contents(tmp_path) == _OLD
