from pathlib import Path

import pytest

from fast_agent.core.logging.logger import get_logger
from fast_agent.tools.environment_transfer import copy_file, copy_tree
from fast_agent.tools.local_shell_executor import LocalEnvironment


def _local_environment(root: Path) -> LocalEnvironment:
    return LocalEnvironment(logger=get_logger(__name__), working_directory=root)


@pytest.mark.asyncio
async def test_copy_file_preserves_binary_content(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    source_root.mkdir()
    target_root.mkdir()
    payload = b"\x00fast-agent\xff"
    (source_root / "input.bin").write_bytes(payload)

    report = await copy_file(
        _local_environment(source_root),
        "input.bin",
        _local_environment(target_root),
        "nested/output.bin",
    )

    assert (target_root / "nested" / "output.bin").read_bytes() == payload
    assert report.files == 1
    assert report.bytes == len(payload)


@pytest.mark.asyncio
async def test_copy_tree_recursively_copies_files(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    (source_root / "tree" / "nested").mkdir(parents=True)
    target_root.mkdir()
    (source_root / "tree" / "a.txt").write_text("alpha", encoding="utf-8")
    (source_root / "tree" / "nested" / "b.bin").write_bytes(b"\x00beta")

    report = await copy_tree(
        _local_environment(source_root),
        "tree",
        _local_environment(target_root),
        "copied",
    )

    assert (target_root / "copied" / "a.txt").read_text(encoding="utf-8") == "alpha"
    assert (target_root / "copied" / "nested" / "b.bin").read_bytes() == b"\x00beta"
    assert report.files == 2
    assert report.directories == 2
    assert report.bytes == len("alpha".encode()) + len(b"\x00beta")


@pytest.mark.asyncio
async def test_copy_tree_does_not_follow_directory_symlinks(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    tree = source_root / "tree"
    tree.mkdir(parents=True)
    target_root.mkdir()
    (tree / "a.txt").write_text("alpha", encoding="utf-8")
    (tree / "self").symlink_to(tree, target_is_directory=True)

    report = await copy_tree(
        _local_environment(source_root),
        "tree",
        _local_environment(target_root),
        "copied",
    )

    assert (target_root / "copied" / "a.txt").read_text(encoding="utf-8") == "alpha"
    assert not (target_root / "copied" / "self").exists()
    assert report.files == 1
    assert report.directories == 1
