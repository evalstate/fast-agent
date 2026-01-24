from __future__ import annotations

from pathlib import Path

import pytest

from fast_agent.patch.errors import InvalidHunkError, InvalidPatchError
from fast_agent.patch.parser import (
    AddFileHunk,
    ApplyPatchArgs,
    DeleteFileHunk,
    ParseMode,
    UpdateFileChunk,
    UpdateFileHunk,
    parse_patch_text,
)


def test_parse_patch_invalid_markers() -> None:
    with pytest.raises(InvalidPatchError, match=r"first line.*Begin Patch"):
        parse_patch_text("bad", ParseMode.STRICT)

    with pytest.raises(InvalidPatchError, match=r"last line.*End Patch"):
        parse_patch_text("*** Begin Patch\nbad", ParseMode.STRICT)


def test_parse_patch_hunks() -> None:
    patch = (
        "*** Begin Patch\n"
        "*** Add File: path/add.py\n"
        "+abc\n"
        "+def\n"
        "*** Delete File: path/delete.py\n"
        "*** Update File: path/update.py\n"
        "*** Move to: path/update2.py\n"
        "@@ def f():\n"
        "-    pass\n"
        "+    return 123\n"
        "*** End Patch"
    )
    parsed = parse_patch_text(patch, ParseMode.STRICT)
    assert parsed.hunks == [
        AddFileHunk(kind="add", path=Path("path/add.py"), contents="abc\ndef\n"),
        DeleteFileHunk(kind="delete", path=Path("path/delete.py")),
        UpdateFileHunk(
            kind="update",
            path=Path("path/update.py"),
            move_path=Path("path/update2.py"),
            chunks=[
                UpdateFileChunk(
                    change_context="def f():",
                    old_lines=["    pass"],
                    new_lines=["    return 123"],
                    is_end_of_file=False,
                )
            ],
        ),
    ]


def test_parse_update_file_hunk_empty() -> None:
    patch = "*** Begin Patch\n*** Update File: test.py\n*** End Patch"
    with pytest.raises(InvalidHunkError, match=r"Update file hunk.*empty"):
        parse_patch_text(patch, ParseMode.STRICT)


def test_parse_update_file_hunk_without_context_header() -> None:
    patch = (
        "*** Begin Patch\n"
        "*** Update File: file2.py\n"
        " import foo\n"
        "+bar\n"
        "*** End Patch"
    )
    parsed = parse_patch_text(patch, ParseMode.STRICT)
    assert parsed.hunks == [
        UpdateFileHunk(
            kind="update",
            path=Path("file2.py"),
            move_path=None,
            chunks=[
                UpdateFileChunk(
                    change_context=None,
                    old_lines=["import foo"],
                    new_lines=["import foo", "bar"],
                    is_end_of_file=False,
                )
            ],
        )
    ]


def test_parse_patch_lenient_heredoc() -> None:
    patch_text = (
        "*** Begin Patch\n"
        "*** Update File: file2.py\n"
        " import foo\n"
        "+bar\n"
        "*** End Patch"
    )
    expected = ApplyPatchArgs(
        patch=patch_text,
        hunks=[
            UpdateFileHunk(
                kind="update",
                path=Path("file2.py"),
                move_path=None,
                chunks=[
                    UpdateFileChunk(
                        change_context=None,
                        old_lines=["import foo"],
                        new_lines=["import foo", "bar"],
                        is_end_of_file=False,
                    )
                ],
            )
        ],
        workdir=None,
    )

    heredoc = f"<<EOF\n{patch_text}\nEOF\n"
    with pytest.raises(InvalidPatchError):
        parse_patch_text(heredoc, ParseMode.STRICT)

    assert parse_patch_text(heredoc, ParseMode.LENIENT) == expected
