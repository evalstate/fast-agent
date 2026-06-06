from __future__ import annotations

from pathlib import Path
from typing import TypeGuard

from fast_agent.tools.edit_file_engine import (
    EditFileError,
    EditFileResult,
    EditFileSuccess,
    edit_file,
)


def _is_success(result: EditFileResult) -> TypeGuard[EditFileSuccess]:
    return result["success"] is True


def _is_error(result: EditFileResult) -> TypeGuard[EditFileError]:
    return result["success"] is False


def test_edit_file_reports_missing_file_before_reading(tmp_path: Path) -> None:
    result = edit_file(
        tmp_path / "missing.txt",
        display_path="missing.txt",
        old_string="before",
        new_string="after",
    )

    assert _is_error(result)
    assert result["error"] == "file_not_found"


def test_edit_file_reports_multiple_matches_with_locations(tmp_path: Path) -> None:
    target_file = tmp_path / "repeat.txt"
    target_file.write_text("one\none\n", encoding="utf-8", newline="")

    result = edit_file(
        target_file,
        display_path="repeat.txt",
        old_string="one",
        new_string="two",
    )

    assert _is_error(result)
    assert result["error"] == "multiple_matches"
    assert result["matches"] == [
        {"line_start": 1, "line_end": 1},
        {"line_start": 2, "line_end": 2},
    ]
    assert target_file.read_text(encoding="utf-8", newline="") == "one\none\n"


def test_edit_file_reports_overlapping_matches_as_ambiguous(tmp_path: Path) -> None:
    target_file = tmp_path / "overlap.txt"
    target_file.write_text("aaa", encoding="utf-8", newline="")

    result = edit_file(
        target_file,
        display_path="overlap.txt",
        old_string="aa",
        new_string="b",
    )

    assert _is_error(result)
    assert result["error"] == "multiple_matches"
    assert result["matches"] == [
        {"line_start": 1, "line_end": 1},
        {"line_start": 1, "line_end": 1},
    ]
    assert target_file.read_text(encoding="utf-8", newline="") == "aaa"


def test_edit_file_replace_all_uses_non_overlapping_single_pass(tmp_path: Path) -> None:
    target_file = tmp_path / "repeat.txt"
    target_file.write_text("aaaa", encoding="utf-8", newline="")

    result = edit_file(
        Path(target_file),
        display_path="repeat.txt",
        old_string="aa",
        new_string="b",
        replace_all=True,
    )

    assert _is_success(result)
    success = result
    assert target_file.read_text(encoding="utf-8", newline="") == "bb"
    assert success["replacements"] == 2
    assert success["line_start"] == 1
    assert success["line_end"] == 1


def test_edit_file_replace_all_does_not_loop_when_new_contains_old(tmp_path: Path) -> None:
    target_file = tmp_path / "expand.txt"
    target_file.write_text("aa", encoding="utf-8", newline="")

    result = edit_file(
        Path(target_file),
        display_path="expand.txt",
        old_string="a",
        new_string="aa",
        replace_all=True,
    )

    assert _is_success(result)
    success = result
    assert target_file.read_text(encoding="utf-8", newline="") == "aaaa"
    assert success["replacements"] == 2
    assert success["line_start"] == 1
    assert success["line_end"] == 1


def test_edit_file_reports_no_op_before_writing(tmp_path: Path) -> None:
    target_file = tmp_path / "same.txt"
    target_file.write_text("keep\n", encoding="utf-8", newline="")

    result = edit_file(
        target_file,
        display_path="same.txt",
        old_string="keep",
        new_string="keep",
    )

    assert _is_error(result)
    assert result["error"] == "no_op"
    assert target_file.read_text(encoding="utf-8", newline="") == "keep\n"
