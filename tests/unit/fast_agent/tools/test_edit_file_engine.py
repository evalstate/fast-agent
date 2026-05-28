from __future__ import annotations

from pathlib import Path
from typing import TypeGuard

from fast_agent.tools.edit_file_engine import EditFileResult, EditFileSuccess, edit_file


def _is_success(result: EditFileResult) -> TypeGuard[EditFileSuccess]:
    return result["success"] is True


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
