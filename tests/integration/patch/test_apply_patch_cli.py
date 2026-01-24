from __future__ import annotations

import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest


@pytest.mark.integration
def test_apply_patch_cli_add_and_update() -> None:
    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        file_name = "cli_test.txt"
        absolute_path = tmp_path / file_name

        add_patch = (
            "*** Begin Patch\n"
            f"*** Add File: {file_name}\n"
            "+hello\n"
            "*** End Patch"
        )
        add_result = _run_cli(add_patch, tmp_path)
        assert add_result.returncode == 0
        assert add_result.stdout == (
            f"Success. Updated the following files:\nA {file_name}\n"
        )
        assert absolute_path.read_text(encoding="utf-8", newline="") == "hello\n"

        update_patch = (
            "*** Begin Patch\n"
            f"*** Update File: {file_name}\n"
            "@@\n"
            "-hello\n"
            "+world\n"
            "*** End Patch"
        )
        update_result = _run_cli(update_patch, tmp_path)
        assert update_result.returncode == 0
        assert update_result.stdout == (
            f"Success. Updated the following files:\nM {file_name}\n"
        )
        assert absolute_path.read_text(encoding="utf-8", newline="") == "world\n"


@pytest.mark.integration
def test_apply_patch_cli_stdin_add_and_update() -> None:
    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        file_name = "cli_test_stdin.txt"
        absolute_path = tmp_path / file_name

        add_patch = (
            "*** Begin Patch\n"
            f"*** Add File: {file_name}\n"
            "+hello\n"
            "*** End Patch"
        )
        add_result = _run_cli(None, tmp_path, stdin_patch=add_patch)
        assert add_result.returncode == 0
        assert add_result.stdout == (
            f"Success. Updated the following files:\nA {file_name}\n"
        )
        assert absolute_path.read_text(encoding="utf-8", newline="") == "hello\n"

        update_patch = (
            "*** Begin Patch\n"
            f"*** Update File: {file_name}\n"
            "@@\n"
            "-hello\n"
            "+world\n"
            "*** End Patch"
        )
        update_result = _run_cli(None, tmp_path, stdin_patch=update_patch)
        assert update_result.returncode == 0
        assert update_result.stdout == (
            f"Success. Updated the following files:\nM {file_name}\n"
        )
        assert absolute_path.read_text(encoding="utf-8", newline="") == "world\n"


@pytest.mark.integration
def test_apply_patch_cli_applies_multiple_operations() -> None:
    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        modify_path = tmp_path / "modify.txt"
        delete_path = tmp_path / "delete.txt"

        modify_path.write_text("line1\nline2\n", encoding="utf-8", newline="")
        delete_path.write_text("obsolete\n", encoding="utf-8", newline="")

        patch = (
            "*** Begin Patch\n"
            "*** Add File: nested/new.txt\n"
            "+created\n"
            "*** Delete File: delete.txt\n"
            "*** Update File: modify.txt\n"
            "@@\n"
            "-line2\n"
            "+changed\n"
            "*** End Patch"
        )

        result = _run_cli(patch, tmp_path)
        assert result.returncode == 0
        assert result.stdout == (
            "Success. Updated the following files:\n"
            "A nested/new.txt\n"
            "M modify.txt\n"
            "D delete.txt\n"
        )

        assert (tmp_path / "nested/new.txt").read_text(encoding="utf-8", newline="") == (
            "created\n"
        )
        assert modify_path.read_text(encoding="utf-8", newline="") == "line1\nchanged\n"
        assert not delete_path.exists()


@pytest.mark.integration
def test_apply_patch_cli_applies_multiple_chunks() -> None:
    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        target_path = tmp_path / "multi.txt"
        target_path.write_text(
            "line1\nline2\nline3\nline4\n", encoding="utf-8", newline=""
        )

        patch = (
            "*** Begin Patch\n"
            "*** Update File: multi.txt\n"
            "@@\n"
            "-line2\n"
            "+changed2\n"
            "@@\n"
            "-line4\n"
            "+changed4\n"
            "*** End Patch"
        )

        result = _run_cli(patch, tmp_path)
        assert result.returncode == 0
        assert result.stdout == "Success. Updated the following files:\nM multi.txt\n"
        assert (
            target_path.read_text(encoding="utf-8", newline="")
            == "line1\nchanged2\nline3\nchanged4\n"
        )


@pytest.mark.integration
def test_apply_patch_cli_moves_file_to_new_directory() -> None:
    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        original_path = tmp_path / "old/name.txt"
        new_path = tmp_path / "renamed/dir/name.txt"
        original_path.parent.mkdir(parents=True, exist_ok=True)
        original_path.write_text("old content\n", encoding="utf-8", newline="")

        patch = (
            "*** Begin Patch\n"
            "*** Update File: old/name.txt\n"
            "*** Move to: renamed/dir/name.txt\n"
            "@@\n"
            "-old content\n"
            "+new content\n"
            "*** End Patch"
        )

        result = _run_cli(patch, tmp_path)
        assert result.returncode == 0
        assert result.stdout == (
            "Success. Updated the following files:\nM renamed/dir/name.txt\n"
        )
        assert not original_path.exists()
        assert new_path.read_text(encoding="utf-8", newline="") == "new content\n"


@pytest.mark.integration
def test_apply_patch_cli_rejects_empty_patch() -> None:
    with TemporaryDirectory() as tmp:
        result = _run_cli("*** Begin Patch\n*** End Patch", Path(tmp))
        assert result.returncode == 1
        assert result.stderr == "No files were modified.\n"


@pytest.mark.integration
def test_apply_patch_cli_reports_missing_context() -> None:
    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        target_path = tmp_path / "modify.txt"
        target_path.write_text("line1\nline2\n", encoding="utf-8", newline="")

        patch = (
            "*** Begin Patch\n"
            "*** Update File: modify.txt\n"
            "@@\n"
            "-missing\n"
            "+changed\n"
            "*** End Patch"
        )
        result = _run_cli(patch, tmp_path)
        assert result.returncode == 1
        assert result.stderr == "Failed to find expected lines in modify.txt:\nmissing\n"
        assert target_path.read_text(encoding="utf-8", newline="") == "line1\nline2\n"


@pytest.mark.integration
def test_apply_patch_cli_rejects_missing_file_delete() -> None:
    with TemporaryDirectory() as tmp:
        result = _run_cli(
            "*** Begin Patch\n*** Delete File: missing.txt\n*** End Patch",
            Path(tmp),
        )
        assert result.returncode == 1
        assert result.stderr == "Failed to delete file missing.txt\n"


@pytest.mark.integration
def test_apply_patch_cli_rejects_empty_update_hunk() -> None:
    with TemporaryDirectory() as tmp:
        result = _run_cli(
            "*** Begin Patch\n*** Update File: foo.txt\n*** End Patch",
            Path(tmp),
        )
        assert result.returncode == 1
        assert (
            result.stderr
            == "Invalid patch hunk on line 2: Update file hunk for path 'foo.txt' is empty\n"
        )


@pytest.mark.integration
def test_apply_patch_cli_requires_existing_file_for_update() -> None:
    with TemporaryDirectory() as tmp:
        result = _run_cli(
            "*** Begin Patch\n*** Update File: missing.txt\n@@\n-old\n+new\n*** End Patch",
            Path(tmp),
        )
        assert result.returncode == 1
        assert (
            result.stderr
            == "Failed to read file to update missing.txt: No such file or directory (os error 2)\n"
        )


@pytest.mark.integration
def test_apply_patch_cli_move_overwrites_existing_destination() -> None:
    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        original_path = tmp_path / "old/name.txt"
        destination = tmp_path / "renamed/dir/name.txt"
        original_path.parent.mkdir(parents=True, exist_ok=True)
        destination.parent.mkdir(parents=True, exist_ok=True)
        original_path.write_text("from\n", encoding="utf-8", newline="")
        destination.write_text("existing\n", encoding="utf-8", newline="")

        patch = (
            "*** Begin Patch\n"
            "*** Update File: old/name.txt\n"
            "*** Move to: renamed/dir/name.txt\n"
            "@@\n"
            "-from\n"
            "+new\n"
            "*** End Patch"
        )

        result = _run_cli(patch, tmp_path)
        assert result.returncode == 0
        assert result.stdout == (
            "Success. Updated the following files:\nM renamed/dir/name.txt\n"
        )
        assert not original_path.exists()
        assert destination.read_text(encoding="utf-8", newline="") == "new\n"


@pytest.mark.integration
def test_apply_patch_cli_add_overwrites_existing_file() -> None:
    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        path = tmp_path / "duplicate.txt"
        path.write_text("old content\n", encoding="utf-8", newline="")

        patch = (
            "*** Begin Patch\n"
            "*** Add File: duplicate.txt\n"
            "+new content\n"
            "*** End Patch"
        )
        result = _run_cli(patch, tmp_path)
        assert result.returncode == 0
        assert result.stdout == "Success. Updated the following files:\nA duplicate.txt\n"
        assert path.read_text(encoding="utf-8", newline="") == "new content\n"


@pytest.mark.integration
def test_apply_patch_cli_delete_directory_fails() -> None:
    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "dir").mkdir()

        result = _run_cli("*** Begin Patch\n*** Delete File: dir\n*** End Patch", tmp_path)
        assert result.returncode == 1
        assert result.stderr == "Failed to delete file dir\n"


@pytest.mark.integration
def test_apply_patch_cli_rejects_invalid_hunk_header() -> None:
    with TemporaryDirectory() as tmp:
        result = _run_cli(
            "*** Begin Patch\n*** Frobnicate File: foo\n*** End Patch",
            Path(tmp),
        )
        assert result.returncode == 1
        assert result.stderr == (
            "Invalid patch hunk on line 2: '*** Frobnicate File: foo' is not a valid hunk header. "
            "Valid hunk headers: '*** Add File: {path}', '*** Delete File: {path}', "
            "'*** Update File: {path}'\n"
        )


@pytest.mark.integration
def test_apply_patch_cli_updates_file_appends_trailing_newline() -> None:
    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        target_path = tmp_path / "no_newline.txt"
        target_path.write_text("no newline at end", encoding="utf-8", newline="")

        patch = (
            "*** Begin Patch\n"
            "*** Update File: no_newline.txt\n"
            "@@\n"
            "-no newline at end\n"
            "+first line\n"
            "+second line\n"
            "*** End Patch"
        )

        result = _run_cli(patch, tmp_path)
        assert result.returncode == 0
        assert result.stdout == (
            "Success. Updated the following files:\nM no_newline.txt\n"
        )
        contents = target_path.read_text(encoding="utf-8", newline="")
        assert contents.endswith("\n")
        assert contents == "first line\nsecond line\n"


@pytest.mark.integration
def test_apply_patch_cli_failure_after_partial_success_leaves_changes() -> None:
    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        new_file = tmp_path / "created.txt"

        patch = (
            "*** Begin Patch\n"
            "*** Add File: created.txt\n"
            "+hello\n"
            "*** Update File: missing.txt\n"
            "@@\n"
            "-old\n"
            "+new\n"
            "*** End Patch"
        )

        result = _run_cli(patch, tmp_path)
        assert result.returncode == 1
        assert result.stdout == ""
        assert (
            result.stderr
            == "Failed to read file to update missing.txt: No such file or directory (os error 2)\n"
        )
        assert new_file.read_text(encoding="utf-8", newline="") == "hello\n"


def _run_cli(
    patch: str | None,
    cwd: Path,
    *,
    stdin_patch: str | None = None,
) -> subprocess.CompletedProcess[str]:
    args = ["uv", "run", "python", "-m", "fast_agent.patch.cli"]
    if patch is not None:
        args.append(patch)
    return subprocess.run(
        args,
        input=stdin_patch,
        text=True,
        capture_output=True,
        cwd=cwd,
    )
