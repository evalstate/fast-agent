from __future__ import annotations

from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

from fast_agent.patch.engine import apply_patch


def _run_patch(patch: str) -> tuple[str, str]:
    stdout = StringIO()
    stderr = StringIO()
    apply_patch(patch, stdout, stderr)
    return stdout.getvalue(), stderr.getvalue()


def test_update_file_hunk_modifies_content() -> None:
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "update.txt"
        path.write_text("foo\nbar\n", encoding="utf-8", newline="")

        patch = (
            "*** Begin Patch\n"
            f"*** Update File: {path}\n"
            "@@\n"
            " foo\n"
            "-bar\n"
            "+baz\n"
            "*** End Patch"
        )

        stdout, stderr = _run_patch(patch)

        assert stdout == f"Success. Updated the following files:\nM {path}\n"
        assert stderr == ""
        assert path.read_text(encoding="utf-8", newline="") == "foo\nbaz\n"


def test_update_file_hunk_can_move_file() -> None:
    with TemporaryDirectory() as tmp:
        src = Path(tmp) / "src.txt"
        dest = Path(tmp) / "dst.txt"
        src.write_text("line\n", encoding="utf-8", newline="")

        patch = (
            "*** Begin Patch\n"
            f"*** Update File: {src}\n"
            f"*** Move to: {dest}\n"
            "@@\n"
            "-line\n"
            "+line2\n"
            "*** End Patch"
        )

        stdout, stderr = _run_patch(patch)

        assert stdout == f"Success. Updated the following files:\nM {dest}\n"
        assert stderr == ""
        assert not src.exists()
        assert dest.read_text(encoding="utf-8", newline="") == "line2\n"


def test_update_file_end_of_file_anchor() -> None:
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "interleaved.txt"
        path.write_text("a\nb\nc\n", encoding="utf-8", newline="")

        patch = (
            "*** Begin Patch\n"
            f"*** Update File: {path}\n"
            "@@\n"
            " c\n"
            "+d\n"
            "*** End of File\n"
            "*** End Patch"
        )

        stdout, stderr = _run_patch(patch)

        assert stdout == f"Success. Updated the following files:\nM {path}\n"
        assert stderr == ""
        assert path.read_text(encoding="utf-8", newline="") == "a\nb\nc\nd\n"


def test_update_file_change_context_disambiguates() -> None:
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "context.txt"
        path.write_text("header\nfoo\nold\nfoo\nold\n", encoding="utf-8", newline="")

        patch = (
            "*** Begin Patch\n"
            f"*** Update File: {path}\n"
            "@@ foo\n"
            "-old\n"
            "+new\n"
            "*** End Patch"
        )

        stdout, stderr = _run_patch(patch)

        assert stdout == f"Success. Updated the following files:\nM {path}\n"
        assert stderr == ""
        assert path.read_text(encoding="utf-8", newline="") == "header\nfoo\nnew\nfoo\nold\n"


def test_update_line_with_unicode_dash() -> None:
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "unicode.py"
        original = "import asyncio  # local import \u2013 avoids top\u2011level dep\n"
        path.write_text(original, encoding="utf-8", newline="")

        patch = (
            "*** Begin Patch\n"
            f"*** Update File: {path}\n"
            "@@\n"
            "-import asyncio  # local import - avoids top-level dep\n"
            "+import asyncio  # HELLO\n"
            "*** End Patch"
        )

        stdout, stderr = _run_patch(patch)

        assert stdout == f"Success. Updated the following files:\nM {path}\n"
        assert stderr == ""
        assert path.read_text(encoding="utf-8", newline="") == "import asyncio  # HELLO\n"


def test_pure_addition_chunk_followed_by_removal() -> None:
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "panic.txt"
        path.write_text("line1\nline2\nline3\n", encoding="utf-8", newline="")

        patch = (
            "*** Begin Patch\n"
            f"*** Update File: {path}\n"
            "@@\n"
            "+after-context\n"
            "+second-line\n"
            "@@\n"
            " line1\n"
            "-line2\n"
            "-line3\n"
            "+line2-replacement\n"
            "*** End Patch"
        )

        stdout, stderr = _run_patch(patch)

        assert stdout == f"Success. Updated the following files:\nM {path}\n"
        assert stderr == ""
        assert (
            path.read_text(encoding="utf-8", newline="")
            == "line1\nline2-replacement\nafter-context\nsecond-line\n"
        )
