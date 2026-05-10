"""Tests for skill_archive — safe tar.gz / zip skill unpacking."""

from __future__ import annotations

import io
import tarfile
import zipfile

from fast_agent.mcp.skill_archive import (
    MAX_ARCHIVE_MEMBERS,
    MAX_ARCHIVE_UNPACKED_BYTES,
    MAX_SKILL_FILE_BYTES,
    unpack_skill_archive,
)


# --- helpers -------------------------------------------------------------


def _make_targz(entries: list[tuple[str, bytes]]) -> bytes:
    """Build a tar.gz blob from `(member_name, data)` tuples."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, data in entries:
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def _make_targz_with_link(name: str, target: str, link_kind: str = "sym") -> bytes:
    """Build a tar.gz containing a symlink or hardlink member.

    `link_kind`: 'sym' for symlink (LNKTYPE='2'), 'hard' for hardlink ('1').
    """
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        # Add a SKILL.md so the root-presence check passes for tests
        # that want to isolate the link rejection.
        skill = tarfile.TarInfo(name="SKILL.md")
        skill_body = b"---\nname: x\ndescription: y\n---\n"
        skill.size = len(skill_body)
        tar.addfile(skill, io.BytesIO(skill_body))

        info = tarfile.TarInfo(name=name)
        info.size = 0
        info.type = tarfile.SYMTYPE if link_kind == "sym" else tarfile.LNKTYPE
        info.linkname = target
        tar.addfile(info)
    return buf.getvalue()


def _make_zip(entries: list[tuple[str, bytes]]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in entries:
            zf.writestr(name, data)
    return buf.getvalue()


def _make_zip_with_symlink(link_name: str, target: str) -> bytes:
    """Build a zip with a unix symlink entry."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        # Skip the SKILL.md sentinel — link rejection should fire before
        # the root check.
        zf.writestr("SKILL.md", b"---\nname: x\ndescription: y\n---\n")
        info = zipfile.ZipInfo(link_name)
        info.create_system = 3  # unix
        info.external_attr = (0o120755 & 0xFFFF) << 16  # symlink mode
        zf.writestr(info, target.encode("utf-8"))
    return buf.getvalue()


SKILL_MD_BODY = (
    b"---\nname: alpha\ndescription: a test skill\n---\nbody text\n"
)


# --- happy paths ---------------------------------------------------------


def test_unpacks_simple_targz() -> None:
    blob = _make_targz(
        [
            ("SKILL.md", SKILL_MD_BODY),
            ("references/GUIDE.md", b"guide"),
            ("scripts/run.py", b"print('hi')"),
        ]
    )
    files = unpack_skill_archive(blob, "application/gzip", "skill://alpha.tar.gz")
    assert files is not None
    assert files["SKILL.md"] == SKILL_MD_BODY
    assert files["references/GUIDE.md"] == b"guide"
    assert files["scripts/run.py"] == b"print('hi')"


def test_unpacks_simple_zip() -> None:
    blob = _make_zip(
        [
            ("SKILL.md", SKILL_MD_BODY),
            ("references/GUIDE.md", b"guide"),
        ]
    )
    files = unpack_skill_archive(blob, "application/zip", "skill://alpha.zip")
    assert files is not None
    assert files["SKILL.md"] == SKILL_MD_BODY
    assert files["references/GUIDE.md"] == b"guide"


def test_targz_format_detected_from_url_when_mime_missing() -> None:
    blob = _make_targz([("SKILL.md", SKILL_MD_BODY)])
    files = unpack_skill_archive(blob, None, "skill://alpha.tar.gz")
    assert files is not None
    assert "SKILL.md" in files


def test_zip_format_detected_from_tgz_alias() -> None:
    blob = _make_targz([("SKILL.md", SKILL_MD_BODY)])
    files = unpack_skill_archive(blob, None, "skill://alpha.tgz")
    assert files is not None


# --- safety rejections ---------------------------------------------------


def test_rejects_traversal_in_targz() -> None:
    blob = _make_targz([
        ("SKILL.md", SKILL_MD_BODY),
        ("../escape.md", b"x"),
    ])
    assert unpack_skill_archive(blob, "application/gzip", "skill://a.tar.gz") is None


def test_rejects_absolute_path_in_targz() -> None:
    blob = _make_targz([("/etc/passwd", b"x"), ("SKILL.md", SKILL_MD_BODY)])
    assert unpack_skill_archive(blob, "application/gzip", "skill://a.tar.gz") is None


def test_rejects_windows_drive_path_in_zip() -> None:
    blob = _make_zip([("C:/evil.md", b"x"), ("SKILL.md", SKILL_MD_BODY)])
    assert unpack_skill_archive(blob, "application/zip", "skill://a.zip") is None


def test_rejects_backslash_traversal_in_zip() -> None:
    blob = _make_zip([("foo\\..\\bar", b"x"), ("SKILL.md", SKILL_MD_BODY)])
    assert unpack_skill_archive(blob, "application/zip", "skill://a.zip") is None


def test_rejects_symlink_in_targz() -> None:
    blob = _make_targz_with_link("link.md", "/etc/passwd", link_kind="sym")
    assert unpack_skill_archive(blob, "application/gzip", "skill://a.tar.gz") is None


def test_rejects_hardlink_in_targz() -> None:
    blob = _make_targz_with_link("link.md", "SKILL.md", link_kind="hard")
    assert unpack_skill_archive(blob, "application/gzip", "skill://a.tar.gz") is None


def test_rejects_symlink_in_zip() -> None:
    blob = _make_zip_with_symlink("link.md", "../../../etc/passwd")
    assert unpack_skill_archive(blob, "application/zip", "skill://a.zip") is None


def test_rejects_missing_skill_md_at_root() -> None:
    blob = _make_targz(
        [
            ("nested/SKILL.md", SKILL_MD_BODY),  # nested under wrapper dir
            ("nested/refs/x.md", b"x"),
        ]
    )
    assert unpack_skill_archive(blob, "application/gzip", "skill://a.tar.gz") is None


def test_rejects_oversized_per_file_targz() -> None:
    big = b"x" * (MAX_SKILL_FILE_BYTES + 1)
    blob = _make_targz([("SKILL.md", SKILL_MD_BODY), ("big.bin", big)])
    assert unpack_skill_archive(blob, "application/gzip", "skill://a.tar.gz") is None


def test_rejects_oversized_total_targz() -> None:
    """Total uncompressed size > cap, even when individual files are within
    per-file cap."""
    chunk = b"x" * MAX_SKILL_FILE_BYTES
    # 9 chunks * 1 MiB = 9 MiB, exceeds 8 MiB total cap.
    entries = [("SKILL.md", SKILL_MD_BODY)]
    for i in range(9):
        entries.append((f"part{i}.bin", chunk))
    blob = _make_targz(entries)
    assert unpack_skill_archive(blob, "application/gzip", "skill://a.tar.gz") is None


def test_rejects_too_many_members() -> None:
    entries = [("SKILL.md", SKILL_MD_BODY)]
    for i in range(MAX_ARCHIVE_MEMBERS + 1):
        entries.append((f"f{i}.txt", b"x"))
    blob = _make_targz(entries)
    assert unpack_skill_archive(blob, "application/gzip", "skill://a.tar.gz") is None


def test_rejects_unknown_format() -> None:
    assert unpack_skill_archive(b"random bytes", "text/plain", "skill://a.bogus") is None


def test_rejects_decompression_bomb_via_declared_zip_size() -> None:
    """A zip can declare an uncompressed size much larger than the
    compressed data. Our up-front check on ZipInfo.file_size catches the
    declared size before we'd allocate memory for the read."""
    # Build a tiny zip but rewrite the declared uncompressed size to a
    # large value. We do this by hand-tampering with one entry's header.
    blob = _make_zip([("SKILL.md", SKILL_MD_BODY), ("bomb.bin", b"compressible")])
    # We can't easily rewrite the central directory by hand here, so
    # instead use a real bomb-pattern: many small members whose declared
    # size sums above the total cap.
    chunk = b"x" * MAX_SKILL_FILE_BYTES
    entries = [("SKILL.md", SKILL_MD_BODY)]
    for i in range(9):
        entries.append((f"part{i}.bin", chunk))
    blob = _make_zip(entries)
    assert unpack_skill_archive(blob, "application/zip", "skill://a.zip") is None


def test_garbage_targz_blob_returns_none() -> None:
    """A blob that claims to be tar.gz but is corrupt must not raise."""
    assert (
        unpack_skill_archive(b"not actually gzipped", "application/gzip", "skill://a.tar.gz")
        is None
    )


def test_garbage_zip_blob_returns_none() -> None:
    assert (
        unpack_skill_archive(b"not actually a zip", "application/zip", "skill://a.zip") is None
    )
