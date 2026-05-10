"""Safe unpacker for `type: "archive"` skill index entries.

Per SEP-2640, hosts MUST support `.tar.gz` and `.zip` archive distribution
and MUST apply the Agent Skills archive-safety requirements:

- Reject path traversal sequences (`..`) and absolute paths.
- Reject symlinks / hardlinks resolving outside the skill directory.
  (We reject all link types for now — the SEP allows safe-relative
  links but a robust resolver is harder to get right than to defer.)
- Bound total uncompressed size to prevent decompression bombs.
- Require `SKILL.md` at the archive root, not inside a wrapper directory.

Returns the archive's files as a flat dict keyed by relative POSIX path.
The caller (mcp_skills_loader) translates these into the skill's virtual
`skill://<skill-path>/<file-path>` namespace and seeds the SkillReader's
in-memory cache. Nothing touches disk.
"""

from __future__ import annotations

import io
import tarfile
import zipfile
from typing import Iterable

from fast_agent.core.logging.logger import get_logger

logger = get_logger(__name__)

# Sized for typical skills: a SKILL.md is short, supporting files
# (references, scripts, small assets) cap at a few MB total. A skill
# materially larger than this is a smell, not a feature.
MAX_ARCHIVE_UNPACKED_BYTES = 8 * 1024 * 1024  # 8 MiB total
MAX_SKILL_FILE_BYTES = 1 * 1024 * 1024  # 1 MiB per file
MAX_ARCHIVE_MEMBERS = 1024


def unpack_skill_archive(
    blob: bytes,
    mime_type: str | None,
    url: str,
) -> dict[str, bytes] | None:
    """Decompress a skill archive blob into a `<file-path> -> bytes` map.

    Returns None on any safety failure or unsupported format. Never
    returns a partially-unpacked map: a single bad member rejects the
    whole archive.
    """

    fmt = _detect_format(mime_type, url)
    if fmt is None:
        logger.warning(
            "Skill archive has unrecognized format",
            data={"url": url, "mime_type": mime_type},
        )
        return None

    try:
        if fmt == "tar.gz":
            files = _unpack_targz(blob, url)
        else:
            files = _unpack_zip(blob, url)
    except Exception as exc:
        logger.warning(
            "Skill archive failed to decompress",
            data={"url": url, "format": fmt, "error": str(exc)},
        )
        return None

    if files is None:
        return None

    if "SKILL.md" not in files:
        logger.warning(
            "Skill archive does not contain SKILL.md at root",
            data={"url": url, "members": sorted(files.keys())[:20]},
        )
        return None

    return files


def _detect_format(mime_type: str | None, url: str) -> str | None:
    """Pick `tar.gz` or `zip` from mime first, URL suffix as fallback."""
    if mime_type:
        m = mime_type.lower().split(";")[0].strip()
        if m == "application/gzip" or m == "application/x-gzip" or m == "application/x-tar+gzip":
            return "tar.gz"
        if m == "application/zip" or m == "application/x-zip-compressed":
            return "zip"
    lowered = url.lower().rstrip("/")
    if lowered.endswith(".tar.gz") or lowered.endswith(".tgz"):
        return "tar.gz"
    if lowered.endswith(".zip"):
        return "zip"
    return None


def _unpack_targz(blob: bytes, url: str) -> dict[str, bytes] | None:
    files: dict[str, bytes] = {}
    total = 0
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
        members = tar.getmembers()
        if not _check_member_count(members, url):
            return None
        for member in members:
            name = member.name
            if member.issym() or member.islnk():
                logger.warning(
                    "Skill archive contains link member; rejecting",
                    data={"url": url, "member": name, "kind": "symlink/hardlink"},
                )
                return None
            if member.isdir():
                continue
            if not member.isfile():
                logger.warning(
                    "Skill archive contains non-regular member; rejecting",
                    data={"url": url, "member": name, "type": _tar_typeflag(member)},
                )
                return None
            if not _is_safe_relative_path(name):
                logger.warning(
                    "Skill archive contains unsafe path; rejecting",
                    data={"url": url, "member": name},
                )
                return None
            if member.size > MAX_SKILL_FILE_BYTES:
                logger.warning(
                    "Skill archive member exceeds per-file size limit",
                    data={
                        "url": url,
                        "member": name,
                        "bytes": member.size,
                        "limit": MAX_SKILL_FILE_BYTES,
                    },
                )
                return None
            total += member.size
            if total > MAX_ARCHIVE_UNPACKED_BYTES:
                logger.warning(
                    "Skill archive total uncompressed size exceeds limit",
                    data={
                        "url": url,
                        "total": total,
                        "limit": MAX_ARCHIVE_UNPACKED_BYTES,
                    },
                )
                return None
            extracted = tar.extractfile(member)
            if extracted is None:
                logger.warning(
                    "Skill archive member could not be opened",
                    data={"url": url, "member": name},
                )
                return None
            data = extracted.read(MAX_SKILL_FILE_BYTES + 1)
            if len(data) > MAX_SKILL_FILE_BYTES:
                logger.warning(
                    "Skill archive member exceeds per-file size after read",
                    data={"url": url, "member": name, "limit": MAX_SKILL_FILE_BYTES},
                )
                return None
            files[_normalize(name)] = data
    return files


def _unpack_zip(blob: bytes, url: str) -> dict[str, bytes] | None:
    files: dict[str, bytes] = {}
    total = 0
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        infos = zf.infolist()
        if not _check_member_count(infos, url):
            return None
        for info in infos:
            name = info.filename
            if name.endswith("/"):
                continue  # directory entry
            if _zip_is_link(info):
                logger.warning(
                    "Skill archive (zip) contains link member; rejecting",
                    data={"url": url, "member": name},
                )
                return None
            if not _is_safe_relative_path(name):
                logger.warning(
                    "Skill archive (zip) contains unsafe path; rejecting",
                    data={"url": url, "member": name},
                )
                return None
            # ZipInfo.file_size is the *declared* uncompressed size — trust
            # it for the up-front cap (decompression-bomb defense), then
            # verify against the actual read length below.
            if info.file_size > MAX_SKILL_FILE_BYTES:
                logger.warning(
                    "Skill archive (zip) member declared size exceeds per-file limit",
                    data={
                        "url": url,
                        "member": name,
                        "declared": info.file_size,
                        "limit": MAX_SKILL_FILE_BYTES,
                    },
                )
                return None
            total += info.file_size
            if total > MAX_ARCHIVE_UNPACKED_BYTES:
                logger.warning(
                    "Skill archive (zip) declared total exceeds limit",
                    data={
                        "url": url,
                        "total": total,
                        "limit": MAX_ARCHIVE_UNPACKED_BYTES,
                    },
                )
                return None
            with zf.open(info, "r") as fh:
                data = fh.read(MAX_SKILL_FILE_BYTES + 1)
            if len(data) > MAX_SKILL_FILE_BYTES:
                logger.warning(
                    "Skill archive (zip) member actual size exceeds limit",
                    data={"url": url, "member": name, "limit": MAX_SKILL_FILE_BYTES},
                )
                return None
            files[_normalize(name)] = data
    return files


def _check_member_count(members: Iterable, url: str) -> bool:
    count = sum(1 for _ in members)
    if count > MAX_ARCHIVE_MEMBERS:
        logger.warning(
            "Skill archive exceeds member count limit",
            data={"url": url, "members": count, "limit": MAX_ARCHIVE_MEMBERS},
        )
        return False
    return True


def _is_safe_relative_path(name: str) -> bool:
    """Reject paths that escape the archive root.

    The Agent Skills archive-safety requirement: no `..` segments, no
    absolute paths, no NUL. We normalize separators and walk segments
    instead of trusting `os.path` so that a tar member named
    `foo\\..\\bar` on a Linux host doesn't get a free pass.
    """
    if not name or "\x00" in name:
        return False
    # Reject absolute paths (Unix `/`, Windows `\\` or `C:\\`, UNC, etc.)
    if name.startswith("/") or name.startswith("\\"):
        return False
    if len(name) >= 2 and name[1] == ":":  # drive letter
        return False
    # Normalize backslashes to forward slashes for segment inspection.
    normalized = name.replace("\\", "/")
    for segment in normalized.split("/"):
        if segment in ("", ".", ".."):
            # Empty segment is from a leading slash or doubled slash;
            # `.` and `..` are traversal markers.
            if segment == "":
                # Allow nothing — if we got here, the leading-/ check
                # above was bypassed somehow; treat empty segments as
                # unsafe to avoid `foo//bar` ambiguity.
                return False
            return False
    return True


def _normalize(name: str) -> str:
    """Canonicalize archive member names to forward-slash form."""
    return name.replace("\\", "/")


def _tar_typeflag(member: tarfile.TarInfo) -> str:
    return member.type.decode() if isinstance(member.type, bytes) else str(member.type)


def _zip_is_link(info: zipfile.ZipInfo) -> bool:
    """Detect symlinks in a zip via the high bits of `external_attr`.

    Zip files store unix mode in the upper 16 bits of `external_attr`
    when `create_system == 3` (unix). Symlinks set `S_IFLNK` (0o120000).
    """
    if info.create_system != 3:
        return False
    mode = (info.external_attr >> 16) & 0xFFFF
    # 0o120000 == S_IFLNK
    return (mode & 0o170000) == 0o120000
