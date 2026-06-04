"""Shared helpers for Skills-over-MCP resource URIs.

Centralizes the `/SKILL.md` suffix handling used by the discovery loader,
the skill registry (prompt formatting), and the read-tool dispatcher.
Kept scheme-agnostic per the SEP — `skill://` is the default but any
scheme listed in `skill://index.json` is valid.
"""

from __future__ import annotations

SKILL_MD_SUFFIX = "/SKILL.md"


def strip_skill_md(uri: str) -> str:
    """Return the skill root URI — the SKILL.md URI minus `/SKILL.md`.

    Used both to derive the `<directory>` URI for prompt formatting and
    the root prefix for the read-tool trust boundary. Returns the URI
    unchanged if the suffix is absent. Tolerates a buggy trailing slash
    (`.../SKILL.md/`) so a misbehaving server can't seed an oversized
    root into the reader's allow-list.
    """
    trimmed = uri.rstrip("/")
    if trimmed.endswith(SKILL_MD_SUFFIX):
        return trimmed[: -len(SKILL_MD_SUFFIX)]
    return uri


def skill_name_from_uri(uri: str) -> str | None:
    """Return the final `<skill-path>` segment from a `<scheme>://.../SKILL.md` URI.

    `skill://git-workflow/SKILL.md` yields `git-workflow`;
    `github://owner/repo/skills/refunds/SKILL.md` yields `refunds`.
    Returns None if the URI doesn't end in `/SKILL.md` or has no
    skill-path segment before the suffix (e.g. `skill://SKILL.md`).
    """
    trimmed = uri.rstrip("/")
    if not trimmed.endswith(SKILL_MD_SUFFIX):
        return None
    stem = trimmed[: -len(SKILL_MD_SUFFIX)]
    scheme_sep = stem.find("://")
    if scheme_sep != -1:
        stem = stem[scheme_sep + 3 :]
    if "/" in stem:
        tail = stem.rsplit("/", 1)[-1]
        return tail or None
    return stem or None
