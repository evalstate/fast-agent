"""Load skills served by connected MCP servers per the Skills-over-MCP SEP.

Implements the `io.modelcontextprotocol/skills` extension: fetches the
well-known `skill://index.json` resource from each connected server, parses
its entries, and builds `SkillManifest` objects backed by the URIs listed
in the index. Entry URIs may use any scheme (`skill://` is the SEP's
default but servers MAY use `github://`, `repo://`, etc.).

SEP: https://github.com/modelcontextprotocol/experimental-ext-skills/pull/69
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable, Sequence

from mcp.types import BlobResourceContents, TextResourceContents

from fast_agent.core.logging.logger import get_logger
from fast_agent.mcp.skill_archive import unpack_skill_archive
from fast_agent.mcp.skill_uri import skill_name_from_uri
from fast_agent.skills.registry import SkillManifest, SkillRegistry

if TYPE_CHECKING:
    from fast_agent.mcp.mcp_aggregator import MCPAggregator

logger = get_logger(__name__)

INDEX_URI = "skill://index.json"

# Soft ceilings on server-returned text to keep a hostile or misbehaving
# server from pinning megabytes of memory per discovery pass. An index
# listing thousands of skills still fits comfortably under 1MB; a single
# SKILL.md is conventionally a short instruction document.
MAX_INDEX_BYTES = 1_048_576  # 1 MiB
MAX_SKILL_MD_BYTES = 262_144  # 256 KiB

# Schema versions of the Agent Skills discovery index this host knows how to
# parse. Per SEP-2640: clients SHOULD match `$schema` against known URIs
# before processing. An unknown schema does not abort parsing — we proceed
# best-effort so a newer index that adds fields stays readable for the
# entries we still recognize — but it does emit a warning so the operator
# knows the host may be missing functionality the server expects.
KNOWN_INDEX_SCHEMAS = frozenset(
    {"https://schemas.agentskills.io/discovery/0.2.0/schema.json"}
)

# Maximum size of an archive blob we'll fetch from a server. Distinct from
# the per-archive *unpacked* limit enforced by skill_archive — this is the
# wire-bytes ceiling. A skill that needs >4 MiB compressed should be
# distributed as individual files so the host can stage progressively.
MAX_ARCHIVE_BLOB_BYTES = 4 * 1024 * 1024  # 4 MiB

ARCHIVE_SUFFIXES = (".tar.gz", ".tgz", ".zip")


@dataclass
class LoadedSkills:
    """Result of `load_mcp_skill_manifests`.

    The loader discovers two distinct kinds of artifact:

    - `manifests` — concrete skills (both `skill-md` and `archive` index
      entries reduce to one `SkillManifest` each).
    - `archive_cache` — for archive-distributed skills, a per-skill
      in-memory file map keyed by the skill's *root* URI (the URI with
      `/SKILL.md` stripped, NOT the original `.tar.gz` URL). The
      SkillReader checks this cache before issuing `resources/read`,
      so archive-backed reads stay local after the initial fetch.

    Future fields land here (template entries in Feature 3) so the
    loader's signature stops growing.
    """

    manifests: list[SkillManifest] = field(default_factory=list)
    archive_cache: dict[str, dict[str, bytes]] = field(default_factory=dict)


def merge_filesystem_and_mcp_manifests(
    filesystem_manifests: Sequence[SkillManifest],
    mcp_manifests: Sequence[SkillManifest],
) -> tuple[list[SkillManifest], list[str]]:
    """Merge MCP-discovered manifests into the filesystem set.

    Filesystem manifests win on name collision (consistent with
    `SkillRegistry` dedup semantics). Within the MCP batch, the first
    manifest with a given name wins; later ones are hidden with a
    warning. Returns the merged list and a list of human-readable
    warnings for hidden manifests.
    """
    filesystem_names = {m.name.lower() for m in filesystem_manifests}
    mcp_winner_by_name: dict[str, str | None] = {}
    merged: list[SkillManifest] = list(filesystem_manifests)
    warnings: list[str] = []
    for mcp_manifest in mcp_manifests:
        key = mcp_manifest.name.lower()
        if key in filesystem_names:
            warnings.append(
                f"MCP-served skill '{mcp_manifest.name}' from server "
                f"'{mcp_manifest.server_name}' hidden by local filesystem skill."
            )
            continue
        if key in mcp_winner_by_name:
            winner = mcp_winner_by_name[key] or "<unknown>"
            warnings.append(
                f"MCP-served skill '{mcp_manifest.name}' from server "
                f"'{mcp_manifest.server_name}' hidden by an earlier MCP-served "
                f"skill of the same name from server '{winner}'."
            )
            continue
        merged.append(mcp_manifest)
        mcp_winner_by_name[key] = mcp_manifest.server_name
    return merged, warnings


async def load_mcp_skill_manifests(
    aggregator: "MCPAggregator",
    server_names: Sequence[str],
    *,
    enabled_servers: set[str] | None = None,
) -> LoadedSkills:
    """Fetch and parse skill manifests from connected MCP servers.

    For each server, reads `skill://index.json` (optional; missing index is
    silently skipped), then walks each entry:

    - `type: "skill-md"` — fetches the `SKILL.md` and parses frontmatter.
    - `type: "archive"` — fetches the archive blob and unpacks it
      in-memory; the resulting file map seeds `LoadedSkills.archive_cache`
      and the SKILL.md inside the archive is parsed identically to a
      direct `skill-md` entry.
    - `mcp-resource-template` — logged and skipped (Feature 3 work).

    Errors from a single server or entry are logged as warnings; a failure
    never aborts the whole batch.
    """

    result = LoadedSkills()
    for server_name in server_names:
        if enabled_servers is not None and server_name not in enabled_servers:
            logger.debug(
                "MCP skill discovery disabled for server",
                data={"server": server_name},
            )
            continue

        entries = await _read_index(aggregator, server_name)
        if not entries:
            continue

        for entry in entries:
            entry_type = entry.get("type")
            if entry_type == "mcp-resource-template":
                logger.debug(
                    "Skipping MCP skill template entry (not yet supported)",
                    data={
                        "server": server_name,
                        "url": entry.get("url"),
                    },
                )
                continue
            if entry_type == "skill-md":
                manifest = await _load_concrete_entry(aggregator, server_name, entry)
                if manifest is not None:
                    result.manifests.append(manifest)
                continue
            if entry_type == "archive":
                pair = await _load_archive_entry(aggregator, server_name, entry)
                if pair is not None:
                    manifest, files = pair
                    result.manifests.append(manifest)
                    # Cache key is the skill's root URI (post-strip), so
                    # the reader's per-segment lookup matches without
                    # round-tripping back through the archive URL.
                    root_uri = manifest.uri.removesuffix("/SKILL.md")
                    result.archive_cache[root_uri] = files
                continue
            logger.debug(
                "Skipping MCP skill entry with unrecognized type",
                data={"server": server_name, "type": entry_type},
            )

    return result


async def _read_index(
    aggregator: "MCPAggregator", server_name: str
) -> list[dict] | None:
    """Read and parse `skill://index.json` from a server; returns None if absent."""
    try:
        result = await aggregator.get_resource(INDEX_URI, server_name=server_name)
    except Exception as exc:
        # The SEP treats the index as optional. Absence / lack of resources
        # support / network error all fall through to "no indexed skills".
        logger.debug(
            "No skill index from server",
            data={"server": server_name, "error": str(exc)},
        )
        return None

    text = _first_text(result.contents)
    if not text:
        logger.warning(
            "Skill index has no text content",
            data={"server": server_name},
        )
        return None

    byte_len = len(text.encode("utf-8"))
    if byte_len > MAX_INDEX_BYTES:
        logger.warning(
            "Skill index exceeds size limit; refusing to parse",
            data={
                "server": server_name,
                "bytes": byte_len,
                "limit": MAX_INDEX_BYTES,
            },
        )
        return None

    try:
        parsed = json.loads(text)
    except Exception as exc:
        logger.warning(
            "Failed to parse skill index JSON",
            data={"server": server_name, "error": str(exc)},
        )
        return None

    if isinstance(parsed, dict):
        schema = parsed.get("$schema")
        if isinstance(schema, str) and schema not in KNOWN_INDEX_SCHEMAS:
            logger.warning(
                "Skill index $schema is not in the known set; parsing best-effort",
                data={
                    "server": server_name,
                    "schema": schema,
                    "supported": sorted(KNOWN_INDEX_SCHEMAS),
                },
            )

    skills = parsed.get("skills") if isinstance(parsed, dict) else None
    if not isinstance(skills, list):
        logger.warning(
            "Skill index missing `skills` array",
            data={"server": server_name, "top_level_type": type(parsed).__name__},
        )
        return None
    return [entry for entry in skills if isinstance(entry, dict)]


async def _load_concrete_entry(
    aggregator: "MCPAggregator",
    server_name: str,
    entry: dict,
) -> SkillManifest | None:
    url = entry.get("url")
    if not isinstance(url, str) or not url:
        logger.warning(
            "Skill entry missing `url`",
            data={"server": server_name, "entry": entry},
        )
        return None

    # Reject `file://` skill URIs: the trust model for Skills-over-MCP is
    # "the MCP server is the authority for content under its published
    # roots". A `file://` root collapses that to "whatever the server's
    # process can read from disk" — something the user thought they were
    # delegating through MCP ends up reading the local filesystem without
    # the usual ACP / filesystem-runtime path guardrails. Keep the root
    # out of the reader's allow-list entirely.
    if url.lower().startswith("file://"):
        logger.warning(
            "Rejecting `file://` skill URI: not allowed for Skills-over-MCP",
            data={"server": server_name, "url": url},
        )
        return None

    try:
        result = await aggregator.get_resource(url, server_name=server_name)
    except Exception as exc:
        logger.warning(
            "Failed to read MCP skill SKILL.md",
            data={"server": server_name, "url": url, "error": str(exc)},
        )
        return None

    text = _first_text(result.contents)
    if not text:
        logger.warning(
            "MCP skill SKILL.md has no text content",
            data={"server": server_name, "url": url},
        )
        return None

    byte_len = len(text.encode("utf-8"))
    if byte_len > MAX_SKILL_MD_BYTES:
        logger.warning(
            "MCP skill SKILL.md exceeds size limit; refusing to parse",
            data={
                "server": server_name,
                "url": url,
                "bytes": byte_len,
                "limit": MAX_SKILL_MD_BYTES,
            },
        )
        return None

    manifest, parse_error = SkillRegistry.parse_manifest_text(text)
    if manifest is None:
        logger.warning(
            "Failed to parse MCP skill frontmatter",
            data={"server": server_name, "url": url, "error": parse_error},
        )
        return None

    # SEP: the final segment of <skill-path> MUST equal the frontmatter name.
    # A URI with no skill-path segment (e.g. `skill://SKILL.md`) is rejected
    # outright: stripping `/SKILL.md` would yield `skill:/`, which the reader
    # would seed into its allowed-roots set and then admit every `skill://...`
    # URI via the `startswith(root + "/")` check — the trust boundary must
    # not rely on server-published URIs being well-formed.
    url_name = skill_name_from_uri(url)
    if not url_name:
        logger.warning(
            "Skill entry URI has no path segment before `/SKILL.md`; refusing to register",
            data={"server": server_name, "url": url},
        )
        return None
    if url_name != manifest.name:
        # If index or URI disagree with the frontmatter, frontmatter wins — the
        # spec says the skill's identity is its `name` field — but log it.
        logger.warning(
            "MCP skill URI final segment differs from frontmatter name",
            data={
                "server": server_name,
                "url": url,
                "url_name": url_name,
                "frontmatter_name": manifest.name,
            },
        )

    return SkillManifest(
        name=manifest.name,
        description=manifest.description,
        body=manifest.body,
        path=None,
        license=manifest.license,
        compatibility=manifest.compatibility,
        metadata=manifest.metadata,
        allowed_tools=manifest.allowed_tools,
        uri=url,
        server_name=server_name,
    )


def _first_text(contents: Iterable) -> str | None:
    for item in contents:
        if isinstance(item, TextResourceContents):
            return item.text
    return None


def _first_blob(contents: Iterable) -> tuple[bytes, str | None] | None:
    """Return `(blob_bytes, mime_type)` from the first BlobResourceContents.

    `BlobResourceContents.blob` is base64-encoded per the MCP schema;
    decode here so callers receive raw bytes.
    """
    import base64

    for item in contents:
        if isinstance(item, BlobResourceContents):
            try:
                return base64.b64decode(item.blob), item.mimeType
            except Exception:
                return None
    return None


def _strip_archive_suffix(url: str) -> str | None:
    """Return the URL with its archive suffix removed, or None if none match.

    `skill://pdf-processing.tar.gz` -> `skill://pdf-processing`
    `skill://acme/billing/refunds.zip` -> `skill://acme/billing/refunds`
    """
    lowered = url.lower()
    for suffix in ARCHIVE_SUFFIXES:
        if lowered.endswith(suffix):
            return url[: -len(suffix)]
    return None


async def _load_archive_entry(
    aggregator: "MCPAggregator",
    server_name: str,
    entry: dict,
) -> tuple[SkillManifest, dict[str, bytes]] | None:
    """Fetch and unpack an `type: "archive"` index entry.

    Returns `(manifest, files)` where `files` is the unpacked file map
    keyed by archive-relative POSIX path. The manifest's URI is
    rewritten from the archive URL (`...skill.tar.gz`) to the post-unpack
    SKILL.md URI (`.../skill/SKILL.md`) so the rest of the host treats
    the result identically to a `skill-md` entry.
    """
    url = entry.get("url")
    if not isinstance(url, str) or not url:
        logger.warning(
            "Skill archive entry missing `url`",
            data={"server": server_name, "entry": entry},
        )
        return None

    # Same `file://` rejection as concrete entries: the SEP's trust model
    # delegates content authority to the MCP server, so a local-disk root
    # is not admissible regardless of distribution mechanism.
    if url.lower().startswith("file://"):
        logger.warning(
            "Rejecting `file://` skill archive URI",
            data={"server": server_name, "url": url},
        )
        return None

    skill_root = _strip_archive_suffix(url)
    if skill_root is None:
        logger.warning(
            "Skill archive URL has no recognized suffix (.tar.gz/.tgz/.zip)",
            data={"server": server_name, "url": url},
        )
        return None

    try:
        result = await aggregator.get_resource(url, server_name=server_name)
    except Exception as exc:
        logger.warning(
            "Failed to read MCP skill archive",
            data={"server": server_name, "url": url, "error": str(exc)},
        )
        return None

    blob_pair = _first_blob(result.contents)
    if blob_pair is None:
        logger.warning(
            "MCP skill archive resource returned no binary content",
            data={"server": server_name, "url": url},
        )
        return None
    blob, mime_type = blob_pair

    if len(blob) > MAX_ARCHIVE_BLOB_BYTES:
        logger.warning(
            "Skill archive blob exceeds wire-bytes limit",
            data={
                "server": server_name,
                "url": url,
                "bytes": len(blob),
                "limit": MAX_ARCHIVE_BLOB_BYTES,
            },
        )
        return None

    files = unpack_skill_archive(blob, mime_type, url)
    if files is None:
        # skill_archive already logged the specific safety failure.
        return None

    skill_md_bytes = files.get("SKILL.md")
    if skill_md_bytes is None:
        # unpack_skill_archive enforces this, but guard defensively.
        logger.warning(
            "Unpacked skill archive missing SKILL.md",
            data={"server": server_name, "url": url},
        )
        return None

    try:
        skill_md_text = skill_md_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        logger.warning(
            "Skill archive SKILL.md is not valid UTF-8",
            data={"server": server_name, "url": url, "error": str(exc)},
        )
        return None

    parsed, parse_error = SkillRegistry.parse_manifest_text(skill_md_text)
    if parsed is None:
        logger.warning(
            "Failed to parse SKILL.md from skill archive",
            data={"server": server_name, "url": url, "error": parse_error},
        )
        return None

    # Sanity check: archive URL final segment (post-suffix-strip) should
    # equal the frontmatter name. We do not enforce — the SEP says the
    # frontmatter `name` is the source of truth — but a mismatch is worth
    # surfacing as a warning so the operator can fix the index.
    archive_url_name = skill_root.rsplit("/", 1)[-1]
    if archive_url_name != parsed.name:
        logger.warning(
            "Skill archive URL final segment differs from frontmatter name",
            data={
                "server": server_name,
                "url": url,
                "url_name": archive_url_name,
                "frontmatter_name": parsed.name,
            },
        )

    skill_md_uri = f"{skill_root}/SKILL.md"

    manifest = SkillManifest(
        name=parsed.name,
        description=parsed.description,
        body=parsed.body,
        path=None,
        license=parsed.license,
        compatibility=parsed.compatibility,
        metadata=parsed.metadata,
        allowed_tools=parsed.allowed_tools,
        uri=skill_md_uri,
        server_name=server_name,
    )
    return manifest, files
