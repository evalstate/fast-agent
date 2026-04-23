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
from typing import TYPE_CHECKING, Iterable, Sequence

from mcp.types import TextResourceContents

from fast_agent.core.logging.logger import get_logger
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
) -> list[SkillManifest]:
    """Fetch and parse skill manifests from connected MCP servers.

    For each server, reads `skill://index.json` (optional; missing index is
    silently skipped), then fetches each listed `SKILL.md` resource and parses
    its frontmatter. Returns one `SkillManifest` per concrete `skill-md` index
    entry. `mcp-resource-template` entries are logged and skipped (future work).

    Errors from a single server or entry are logged as warnings; a failure
    never aborts the whole batch.
    """

    manifests: list[SkillManifest] = []
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
            if entry_type != "skill-md":
                logger.debug(
                    "Skipping MCP skill entry with unrecognized type",
                    data={"server": server_name, "type": entry_type},
                )
                continue
            manifest = await _load_concrete_entry(aggregator, server_name, entry)
            if manifest is not None:
                manifests.append(manifest)

    return manifests


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
