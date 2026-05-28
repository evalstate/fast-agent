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

from mcp.types import TextResourceContents

from fast_agent.core.logging.logger import get_logger
from fast_agent.mcp.skill_uri import skill_name_from_uri
from fast_agent.skills.registry import SkillManifest, SkillRegistry

if TYPE_CHECKING:
    from fast_agent.mcp.mcp_aggregator import MCPAggregator

logger = get_logger(__name__)

INDEX_URI = "skill://index.json"

# Soft ceilings on server-returned bytes: bound memory against a misbehaving server.
MAX_INDEX_BYTES = 1_048_576  # 1 MiB
MAX_SKILL_MD_BYTES = 262_144  # 256 KiB

# Per SEP-2640: clients SHOULD match `$schema` against known URIs before processing.
# Unknown schema parses best-effort with a warning rather than aborting.
KNOWN_INDEX_SCHEMAS = frozenset(
    {"https://schemas.agentskills.io/discovery/0.2.0/schema.json"}
)


@dataclass(frozen=True)
class SkillTemplateEntry:
    """A parameterized skill namespace from `skill://index.json`.

    `type: "mcp-resource-template"` entries describe a skill space the
    user navigates rather than a concrete skill — the URL is an RFC 6570
    URI template (e.g. `skill://docs/{product}/SKILL.md`) which a host
    resolves by filling each variable, typically via the MCP completion
    API. Resolved URIs become regular `skill-md` entries; the template
    itself is never registered as a `SkillManifest`.
    """

    server_name: str
    url_template: str
    description: str

    def variable_names(self) -> list[str]:
        """Return RFC 6570 variable names (just simple `{var}` form)."""
        import re

        return re.findall(r"\{([A-Za-z0-9_.\-]+)\}", self.url_template)


def expand_uri_template(template: str, values: dict[str, str]) -> str:
    """Resolve a simple RFC 6570 template by substituting `{var}` literals.

    Only handles the bare `{var}` form — no `{?var}`, `{+var}`, etc. The
    SEP example (`skill://docs/{product}/SKILL.md`) is the bound case
    we're targeting; if a server uses an extended form we'll need to
    pull in the `uritemplate` package or hand-roll more shapes.
    """
    import re
    from urllib.parse import quote

    def _sub(match: "re.Match[str]") -> str:
        name = match.group(1)
        if name not in values:
            raise KeyError(f"template variable not provided: {name}")
        return quote(values[name], safe="")

    return re.sub(r"\{([A-Za-z0-9_.\-]+)\}", _sub, template)


@dataclass
class LoadedSkills:
    """Result of `load_mcp_skill_manifests`.

    The loader discovers two distinct kinds of artifact:

    - `manifests` — concrete `skill-md` entries reduced to `SkillManifest`.
    - `template_entries` — `mcp-resource-template` index entries left
      unresolved. The host surfaces these in its UI; the user fills
      variables (via the MCP completion API) and the resolved URI
      registers as a regular `skill-md` entry.
    """

    manifests: list[SkillManifest] = field(default_factory=list)
    template_entries: list[SkillTemplateEntry] = field(default_factory=list)


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
    - `mcp-resource-template` — held as a `SkillTemplateEntry` for the
      user to resolve via `/skills resolve`.

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
                url = entry.get("url")
                description = entry.get("description") or ""
                if not isinstance(url, str) or not url:
                    logger.warning(
                        "Skill template entry missing `url`",
                        data={"server": server_name, "entry": entry},
                    )
                    continue
                if url.lower().startswith("file://"):
                    # See `_load_concrete_entry` for the `file://` trust rationale.
                    logger.warning(
                        "Rejecting `file://` skill template URI",
                        data={"server": server_name, "url": url},
                    )
                    continue
                result.template_entries.append(
                    SkillTemplateEntry(
                        server_name=server_name,
                        url_template=url,
                        description=description,
                    )
                )
                continue
            if entry_type == "skill-md":
                manifest = await _load_concrete_entry(aggregator, server_name, entry)
                if manifest is not None:
                    result.manifests.append(manifest)
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


async def resolve_skill_template(
    aggregator: "MCPAggregator",
    template: SkillTemplateEntry,
    variables: dict[str, str],
) -> SkillManifest | None:
    """Expand `template` with `variables` and fetch the resulting SKILL.md.

    Returns a `SkillManifest` ready to register, or `None` if expansion
    fails (missing variable) or the resulting resource can't be loaded.
    The host treats the manifest exactly as if a `skill-md` index entry
    had named the resolved URI — no special handling is needed downstream.
    """
    try:
        resolved_url = expand_uri_template(template.url_template, variables)
    except KeyError as exc:
        logger.warning(
            "Skill template expansion missing variable",
            data={
                "server": template.server_name,
                "url_template": template.url_template,
                "missing": str(exc),
            },
        )
        return None

    synthetic_entry = {
        "type": "skill-md",
        "url": resolved_url,
    }
    return await _load_concrete_entry(aggregator, template.server_name, synthetic_entry)


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

    # Reject `file://` skill URIs: Skills-over-MCP delegates content authority to
    # the publishing server, but `file://` collapses that to the server process's
    # local disk view — bypassing the ACP / filesystem-runtime path guardrails.
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

    # A URI with no skill-path segment (e.g. `skill://SKILL.md`) is rejected: stripping
    # `/SKILL.md` would leave `skill:/` in the reader's allowed-roots set and admit every
    # `skill://...` URI via the prefix check.
    url_name = skill_name_from_uri(url)
    if not url_name:
        logger.warning(
            "Skill entry URI has no path segment before `/SKILL.md`; refusing to register",
            data={"server": server_name, "url": url},
        )
        return None
    if url_name != manifest.name:
        # Frontmatter `name` is the spec's source of truth; log the mismatch.
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
