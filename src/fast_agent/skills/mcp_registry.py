"""SEP-2640 Skills over MCP registry support.

This module implements the registry/install/update portion of the Skills
Extension SEP (SEP-2640): servers that advertise ``io.modelcontextprotocol/skills``
can be used as a source for installing SHA256-verified skills into the normal
managed skills directory.

The index format follows the current SEP: each ``skills[]`` entry carries a
verbatim ``frontmatter`` object (the skill's ``SKILL.md`` YAML rendered as JSON)
plus an optional direct ``url``/``digest`` for ``SKILL.md`` and/or an
``archives[]`` array of pre-packed forms. There is no ``type`` field; an entry
must supply a usable ``url``, a non-empty ``archives``, or both.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import re
import shutil
import stat
import tarfile
import tempfile
import unicodedata
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Iterable, Literal, Mapping, Protocol
from urllib.parse import urlparse

import frontmatter
from mcp.types import BlobResourceContents, ServerCapabilities, TextResourceContents
from pydantic import AnyUrl

from fast_agent.core.logging.logger import get_logger
from fast_agent.marketplace import git_sources as marketplace_git_sources
from fast_agent.skills.provenance import (
    build_mcp_installed_skill_source,
    compute_skill_content_fingerprint,
    write_installed_skill_source,
)
from fast_agent.skills.registry import SkillRegistry
from fast_agent.utils.async_utils import run_coroutine

if TYPE_CHECKING:
    from mcp.types import ListResourcesResult, ReadResourceResult


class McpSkillRegistryClient(Protocol):
    async def get_capabilities(self, server_name: str) -> ServerCapabilities | None: ...

    async def get_resource(
        self, resource_uri: str, *, server_name: str | None = None
    ) -> "ReadResourceResult": ...


class McpSkillInstallClient(McpSkillRegistryClient, Protocol):
    """Client used to install skills; also walks directories for supporting files."""

    async def read_directory(
        self, uri: str, *, server_name: str | None = None, cursor: str | None = None
    ) -> "ListResourcesResult": ...


class McpSkillRegistryStatusClient(McpSkillRegistryClient, Protocol):
    async def collect_server_status(self) -> Mapping[str, Any]: ...


logger = get_logger(__name__)

SKILLS_EXTENSION = "io.modelcontextprotocol/skills"
INDEX_URI = "skill://index.json"
MAX_INDEX_BYTES = 1_048_576
MAX_SKILL_MD_BYTES = 262_144
MAX_ARCHIVE_BYTES = 10 * 1_048_576
MAX_UNPACKED_ARCHIVE_BYTES = 50 * 1_048_576
# Per-resource cap for a supporting file, bounding any single file under the
# cumulative ``MAX_UNPACKED_ARCHIVE_BYTES`` budget.
MAX_SUPPORTING_FILE_BYTES = 10 * 1_048_576
# Bounds on a ``resources/directory/read`` walk so a buggy or hostile server
# cannot hang the install: total pages (never-terminating cursor), entries, depth.
MAX_WALK_PAGES = 1_000
MAX_WALK_ENTRIES = 10_000
MAX_WALK_DEPTH = 32
DIRECTORY_MIME_TYPE = "inode/directory"
ArtifactType = Literal["skill-md", "archive"]
SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")

# Supported archive media types mapped to the extraction strategy.
ARCHIVE_MEDIA_TYPES: dict[str, Literal["tar", "zip"]] = {
    "application/gzip": "tar",
    "application/x-gzip": "tar",
    "application/x-tar": "tar",
    "application/x-gtar": "tar",
    "application/zip": "zip",
    "application/x-zip-compressed": "zip",
}


@dataclass(frozen=True)
class McpRegistryArchive:
    url: str
    mime_type: str
    digest: str


@dataclass(frozen=True)
class McpRegistrySkill:
    name: str
    description: str | None
    source_url: str
    server_name: str
    digest: str
    artifact_type: ArtifactType = "skill-md"
    server_version: str | None = None
    frontmatter: dict[str, Any] = field(default_factory=dict)
    archives: tuple[McpRegistryArchive, ...] = ()
    artifact_mime_type: str | None = None

    @property
    def install_dir_name(self) -> str:
        return self.name


@dataclass(frozen=True)
class McpSkillRegistry:
    server_name: str
    server_version: str | None
    skills: list[McpRegistrySkill]

    @property
    def display_name(self) -> str:
        version = f"@{self.server_version}" if self.server_version else ""
        return f"mcp-server {self.server_name}{version}"


def _extension_settings(capabilities: ServerCapabilities | None) -> Mapping[str, Any] | None:
    if capabilities is None:
        return None
    extras = capabilities.model_extra or {}
    extensions = extras.get("extensions")
    if not isinstance(extensions, Mapping):
        return None
    settings = extensions.get(SKILLS_EXTENSION)
    return settings if isinstance(settings, Mapping) else None


def server_supports_mcp_skills(capabilities: ServerCapabilities | None) -> bool:
    if capabilities is None:
        return False
    extras = capabilities.model_extra or {}
    extensions = extras.get("extensions")
    if not isinstance(extensions, Mapping):
        return False
    return SKILLS_EXTENSION in extensions


def server_supports_directory_read(capabilities: ServerCapabilities | None) -> bool:
    """Whether the server declared ``directoryRead`` for the skills extension."""
    settings = _extension_settings(capabilities)
    if settings is None:
        return False
    return settings.get("directoryRead") is True


async def scan_mcp_skill_registry(
    aggregator: McpSkillRegistryClient,
    server_name: str,
    *,
    server_version: str | None = None,
) -> McpSkillRegistry | None:
    capabilities = await aggregator.get_capabilities(server_name)
    if not server_supports_mcp_skills(capabilities):
        return None

    try:
        result = await aggregator.get_resource(INDEX_URI, server_name=server_name)
    except Exception as exc:  # noqa: BLE001 - optional index; registry simply has no entries.
        logger.debug(
            "SEP-2640 skills index unavailable",
            data={"server": server_name, "error": str(exc)},
        )
        return McpSkillRegistry(server_name=server_name, server_version=server_version, skills=[])

    entries = _parse_index(result, server_name)
    skills: list[McpRegistrySkill] = []
    for entry in entries:
        skill = _build_registry_skill(entry, server_name=server_name, server_version=server_version)
        if skill is not None:
            skills.append(skill)
    return McpSkillRegistry(server_name=server_name, server_version=server_version, skills=skills)


def _build_registry_skill(
    entry: dict[str, Any],
    *,
    server_name: str,
    server_version: str | None,
) -> McpRegistrySkill | None:
    frontmatter = entry.get("frontmatter")
    if not isinstance(frontmatter, Mapping):
        logger.warning("MCP skill entry missing frontmatter", data={"server": server_name})
        return None
    name = frontmatter.get("name")
    if not isinstance(name, str) or not name.strip():
        logger.warning("MCP skill entry frontmatter missing name", data={"server": server_name})
        return None
    name = name.strip()
    description = frontmatter.get("description")
    description = description.strip() if isinstance(description, str) else None

    direct = _validate_url_and_digest(
        url=entry.get("url"),
        digest=entry.get("digest"),
        server_name=server_name,
        name=name,
        label="url",
    )
    archives = _parse_archives(entry, server_name=server_name, name=name)

    # Prefer an archive (atomic, complete multi-file skill) when one is offered;
    # otherwise fall back to the direct SKILL.md entry.
    if archives:
        chosen = archives[0]
        artifact_type: ArtifactType = "archive"
        source_url = chosen.url
        digest = chosen.digest
        artifact_mime_type: str | None = chosen.mime_type
    elif direct is not None:
        artifact_type = "skill-md"
        source_url, digest = direct
        artifact_mime_type = None
    else:
        logger.warning(
            "MCP skill entry has no usable url or archives",
            data={"server": server_name, "name": name},
        )
        return None

    return McpRegistrySkill(
        name=name,
        description=description,
        source_url=source_url,
        server_name=server_name,
        digest=digest,
        artifact_type=artifact_type,
        server_version=server_version,
        frontmatter=dict(frontmatter),
        archives=tuple(archives),
        artifact_mime_type=artifact_mime_type,
    )


def _validate_url_and_digest(
    *, url: Any, digest: Any, server_name: str, name: str, label: str
) -> tuple[str, str] | None:
    """Validate an artifact's ``url``/``digest`` pair, shared by direct entries and archives.

    Returns ``(resolved_url, stripped_digest)`` or ``None`` when the url is
    missing/empty (silently), points at ``file://``, or the digest is not a valid
    sha256. ``label`` ("url" / "archive") is woven into the rejection warnings.
    """
    if not isinstance(url, str) or not url.strip():
        return None
    source_url = url.strip()
    if source_url.lower().startswith("file://"):
        logger.warning(
            f"Rejecting file:// MCP skill {label}",
            data={"server": server_name, "name": name, "url": source_url},
        )
        return None
    if not isinstance(digest, str) or not _is_valid_sha256_digest(digest):
        logger.warning(
            f"MCP skill {label} missing valid sha256 digest",
            data={"server": server_name, "name": name},
        )
        return None
    return _resolve_entry_url(source_url), digest.strip()


def _parse_archives(
    entry: Mapping[str, Any], *, server_name: str, name: str
) -> list[McpRegistryArchive]:
    raw_archives = entry.get("archives")
    if not isinstance(raw_archives, list):
        return []
    archives: list[McpRegistryArchive] = []
    for raw in raw_archives:
        if not isinstance(raw, Mapping):
            continue
        mime_type = raw.get("mimeType")
        if not isinstance(mime_type, str) or mime_type.strip() not in ARCHIVE_MEDIA_TYPES:
            logger.warning(
                "Skipping MCP skill archive with unsupported media type",
                data={"server": server_name, "name": name, "mimeType": mime_type},
            )
            continue
        validated = _validate_url_and_digest(
            url=raw.get("url"),
            digest=raw.get("digest"),
            server_name=server_name,
            name=name,
            label="archive",
        )
        if validated is None:
            continue
        url, digest = validated
        archives.append(
            McpRegistryArchive(url=url, mime_type=mime_type.strip(), digest=digest)
        )
    return archives


async def list_mcp_skill_registries(
    aggregator: McpSkillRegistryStatusClient, server_names: Iterable[str]
) -> list[McpSkillRegistry]:
    registries: list[McpSkillRegistry] = []
    for server_name in server_names:
        server_version = await _server_version(aggregator, server_name)
        registry = await scan_mcp_skill_registry(
            aggregator, server_name, server_version=server_version
        )
        if registry is not None:
            registries.append(registry)
    return registries


async def install_mcp_registry_skill(
    aggregator: McpSkillInstallClient,
    skill: McpRegistrySkill,
    *,
    destination_root: Path,
) -> Path:
    install_dir = destination_root.resolve() / _safe_install_dir_name(skill.name)
    if install_dir.exists():
        raise FileExistsError(f"Skill already exists: {install_dir}")

    try:
        await _write_verified_mcp_skill(aggregator, skill, install_dir)
    except Exception:
        if install_dir.exists():
            shutil.rmtree(install_dir)
        raise
    return install_dir


async def update_mcp_registry_skill(
    aggregator: McpSkillInstallClient,
    skill: McpRegistrySkill,
    *,
    skill_dir: Path,
) -> Path:
    skill_dir = skill_dir.resolve()
    parent_dir = skill_dir.parent
    with tempfile.TemporaryDirectory(
        dir=parent_dir,
        prefix=f".{skill_dir.name}.update-",
    ) as temp_dir_str:
        staged_dir = Path(temp_dir_str) / skill_dir.name
        await _write_verified_mcp_skill(aggregator, skill, staged_dir)
        marketplace_git_sources.atomic_replace_directory(
            existing_dir=skill_dir,
            staged_dir=staged_dir,
        )
    return skill_dir


def select_mcp_registry_skill(
    entries: Iterable[McpRegistrySkill],
    selector: str,
) -> McpRegistrySkill | None:
    selector_clean = selector.strip()
    if not selector_clean:
        return None
    entries_list = list(entries)
    if selector_clean.isdigit():
        index = int(selector_clean)
        if 1 <= index <= len(entries_list):
            return entries_list[index - 1]
        return None
    selector_lower = selector_clean.lower()
    for entry in entries_list:
        if entry.name.lower() == selector_lower:
            return entry
    return None


async def _server_version(aggregator: McpSkillRegistryStatusClient, server_name: str) -> str | None:
    status_map = await aggregator.collect_server_status()
    status = status_map.get(server_name)
    if status is None:
        return None
    return status.implementation_version


def _parse_index(result: "ReadResourceResult", server_name: str) -> list[dict[str, Any]]:
    text = _first_text(result)
    if not text:
        return []
    if len(text.encode("utf-8")) > MAX_INDEX_BYTES:
        logger.warning(
            "MCP skill index exceeds size limit",
            data={"server": server_name, "limit": MAX_INDEX_BYTES},
        )
        return []
    try:
        parsed = json.loads(text)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Failed to parse MCP skill index",
            data={"server": server_name, "error": str(exc)},
        )
        return []
    skills = parsed.get("skills") if isinstance(parsed, dict) else None
    if not isinstance(skills, list):
        return []
    return [entry for entry in skills if isinstance(entry, dict)]


def _first_text(result: "ReadResourceResult") -> str | None:
    for item in result.contents:
        if isinstance(item, TextResourceContents):
            return item.text
    return None


def _first_bytes(result: "ReadResourceResult") -> bytes | None:
    for item in result.contents:
        if isinstance(item, TextResourceContents):
            return item.text.encode("utf-8")
        if isinstance(item, BlobResourceContents):
            return base64.b64decode(item.blob)
    return None


async def _write_verified_mcp_skill(
    aggregator: McpSkillInstallClient,
    skill: McpRegistrySkill,
    install_dir: Path,
) -> None:
    result = await aggregator.get_resource(skill.source_url, server_name=skill.server_name)
    artifact = _first_bytes(result)
    if artifact is None:
        raise ValueError(f"MCP skill resource returned no content: {skill.source_url}")
    _verify_artifact_digest(artifact, skill.digest)

    if skill.artifact_type == "skill-md":
        _write_skill_md_artifact(skill, artifact, install_dir)
        await _materialize_supporting_files(aggregator, skill, install_dir)
    else:
        _write_archive_artifact(skill, artifact, install_dir)

    fingerprint = compute_skill_content_fingerprint(install_dir)
    write_installed_skill_source(
        install_dir,
        build_mcp_installed_skill_source(
            server_name=skill.server_name,
            server_version=skill.server_version,
            skill_uri=skill.source_url,
            fingerprint=fingerprint,
            artifact_digest=skill.digest,
            artifact_type=skill.artifact_type,
        ),
    )


def _canonical_frontmatter(value: Any) -> Any:
    """Normalize a frontmatter value for content comparison.

    Strips surrounding whitespace from scalars (so a YAML folded-scalar trailing
    newline does not read as a mismatch) and recurses into mappings/sequences;
    mappings compare key-and-value, order-independent.
    """
    if isinstance(value, Mapping):
        return {str(key): _canonical_frontmatter(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical_frontmatter(item) for item in value]
    if isinstance(value, str):
        return value.strip()
    return value


def _verify_frontmatter_matches_index(skill_text: str, skill: McpRegistrySkill) -> None:
    """Re-verify the served SKILL.md frontmatter against the index entry, field by field.

    SEP-2640 requires the index ``frontmatter`` object to be identical in content
    to the SKILL.md it describes — the index is not authoritative. fast-agent
    pins the SKILL.md bytes by SHA-256, but the digest covers the file we fetched,
    not the index's *claim* about it, so a server can still publish an index
    ``frontmatter`` that diverges from the file (advertising a tamer description,
    omitting a self-granted field, etc.). Re-parse the file we actually fetched
    and compare every field; treat any divergence as a verification failure.
    Previously only ``manifest.name == skill.name`` was checked.
    """
    served = frontmatter.loads(skill_text).metadata or {}
    if _canonical_frontmatter(served) != _canonical_frontmatter(skill.frontmatter or {}):
        raise ValueError(
            "MCP skill index frontmatter does not match the served SKILL.md "
            f"frontmatter for '{skill.name}' (index frontmatter is not authoritative)"
        )


# Frontmatter fields that widen a skill's host permissions or register executable
# behaviour. Honoring these for a *remote* (MCP-origin) skill lets the serving
# server self-grant access on the host, so they are stripped on install.
PERMISSION_WIDENING_FRONTMATTER = ("allowed-tools", "hooks")


def _strip_permission_widening_frontmatter(install_dir: Path) -> None:
    """Strip permission-widening / exec-registering frontmatter from the installed SKILL.md.

    Every skill installed through this module is MCP-origin (remote). Fields like
    ``allowed-tools`` and ``hooks`` scope which tools / automation a *local* skill
    may use; honoring them for a remote skill lets the serving MCP server widen
    its own access on the host. This install path has no per-skill approval
    channel, so strip them — the skill still installs, with the self-grant
    neutralized — rather than honor a server self-grant.
    """
    skill_md = install_dir / "SKILL.md"
    if not skill_md.is_file():
        return
    post = frontmatter.loads(skill_md.read_text(encoding="utf-8"))
    removed = [key for key in PERMISSION_WIDENING_FRONTMATTER if key in post.metadata]
    if not removed:
        return
    for key in removed:
        post.metadata.pop(key, None)
    skill_md.write_text(frontmatter.dumps(post), encoding="utf-8")
    logger.warning(
        "Stripped permission-widening frontmatter from MCP-origin skill",
        data={"skill": install_dir.name, "fields": removed},
    )


def _write_skill_md_artifact(skill: McpRegistrySkill, artifact: bytes, install_dir: Path) -> None:
    if len(artifact) > MAX_SKILL_MD_BYTES:
        raise ValueError(f"MCP skill SKILL.md exceeds size limit: {skill.source_url}")
    skill_text = artifact.decode("utf-8")
    manifest, parse_error = SkillRegistry.parse_manifest_text(skill_text)
    if manifest is None:
        raise ValueError(f"Failed to parse MCP skill manifest: {parse_error}")
    if manifest.name != skill.name:
        raise ValueError(
            f"MCP skill index name '{skill.name}' does not match manifest name '{manifest.name}'"
        )
    _verify_frontmatter_matches_index(skill_text, skill)

    install_dir.mkdir(parents=True, exist_ok=False)
    (install_dir / "SKILL.md").write_text(skill_text, encoding="utf-8")
    _strip_permission_widening_frontmatter(install_dir)


def _write_archive_artifact(skill: McpRegistrySkill, artifact: bytes, install_dir: Path) -> None:
    if len(artifact) > MAX_ARCHIVE_BYTES:
        raise ValueError(f"MCP skill archive exceeds size limit: {skill.source_url}")
    install_dir.mkdir(parents=True, exist_ok=False)
    try:
        if _archive_strategy(skill) == "zip":
            _extract_zip_safely(artifact, install_dir)
        else:
            _extract_tar_safely(artifact, install_dir)
        manifest_path = install_dir / "SKILL.md"
        if not manifest_path.is_file():
            raise ValueError("MCP skill archive must contain SKILL.md at the root")
        manifest_text = manifest_path.read_text(encoding="utf-8")
        manifest, parse_error = SkillRegistry.parse_manifest_text(manifest_text)
        if manifest is None:
            raise ValueError(f"Failed to parse MCP skill manifest: {parse_error}")
        if manifest.name != skill.name:
            raise ValueError(
                f"MCP skill index name '{skill.name}' does not match manifest name '{manifest.name}'"
            )
        _verify_frontmatter_matches_index(manifest_text, skill)
        _strip_permission_widening_frontmatter(install_dir)
    except Exception:
        if install_dir.exists():
            shutil.rmtree(install_dir)
        raise


def _archive_strategy(skill: McpRegistrySkill) -> Literal["tar", "zip"]:
    # ``_parse_archives`` only admits archives whose ``mimeType`` is known, so
    # an index-sourced archive always resolves here.
    mime_type = skill.artifact_mime_type
    if mime_type is None or mime_type not in ARCHIVE_MEDIA_TYPES:
        raise ValueError(f"MCP skill archive has no recognized media type: {skill.source_url}")
    return ARCHIVE_MEDIA_TYPES[mime_type]


async def _materialize_supporting_files(
    aggregator: McpSkillInstallClient,
    skill: McpRegistrySkill,
    install_dir: Path,
) -> None:
    """Fetch a direct-entry skill's supporting files via ``resources/directory/read``.

    A direct ``url`` entry only addresses ``SKILL.md``; supporting files are
    sibling resources. When the server advertises ``directoryRead``, walk the
    skill root and materialize each child. All-or-nothing: the walk stages into a
    temp dir and merges only on full success, so a mid-walk failure leaves the
    verified single-file skill intact rather than a half-written tree.

    Unlike the digest-checked ``SKILL.md`` and archive artifacts, supporting
    files carry no digests, so they rest on trusting this server and the
    transport; path traversal is still blocked and total size bounded.
    """
    capabilities = await aggregator.get_capabilities(skill.server_name)
    if not server_supports_directory_read(capabilities):
        return
    root_uri = _skill_root_uri(skill.source_url)
    if root_uri is None:
        return
    with tempfile.TemporaryDirectory(
        dir=install_dir.parent, prefix=f".{install_dir.name}.support-"
    ) as staging_str:
        staging = Path(staging_str)
        try:
            await _walk_skill_directory(
                aggregator,
                server_name=skill.server_name,
                root_uri=root_uri,
                dir_uri=root_uri,
                dest_dir=staging,
                budget=_ByteBudget(MAX_UNPACKED_ARCHIVE_BYTES),
                limits=_WalkLimits(),
                depth=0,
            )
        except Exception as exc:  # noqa: BLE001 - supporting files are best-effort.
            # Staging is discarded on exit; the verified skill remains intact.
            logger.warning(
                "Failed to materialize MCP skill supporting files",
                data={"server": skill.server_name, "skill": skill.name, "error": str(exc)},
            )
            return
        # Walk succeeded: merge the staged tree alongside the verified SKILL.md.
        shutil.copytree(staging, install_dir, dirs_exist_ok=True)


class _ByteBudget:
    def __init__(self, limit: int) -> None:
        self._limit = limit
        self._used = 0

    def add(self, size: int) -> None:
        self._used += size
        if self._used > self._limit:
            raise ValueError("MCP skill supporting files exceed unpacked size limit")


class _WalkLimits:
    """Shared counters bounding a directory walk across all recursion branches."""

    def __init__(self) -> None:
        self._pages = 0
        self._entries = 0

    def count_page(self) -> None:
        self._pages += 1
        if self._pages > MAX_WALK_PAGES:
            raise ValueError("MCP skill directory walk exceeded page limit")

    def count_entry(self) -> None:
        self._entries += 1
        if self._entries > MAX_WALK_ENTRIES:
            raise ValueError("MCP skill directory walk exceeded entry limit")


async def _walk_skill_directory(
    aggregator: McpSkillInstallClient,
    *,
    server_name: str,
    root_uri: str,
    dir_uri: str,
    dest_dir: Path,
    budget: _ByteBudget,
    limits: _WalkLimits,
    depth: int,
) -> None:
    if depth > MAX_WALK_DEPTH:
        raise ValueError("MCP skill directory walk exceeded depth limit")
    cursor: str | None = None
    while True:
        limits.count_page()
        listing = await aggregator.read_directory(dir_uri, server_name=server_name, cursor=cursor)
        for resource in listing.resources:
            limits.count_entry()
            child_uri = str(resource.uri)
            relative = _relative_uri_path(root_uri, child_uri)
            if relative is None:
                logger.debug(
                    "Skipping MCP skill resource outside skill root",
                    data={"server": server_name, "root": root_uri, "uri": child_uri},
                )
                continue
            if resource.mimeType == DIRECTORY_MIME_TYPE:
                await _walk_skill_directory(
                    aggregator,
                    server_name=server_name,
                    root_uri=root_uri,
                    dir_uri=child_uri,
                    dest_dir=dest_dir,
                    budget=budget,
                    limits=limits,
                    depth=depth + 1,
                )
                continue
            if relative.casefold() == "skill.md":
                # Already written (digest-verified) from the direct entry; skip
                # case-insensitively so a sibling can't clobber it on Windows/macOS.
                continue
            _validate_archive_name(relative)
            content = await aggregator.get_resource(child_uri, server_name=server_name)
            data = _first_bytes(content)
            if data is None:
                # No bytes -- often a directory the server failed to tag with
                # ``inode/directory``, so its children are silently dropped. Warn
                # so the gap is visible.
                logger.warning(
                    "MCP skill supporting resource returned no content; skipping "
                    "(its children, if it is an untagged directory, are not installed)",
                    data={
                        "server": server_name,
                        "uri": child_uri,
                        "mimeType": resource.mimeType,
                    },
                )
                continue
            if len(data) > MAX_SUPPORTING_FILE_BYTES:
                raise ValueError(
                    f"MCP skill supporting file exceeds size limit: {child_uri}"
                )
            budget.add(len(data))
            destination = dest_dir / PurePosixPath(relative)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
        cursor = listing.nextCursor
        if not cursor:
            break


def _skill_root_uri(source_url: str) -> str | None:
    # Case-insensitive: the manifest may be addressed as ``/skill.md``.
    suffix = "/SKILL.md"
    if source_url.lower().endswith(suffix.lower()):
        return _normalize_uri(source_url[: -len(suffix)])
    return None


def _normalize_uri(uri: str) -> str:
    """Normalize a URI through ``AnyUrl`` to match directory children.

    Children arrive as ``str(resource.uri)`` (an ``AnyUrl``); the root is sliced
    from the raw index ``url``, so it must be normalized the same way or the
    prefix comparison spuriously fails and drops every child.
    """
    try:
        return str(AnyUrl(uri))
    except Exception:  # noqa: BLE001 - fall back to the raw string if unparseable.
        return uri


def _relative_uri_path(root_uri: str, child_uri: str) -> str | None:
    prefix = root_uri.rstrip("/") + "/"
    if not child_uri.startswith(prefix):
        return None
    relative = child_uri[len(prefix) :]
    return relative or None


def _check_archive_name_collisions(names: Iterable[str]) -> None:
    """Reject archives whose entry names collide under Unicode normalization and case-folding.

    On a case-insensitive or Unicode-normalizing filesystem (Windows NTFS, macOS
    APFS) two distinct entry names can resolve to the same path, so a later entry
    silently overwrites an earlier one — e.g. ``Skill.md`` clobbering the
    digest-verified ``SKILL.md``. A raw byte-equality / ``set`` duplicate check
    misses this, so fold each name the way such a filesystem would (NFC + case)
    before checking for duplicates.
    """
    seen: dict[str, str] = {}
    for name in names:
        # Normalize the whole POSIX path as a case-insensitive, NFC-normalizing
        # filesystem would; drop a trailing slash so a directory entry and a file
        # of the same name also collide.
        key = unicodedata.normalize("NFC", name).rstrip("/").casefold()
        if not key:
            continue
        if key in seen:
            raise ValueError(
                "MCP skill archive entries collide under case/Unicode normalization: "
                f"{seen[key]!r} and {name!r}"
            )
        seen[key] = name


def _extract_tar_safely(artifact: bytes, destination: Path) -> None:
    total_size = 0
    with tarfile.open(fileobj=io.BytesIO(artifact), mode="r:*") as archive:
        members = archive.getmembers()
        _check_archive_name_collisions(member.name for member in members)
        for member in members:
            _validate_archive_name(member.name)
            if member.issym() or member.islnk():
                raise ValueError("MCP skill archives must not contain links")
            if member.isfile():
                total_size += member.size
                if total_size > MAX_UNPACKED_ARCHIVE_BYTES:
                    raise ValueError("MCP skill archive unpacked size exceeds limit")
        archive.extractall(destination, filter="data")


def _extract_zip_safely(artifact: bytes, destination: Path) -> None:
    total_size = 0
    with zipfile.ZipFile(io.BytesIO(artifact)) as archive:
        infos = archive.infolist()
        _check_archive_name_collisions(info.filename for info in infos)
        for info in infos:
            _validate_archive_name(info.filename)
            mode = info.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise ValueError("MCP skill archives must not contain links")
            total_size += info.file_size
            if total_size > MAX_UNPACKED_ARCHIVE_BYTES:
                raise ValueError("MCP skill archive unpacked size exceeds limit")
        archive.extractall(destination)


_WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:")


def _validate_archive_name(name: str) -> None:
    # Names are later joined with the OS path API. PurePosixPath ignores Windows
    # separators and drive anchors, so reject those too or "..\\.." / "C:\\..."
    # could escape the install dir on Windows.
    if "\\" in name:
        raise ValueError(f"Unsafe path in MCP skill archive: {name}")
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe path in MCP skill archive: {name}")
    if any(_WINDOWS_DRIVE_RE.match(part) for part in path.parts):
        raise ValueError(f"Unsafe path in MCP skill archive: {name}")


def _verify_artifact_digest(artifact: bytes, expected: str) -> None:
    if not _is_valid_sha256_digest(expected):
        raise ValueError("MCP skill entry is missing a valid SHA256 digest")
    actual = f"sha256:{hashlib.sha256(artifact).hexdigest()}"
    if actual != expected:
        raise ValueError(f"MCP skill SHA256 mismatch: expected {expected}, got {actual}")


def _is_valid_sha256_digest(value: str) -> bool:
    return bool(SHA256_RE.fullmatch(value))


def _resolve_entry_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme:
        return url
    return f"skill://{url.lstrip('/')}"


def _safe_install_dir_name(name: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        raise ValueError(f"Invalid MCP skill name for local install: {name}")
    return name


def scan_mcp_skill_registry_sync(
    aggregator: McpSkillRegistryClient, server_name: str
) -> McpSkillRegistry | None:
    return run_coroutine(scan_mcp_skill_registry(aggregator, server_name))
