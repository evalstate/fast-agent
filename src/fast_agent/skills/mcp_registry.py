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
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Iterable, Literal, Mapping, Protocol
from urllib.parse import urlparse

from mcp.types import BlobResourceContents, ServerCapabilities, TextResourceContents

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

    direct_url, direct_digest = _parse_direct_entry(entry, server_name=server_name, name=name)
    archives = _parse_archives(entry, server_name=server_name, name=name)

    # Prefer an archive (atomic, complete multi-file skill) when one is offered;
    # otherwise fall back to the direct SKILL.md entry.
    if archives:
        chosen = archives[0]
        artifact_type: ArtifactType = "archive"
        source_url = chosen.url
        digest = chosen.digest
        artifact_mime_type: str | None = chosen.mime_type
    elif direct_url is not None and direct_digest is not None:
        artifact_type = "skill-md"
        source_url = direct_url
        digest = direct_digest
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


def _parse_direct_entry(
    entry: Mapping[str, Any], *, server_name: str, name: str
) -> tuple[str | None, str | None]:
    url = entry.get("url")
    if not isinstance(url, str) or not url.strip():
        return None, None
    source_url = url.strip()
    if source_url.lower().startswith("file://"):
        logger.warning(
            "Rejecting file:// MCP skill URL",
            data={"server": server_name, "url": source_url},
        )
        return None, None
    digest = entry.get("digest")
    if not isinstance(digest, str) or not _is_valid_sha256_digest(digest):
        logger.warning(
            "MCP skill entry url missing valid sha256 digest",
            data={"server": server_name, "name": name},
        )
        return None, None
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
        url = raw.get("url")
        mime_type = raw.get("mimeType")
        digest = raw.get("digest")
        if not isinstance(url, str) or not url.strip():
            continue
        if url.strip().lower().startswith("file://"):
            logger.warning(
                "Rejecting file:// MCP skill archive URL",
                data={"server": server_name, "name": name},
            )
            continue
        if not isinstance(mime_type, str) or mime_type.strip() not in ARCHIVE_MEDIA_TYPES:
            logger.warning(
                "Skipping MCP skill archive with unsupported media type",
                data={"server": server_name, "name": name, "mimeType": mime_type},
            )
            continue
        if not isinstance(digest, str) or not _is_valid_sha256_digest(digest):
            logger.warning(
                "MCP skill archive missing valid sha256 digest",
                data={"server": server_name, "name": name},
            )
            continue
        archives.append(
            McpRegistryArchive(
                url=_resolve_entry_url(url.strip()),
                mime_type=mime_type.strip(),
                digest=digest.strip(),
            )
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

    install_dir.mkdir(parents=True, exist_ok=False)
    (install_dir / "SKILL.md").write_text(skill_text, encoding="utf-8")


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
        manifest, parse_error = SkillRegistry.parse_manifest_text(
            manifest_path.read_text(encoding="utf-8")
        )
        if manifest is None:
            raise ValueError(f"Failed to parse MCP skill manifest: {parse_error}")
        if manifest.name != skill.name:
            raise ValueError(
                f"MCP skill index name '{skill.name}' does not match manifest name '{manifest.name}'"
            )
    except Exception:
        if install_dir.exists():
            shutil.rmtree(install_dir)
        raise


def _archive_strategy(skill: McpRegistrySkill) -> Literal["tar", "zip"]:
    if skill.artifact_mime_type is not None:
        strategy = ARCHIVE_MEDIA_TYPES.get(skill.artifact_mime_type)
        if strategy is not None:
            return strategy
    # Fall back to the URL suffix when no (recognized) media type is available.
    return "zip" if skill.source_url.lower().endswith(".zip") else "tar"


async def _materialize_supporting_files(
    aggregator: McpSkillInstallClient,
    skill: McpRegistrySkill,
    install_dir: Path,
) -> None:
    """Fetch a direct-entry skill's supporting files via ``resources/directory/read``.

    A direct ``url`` entry only addresses ``SKILL.md``; its supporting files are
    sibling resources. When the server advertises the ``directoryRead`` capability,
    walk the skill root and materialize each child into ``install_dir``. Best-effort:
    a failure leaves the (valid) single-file skill in place.
    """
    capabilities = await aggregator.get_capabilities(skill.server_name)
    if not server_supports_directory_read(capabilities):
        return
    root_uri = _skill_root_uri(skill.source_url)
    if root_uri is None:
        return
    try:
        await _walk_skill_directory(
            aggregator,
            server_name=skill.server_name,
            root_uri=root_uri,
            dir_uri=root_uri,
            install_dir=install_dir,
            budget=_ByteBudget(MAX_UNPACKED_ARCHIVE_BYTES),
        )
    except Exception as exc:  # noqa: BLE001 - supporting files are best-effort.
        logger.warning(
            "Failed to materialize MCP skill supporting files",
            data={"server": skill.server_name, "skill": skill.name, "error": str(exc)},
        )


class _ByteBudget:
    def __init__(self, limit: int) -> None:
        self._limit = limit
        self._used = 0

    def add(self, size: int) -> None:
        self._used += size
        if self._used > self._limit:
            raise ValueError("MCP skill supporting files exceed unpacked size limit")


async def _walk_skill_directory(
    aggregator: McpSkillInstallClient,
    *,
    server_name: str,
    root_uri: str,
    dir_uri: str,
    install_dir: Path,
    budget: _ByteBudget,
) -> None:
    cursor: str | None = None
    while True:
        listing = await aggregator.read_directory(dir_uri, server_name=server_name, cursor=cursor)
        for resource in listing.resources:
            child_uri = str(resource.uri)
            relative = _relative_uri_path(root_uri, child_uri)
            if relative is None:
                continue
            if resource.mimeType == DIRECTORY_MIME_TYPE:
                await _walk_skill_directory(
                    aggregator,
                    server_name=server_name,
                    root_uri=root_uri,
                    dir_uri=child_uri,
                    install_dir=install_dir,
                    budget=budget,
                )
                continue
            if relative == "SKILL.md":
                continue  # already written from the direct entry
            _validate_archive_name(relative)
            content = await aggregator.get_resource(child_uri, server_name=server_name)
            data = _first_bytes(content)
            if data is None:
                continue
            budget.add(len(data))
            destination = install_dir / PurePosixPath(relative)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
        cursor = listing.nextCursor
        if not cursor:
            break


def _skill_root_uri(source_url: str) -> str | None:
    suffix = "/SKILL.md"
    if source_url.endswith(suffix):
        return source_url[: -len(suffix)]
    return None


def _relative_uri_path(root_uri: str, child_uri: str) -> str | None:
    prefix = root_uri.rstrip("/") + "/"
    if not child_uri.startswith(prefix):
        return None
    relative = child_uri[len(prefix) :]
    return relative or None


def _extract_tar_safely(artifact: bytes, destination: Path) -> None:
    total_size = 0
    with tarfile.open(fileobj=io.BytesIO(artifact), mode="r:*") as archive:
        for member in archive.getmembers():
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
        for info in archive.infolist():
            _validate_archive_name(info.filename)
            mode = info.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise ValueError("MCP skill archives must not contain links")
            total_size += info.file_size
            if total_size > MAX_UNPACKED_ARCHIVE_BYTES:
                raise ValueError("MCP skill archive unpacked size exceeds limit")
        archive.extractall(destination)


def _validate_archive_name(name: str) -> None:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
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
