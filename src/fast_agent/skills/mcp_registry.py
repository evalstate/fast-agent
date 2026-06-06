"""SEP-2640 Skills over MCP registry support.

This module intentionally implements the registry/install/update portion of the
draft SEP: servers that advertise ``io.modelcontextprotocol/skills`` can be
used as a source for installing SHA256-verified ``skill-md`` or archive entries
into the normal managed skills directory.
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
from dataclasses import dataclass
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
    from mcp.types import ReadResourceResult


class McpSkillRegistryClient(Protocol):
    async def get_capabilities(self, server_name: str) -> ServerCapabilities | None: ...

    async def get_resource(
        self, resource_uri: str, *, server_name: str | None = None
    ) -> "ReadResourceResult": ...


class McpSkillRegistryStatusClient(McpSkillRegistryClient, Protocol):
    async def collect_server_status(self) -> Mapping[str, Any]: ...


logger = get_logger(__name__)

SKILLS_EXTENSION = "io.modelcontextprotocol/skills"
INDEX_URI = "skill://index.json"
MAX_INDEX_BYTES = 1_048_576
MAX_SKILL_MD_BYTES = 262_144
MAX_ARCHIVE_BYTES = 10 * 1_048_576
MAX_UNPACKED_ARCHIVE_BYTES = 50 * 1_048_576
ArtifactType = Literal["skill-md", "archive"]
SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


@dataclass(frozen=True)
class McpRegistrySkill:
    name: str
    description: str | None
    source_url: str
    server_name: str
    digest: str
    artifact_type: ArtifactType = "skill-md"
    server_version: str | None = None

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


def server_supports_mcp_skills(capabilities: ServerCapabilities | None) -> bool:
    if capabilities is None:
        return False
    extras = capabilities.model_extra or {}
    extensions = extras.get("extensions")
    if not isinstance(extensions, Mapping):
        return False
    return SKILLS_EXTENSION in extensions


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
        entry_type = entry.get("type")
        if entry_type not in {"skill-md", "archive"}:
            logger.warning(
                "Skipping unsupported MCP skill entry type",
                data={"server": server_name, "type": entry_type},
            )
            continue
        name = entry.get("name")
        url = entry.get("url")
        digest = entry.get("digest")
        description = entry.get("description")
        if not isinstance(name, str) or not name.strip():
            logger.warning("MCP skill entry missing name", data={"server": server_name})
            continue
        if not isinstance(url, str) or not url.strip():
            logger.warning("MCP skill entry missing url", data={"server": server_name})
            continue
        source_url = url.strip()
        if source_url.lower().startswith("file://"):
            logger.warning(
                "Rejecting file:// MCP skill URL",
                data={"server": server_name, "url": source_url},
            )
            continue
        if not isinstance(digest, str) or not _is_valid_sha256_digest(digest):
            logger.warning(
                "MCP skill entry missing valid sha256 digest",
                data={"server": server_name, "name": name},
            )
            continue
        artifact_type: ArtifactType = "archive" if entry_type == "archive" else "skill-md"
        skills.append(
            McpRegistrySkill(
                name=name.strip(),
                description=description.strip() if isinstance(description, str) else None,
                source_url=_resolve_entry_url(source_url),
                server_name=server_name,
                digest=digest.strip(),
                artifact_type=artifact_type,
                server_version=server_version,
            )
        )
    return McpSkillRegistry(server_name=server_name, server_version=server_version, skills=skills)


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
    aggregator: McpSkillRegistryClient,
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
    aggregator: McpSkillRegistryClient,
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
    aggregator: McpSkillRegistryClient,
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
        if skill.source_url.lower().endswith(".zip"):
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
