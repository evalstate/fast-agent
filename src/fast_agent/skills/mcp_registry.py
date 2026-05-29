"""SEP-2640 Skills over MCP registry support.

This module intentionally implements only the registry/install part of the
draft SEP: servers that advertise ``io.modelcontextprotocol/skills`` can be
used as a source for installing concrete ``skill-md`` entries into the normal
managed skills directory.
"""

from __future__ import annotations

import asyncio
import json
import re
import shutil
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Protocol

from mcp.types import ServerCapabilities, TextResourceContents

from fast_agent.core.logging.logger import get_logger
from fast_agent.skills.provenance import (
    build_mcp_installed_skill_source,
    compute_skill_content_fingerprint,
    write_installed_skill_source,
)
from fast_agent.skills.registry import SkillRegistry

if TYPE_CHECKING:
    from pathlib import Path

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


@dataclass(frozen=True)
class McpRegistrySkill:
    name: str
    description: str | None
    source_url: str
    server_name: str
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
        if entry.get("type") != "skill-md":
            continue
        name = entry.get("name")
        url = entry.get("url")
        description = entry.get("description")
        if not isinstance(name, str) or not name.strip():
            logger.warning("MCP skill entry missing name", data={"server": server_name})
            continue
        if not isinstance(url, str) or not url.strip():
            logger.warning("MCP skill entry missing url", data={"server": server_name})
            continue
        if url.lower().startswith("file://"):
            logger.warning(
                "Rejecting file:// MCP skill URL",
                data={"server": server_name, "url": url},
            )
            continue
        skills.append(
            McpRegistrySkill(
                name=name.strip(),
                description=description.strip() if isinstance(description, str) else None,
                source_url=url.strip(),
                server_name=server_name,
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
    result = await aggregator.get_resource(skill.source_url, server_name=skill.server_name)
    skill_text = _first_text(result)
    if not skill_text:
        raise ValueError(f"MCP skill resource returned no text content: {skill.source_url}")
    if len(skill_text.encode("utf-8")) > MAX_SKILL_MD_BYTES:
        raise ValueError(f"MCP skill SKILL.md exceeds size limit: {skill.source_url}")

    manifest, parse_error = SkillRegistry.parse_manifest_text(skill_text)
    if manifest is None:
        raise ValueError(f"Failed to parse MCP skill manifest: {parse_error}")
    if manifest.name != skill.name:
        raise ValueError(
            f"MCP skill index name '{skill.name}' does not match manifest name '{manifest.name}'"
        )

    install_dir = destination_root.resolve() / _safe_install_dir_name(skill.name)
    if install_dir.exists():
        raise FileExistsError(f"Skill already exists: {install_dir}")

    try:
        install_dir.mkdir(parents=True, exist_ok=False)
        (install_dir / "SKILL.md").write_text(skill_text, encoding="utf-8")
        fingerprint = compute_skill_content_fingerprint(install_dir)
        write_installed_skill_source(
            install_dir,
            build_mcp_installed_skill_source(
                server_name=skill.server_name,
                server_version=skill.server_version,
                skill_uri=skill.source_url,
                fingerprint=fingerprint,
            ),
        )
    except Exception:
        if install_dir.exists():
            shutil.rmtree(install_dir)
        raise
    return install_dir


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


async def _server_version(
    aggregator: McpSkillRegistryStatusClient, server_name: str
) -> str | None:
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


def _safe_install_dir_name(name: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        raise ValueError(f"Invalid MCP skill name for local install: {name}")
    return name


def scan_mcp_skill_registry_sync(
    aggregator: McpSkillRegistryClient, server_name: str
) -> McpSkillRegistry | None:
    return asyncio.run(scan_mcp_skill_registry(aggregator, server_name))
