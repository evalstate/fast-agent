from __future__ import annotations

import json

import pytest
from mcp.types import ReadResourceResult, ServerCapabilities, TextResourceContents
from pydantic import AnyUrl

from fast_agent.skills.mcp_registry import (
    INDEX_URI,
    McpRegistrySkill,
    install_mcp_registry_skill,
    scan_mcp_skill_registry,
    server_supports_mcp_skills,
)
from fast_agent.skills.provenance import read_installed_skill_source


def _text(uri: str, body: str) -> ReadResourceResult:
    return ReadResourceResult(
        contents=[TextResourceContents(uri=AnyUrl(uri), mimeType="text/plain", text=body)]
    )


class _Aggregator:
    def __init__(self, *, capabilities: ServerCapabilities, responses: dict[str, str]) -> None:
        self.capabilities = capabilities
        self.responses = responses

    async def get_capabilities(self, server_name: str) -> ServerCapabilities:
        del server_name
        return self.capabilities

    async def get_resource(
        self, resource_uri: str, *, server_name: str | None = None
    ) -> ReadResourceResult:
        del server_name
        return _text(resource_uri, self.responses[resource_uri])


def _skills_capabilities() -> ServerCapabilities:
    return ServerCapabilities.model_validate(
        {"extensions": {"io.modelcontextprotocol/skills": {}}}
    )


def test_server_supports_mcp_skills_from_extensions_extra() -> None:
    assert server_supports_mcp_skills(_skills_capabilities())
    assert not server_supports_mcp_skills(ServerCapabilities())


@pytest.mark.asyncio
async def test_scan_mcp_skill_registry_requires_capability() -> None:
    index = json.dumps(
        {
            "skills": [
                {
                    "type": "skill-md",
                    "name": "demo",
                    "description": "Demo skill",
                    "url": "skill://demo/SKILL.md",
                }
            ]
        }
    )
    aggregator = _Aggregator(capabilities=ServerCapabilities(), responses={INDEX_URI: index})

    assert await scan_mcp_skill_registry(aggregator, "demo") is None


@pytest.mark.asyncio
async def test_scan_mcp_skill_registry_reads_skill_md_entries_only() -> None:
    index = json.dumps(
        {
            "skills": [
                {
                    "type": "skill-md",
                    "name": "demo",
                    "description": "Demo skill",
                    "url": "skill://demo/SKILL.md",
                },
                {
                    "type": "mcp-resource-template",
                    "description": "Template",
                    "url": "skill://docs/{product}/SKILL.md",
                },
            ]
        }
    )
    aggregator = _Aggregator(capabilities=_skills_capabilities(), responses={INDEX_URI: index})

    registry = await scan_mcp_skill_registry(aggregator, "hf", server_version="1.2.3")

    assert registry is not None
    assert registry.server_name == "hf"
    assert registry.server_version == "1.2.3"
    assert [skill.name for skill in registry.skills] == ["demo"]
    assert registry.skills[0].source_url == "skill://demo/SKILL.md"


@pytest.mark.asyncio
async def test_install_mcp_registry_skill_writes_provenance(tmp_path) -> None:
    skill_text = "---\nname: demo\ndescription: Demo skill\n---\nBody\n"
    skill = McpRegistrySkill(
        name="demo",
        description="Demo skill",
        source_url="skill://demo/SKILL.md",
        server_name="hf",
        server_version="1.2.3",
    )
    aggregator = _Aggregator(
        capabilities=_skills_capabilities(),
        responses={"skill://demo/SKILL.md": skill_text},
    )

    install_dir = await install_mcp_registry_skill(
        aggregator,
        skill,
        destination_root=tmp_path,
    )

    assert (install_dir / "SKILL.md").read_text(encoding="utf-8") == skill_text
    source, error = read_installed_skill_source(install_dir)
    assert error is None
    assert source is not None
    assert source.installed_via == "mcp"
    assert source.source_origin == "mcp"
    assert source.mcp_server_name == "hf"
    assert source.mcp_server_version == "1.2.3"
    assert source.source_url == "skill://demo/SKILL.md"
