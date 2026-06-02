from __future__ import annotations

import base64
import json
import tarfile
from hashlib import sha256
from io import BytesIO

import pytest
from mcp.types import (
    BlobResourceContents,
    ReadResourceResult,
    ServerCapabilities,
    TextResourceContents,
)
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


def _blob(uri: str, body: bytes) -> ReadResourceResult:
    return ReadResourceResult(
        contents=[
            BlobResourceContents(
                uri=AnyUrl(uri),
                mimeType="application/octet-stream",
                blob=base64.b64encode(body).decode("ascii"),
            )
        ]
    )


def _digest(body: bytes | str) -> str:
    data = body.encode("utf-8") if isinstance(body, str) else body
    return f"sha256:{sha256(data).hexdigest()}"


class _Aggregator:
    def __init__(
        self,
        *,
        capabilities: ServerCapabilities,
        responses: dict[str, str | bytes],
    ) -> None:
        self.capabilities = capabilities
        self.responses = responses

    async def get_capabilities(self, server_name: str) -> ServerCapabilities:
        del server_name
        return self.capabilities

    async def get_resource(
        self, resource_uri: str, *, server_name: str | None = None
    ) -> ReadResourceResult:
        del server_name
        response = self.responses[resource_uri]
        if isinstance(response, bytes):
            return _blob(resource_uri, response)
        return _text(resource_uri, response)


def _skills_capabilities() -> ServerCapabilities:
    return ServerCapabilities.model_validate(
        {"extensions": {"io.modelcontextprotocol/skills": {}}}
    )


def _tar_gz(files: dict[str, bytes]) -> bytes:
    archive_bytes = BytesIO()
    with tarfile.open(fileobj=archive_bytes, mode="w:gz") as archive:
        for name, content in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, BytesIO(content))
    return archive_bytes.getvalue()


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
                    "digest": "sha256:" + "0" * 64,
                }
            ]
        }
    )
    aggregator = _Aggregator(capabilities=ServerCapabilities(), responses={INDEX_URI: index})

    assert await scan_mcp_skill_registry(aggregator, "demo") is None


@pytest.mark.asyncio
async def test_scan_mcp_skill_registry_reads_verified_skill_entries() -> None:
    skill_text = "---\nname: demo\ndescription: Demo skill\n---\nBody\n"
    archive = _tar_gz({"SKILL.md": skill_text.encode("utf-8")})
    index = json.dumps(
        {
            "skills": [
                {
                    "type": "skill-md",
                    "name": "demo",
                    "description": "Demo skill",
                    "url": "skill://demo/SKILL.md",
                    "digest": _digest(skill_text),
                },
                {
                    "type": "archive",
                    "name": "bundle",
                    "description": "Bundled skill",
                    "url": "bundle/bundle.tar.gz",
                    "digest": _digest(archive),
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
    assert [skill.name for skill in registry.skills] == ["demo", "bundle"]
    assert registry.skills[0].source_url == "skill://demo/SKILL.md"
    assert registry.skills[0].digest == _digest(skill_text)
    assert registry.skills[1].artifact_type == "archive"
    assert registry.skills[1].source_url == "skill://bundle/bundle.tar.gz"


@pytest.mark.asyncio
async def test_scan_mcp_skill_registry_skips_entries_without_sha256() -> None:
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
    aggregator = _Aggregator(capabilities=_skills_capabilities(), responses={INDEX_URI: index})

    registry = await scan_mcp_skill_registry(aggregator, "hf")

    assert registry is not None
    assert registry.skills == []


@pytest.mark.asyncio
async def test_install_mcp_registry_skill_writes_provenance(tmp_path) -> None:
    skill_text = "---\nname: demo\ndescription: Demo skill\n---\nBody\n"
    skill = McpRegistrySkill(
        name="demo",
        description="Demo skill",
        source_url="skill://demo/SKILL.md",
        server_name="hf",
        digest=_digest(skill_text),
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
    assert source.artifact_digest == _digest(skill_text)
    assert source.artifact_type == "skill-md"


@pytest.mark.asyncio
async def test_install_mcp_registry_skill_rejects_digest_mismatch(tmp_path) -> None:
    skill_text = "---\nname: demo\ndescription: Demo skill\n---\nBody\n"
    skill = McpRegistrySkill(
        name="demo",
        description="Demo skill",
        source_url="skill://demo/SKILL.md",
        server_name="hf",
        digest="sha256:" + "0" * 64,
    )
    aggregator = _Aggregator(
        capabilities=_skills_capabilities(),
        responses={"skill://demo/SKILL.md": skill_text},
    )

    with pytest.raises(ValueError, match="SHA256 mismatch"):
        await install_mcp_registry_skill(aggregator, skill, destination_root=tmp_path)

    assert not (tmp_path / "demo").exists()


@pytest.mark.asyncio
async def test_install_mcp_registry_archive_skill(tmp_path) -> None:
    skill_text = "---\nname: demo\ndescription: Demo skill\n---\nBody\n"
    artifact = _tar_gz(
        {
            "SKILL.md": skill_text.encode("utf-8"),
            "scripts/run.sh": b"echo ok\n",
        }
    )
    skill = McpRegistrySkill(
        name="demo",
        description="Demo skill",
        source_url="skill://demo/demo.tar.gz",
        server_name="hf",
        digest=_digest(artifact),
        artifact_type="archive",
    )
    aggregator = _Aggregator(
        capabilities=_skills_capabilities(),
        responses={"skill://demo/demo.tar.gz": artifact},
    )

    install_dir = await install_mcp_registry_skill(aggregator, skill, destination_root=tmp_path)

    assert (install_dir / "SKILL.md").read_text(encoding="utf-8") == skill_text
    assert (install_dir / "scripts" / "run.sh").read_text(encoding="utf-8") == "echo ok\n"
