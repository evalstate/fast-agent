from __future__ import annotations

import base64
import json
import tarfile
from hashlib import sha256
from io import BytesIO

import pytest
from mcp.types import (
    BlobResourceContents,
    ListResourcesResult,
    ReadResourceResult,
    Resource,
    ServerCapabilities,
    TextResourceContents,
)
from pydantic import AnyUrl

from fast_agent.skills.mcp_registry import (
    INDEX_URI,
    MAX_WALK_PAGES,
    McpRegistrySkill,
    install_mcp_registry_skill,
    scan_mcp_skill_registry,
    server_supports_directory_read,
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
        directories: dict[str, list[Resource]] | None = None,
    ) -> None:
        self.capabilities = capabilities
        self.responses = responses
        self.directories = directories or {}

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

    async def read_directory(
        self, uri: str, *, server_name: str | None = None, cursor: str | None = None
    ) -> ListResourcesResult:
        del server_name, cursor
        return ListResourcesResult(resources=self.directories.get(uri, []))


def _skills_capabilities(*, directory_read: bool = False) -> ServerCapabilities:
    settings: dict[str, object] = {"directoryRead": True} if directory_read else {}
    return ServerCapabilities.model_validate(
        {"extensions": {"io.modelcontextprotocol/skills": settings}}
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


def test_server_supports_directory_read() -> None:
    assert server_supports_directory_read(_skills_capabilities(directory_read=True))
    # An empty extension object means supported but without directoryRead.
    assert not server_supports_directory_read(_skills_capabilities(directory_read=False))
    assert not server_supports_directory_read(ServerCapabilities())


@pytest.mark.asyncio
async def test_scan_mcp_skill_registry_requires_capability() -> None:
    index = json.dumps(
        {
            "skills": [
                {
                    "frontmatter": {"name": "demo", "description": "Demo skill"},
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
                    "frontmatter": {"name": "demo", "description": "Demo skill"},
                    "url": "skill://demo/SKILL.md",
                    "digest": _digest(skill_text),
                },
                {
                    "frontmatter": {"name": "bundle", "description": "Bundled skill"},
                    "archives": [
                        {
                            "url": "bundle/bundle.tar.gz",
                            "mimeType": "application/gzip",
                            "digest": _digest(archive),
                        }
                    ],
                },
                {
                    # No url and no archives -> not installable, skipped.
                    "frontmatter": {"name": "broken", "description": "Missing artifacts"},
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
    assert registry.skills[0].artifact_type == "skill-md"
    assert registry.skills[0].source_url == "skill://demo/SKILL.md"
    assert registry.skills[0].digest == _digest(skill_text)
    assert registry.skills[0].description == "Demo skill"
    assert registry.skills[0].frontmatter == {"name": "demo", "description": "Demo skill"}
    assert registry.skills[1].artifact_type == "archive"
    assert registry.skills[1].source_url == "skill://bundle/bundle.tar.gz"
    assert registry.skills[1].artifact_mime_type == "application/gzip"


@pytest.mark.asyncio
async def test_scan_prefers_archive_when_entry_has_both() -> None:
    skill_text = "---\nname: demo\ndescription: Demo skill\n---\nBody\n"
    archive = _tar_gz({"SKILL.md": skill_text.encode("utf-8")})
    index = json.dumps(
        {
            "skills": [
                {
                    "frontmatter": {"name": "demo", "description": "Demo skill"},
                    "url": "skill://demo/SKILL.md",
                    "digest": _digest(skill_text),
                    "archives": [
                        {
                            "url": "skill://demo.tar.gz",
                            "mimeType": "application/gzip",
                            "digest": _digest(archive),
                        }
                    ],
                }
            ]
        }
    )
    aggregator = _Aggregator(capabilities=_skills_capabilities(), responses={INDEX_URI: index})

    registry = await scan_mcp_skill_registry(aggregator, "hf")

    assert registry is not None
    assert registry.skills[0].artifact_type == "archive"
    assert registry.skills[0].source_url == "skill://demo.tar.gz"


@pytest.mark.asyncio
async def test_scan_ignores_unsupported_archive_media_type() -> None:
    index = json.dumps(
        {
            "skills": [
                {
                    "frontmatter": {"name": "demo", "description": "Demo skill"},
                    "archives": [
                        {
                            "url": "skill://demo.7z",
                            "mimeType": "application/x-7z-compressed",
                            "digest": "sha256:" + "0" * 64,
                        }
                    ],
                }
            ]
        }
    )
    aggregator = _Aggregator(capabilities=_skills_capabilities(), responses={INDEX_URI: index})

    registry = await scan_mcp_skill_registry(aggregator, "hf")

    assert registry is not None
    assert registry.skills == []


@pytest.mark.asyncio
async def test_scan_mcp_skill_registry_rejects_padded_file_urls() -> None:
    index = json.dumps(
        {
            "skills": [
                {
                    "frontmatter": {"name": "local", "description": "Local skill"},
                    "url": " file:///tmp/SKILL.md",
                    "digest": "sha256:" + "0" * 64,
                }
            ]
        }
    )
    aggregator = _Aggregator(capabilities=_skills_capabilities(), responses={INDEX_URI: index})

    registry = await scan_mcp_skill_registry(aggregator, "hf")

    assert registry is not None
    assert registry.skills == []


@pytest.mark.asyncio
async def test_scan_mcp_skill_registry_skips_entries_without_sha256() -> None:
    index = json.dumps(
        {
            "skills": [
                {
                    "frontmatter": {"name": "demo", "description": "Demo skill"},
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
    read_result = read_installed_skill_source(install_dir)
    assert read_result.error is None
    source = read_result.source
    assert source is not None
    assert source.installed_via == "mcp"
    assert source.source_origin == "mcp"
    assert source.mcp_server_name == "hf"
    assert source.mcp_server_version == "1.2.3"
    assert source.source_url == "skill://demo/SKILL.md"
    assert source.artifact_digest == _digest(skill_text)
    assert source.artifact_type == "skill-md"


@pytest.mark.asyncio
async def test_install_direct_skill_materializes_supporting_files(tmp_path) -> None:
    skill_text = "---\nname: demo\ndescription: Demo skill\n---\nSee references/GUIDE.md\n"
    guide_text = "# Guide\nbody\n"
    skill = McpRegistrySkill(
        name="demo",
        description="Demo skill",
        source_url="skill://demo/SKILL.md",
        server_name="hf",
        digest=_digest(skill_text),
    )
    aggregator = _Aggregator(
        capabilities=_skills_capabilities(directory_read=True),
        responses={
            "skill://demo/SKILL.md": skill_text,
            "skill://demo/references/GUIDE.md": guide_text,
        },
        directories={
            "skill://demo": [
                Resource(uri=AnyUrl("skill://demo/SKILL.md"), name="SKILL.md"),
                Resource(
                    uri=AnyUrl("skill://demo/references"),
                    name="references",
                    mimeType="inode/directory",
                ),
            ],
            "skill://demo/references": [
                Resource(uri=AnyUrl("skill://demo/references/GUIDE.md"), name="GUIDE.md"),
            ],
        },
    )

    install_dir = await install_mcp_registry_skill(aggregator, skill, destination_root=tmp_path)

    assert (install_dir / "SKILL.md").read_text(encoding="utf-8") == skill_text
    assert (install_dir / "references" / "GUIDE.md").read_text(encoding="utf-8") == guide_text


@pytest.mark.asyncio
async def test_install_direct_skill_bounds_infinite_pagination(tmp_path) -> None:
    """A server returning a never-terminating cursor must not hang the install."""
    skill_text = "---\nname: demo\ndescription: Demo skill\n---\nBody\n"
    skill = McpRegistrySkill(
        name="demo",
        description="Demo skill",
        source_url="skill://demo/SKILL.md",
        server_name="hf",
        digest=_digest(skill_text),
    )

    class _InfinitePagerAggregator(_Aggregator):
        def __init__(self) -> None:
            super().__init__(
                capabilities=_skills_capabilities(directory_read=True),
                responses={"skill://demo/SKILL.md": skill_text},
            )
            self.calls = 0

        async def read_directory(self, uri, *, server_name=None, cursor=None):
            del uri, server_name, cursor
            self.calls += 1
            return ListResourcesResult(resources=[], nextCursor="more")

    aggregator = _InfinitePagerAggregator()

    install_dir = await install_mcp_registry_skill(aggregator, skill, destination_root=tmp_path)

    # Walk is best-effort: it aborts at the page cap and leaves the valid skill.
    assert (install_dir / "SKILL.md").read_text(encoding="utf-8") == skill_text
    assert aggregator.calls <= MAX_WALK_PAGES + 1


@pytest.mark.asyncio
async def test_install_direct_skill_skips_walk_without_capability(tmp_path) -> None:
    skill_text = "---\nname: demo\ndescription: Demo skill\n---\nBody\n"
    skill = McpRegistrySkill(
        name="demo",
        description="Demo skill",
        source_url="skill://demo/SKILL.md",
        server_name="hf",
        digest=_digest(skill_text),
    )
    aggregator = _Aggregator(
        capabilities=_skills_capabilities(directory_read=False),
        responses={"skill://demo/SKILL.md": skill_text},
        directories={
            "skill://demo": [
                Resource(uri=AnyUrl("skill://demo/extra.md"), name="extra.md"),
            ]
        },
    )

    install_dir = await install_mcp_registry_skill(aggregator, skill, destination_root=tmp_path)

    assert (install_dir / "SKILL.md").read_text(encoding="utf-8") == skill_text
    assert not (install_dir / "extra.md").exists()


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
        artifact_mime_type="application/gzip",
    )
    aggregator = _Aggregator(
        capabilities=_skills_capabilities(),
        responses={"skill://demo/demo.tar.gz": artifact},
    )

    install_dir = await install_mcp_registry_skill(aggregator, skill, destination_root=tmp_path)

    assert (install_dir / "SKILL.md").read_text(encoding="utf-8") == skill_text
    assert (install_dir / "scripts" / "run.sh").read_text(encoding="utf-8") == "echo ok\n"
