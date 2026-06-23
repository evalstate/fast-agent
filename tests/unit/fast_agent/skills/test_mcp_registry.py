from __future__ import annotations

import base64
import json
import tarfile
from hashlib import sha256
from io import BytesIO

import frontmatter
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

from fast_agent.skills import mcp_registry
from fast_agent.skills.mcp_registry import (
    INDEX_URI,
    MAX_WALK_PAGES,
    McpRegistrySkill,
    _validate_archive_name,
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
        frontmatter={"name": "demo", "description": "Demo skill"},
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
        frontmatter={"name": "demo", "description": "Demo skill"},
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
async def test_install_direct_skill_walks_lowercase_skill_md_url(tmp_path) -> None:
    """A url addressing the manifest as '/skill.md' still triggers the directory walk.

    The skill root is derived by stripping the manifest suffix case-insensitively;
    a case-sensitive check would silently disable supporting-file materialization.
    """
    skill_text = "---\nname: demo\ndescription: Demo skill\n---\nSee GUIDE.md\n"
    guide_text = "# Guide\n"
    skill = McpRegistrySkill(
        name="demo",
        description="Demo skill",
        source_url="skill://demo/skill.md",
        server_name="hf",
        digest=_digest(skill_text),
        frontmatter={"name": "demo", "description": "Demo skill"},
    )
    aggregator = _Aggregator(
        capabilities=_skills_capabilities(directory_read=True),
        responses={
            "skill://demo/skill.md": skill_text,
            "skill://demo/GUIDE.md": guide_text,
        },
        directories={
            "skill://demo": [
                Resource(uri=AnyUrl("skill://demo/GUIDE.md"), name="GUIDE.md"),
            ],
        },
    )

    install_dir = await install_mcp_registry_skill(aggregator, skill, destination_root=tmp_path)

    assert (install_dir / "SKILL.md").read_text(encoding="utf-8") == skill_text
    assert (install_dir / "GUIDE.md").read_text(encoding="utf-8") == guide_text


@pytest.mark.asyncio
async def test_install_direct_skill_rejects_oversized_supporting_file(
    tmp_path, monkeypatch
) -> None:
    """A supporting file over the per-resource cap aborts the (best-effort) walk."""
    monkeypatch.setattr(mcp_registry, "MAX_SUPPORTING_FILE_BYTES", 16)
    skill_text = "---\nname: demo\ndescription: Demo skill\n---\nBody\n"
    skill = McpRegistrySkill(
        name="demo",
        description="Demo skill",
        source_url="skill://demo/SKILL.md",
        server_name="hf",
        digest=_digest(skill_text),
        frontmatter={"name": "demo", "description": "Demo skill"},
    )
    aggregator = _Aggregator(
        capabilities=_skills_capabilities(directory_read=True),
        responses={
            "skill://demo/SKILL.md": skill_text,
            "skill://demo/BIG.md": "x" * 1024,  # exceeds the patched 16-byte cap
        },
        directories={
            "skill://demo": [
                Resource(uri=AnyUrl("skill://demo/BIG.md"), name="BIG.md"),
            ],
        },
    )

    install_dir = await install_mcp_registry_skill(aggregator, skill, destination_root=tmp_path)

    # Verified single-file skill survives; the oversized sibling is not written.
    assert (install_dir / "SKILL.md").read_text(encoding="utf-8") == skill_text
    assert not (install_dir / "BIG.md").exists()


def test_validate_archive_name_rejects_windows_traversal() -> None:
    # Forward-slash traversal and POSIX absolute paths (pre-existing guard).
    for bad in ("../evil.md", "/etc/passwd", "a/../../b"):
        with pytest.raises(ValueError):
            _validate_archive_name(bad)
    # Windows separators and drive anchors: a PurePosixPath-only check would
    # let these escape install_dir when later joined with the OS path API.
    for bad in ("..\\..\\evil.md", "sub\\evil.md", "C:\\Windows\\evil", "C:/Windows/evil"):
        with pytest.raises(ValueError):
            _validate_archive_name(bad)
    # Legitimate nested names still pass.
    _validate_archive_name("references/GUIDE.md")
    _validate_archive_name("a/b/c.txt")


@pytest.mark.asyncio
async def test_install_direct_skill_does_not_overwrite_verified_skill_md(tmp_path) -> None:
    """A sibling named 'skill.md' must not clobber the digest-verified SKILL.md.

    On case-insensitive filesystems (Windows, macOS) an unverified sibling whose
    name differs only in case would otherwise overwrite the verified SKILL.md.
    """
    skill_text = "---\nname: demo\ndescription: Demo skill\n---\nVerified body\n"
    skill = McpRegistrySkill(
        name="demo",
        description="Demo skill",
        source_url="skill://demo/SKILL.md",
        server_name="hf",
        digest=_digest(skill_text),
        frontmatter={"name": "demo", "description": "Demo skill"},
    )
    aggregator = _Aggregator(
        capabilities=_skills_capabilities(directory_read=True),
        responses={
            "skill://demo/SKILL.md": skill_text,
            "skill://demo/skill.md": "UNVERIFIED OVERWRITE",
        },
        directories={
            "skill://demo": [
                Resource(uri=AnyUrl("skill://demo/SKILL.md"), name="SKILL.md"),
                Resource(uri=AnyUrl("skill://demo/skill.md"), name="skill.md"),
            ],
        },
    )

    install_dir = await install_mcp_registry_skill(aggregator, skill, destination_root=tmp_path)

    assert (install_dir / "SKILL.md").read_text(encoding="utf-8") == skill_text


@pytest.mark.asyncio
async def test_install_direct_skill_rolls_back_partial_supporting_files(tmp_path) -> None:
    """A mid-walk failure leaves only the verified SKILL.md, not a half-written tree."""
    skill_text = "---\nname: demo\ndescription: Demo skill\n---\nBody\n"
    skill = McpRegistrySkill(
        name="demo",
        description="Demo skill",
        source_url="skill://demo/SKILL.md",
        server_name="hf",
        digest=_digest(skill_text),
        frontmatter={"name": "demo", "description": "Demo skill"},
    )
    aggregator = _Aggregator(
        capabilities=_skills_capabilities(directory_read=True),
        responses={
            "skill://demo/SKILL.md": skill_text,
            "skill://demo/GUIDE.md": "# Guide\n",
        },
        directories={
            # GUIDE.md is staged first, then the unsafe backslash sibling raises
            # mid-walk (validated by _validate_archive_name).
            "skill://demo": [
                Resource(uri=AnyUrl("skill://demo/GUIDE.md"), name="GUIDE.md"),
                Resource(uri=AnyUrl(r"skill://demo/..\..\evil.md"), name="evil.md"),
            ],
        },
    )

    install_dir = await install_mcp_registry_skill(aggregator, skill, destination_root=tmp_path)

    assert (install_dir / "SKILL.md").read_text(encoding="utf-8") == skill_text
    # The walk failed after staging GUIDE.md, so nothing is merged in.
    assert not (install_dir / "GUIDE.md").exists()


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
        frontmatter={"name": "demo", "description": "Demo skill"},
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
        frontmatter={"name": "demo", "description": "Demo skill"},
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
        frontmatter={"name": "demo", "description": "Demo skill"},
    )
    aggregator = _Aggregator(
        capabilities=_skills_capabilities(),
        responses={"skill://demo/demo.tar.gz": artifact},
    )

    install_dir = await install_mcp_registry_skill(aggregator, skill, destination_root=tmp_path)

    assert (install_dir / "SKILL.md").read_text(encoding="utf-8") == skill_text
    assert (install_dir / "scripts" / "run.sh").read_text(encoding="utf-8") == "echo ok\n"


# --- SEP-2640 install-path hardening guards ---------------------------------


@pytest.mark.asyncio
async def test_archive_rejects_case_fold_collision(tmp_path) -> None:
    """An archive entry that case-folds onto SKILL.md (Skill.md) is rejected before unpack,
    so it cannot silently overwrite the digest-verified manifest on a case-insensitive FS."""
    skill_text = "---\nname: demo\ndescription: Demo skill\n---\nVerified\n"
    artifact = _tar_gz(
        {
            "SKILL.md": skill_text.encode("utf-8"),
            "Skill.md": b"---\nname: demo\ndescription: Overwrite\n---\nUNVERIFIED\n",
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
        frontmatter={"name": "demo", "description": "Demo skill"},
    )
    aggregator = _Aggregator(
        capabilities=_skills_capabilities(),
        responses={"skill://demo/demo.tar.gz": artifact},
    )

    with pytest.raises(ValueError, match="collide"):
        await install_mcp_registry_skill(aggregator, skill, destination_root=tmp_path)

    assert not (tmp_path / "demo").exists()


@pytest.mark.asyncio
async def test_install_rejects_frontmatter_index_mismatch(tmp_path) -> None:
    """The served SKILL.md frontmatter must match the index entry field-by-field; a server
    advertising different frontmatter than it serves is a verification failure."""
    skill_text = "---\nname: demo\ndescription: Honest served description\n---\nBody\n"
    skill = McpRegistrySkill(
        name="demo",
        description="Demo skill",
        source_url="skill://demo/SKILL.md",
        server_name="hf",
        digest=_digest(skill_text),
        frontmatter={
            "name": "demo",
            "description": "Index claims something tamer",
            "allowed-tools": ["Bash"],
        },
    )
    aggregator = _Aggregator(
        capabilities=_skills_capabilities(),
        responses={"skill://demo/SKILL.md": skill_text},
    )

    with pytest.raises(ValueError, match="does not match"):
        await install_mcp_registry_skill(aggregator, skill, destination_root=tmp_path)

    assert not (tmp_path / "demo").exists()


@pytest.mark.asyncio
async def test_install_strips_permission_widening_frontmatter(tmp_path) -> None:
    """allowed-tools / hooks are stripped from an installed MCP-origin SKILL.md so a remote
    server cannot self-grant host permissions; the skill still installs."""
    skill_text = (
        "---\nname: demo\ndescription: Demo skill\n"
        "allowed-tools:\n  - Bash\n  - Write\n---\nBody\n"
    )
    skill = McpRegistrySkill(
        name="demo",
        description="Demo skill",
        source_url="skill://demo/SKILL.md",
        server_name="hf",
        digest=_digest(skill_text),
        frontmatter={
            "name": "demo",
            "description": "Demo skill",
            "allowed-tools": ["Bash", "Write"],
        },
    )
    aggregator = _Aggregator(
        capabilities=_skills_capabilities(),
        responses={"skill://demo/SKILL.md": skill_text},
    )

    install_dir = await install_mcp_registry_skill(aggregator, skill, destination_root=tmp_path)

    installed = frontmatter.loads((install_dir / "SKILL.md").read_text(encoding="utf-8"))
    assert "allowed-tools" not in installed.metadata
    assert installed.metadata["name"] == "demo"


@pytest.mark.asyncio
async def test_install_warns_on_unverified_supporting_files(tmp_path, monkeypatch) -> None:
    """Url-only supporting files are still materialized, but flagged as unverified (no digest
    covers bytes fetched via the directory walk)."""
    messages: list[str] = []
    monkeypatch.setattr(
        mcp_registry.logger,
        "warning",
        lambda message, *args, **kwargs: messages.append(str(message)),
    )
    skill_text = "---\nname: demo\ndescription: Demo skill\n---\nSee references/GUIDE.md\n"
    skill = McpRegistrySkill(
        name="demo",
        description="Demo skill",
        source_url="skill://demo/SKILL.md",
        server_name="hf",
        digest=_digest(skill_text),
        frontmatter={"name": "demo", "description": "Demo skill"},
    )
    aggregator = _Aggregator(
        capabilities=_skills_capabilities(directory_read=True),
        responses={
            "skill://demo/SKILL.md": skill_text,
            "skill://demo/references/GUIDE.md": "# Guide\n",
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

    # Feature preserved: the supporting file is still installed...
    assert (install_dir / "references" / "GUIDE.md").read_text(encoding="utf-8") == "# Guide\n"
    # ...but flagged as unverified.
    assert any("unverified" in message.lower() for message in messages)


@pytest.mark.asyncio
async def test_archive_enforces_cumulative_per_server_budget(tmp_path, monkeypatch) -> None:
    """Each archive fits the per-archive cap, but a second archive from the same server that
    pushes the server's cumulative on-disk total over budget is rejected."""

    def _part(name: str) -> tuple[bytes, str]:
        text = f"---\nname: {name}\ndescription: Budget part\n---\n" + "padding " * 200 + "\n"
        return _tar_gz({"SKILL.md": text.encode("utf-8")}), text

    artifact_a, text_a = _part("budget-a")
    artifact_b, _ = _part("budget-b")
    unpacked = len(text_a.encode("utf-8"))
    # Room for one part's unpacked bytes plus headroom, but not two.
    monkeypatch.setattr(mcp_registry, "MAX_SERVER_UNPACKED_BYTES", unpacked + unpacked // 2)

    def _part_skill(name: str, artifact: bytes) -> McpRegistrySkill:
        return McpRegistrySkill(
            name=name,
            description="Budget part",
            source_url=f"skill://{name}.tar.gz",
            server_name="budget-srv",
            digest=_digest(artifact),
            artifact_type="archive",
            artifact_mime_type="application/gzip",
            frontmatter={"name": name, "description": "Budget part"},
        )

    aggregator = _Aggregator(
        capabilities=_skills_capabilities(),
        responses={
            "skill://budget-a.tar.gz": artifact_a,
            "skill://budget-b.tar.gz": artifact_b,
        },
    )

    await install_mcp_registry_skill(
        aggregator, _part_skill("budget-a", artifact_a), destination_root=tmp_path
    )
    with pytest.raises(ValueError, match="cumulative"):
        await install_mcp_registry_skill(
            aggregator, _part_skill("budget-b", artifact_b), destination_root=tmp_path
        )
    assert not (tmp_path / "budget-b").exists()


@pytest.mark.asyncio
async def test_rolled_back_install_frees_cumulative_budget(tmp_path, monkeypatch) -> None:
    """A failed install (extract succeeds, a post-extract check fails and rolls back) must not
    charge the per-server budget — a later legitimate install still fits. The accumulator this
    replaced leaked the rolled-back bytes and would reject the good install with 'cumulative'."""
    good_text = "---\nname: good\ndescription: Good skill\n---\n" + "padding " * 200 + "\n"
    good_artifact = _tar_gz({"SKILL.md": good_text.encode("utf-8")})
    unpacked = len(good_text.encode("utf-8"))
    monkeypatch.setattr(mcp_registry, "MAX_SERVER_UNPACKED_BYTES", unpacked + unpacked // 2)

    # Bad skill: archive extracts cleanly but its served frontmatter diverges from the index,
    # so _write_archive_artifact rolls the directory back after extraction has been charged.
    bad_text = "---\nname: bad\ndescription: Served description\n---\n" + "padding " * 200 + "\n"
    bad_artifact = _tar_gz({"SKILL.md": bad_text.encode("utf-8")})
    bad_skill = McpRegistrySkill(
        name="bad",
        description="Bad skill",
        source_url="skill://bad.tar.gz",
        server_name="srv",
        digest=_digest(bad_artifact),
        artifact_type="archive",
        artifact_mime_type="application/gzip",
        frontmatter={"name": "bad", "description": "Index claims otherwise"},
    )
    good_skill = McpRegistrySkill(
        name="good",
        description="Good skill",
        source_url="skill://good.tar.gz",
        server_name="srv",
        digest=_digest(good_artifact),
        artifact_type="archive",
        artifact_mime_type="application/gzip",
        frontmatter={"name": "good", "description": "Good skill"},
    )
    aggregator = _Aggregator(
        capabilities=_skills_capabilities(),
        responses={
            "skill://bad.tar.gz": bad_artifact,
            "skill://good.tar.gz": good_artifact,
        },
    )

    with pytest.raises(ValueError, match="does not match"):
        await install_mcp_registry_skill(aggregator, bad_skill, destination_root=tmp_path)
    assert not (tmp_path / "bad").exists()

    install_dir = await install_mcp_registry_skill(
        aggregator, good_skill, destination_root=tmp_path
    )
    assert (install_dir / "SKILL.md").read_text(encoding="utf-8") == good_text


@pytest.mark.asyncio
async def test_supporting_files_count_against_server_budget(tmp_path, monkeypatch) -> None:
    """Walk-fetched supporting files share the cumulative per-server budget with archives:
    once a prior install has consumed the cap, a further supporting file is refused (the walk
    is best-effort), though the digest-verified SKILL.md still installs."""
    base_text = "---\nname: base\ndescription: Base\n---\n" + "padding " * 200 + "\n"
    base_artifact = _tar_gz({"SKILL.md": base_text.encode("utf-8")})
    base_unpacked = len(base_text.encode("utf-8"))
    # Budget leaves only a few dozen bytes of headroom after the base skill installs.
    monkeypatch.setattr(mcp_registry, "MAX_SERVER_UNPACKED_BYTES", base_unpacked + 64)

    base_skill = McpRegistrySkill(
        name="base",
        description="Base",
        source_url="skill://base.tar.gz",
        server_name="srv",
        digest=_digest(base_artifact),
        artifact_type="archive",
        artifact_mime_type="application/gzip",
        frontmatter={"name": "base", "description": "Base"},
    )
    skill_text = "---\nname: demo\ndescription: Demo skill\n---\nSee GUIDE.md\n"
    big_support = "x" * 4096  # well over the remaining headroom, under the per-file cap
    demo_skill = McpRegistrySkill(
        name="demo",
        description="Demo skill",
        source_url="skill://demo/SKILL.md",
        server_name="srv",
        digest=_digest(skill_text),
        frontmatter={"name": "demo", "description": "Demo skill"},
    )
    aggregator = _Aggregator(
        capabilities=_skills_capabilities(directory_read=True),
        responses={
            "skill://base.tar.gz": base_artifact,
            "skill://demo/SKILL.md": skill_text,
            "skill://demo/GUIDE.md": big_support,
        },
        directories={
            "skill://demo": [
                Resource(uri=AnyUrl("skill://demo/GUIDE.md"), name="GUIDE.md"),
            ],
        },
    )

    await install_mcp_registry_skill(aggregator, base_skill, destination_root=tmp_path)
    install_dir = await install_mcp_registry_skill(
        aggregator, demo_skill, destination_root=tmp_path
    )

    # SKILL.md is digest-verified and always installs; the supporting file would breach the
    # per-server budget the base skill already consumed, so the best-effort walk drops it.
    assert (install_dir / "SKILL.md").read_text(encoding="utf-8") == skill_text
    assert not (install_dir / "GUIDE.md").exists()


@pytest.mark.asyncio
async def test_install_accepts_yaml_date_frontmatter(tmp_path) -> None:
    """A SKILL.md whose YAML frontmatter carries a date scalar matches the index's JSON string
    rendering of the same value; YAML->JSON type coercion must not cause a false rejection."""
    skill_text = "---\nname: demo\ndescription: Demo skill\nupdated: 2024-01-15\n---\nBody\n"
    skill = McpRegistrySkill(
        name="demo",
        description="Demo skill",
        source_url="skill://demo/SKILL.md",
        server_name="hf",
        digest=_digest(skill_text),
        frontmatter={"name": "demo", "description": "Demo skill", "updated": "2024-01-15"},
    )
    aggregator = _Aggregator(
        capabilities=_skills_capabilities(),
        responses={"skill://demo/SKILL.md": skill_text},
    )

    install_dir = await install_mcp_registry_skill(aggregator, skill, destination_root=tmp_path)

    assert (install_dir / "SKILL.md").read_text(encoding="utf-8") == skill_text


@pytest.mark.asyncio
async def test_archive_allows_identical_duplicate_entries(tmp_path) -> None:
    """Byte-identical duplicate archive entries (some tar invocations emit a directory and a
    file within it, or the same path twice) must not be mistaken for a case-fold collision."""
    skill_text = "---\nname: demo\ndescription: Demo skill\n---\nBody\n"
    run_sh = b"echo ok\n"
    buffer = BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, content in (
            ("SKILL.md", skill_text.encode("utf-8")),
            ("scripts/run.sh", run_sh),
            ("scripts/run.sh", run_sh),  # duplicate: identical path and content
        ):
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, BytesIO(content))
    artifact = buffer.getvalue()
    skill = McpRegistrySkill(
        name="demo",
        description="Demo skill",
        source_url="skill://demo/demo.tar.gz",
        server_name="hf",
        digest=_digest(artifact),
        artifact_type="archive",
        artifact_mime_type="application/gzip",
        frontmatter={"name": "demo", "description": "Demo skill"},
    )
    aggregator = _Aggregator(
        capabilities=_skills_capabilities(),
        responses={"skill://demo/demo.tar.gz": artifact},
    )

    install_dir = await install_mcp_registry_skill(aggregator, skill, destination_root=tmp_path)

    assert (install_dir / "scripts" / "run.sh").read_bytes() == run_sh
