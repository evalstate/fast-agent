from __future__ import annotations

import base64
from hashlib import sha256

import pytest
from mcp_types import (
    BlobResourceContents,
    ReadResourceResult,
    ServerCapabilities,
    TextResourceContents,
)

from fast_agent.mcp.skills_extension import (
    GetSkillResult,
    ListSkillsResult,
    SkillEntry,
    SkillResource,
)
from fast_agent.skills import mcp_registry
from fast_agent.skills.mcp_registry import (
    McpRegistrySkill,
    McpSkillRegistry,
    get_mcp_registry_skill,
    install_mcp_registry_skill,
    scan_mcp_skill_registry,
    select_mcp_registry_skill,
    update_mcp_registry_skill,
)
from fast_agent.skills.mcp_source import McpSkillSource
from fast_agent.skills.models import McpSkillResource, SkillUpdateInfo
from fast_agent.skills.provenance import (
    build_mcp_installed_skill_source,
    compute_skill_content_fingerprint,
    read_installed_skill_source,
    write_installed_skill_source,
)


def _digest(value: bytes | str) -> str:
    data = value.encode() if isinstance(value, str) else value
    return f"sha256:{sha256(data).hexdigest()}"


def _text(uri: str, value: str) -> ReadResourceResult:
    return ReadResourceResult(
        contents=[TextResourceContents(uri=uri, mime_type="text/plain", text=value)]
    )


def _blob(uri: str, value: bytes) -> ReadResourceResult:
    return ReadResourceResult(
        contents=[
            BlobResourceContents(
                uri=uri,
                mime_type="application/octet-stream",
                blob=base64.b64encode(value).decode(),
            )
        ]
    )


class _SkillsServer:
    def __init__(
        self,
        *,
        pages: dict[str | None, ListSkillsResult] | None = None,
        skills: dict[str, SkillEntry] | None = None,
        resources: dict[str, bytes | str] | None = None,
    ) -> None:
        self.pages = pages or {None: ListSkillsResult(skills=[])}
        self.skills = skills or {}
        self.resources = resources or {}
        self.calls: list[tuple[str, str | None]] = []
        self.resource_cache_modes: list[str] = []

    async def get_capabilities(self, server_name: str) -> ServerCapabilities:
        del server_name
        return ServerCapabilities.model_validate(
            {"extensions": {"io.modelcontextprotocol/skills": {}}}
        )

    async def list_skills(self, server_name: str, cursor: str | None) -> ListSkillsResult:
        del server_name
        self.calls.append(("list", cursor))
        return self.pages[cursor]

    async def get_skill(self, uri: str, server_name: str) -> GetSkillResult:
        del server_name
        self.calls.append(("get", uri))
        return GetSkillResult(skill=self.skills[uri])

    async def get_resource(
        self,
        resource_uri: str,
        *,
        server_name: str | None = None,
        cache_mode: str = "use",
    ) -> ReadResourceResult:
        del server_name
        self.calls.append(("resource", resource_uri))
        self.resource_cache_modes.append(cache_mode)
        value = self.resources[resource_uri]
        return (
            _blob(resource_uri, value) if isinstance(value, bytes) else _text(resource_uri, value)
        )


def _entry(
    name: str,
    content: str,
    *,
    uri: str | None = None,
    extra: dict[str, bytes | str] | None = None,
    frontmatter: dict[str, object] | None = None,
) -> tuple[SkillEntry, dict[str, bytes | str]]:
    uri = uri or f"skill://catalog/{name}/SKILL.md"
    files = {uri: content, **(extra or {})}
    resources = [
        SkillResource(uri=resource_uri, digest=_digest(value))
        for resource_uri, value in files.items()
    ]
    return (
        SkillEntry(
            uri=uri,
            frontmatter=frontmatter or {"name": name, "description": f"{name} description"},
            resources=resources,
        ),
        files,
    )


@pytest.mark.asyncio
async def test_scan_paginates_and_preserves_duplicate_names() -> None:
    first, _ = _entry("same", "---\nname: same\ndescription: same description\n---\nfirst\n")
    second, _ = _entry(
        "same",
        "---\nname: same\ndescription: same description\n---\nsecond\n",
        uri="skill://other/same/SKILL.md",
    )
    server = _SkillsServer(
        pages={
            None: ListSkillsResult(skills=[first], next_cursor="next"),
            "next": ListSkillsResult(skills=[second]),
        }
    )

    registry = await scan_mcp_skill_registry(server, "server")

    assert registry is not None
    assert [skill.uri for skill in registry.skills] == [first.uri, second.uri]
    assert server.calls == [("list", None), ("list", "next")]
    with pytest.raises(LookupError, match="ambiguous"):
        select_mcp_registry_skill(registry.skills, "same")


@pytest.mark.asyncio
async def test_scan_returns_empty_registry_for_repeated_cursor_or_invalid_resource_set() -> None:
    valid, _ = _entry("demo", "---\nname: demo\ndescription: demo description\n---\n")
    malformed = SkillEntry(
        uri=valid.uri,
        frontmatter=valid.frontmatter,
        resources=[SkillResource(uri="skill://catalog/demo/../escape", digest=_digest("no"))],
    )
    for server in (
        _SkillsServer(
            pages={
                None: ListSkillsResult(skills=[valid], next_cursor="again"),
                "again": ListSkillsResult(skills=[], next_cursor="again"),
            }
        ),
        _SkillsServer(pages={None: ListSkillsResult(skills=[malformed])}),
    ):
        registry = await scan_mcp_skill_registry(server, "server")
        assert registry is not None
        assert registry.skills == []


@pytest.mark.asyncio
async def test_get_supports_unlisted_uri_and_install_verifies_complete_resource_set(
    tmp_path,
) -> None:
    body = "---\nname: demo\ndescription: demo description\n---\nUse GUIDE.md\n"
    entry, files = _entry("demo", body, extra={"skill://catalog/demo/GUIDE.md": "guide"})
    server = _SkillsServer(skills={entry.uri: entry}, resources=files)
    registry = McpSkillRegistry(server_name="server", server_version="1", skills=[])
    source = McpSkillSource(aggregator=server, registry=registry)

    result = await source.install_skill(entry.uri, destination_root=tmp_path)
    installed = read_installed_skill_source(result.skill_dir).source

    assert (result.skill_dir / "SKILL.md").read_text() == body
    assert (result.skill_dir / "GUIDE.md").read_text() == "guide"
    assert installed is not None
    assert installed.mcp_resources == tuple(
        McpSkillResource(uri=resource.uri, digest=resource.digest)
        for resource in entry.resources or []
    )
    assert server.calls[0] == ("get", entry.uri)
    assert server.calls[1] == ("get", entry.uri)  # install refreshes immediately before fetch
    assert server.resource_cache_modes == ["refresh", "refresh"]


@pytest.mark.asyncio
async def test_install_rejects_digest_failure_without_partial_tree(tmp_path) -> None:
    body = "---\nname: demo\ndescription: demo description\n---\n"
    entry, files = _entry("demo", body, extra={"skill://catalog/demo/GUIDE.md": "expected"})
    files["skill://catalog/demo/GUIDE.md"] = "tampered"
    server = _SkillsServer(skills={entry.uri: entry}, resources=files)
    skill = await get_mcp_registry_skill(server, entry.uri, "server")

    with pytest.raises(ValueError, match="SHA256 mismatch"):
        await install_mcp_registry_skill(server, skill, destination_root=tmp_path)

    assert not (tmp_path / "demo").exists()


@pytest.mark.asyncio
async def test_install_rejects_wrong_or_ambiguous_resource_content(tmp_path) -> None:
    body = "---\nname: demo\ndescription: demo description\n---\n"
    entry, files = _entry("demo", body)

    class _WrongResourceServer(_SkillsServer):
        async def get_resource(
            self,
            resource_uri: str,
            *,
            server_name: str | None = None,
            cache_mode: str = "use",
        ) -> ReadResourceResult:
            del server_name, cache_mode
            content = files[resource_uri]
            assert isinstance(content, str)
            result = _text("skill://catalog/demo/OTHER.md", content)
            result.contents.append(_text(resource_uri, content).contents[0])
            return result

    server = _WrongResourceServer(skills={entry.uri: entry}, resources=files)
    skill = await get_mcp_registry_skill(server, entry.uri, "server")

    with pytest.raises(ValueError, match="ambiguous"):
        await install_mcp_registry_skill(server, skill, destination_root=tmp_path)

    assert not (tmp_path / "demo").exists()


@pytest.mark.asyncio
async def test_install_rejects_skill_changed_after_selection(tmp_path) -> None:
    old_body = "---\nname: demo\ndescription: demo description\n---\nold\n"
    new_body = "---\nname: demo\ndescription: demo description\n---\nnew\n"
    old_entry, _ = _entry("demo", old_body)
    new_entry, files = _entry("demo", new_body)
    server = _SkillsServer(skills={old_entry.uri: new_entry}, resources=files)
    selected = McpRegistrySkill(
        name="demo",
        description="demo description",
        uri=old_entry.uri,
        server_name="server",
        frontmatter=old_entry.frontmatter,
        resources=tuple(old_entry.resources or []),
    )

    with pytest.raises(ValueError, match="changed since it was selected"):
        await install_mcp_registry_skill(server, selected, destination_root=tmp_path)

    assert not (tmp_path / "demo").exists()


@pytest.mark.asyncio
async def test_update_check_refreshes_an_unlisted_persisted_uri(tmp_path) -> None:
    old_body = "---\nname: demo\ndescription: demo description\n---\nold\n"
    fresh_body = "---\nname: demo\ndescription: demo description\n---\nfresh\n"
    entry, files = _entry("demo", fresh_body)
    skill_dir = tmp_path / "demo"
    skill_dir.mkdir()
    skill_dir.joinpath("SKILL.md").write_text(old_body)
    old_resource = McpSkillResource(uri=entry.uri, digest=_digest(old_body))
    write_installed_skill_source(
        skill_dir,
        build_mcp_installed_skill_source(
            server_name="server",
            server_version=None,
            skill_uri=entry.uri,
            fingerprint=compute_skill_content_fingerprint(skill_dir),
            resources=(old_resource,),
            revision=_digest(old_body),
        ),
    )
    server = _SkillsServer(skills={entry.uri: entry}, resources=files)
    source = McpSkillSource(
        aggregator=server,
        registry=McpSkillRegistry(server_name="server", server_version=None, skills=[]),
    )
    installed = read_installed_skill_source(skill_dir).source
    assert installed is not None

    updates = await source.check_updates(
        [
            SkillUpdateInfo(
                index=1,
                name="demo",
                skill_dir=skill_dir,
                status="up_to_date",
                managed_source=installed,
            )
        ]
    )

    assert updates[0].status == "update_available"
    assert server.calls == [("get", entry.uri)]


@pytest.mark.asyncio
async def test_update_rolls_back_and_resource_revision_is_order_invariant(tmp_path) -> None:
    old = tmp_path / "demo"
    old.mkdir()
    (old / "SKILL.md").write_text("old")
    body = "---\nname: demo\ndescription: demo description\n---\n"
    entry, files = _entry("demo", body, extra={"skill://catalog/demo/GUIDE.md": "expected"})
    reversed_entry = SkillEntry(
        uri=entry.uri,
        frontmatter=entry.frontmatter,
        resources=list(reversed(entry.resources or [])),
    )
    files["skill://catalog/demo/GUIDE.md"] = "tampered"
    server = _SkillsServer(skills={entry.uri: entry}, resources=files)
    skill = await get_mcp_registry_skill(server, entry.uri, "server")
    reversed_skill = McpRegistrySkill(
        name=skill.name,
        description=skill.description,
        uri=skill.uri,
        server_name=skill.server_name,
        frontmatter=skill.frontmatter,
        resources=tuple(reversed_entry.resources or []),
    )

    with pytest.raises(ValueError, match="SHA256 mismatch"):
        await update_mcp_registry_skill(server, skill, skill_dir=old)

    assert (old / "SKILL.md").read_text() == "old"
    assert skill.revision == reversed_skill.revision


@pytest.mark.asyncio
async def test_budget_and_permission_stripping_apply_after_full_verification(
    tmp_path, monkeypatch
) -> None:
    body = "---\nname: demo\ndescription: demo description\nallowed-tools: [shell]\nhooks: [run]\n---\n"
    entry, files = _entry(
        "demo",
        body,
        frontmatter={
            "name": "demo",
            "description": "demo description",
            "allowed-tools": ["shell"],
            "hooks": ["run"],
        },
    )
    server = _SkillsServer(skills={entry.uri: entry}, resources=files)
    skill = await get_mcp_registry_skill(server, entry.uri, "server")
    monkeypatch.setattr(mcp_registry, "MAX_SKILL_BYTES", len(body.encode()) - 1)

    with pytest.raises(ValueError, match="total-size"):
        await install_mcp_registry_skill(server, skill, destination_root=tmp_path)
    assert not (tmp_path / "demo").exists()

    monkeypatch.setattr(mcp_registry, "MAX_SKILL_BYTES", 50 * 1_048_576)
    installed = await install_mcp_registry_skill(server, skill, destination_root=tmp_path)
    metadata = installed.joinpath("SKILL.md").read_text()
    assert "allowed-tools" not in metadata
    assert "hooks" not in metadata


def test_duplicate_names_use_uri_selection_options() -> None:
    body = "---\nname: same\ndescription: same description\n---\n"
    first, _ = _entry("same", body)
    second, _ = _entry("same", body, uri="skill://other/same/SKILL.md")
    skills = [
        McpRegistrySkill(
            name="same",
            description="same description",
            uri=entry.uri,
            server_name="server",
            frontmatter=entry.frontmatter,
            resources=tuple(entry.resources or []),
        )
        for entry in (first, second)
    ]
    source = McpSkillSource(
        aggregator=_SkillsServer(),
        registry=McpSkillRegistry(server_name="server", server_version=None, skills=skills),
    )

    assert source.selection_options(skills) == [first.uri, second.uri]


@pytest.mark.asyncio
async def test_explicit_uri_refreshes_stale_list_entry(tmp_path) -> None:
    old_body = "---\nname: demo\ndescription: demo description\n---\nold\n"
    new_body = "---\nname: demo\ndescription: demo description\n---\nnew\n"
    listed, _ = _entry("demo", old_body)
    current, files = _entry("demo", new_body)
    server = _SkillsServer(skills={current.uri: current}, resources=files)
    source = McpSkillSource(
        aggregator=server,
        registry=McpSkillRegistry(
            server_name="server",
            server_version=None,
            skills=[
                McpRegistrySkill(
                    name="demo",
                    description="demo description",
                    uri=listed.uri,
                    server_name="server",
                    frontmatter=listed.frontmatter,
                    resources=tuple(listed.resources or []),
                )
            ],
        ),
    )

    result = await source.install_skill(current.uri, destination_root=tmp_path)

    assert result.skill_dir.joinpath("SKILL.md").read_text().endswith("new\n")


@pytest.mark.parametrize(
    ("uri", "name"),
    [
        ("skill://catalog/UPPER/SKILL.md", "UPPER"),
        ("skill://catalog/demo./SKILL.md", "demo."),
        ("skill://catalog/con/SKILL.md", "con"),
    ],
)
def test_registry_rejects_invalid_or_reserved_skill_names(uri: str, name: str) -> None:
    entry = SkillEntry(
        uri=uri,
        frontmatter={"name": name, "description": "description"},
        resources=[SkillResource(uri=uri, digest=_digest(""))],
    )

    with pytest.raises(ValueError, match="skill name"):
        mcp_registry._registry_skill(entry, server_name="server", server_version=None)


@pytest.mark.parametrize(
    "resource_uri",
    [
        "skill://catalog/demo/.skill-source.json",
        "skill://catalog/demo/SKILL.md%20",
        "skill://catalog/demo/file.txt.",
        "skill://catalog/demo/file:stream",
        "skill://catalog/demo/CON",
    ],
)
def test_registry_rejects_reserved_or_nonportable_resource_paths(resource_uri: str) -> None:
    uri = "skill://catalog/demo/SKILL.md"
    entry = SkillEntry(
        uri=uri,
        frontmatter={"name": "demo", "description": "description"},
        resources=[
            SkillResource(uri=uri, digest=_digest("")),
            SkillResource(uri=resource_uri, digest=_digest("")),
        ],
    )

    with pytest.raises(ValueError):
        mcp_registry._registry_skill(entry, server_name="server", server_version=None)


def test_frontmatter_comparison_preserves_json_scalar_types(tmp_path) -> None:
    skill_dir = tmp_path / "demo"
    skill_dir.mkdir()
    skill_dir.joinpath("SKILL.md").write_text(
        "---\nname: demo\ndescription: description\nflag: 1\n---\n",
        encoding="utf-8",
    )
    skill = McpRegistrySkill(
        name="demo",
        description="description",
        uri="skill://catalog/demo/SKILL.md",
        server_name="server",
        frontmatter={"name": "demo", "description": "description", "flag": True},
    )

    with pytest.raises(ValueError, match="frontmatter"):
        mcp_registry._verify_manifest(skill, skill_dir)
