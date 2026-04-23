"""Tests for SkillReader URI handling (Skills-over-MCP)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from mcp.types import ReadResourceResult, TextResourceContents
from pydantic import AnyUrl

from fast_agent.skills.registry import SkillManifest
from fast_agent.tools.skill_reader import SkillReader


def _text_result(text: str, uri: str) -> ReadResourceResult:
    return ReadResourceResult(
        contents=[TextResourceContents(uri=AnyUrl(uri), mimeType="text/markdown", text=text)]
    )


def _mcp_manifest(name: str = "git-workflow", server: str = "srv") -> SkillManifest:
    return SkillManifest(
        name=name,
        description=f"The {name} skill",
        body="",
        path=None,
        uri=f"skill://{name}/SKILL.md",
        server_name=server,
    )


def _fake_aggregator(responses: dict[str, ReadResourceResult | Exception]) -> Any:
    agg = MagicMock()

    async def get_resource(uri: str, *, server_name: str | None = None) -> ReadResourceResult:
        result = responses.get(uri)
        if result is None:
            raise ValueError(f"unknown uri {uri}")
        if isinstance(result, Exception):
            raise result
        return result

    agg.get_resource = get_resource
    return agg


@pytest.mark.asyncio
async def test_uri_read_dispatches_to_aggregator() -> None:
    manifest = _mcp_manifest()
    agg = _fake_aggregator(
        {"skill://git-workflow/SKILL.md": _text_result("# body", "skill://git-workflow/SKILL.md")}
    )
    reader = SkillReader([manifest], logger=MagicMock(), aggregator=agg)

    result = await reader.execute({"path": "skill://git-workflow/SKILL.md"})

    assert not result.isError
    assert result.content[0].text == "# body"


@pytest.mark.asyncio
async def test_uri_read_allows_descendant_of_skill_root() -> None:
    manifest = _mcp_manifest()
    agg = _fake_aggregator(
        {
            "skill://git-workflow/references/GUIDE.md": _text_result(
                "refs",
                "skill://git-workflow/references/GUIDE.md",
            )
        }
    )
    reader = SkillReader([manifest], logger=MagicMock(), aggregator=agg)

    result = await reader.execute(
        {"path": "skill://git-workflow/references/GUIDE.md"}
    )

    assert not result.isError
    assert result.content[0].text == "refs"


@pytest.mark.asyncio
async def test_uri_outside_known_skill_root_denied() -> None:
    manifest = _mcp_manifest("git-workflow")
    reader = SkillReader([manifest], logger=MagicMock(), aggregator=MagicMock())

    result = await reader.execute({"path": "skill://unknown/SKILL.md"})

    assert result.isError
    assert "Access denied" in result.content[0].text


@pytest.mark.asyncio
async def test_uri_read_with_no_aggregator_errors_clearly() -> None:
    manifest = _mcp_manifest()
    reader = SkillReader([manifest], logger=MagicMock(), aggregator=None)

    result = await reader.execute({"path": "skill://git-workflow/SKILL.md"})

    assert result.isError
    assert "aggregator" in result.content[0].text.lower()


@pytest.mark.asyncio
async def test_non_skill_scheme_uri_dispatches_via_aggregator() -> None:
    """SEP allows servers to publish skills under any scheme (e.g. github://).

    My host MUST route those URIs through the aggregator, not the local
    filesystem — otherwise a `github://...` argument would be Path()-mangled
    into `github:\\...` under cwd on Windows.
    """
    manifest = SkillManifest(
        name="refunds",
        description="d",
        body="",
        path=None,
        uri="github://acme/billing/refunds/SKILL.md",
        server_name="acme-srv",
    )
    agg = _fake_aggregator(
        {
            "github://acme/billing/refunds/SKILL.md": _text_result(
                "# refunds skill", "github://acme/billing/refunds/SKILL.md"
            )
        }
    )
    reader = SkillReader([manifest], logger=MagicMock(), aggregator=agg)

    result = await reader.execute(
        {"path": "github://acme/billing/refunds/SKILL.md"}
    )

    assert not result.isError
    assert "refunds skill" in result.content[0].text


@pytest.mark.asyncio
async def test_unknown_uri_scheme_outside_allowed_roots_denied() -> None:
    """A URI-shaped argument that doesn't match any discovered manifest's
    root must be rejected (security: don't read arbitrary URIs)."""
    manifest = SkillManifest(
        name="known",
        description="d",
        body="",
        path=None,
        uri="skill://known/SKILL.md",
        server_name="srv",
    )
    reader = SkillReader([manifest], logger=MagicMock(), aggregator=MagicMock())

    result = await reader.execute({"path": "https://evil.example/anything"})

    assert result.isError
    assert "Access denied" in result.content[0].text


@pytest.mark.asyncio
async def test_uri_with_parent_traversal_denied() -> None:
    """`skill://good/../evil/SKILL.md` must not slip past the prefix check.

    Defense in depth: the filesystem guard normalizes via Path.resolve();
    the URI guard doesn't resolve URIs (that's server semantics), so it
    rejects any path containing a `..` or `.` segment outright.
    """
    manifest = _mcp_manifest("good")
    reader = SkillReader([manifest], logger=MagicMock(), aggregator=MagicMock())

    traversal = await reader.execute({"path": "skill://good/../evil/SKILL.md"})
    dot_segment = await reader.execute({"path": "skill://good/./SKILL.md"})
    trailing = await reader.execute({"path": "skill://good/.."})

    assert traversal.isError and "Access denied" in traversal.content[0].text
    assert dot_segment.isError and "Access denied" in dot_segment.content[0].text
    assert trailing.isError and "Access denied" in trailing.content[0].text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "uri",
    [
        "skill://good/%2e%2e/evil/SKILL.md",
        "skill://good/%2E%2E/evil/SKILL.md",
        "skill://good/%2E/SKILL.md",
    ],
)
async def test_uri_with_percent_encoded_traversal_denied(uri: str) -> None:
    """Percent-encoded `..` / `.` segments must be rejected too.

    The aggregator forwards the URI to the server as-is and the server
    is the ultimate authority, but the host's trust boundary should not
    rely on that. Decoding just `%2E` (the only RFC-3986 unreserved dot
    encoding) is enough to catch the common bypass.
    """
    manifest = _mcp_manifest("good")
    reader = SkillReader([manifest], logger=MagicMock(), aggregator=MagicMock())

    result = await reader.execute({"path": uri})
    assert result.isError
    assert "Access denied" in result.content[0].text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "uri",
    [
        "skill://good/SKILL.md?redirect=evil",
        "skill://good/SKILL.md#frag",
    ],
)
async def test_uri_with_query_or_fragment_denied(uri: str) -> None:
    """Queries and fragments aren't meaningful for skill reads and would
    let a caller pass the exact-match allow-check with trailing junk."""
    manifest = _mcp_manifest("good")
    reader = SkillReader([manifest], logger=MagicMock(), aggregator=MagicMock())

    result = await reader.execute({"path": uri})
    assert result.isError
    assert "Access denied" in result.content[0].text


@pytest.mark.asyncio
async def test_binary_only_resource_returns_error() -> None:
    """A blob-only resource must error rather than fake a text placeholder.

    The old behavior synthesized `<binary resource: mimeType=..., base64 length=...>`
    and returned it as TextContent — the model would treat that string as the
    actual skill content.
    """
    from mcp.types import BlobResourceContents, ReadResourceResult
    from pydantic import AnyUrl

    manifest = _mcp_manifest("good")
    uri = "skill://good/diagram.png"
    blob_result = ReadResourceResult(
        contents=[
            BlobResourceContents(uri=AnyUrl(uri), mimeType="image/png", blob="AAAA")
        ]
    )
    agg = _fake_aggregator({uri: blob_result})
    reader = SkillReader([manifest], logger=MagicMock(), aggregator=agg)

    result = await reader.execute({"path": uri})
    assert result.isError
    assert "binary" in result.content[0].text.lower()
    assert "image/png" in result.content[0].text


@pytest.mark.asyncio
async def test_file_uri_rejected_even_if_registered() -> None:
    """Defense in depth: even if a manifest somehow declared a `file://` root,
    the trust-boundary check must still refuse the URI. The loader blocks
    file:// at discovery time; this test simulates the invariant being
    violated (e.g. a test fixture or direct construction) and verifies the
    reader refuses regardless."""
    manifest = SkillManifest(
        name="local",
        description="d",
        body="",
        path=None,
        uri="file:///tmp/local/SKILL.md",
        server_name="srv",
    )
    reader = SkillReader([manifest], logger=MagicMock(), aggregator=MagicMock())

    result = await reader.execute({"path": "file:///tmp/local/SKILL.md"})

    assert result.isError
    assert "Access denied" in result.content[0].text


@pytest.mark.asyncio
async def test_unmapped_uri_denied() -> None:
    """Defense in depth for the SkillManifest invariant: if a URI somehow
    lands in allowed-roots without a matching server_name, the aggregator
    would fall through every connected server. The reader must refuse
    before dispatch.
    """
    # Construct by bypassing __post_init__ so we can simulate an invariant
    # violation. In normal use the registry validates this at construction.
    manifest = SkillManifest.__new__(SkillManifest)
    object.__setattr__(manifest, "name", "orphan")
    object.__setattr__(manifest, "description", "d")
    object.__setattr__(manifest, "body", "")
    object.__setattr__(manifest, "path", None)
    object.__setattr__(manifest, "uri", "skill://orphan/SKILL.md")
    object.__setattr__(manifest, "server_name", None)
    object.__setattr__(manifest, "license", None)
    object.__setattr__(manifest, "compatibility", None)
    object.__setattr__(manifest, "metadata", None)
    object.__setattr__(manifest, "allowed_tools", None)

    reader = SkillReader([manifest], logger=MagicMock(), aggregator=MagicMock())
    result = await reader.execute({"path": "skill://orphan/SKILL.md"})

    assert result.isError
    assert "not mapped to a known" in result.content[0].text


@pytest.mark.asyncio
async def test_filesystem_read_still_works(tmp_path) -> None:
    skill_dir = tmp_path / "git-workflow"
    skill_dir.mkdir()
    md = skill_dir / "SKILL.md"
    md.write_text("---\nname: git-workflow\ndescription: d\n---\nbody\n", encoding="utf-8")
    manifest = SkillManifest(
        name="git-workflow",
        description="d",
        body="body",
        path=md,
    )
    reader = SkillReader([manifest], logger=MagicMock())

    result = await reader.execute({"path": str(md)})
    assert not result.isError
    assert "body" in result.content[0].text
