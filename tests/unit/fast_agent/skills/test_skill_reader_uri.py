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
    # MCP-served content is wrapped with an untrusted-source marker per
    # SEP §Security Implications. The body lives between the tags.
    assert "# body" in result.content[0].text
    assert "<untrusted-skill-content" in result.content[0].text


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
    assert "refs" in result.content[0].text
    assert "<untrusted-skill-content" in result.content[0].text


@pytest.mark.asyncio
async def test_non_skill_scheme_outside_known_root_denied() -> None:
    """The trust boundary still rejects non-`skill://` URIs that don't
    descend from any discovered manifest root. SEP narrows the
    unenumerated-load relaxation to `skill://` only — for other schemes
    the host has no evidence the URI is a skill at all."""
    manifest = _mcp_manifest("git-workflow")
    reader = SkillReader([manifest], logger=MagicMock(), aggregator=MagicMock())

    result = await reader.execute({"path": "github://owner/repo/foo.md"})

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
async def test_orphan_skill_uri_fans_out_to_aggregator() -> None:
    """SEP-2640 §Discovery: hosts MUST support loading a skill given only
    its URI, even when the URI never appeared in any index. A manifest
    that's URI-backed but has no `server_name` (or a URI handed to the
    model that doesn't match any discovered root) reaches the aggregator
    *without* a server_name, which fans out across connected servers.
    First responder wins — the documented ambiguity in the SEP.
    """
    # Construct by bypassing __post_init__ so we can simulate the no-
    # server_name case the SEP description matches: an unenumerated URI.
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

    agg = _fake_aggregator(
        {"skill://orphan/SKILL.md": _text_result("# orphan body", "skill://orphan/SKILL.md")}
    )
    reader = SkillReader([manifest], logger=MagicMock(), aggregator=agg)
    result = await reader.execute({"path": "skill://orphan/SKILL.md"})

    assert not result.isError
    assert "orphan body" in result.content[0].text


@pytest.mark.asyncio
async def test_unenumerated_skill_uri_load() -> None:
    """SEP MUST: a `skill://` URI handed to the model (via server
    instructions, user input, or another skill) loads even when no
    manifest covers it. The aggregator fanout finds the publishing
    server; first responder wins.
    """
    agg = _fake_aggregator(
        {
            "skill://surprise/SKILL.md": _text_result(
                "# surprise body", "skill://surprise/SKILL.md"
            )
        }
    )
    # No manifests at all — purely unenumerated.
    reader = SkillReader([], logger=MagicMock(), aggregator=agg)
    result = await reader.execute({"path": "skill://surprise/SKILL.md"})

    assert not result.isError
    assert "surprise body" in result.content[0].text


@pytest.mark.asyncio
async def test_unenumerated_non_skill_scheme_denied() -> None:
    """The SEP narrows unenumerated load to `skill://` only: 'outside
    the index, hosts recognize skills by the skill:// scheme prefix.'
    An unknown `github://` URI has no evidence it's a skill, so reject.
    """
    agg = _fake_aggregator({})
    reader = SkillReader([], logger=MagicMock(), aggregator=agg)
    result = await reader.execute({"path": "github://owner/repo/foo.md"})

    assert result.isError
    assert "not within an allowed skill root" in result.content[0].text


@pytest.mark.asyncio
async def test_unenumerated_uri_with_no_responding_server() -> None:
    """When no connected server serves the URI, return a useful error
    instead of a stack trace from the aggregator's 'not found on any
    server' raise."""
    agg = _fake_aggregator({})  # every read raises
    reader = SkillReader([], logger=MagicMock(), aggregator=agg)
    result = await reader.execute({"path": "skill://missing/SKILL.md"})

    assert result.isError
    assert "not served by any connected MCP server" in result.content[0].text


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


# --- Archive cache reads -------------------------------------------------


def _archive_manifest(name: str = "pdf-processing", server: str = "srv") -> SkillManifest:
    return SkillManifest(
        name=name,
        description=f"The {name} skill",
        body="",
        path=None,
        uri=f"skill://{name}/SKILL.md",
        server_name=server,
    )


@pytest.mark.asyncio
async def test_archive_cache_serves_skill_md_locally() -> None:
    """An archive-backed SKILL.md is served from the cache without
    calling the aggregator. Important property: archive distribution
    delivers the multi-file skill atomically; subsequent reads must not
    silently change the source under the model."""
    manifest = _archive_manifest()
    cache = {
        "skill://pdf-processing": {
            "SKILL.md": b"# pdf body",
        }
    }
    # Aggregator left as MagicMock without a get_resource — any call
    # would raise. The test asserts the archive cache short-circuits.
    agg = MagicMock()
    agg.get_resource = MagicMock(side_effect=AssertionError("must not call aggregator"))

    reader = SkillReader([manifest], logger=MagicMock(), aggregator=agg, archive_cache=cache)
    result = await reader.execute({"path": "skill://pdf-processing/SKILL.md"})
    assert not result.isError
    assert "# pdf body" in result.content[0].text
    # Archive-cached reads are MCP-served too — untrusted-content wrapper
    # applies the same as the live aggregator path.
    assert "<untrusted-skill-content" in result.content[0].text


@pytest.mark.asyncio
async def test_archive_cache_serves_supporting_file_locally() -> None:
    manifest = _archive_manifest()
    cache = {
        "skill://pdf-processing": {
            "SKILL.md": b"# pdf body",
            "references/FORMS.md": b"forms guide",
        }
    }
    agg = MagicMock()
    agg.get_resource = MagicMock(side_effect=AssertionError("must not call aggregator"))

    reader = SkillReader([manifest], logger=MagicMock(), aggregator=agg, archive_cache=cache)
    result = await reader.execute({"path": "skill://pdf-processing/references/FORMS.md"})
    assert not result.isError
    assert "forms guide" in result.content[0].text
    assert "<untrusted-skill-content" in result.content[0].text


@pytest.mark.asyncio
async def test_archive_cache_missing_file_returns_error() -> None:
    """If the model asks for a file the archive doesn't contain, fail
    cleanly — do NOT fall through to the aggregator. The archive is the
    authoritative file set; falling through would mask packaging gaps."""
    manifest = _archive_manifest()
    cache = {"skill://pdf-processing": {"SKILL.md": b"# body"}}
    agg = MagicMock()
    agg.get_resource = MagicMock(side_effect=AssertionError("must not call aggregator"))

    reader = SkillReader([manifest], logger=MagicMock(), aggregator=agg, archive_cache=cache)
    result = await reader.execute({"path": "skill://pdf-processing/missing.md"})
    assert result.isError
    assert "not found in archive" in result.content[0].text


@pytest.mark.asyncio
async def test_archive_cache_trust_boundary_still_enforced() -> None:
    """The cache is a perf/atomicity layer; it doesn't widen the trust
    boundary. A traversal URI is rejected before the cache lookup."""
    manifest = _archive_manifest()
    cache = {"skill://pdf-processing": {"SKILL.md": b"# body"}}
    reader = SkillReader([manifest], logger=MagicMock(), archive_cache=cache)

    result = await reader.execute({"path": "skill://pdf-processing/../escape/SKILL.md"})
    assert result.isError
    assert "not within an allowed skill root" in result.content[0].text


@pytest.mark.asyncio
async def test_non_cached_uri_falls_through_to_aggregator() -> None:
    """A URI that's allowed but not in the archive cache (e.g. a
    `skill-md` entry alongside an archive entry) reaches the aggregator
    normally."""
    archive_m = _archive_manifest("pdf-processing")
    skill_md_m = _mcp_manifest("git-workflow")
    cache = {"skill://pdf-processing": {"SKILL.md": b"pdf body"}}
    agg = _fake_aggregator(
        {
            "skill://git-workflow/SKILL.md": _text_result(
                "# git body", "skill://git-workflow/SKILL.md"
            )
        }
    )
    reader = SkillReader(
        [archive_m, skill_md_m], logger=MagicMock(), aggregator=agg, archive_cache=cache
    )
    result = await reader.execute({"path": "skill://git-workflow/SKILL.md"})
    assert not result.isError
    assert "git body" in result.content[0].text


@pytest.mark.asyncio
async def test_archive_cache_binary_member_rejected_with_clear_error() -> None:
    """A non-UTF-8 supporting file (e.g. a PNG asset) returns an error
    explaining `read_skill` only returns text — same posture as the
    aggregator path's binary-resource branch."""
    manifest = _archive_manifest()
    # 0x80 alone is invalid utf-8.
    cache = {"skill://pdf-processing": {"SKILL.md": b"# body", "logo.png": b"\x80\x81"}}
    reader = SkillReader([manifest], logger=MagicMock(), archive_cache=cache)
    result = await reader.execute({"path": "skill://pdf-processing/logo.png"})
    assert result.isError
    assert "not valid UTF-8" in result.content[0].text or "binary" in result.content[0].text
