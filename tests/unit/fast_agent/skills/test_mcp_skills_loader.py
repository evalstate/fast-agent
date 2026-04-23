"""Tests for mcp_skills_loader — Skills-over-MCP discovery."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from mcp.types import ReadResourceResult, TextResourceContents
from pydantic import AnyUrl

from fast_agent.mcp.mcp_skills_loader import (
    INDEX_URI,
    MAX_INDEX_BYTES,
    MAX_SKILL_MD_BYTES,
    load_mcp_skill_manifests,
    merge_filesystem_and_mcp_manifests,
)
from fast_agent.skills.registry import SkillManifest


def _read_result(text: str, uri: str, mime: str = "application/json") -> ReadResourceResult:
    return ReadResourceResult(
        contents=[
            TextResourceContents(uri=AnyUrl(uri), mimeType=mime, text=text),
        ]
    )


def _index(skills: list[dict]) -> str:
    return json.dumps({"$schema": "https://schemas.agentskills.io/discovery/0.2.0/schema.json",
                       "skills": skills})


def _skill_md(name: str, description: str = "desc", body: str = "body") -> str:
    return f"---\nname: {name}\ndescription: {description}\n---\n{body}\n"


def _make_aggregator(responses: dict[tuple[str, str], ReadResourceResult | Exception]) -> Any:
    agg = MagicMock()

    async def get_resource(uri: str, *, server_name: str | None = None) -> ReadResourceResult:
        key = (server_name or "", uri)
        if key not in responses:
            raise ValueError(f"Resource not found: {key}")
        result = responses[key]
        if isinstance(result, Exception):
            raise result
        return result

    agg.get_resource = get_resource
    agg.get_capabilities = AsyncMock(return_value=None)
    return agg


@pytest.mark.asyncio
async def test_loads_concrete_skill_md_entries() -> None:
    responses = {
        ("srv", INDEX_URI): _read_result(
            _index(
                [
                    {
                        "name": "git-workflow",
                        "type": "skill-md",
                        "description": "Follow git conventions",
                        "url": "skill://git-workflow/SKILL.md",
                    }
                ]
            ),
            INDEX_URI,
        ),
        ("srv", "skill://git-workflow/SKILL.md"): _read_result(
            _skill_md("git-workflow", description="Follow git conventions"),
            "skill://git-workflow/SKILL.md",
            mime="text/markdown",
        ),
    }
    agg = _make_aggregator(responses)

    manifests = await load_mcp_skill_manifests(agg, ["srv"])

    assert len(manifests) == 1
    m = manifests[0]
    assert m.name == "git-workflow"
    assert m.description == "Follow git conventions"
    assert m.uri == "skill://git-workflow/SKILL.md"
    assert m.server_name == "srv"
    assert m.path is None


@pytest.mark.asyncio
async def test_missing_index_yields_empty() -> None:
    agg = _make_aggregator({})  # get_resource on index raises

    manifests = await load_mcp_skill_manifests(agg, ["srv"])

    assert manifests == []


@pytest.mark.asyncio
async def test_malformed_index_yields_empty() -> None:
    responses = {
        ("srv", INDEX_URI): _read_result("not json at all", INDEX_URI),
    }
    agg = _make_aggregator(responses)

    manifests = await load_mcp_skill_manifests(agg, ["srv"])
    assert manifests == []


@pytest.mark.asyncio
async def test_index_without_skills_key_yields_empty() -> None:
    responses = {
        ("srv", INDEX_URI): _read_result(json.dumps({"other": 1}), INDEX_URI),
    }
    agg = _make_aggregator(responses)
    assert await load_mcp_skill_manifests(agg, ["srv"]) == []


@pytest.mark.asyncio
async def test_template_entries_skipped() -> None:
    responses = {
        ("srv", INDEX_URI): _read_result(
            _index(
                [
                    {
                        "type": "mcp-resource-template",
                        "description": "Per-product docs",
                        "url": "skill://docs/{product}/SKILL.md",
                    }
                ]
            ),
            INDEX_URI,
        ),
    }
    agg = _make_aggregator(responses)

    assert await load_mcp_skill_manifests(agg, ["srv"]) == []


@pytest.mark.asyncio
async def test_partial_skill_md_failure_does_not_poison_batch() -> None:
    responses = {
        ("srv", INDEX_URI): _read_result(
            _index(
                [
                    {
                        "name": "good",
                        "type": "skill-md",
                        "description": "ok",
                        "url": "skill://good/SKILL.md",
                    },
                    {
                        "name": "bad",
                        "type": "skill-md",
                        "description": "fails",
                        "url": "skill://bad/SKILL.md",
                    },
                ]
            ),
            INDEX_URI,
        ),
        ("srv", "skill://good/SKILL.md"): _read_result(
            _skill_md("good", description="ok"),
            "skill://good/SKILL.md",
        ),
        ("srv", "skill://bad/SKILL.md"): RuntimeError("boom"),
    }
    agg = _make_aggregator(responses)

    manifests = await load_mcp_skill_manifests(agg, ["srv"])

    assert [m.name for m in manifests] == ["good"]


@pytest.mark.asyncio
async def test_degenerate_url_without_skill_path_segment_rejected() -> None:
    """A URL like `skill://SKILL.md` (no skill-path segment) must be rejected.

    Otherwise `strip_skill_md` would produce `skill:/`, which the reader
    would add to its allowed-URI-roots set, admitting every `skill://...`
    URI via the `startswith(root + "/")` trust-boundary check.
    """
    responses = {
        ("srv", INDEX_URI): _read_result(
            _index(
                [
                    {
                        "name": "good",
                        "type": "skill-md",
                        "description": "ok",
                        "url": "skill://good/SKILL.md",
                    },
                    {
                        "name": "malformed",
                        "type": "skill-md",
                        "description": "no path segment",
                        "url": "skill://SKILL.md",
                    },
                ]
            ),
            INDEX_URI,
        ),
        ("srv", "skill://good/SKILL.md"): _read_result(
            _skill_md("good"),
            "skill://good/SKILL.md",
        ),
        ("srv", "skill://SKILL.md"): _read_result(
            _skill_md("malformed"),
            "skill://SKILL.md",
            mime="text/markdown",
        ),
    }
    agg = _make_aggregator(responses)

    manifests = await load_mcp_skill_manifests(agg, ["srv"])

    assert [m.name for m in manifests] == ["good"]


@pytest.mark.asyncio
async def test_file_uri_entry_rejected() -> None:
    """`file://` skill URIs are rejected at load: the MCP-server-is-authority
    trust model breaks down if local filesystem paths enter the allow list.
    """
    responses = {
        ("srv", INDEX_URI): _read_result(
            _index(
                [
                    {
                        "name": "good",
                        "type": "skill-md",
                        "description": "ok",
                        "url": "skill://good/SKILL.md",
                    },
                    {
                        "name": "local",
                        "type": "skill-md",
                        "description": "local file",
                        "url": "file:///tmp/local/SKILL.md",
                    },
                ]
            ),
            INDEX_URI,
        ),
        ("srv", "skill://good/SKILL.md"): _read_result(
            _skill_md("good"),
            "skill://good/SKILL.md",
        ),
        # No response wired for the file:// URI — rejection must happen
        # before the aggregator is asked.
    }
    agg = _make_aggregator(responses)

    manifests = await load_mcp_skill_manifests(agg, ["srv"])
    assert [m.name for m in manifests] == ["good"]


@pytest.mark.asyncio
async def test_oversize_index_rejected() -> None:
    """A hostile server returning a huge `skill://index.json` must be rejected
    before parsing — don't pin arbitrary memory on `json.loads`."""
    huge = " " * (MAX_INDEX_BYTES + 1)  # padding; json.loads would still try
    responses = {
        ("srv", INDEX_URI): _read_result(huge, INDEX_URI),
    }
    agg = _make_aggregator(responses)

    manifests = await load_mcp_skill_manifests(agg, ["srv"])
    assert manifests == []


@pytest.mark.asyncio
async def test_oversize_skill_md_rejected() -> None:
    """A SKILL.md above the soft limit must be skipped but not abort the batch."""
    big_body = "x" * (MAX_SKILL_MD_BYTES + 1)
    responses = {
        ("srv", INDEX_URI): _read_result(
            _index(
                [
                    {
                        "name": "good",
                        "type": "skill-md",
                        "description": "ok",
                        "url": "skill://good/SKILL.md",
                    },
                    {
                        "name": "huge",
                        "type": "skill-md",
                        "description": "too big",
                        "url": "skill://huge/SKILL.md",
                    },
                ]
            ),
            INDEX_URI,
        ),
        ("srv", "skill://good/SKILL.md"): _read_result(
            _skill_md("good"),
            "skill://good/SKILL.md",
        ),
        ("srv", "skill://huge/SKILL.md"): _read_result(
            _skill_md("huge", body=big_body),
            "skill://huge/SKILL.md",
        ),
    }
    agg = _make_aggregator(responses)

    manifests = await load_mcp_skill_manifests(agg, ["srv"])
    assert [m.name for m in manifests] == ["good"]


@pytest.mark.asyncio
async def test_enabled_servers_filter() -> None:
    """Per-server opt-out: servers absent from enabled_servers are skipped."""
    responses = {
        ("a", INDEX_URI): _read_result(
            _index(
                [
                    {
                        "name": "alpha",
                        "type": "skill-md",
                        "description": "a",
                        "url": "skill://alpha/SKILL.md",
                    }
                ]
            ),
            INDEX_URI,
        ),
        ("a", "skill://alpha/SKILL.md"): _read_result(
            _skill_md("alpha"),
            "skill://alpha/SKILL.md",
        ),
    }
    agg = _make_aggregator(responses)

    # Only server "b" enabled — "a" is suppressed even though its index exists.
    manifests = await load_mcp_skill_manifests(agg, ["a"], enabled_servers={"b"})
    assert manifests == []


def _fs(tmp_path: Path, name: str) -> SkillManifest:
    skill_dir = tmp_path / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = skill_dir / "SKILL.md"
    manifest_path.write_text(
        f"---\nname: {name}\ndescription: d\n---\nbody\n", encoding="utf-8"
    )
    return SkillManifest(name=name, description="d", body="body", path=manifest_path)


def _mcp(name: str, server: str = "srv") -> SkillManifest:
    return SkillManifest(
        name=name,
        description="d",
        body="",
        path=None,
        uri=f"skill://{name}/SKILL.md",
        server_name=server,
    )


class TestMergeFilesystemAndMcpManifests:
    def test_no_collisions_appends_all(self, tmp_path: Path) -> None:
        fs = [_fs(tmp_path, "local-only")]
        mcp = [_mcp("mcp-only")]

        merged, warnings = merge_filesystem_and_mcp_manifests(fs, mcp)

        assert [m.name for m in merged] == ["local-only", "mcp-only"]
        assert warnings == []

    def test_filesystem_wins_on_collision(self, tmp_path: Path) -> None:
        """Filesystem manifest must take precedence over an MCP manifest
        with the same name — consistent with SkillRegistry dedup semantics.
        Without this rule, a malicious or misconfigured server could shadow
        a locally-curated skill."""
        fs = [_fs(tmp_path, "refunds")]
        mcp = [_mcp("refunds", server="acme")]

        merged, warnings = merge_filesystem_and_mcp_manifests(fs, mcp)

        assert len(merged) == 1
        assert merged[0].path is not None  # the filesystem one
        assert merged[0].uri is None
        assert len(warnings) == 1
        assert "refunds" in warnings[0]
        assert "acme" in warnings[0]
        assert "hidden by local filesystem skill" in warnings[0]

    def test_case_insensitive_collision(self, tmp_path: Path) -> None:
        fs = [_fs(tmp_path, "Refunds")]
        mcp = [_mcp("refunds")]

        merged, warnings = merge_filesystem_and_mcp_manifests(fs, mcp)

        assert len(merged) == 1
        assert merged[0].name == "Refunds"
        assert len(warnings) == 1

    def test_first_mcp_wins_within_batch(self) -> None:
        """When two MCP servers publish the same skill name, the first one
        discovered wins and the second is logged. Deterministic behavior
        matters — otherwise the active skill would depend on server-discovery
        ordering."""
        mcp = [_mcp("refunds", server="acme"), _mcp("refunds", server="globex")]

        merged, warnings = merge_filesystem_and_mcp_manifests([], mcp)

        assert [m.server_name for m in merged] == ["acme"]
        assert len(warnings) == 1
        # The warning must name BOTH the loser and the winning server so
        # an operator can act on it without correlating against the index
        # log stream.
        assert "globex" in warnings[0]
        assert "acme" in warnings[0]
        assert "earlier MCP-served skill" in warnings[0]

    def test_empty_mcp_list_is_noop(self, tmp_path: Path) -> None:
        fs = [_fs(tmp_path, "alpha")]
        merged, warnings = merge_filesystem_and_mcp_manifests(fs, [])
        assert [m.name for m in merged] == ["alpha"]
        assert warnings == []
