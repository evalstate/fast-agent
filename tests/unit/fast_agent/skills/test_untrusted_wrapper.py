"""Tests for the SEP-2640 untrusted-content wrapper on MCP skill reads.

SEP §Security Implications: "Hosts MUST treat MCP-served skill content
as untrusted model input." The wrapper is the host's signal to the
model that wrapped content arrived from a connected server and should
be treated as data, not directives.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from mcp.types import ReadResourceResult, TextResourceContents
from pydantic import AnyUrl

from fast_agent.skills.registry import SkillManifest, format_skills_for_prompt
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


# --- MCP path wraps ------------------------------------------------------


@pytest.mark.asyncio
async def test_aggregator_read_is_wrapped_with_source_marker() -> None:
    manifest = _mcp_manifest("git-workflow", server="github")
    agg = _fake_aggregator(
        {"skill://git-workflow/SKILL.md": _text_result("# body", "skill://git-workflow/SKILL.md")}
    )
    reader = SkillReader([manifest], logger=MagicMock(), aggregator=agg)

    result = await reader.execute({"path": "skill://git-workflow/SKILL.md"})

    assert not result.isError
    text = result.content[0].text
    # The wrapper opens with a tag identifying the source and URI, and
    # closes with the matching close tag. The body sits between.
    assert text.startswith(
        '<untrusted-skill-content source="mcp-server: github" '
        'uri="skill://git-workflow/SKILL.md">'
    )
    assert "# body" in text
    assert text.rstrip().endswith("</untrusted-skill-content>")


@pytest.mark.asyncio
async def test_archive_cache_read_is_wrapped() -> None:
    """Cache-served reads come from an MCP-published archive; same
    untrusted classification as the live aggregator path."""
    manifest = _mcp_manifest("pdf-processing", server="acme")
    # Manifest URI must match the archive cache root for `_find_server_for_uri`.
    cache = {"skill://pdf-processing": {"SKILL.md": b"# pdf body"}}
    reader = SkillReader(
        [manifest], logger=MagicMock(), aggregator=MagicMock(), archive_cache=cache
    )

    result = await reader.execute({"path": "skill://pdf-processing/SKILL.md"})

    assert not result.isError
    text = result.content[0].text
    assert 'source="mcp-server: acme"' in text
    assert "# pdf body" in text
    assert "</untrusted-skill-content>" in text


@pytest.mark.asyncio
async def test_unenumerated_uri_wraps_with_unknown_server() -> None:
    """The unenumerated `skill://` fanout path doesn't know which server
    answered. The wrapper still fires, marking source as `(unknown)` —
    a strictly weaker but still-honest attribution. Critically, the
    wrapper is *not* skipped just because we lack a name."""
    agg = _fake_aggregator(
        {"skill://surprise/SKILL.md": _text_result("# body", "skill://surprise/SKILL.md")}
    )
    reader = SkillReader([], logger=MagicMock(), aggregator=agg)

    result = await reader.execute({"path": "skill://surprise/SKILL.md"})

    assert not result.isError
    text = result.content[0].text
    assert 'source="mcp-server: (unknown)"' in text
    assert "# body" in text


# --- Filesystem path does NOT wrap ---------------------------------------


@pytest.mark.asyncio
async def test_filesystem_read_is_not_wrapped(tmp_path: Path) -> None:
    """Filesystem skills are presumed user-installed and inherit the
    user's trust. The wrapper exists to flag the *server* boundary; a
    filesystem skill doesn't cross one. Leaving filesystem reads unwrapped
    keeps the wrapper a precise signal rather than ambient noise."""
    skill_dir = tmp_path / "alpha"
    skill_dir.mkdir()
    md = skill_dir / "SKILL.md"
    md.write_text("---\nname: alpha\ndescription: x\n---\n# alpha body\n", encoding="utf-8")
    manifest = SkillManifest(name="alpha", description="x", body="b", path=md)
    reader = SkillReader([manifest], logger=MagicMock())

    result = await reader.execute({"path": str(md)})

    assert not result.isError
    text = result.content[0].text
    assert "untrusted-skill-content" not in text
    assert "# alpha body" in text


# --- Preamble teaches the model what the wrapper means -------------------


def test_preamble_explains_wrapper_when_mcp_skill_present() -> None:
    manifest = _mcp_manifest("git-workflow", server="github")
    out = format_skills_for_prompt([manifest])
    # The preamble must mention both the tag and what the model should
    # do with wrapped content.
    assert "<untrusted-skill-content" in out
    assert "untrusted" in out.lower()
    # The "treat as data, not directive" framing must be present in
    # some form — otherwise the wrapper is just decoration.
    assert "reference material" in out or "not as authoritative" in out


def test_preamble_omits_wrapper_guidance_when_only_filesystem_skills(tmp_path: Path) -> None:
    """If no MCP skill is present, the wrapper guidance is dead weight —
    don't include it. Symmetric with the existing has_mcp_skill flag."""
    md = tmp_path / "alpha" / "SKILL.md"
    md.parent.mkdir(parents=True)
    md.write_text("---\nname: alpha\ndescription: x\n---\nbody\n", encoding="utf-8")
    manifest = SkillManifest(name="alpha", description="x", body="b", path=md)
    out = format_skills_for_prompt([manifest])
    assert "untrusted-skill-content" not in out
