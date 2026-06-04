"""Tests for `mcp-resource-template` discovery, expansion, and resolution."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest
from mcp.types import ReadResourceResult, TextResourceContents
from pydantic import AnyUrl

from fast_agent.mcp.mcp_skills_loader import (
    INDEX_URI,
    SkillTemplateEntry,
    expand_uri_template,
    load_mcp_skill_manifests,
    resolve_skill_template,
)


def _text(uri: str, body: str, mime: str = "application/json") -> ReadResourceResult:
    return ReadResourceResult(
        contents=[TextResourceContents(uri=AnyUrl(uri), mimeType=mime, text=body)]
    )


def _skill_md(name: str, description: str = "desc") -> str:
    return f"---\nname: {name}\ndescription: {description}\n---\nbody\n"


def _make_aggregator(responses: dict[tuple[str, str], ReadResourceResult]) -> Any:
    agg = MagicMock()

    async def get_resource(uri: str, *, server_name: str | None = None) -> ReadResourceResult:
        key = (server_name or "", uri)
        if key not in responses:
            raise ValueError(f"unknown {key}")
        return responses[key]

    agg.get_resource = get_resource
    return agg


# --- variable parsing & expansion ----------------------------------------


def test_variable_names_extracts_simple_vars() -> None:
    t = SkillTemplateEntry("srv", "skill://docs/{product}/SKILL.md", "x")
    assert t.variable_names() == ["product"]


def test_variable_names_handles_multiple_vars() -> None:
    t = SkillTemplateEntry("srv", "skill://{team}/{project}/SKILL.md", "x")
    assert t.variable_names() == ["team", "project"]


def test_variable_names_empty_for_concrete_url() -> None:
    t = SkillTemplateEntry("srv", "skill://refunds/SKILL.md", "x")
    assert t.variable_names() == []


def test_expand_uri_template_simple() -> None:
    assert (
        expand_uri_template("skill://docs/{product}/SKILL.md", {"product": "anvil"})
        == "skill://docs/anvil/SKILL.md"
    )


def test_expand_uri_template_percent_encodes_slashes() -> None:
    """Simple expansion (RFC 6570 §3.2.2) percent-encodes reserved chars.
    Without this, a user-supplied value containing `/` could escape the
    template's intended segment boundary."""
    expanded = expand_uri_template(
        "skill://docs/{product}/SKILL.md",
        {"product": "a/b"},
    )
    assert expanded == "skill://docs/a%2Fb/SKILL.md"


def test_expand_uri_template_missing_variable_raises() -> None:
    with pytest.raises(KeyError, match="product"):
        expand_uri_template("skill://docs/{product}/SKILL.md", {})


def test_expand_uri_template_no_vars_passthrough() -> None:
    assert (
        expand_uri_template("skill://refunds/SKILL.md", {})
        == "skill://refunds/SKILL.md"
    )


# --- discovery -----------------------------------------------------------


def _index_with_template() -> str:
    return json.dumps(
        {
            "$schema": "https://schemas.agentskills.io/discovery/0.2.0/schema.json",
            "skills": [
                {
                    "type": "mcp-resource-template",
                    "description": "Per-product documentation skill",
                    "url": "skill://docs/{product}/SKILL.md",
                }
            ],
        }
    )


@pytest.mark.asyncio
async def test_template_entry_collected_separately() -> None:
    """A template entry doesn't become a concrete manifest — it lands in
    `template_entries` for the host UI to surface."""
    responses = {("srv", INDEX_URI): _text(INDEX_URI, _index_with_template())}
    agg = _make_aggregator(responses)

    loaded = await load_mcp_skill_manifests(agg, ["srv"])

    assert loaded.manifests == []
    assert len(loaded.template_entries) == 1
    t = loaded.template_entries[0]
    assert t.server_name == "srv"
    assert t.url_template == "skill://docs/{product}/SKILL.md"
    assert t.description == "Per-product documentation skill"


@pytest.mark.asyncio
async def test_template_with_file_scheme_rejected() -> None:
    """`file://` is barred from templates for the same reason concrete
    entries reject it: it would let the server delegate content authority
    to the host's local disk."""
    body = json.dumps(
        {
            "$schema": "https://schemas.agentskills.io/discovery/0.2.0/schema.json",
            "skills": [
                {
                    "type": "mcp-resource-template",
                    "description": "x",
                    "url": "file:///etc/{thing}",
                }
            ],
        }
    )
    responses = {("srv", INDEX_URI): _text(INDEX_URI, body)}
    agg = _make_aggregator(responses)
    loaded = await load_mcp_skill_manifests(agg, ["srv"])
    assert loaded.template_entries == []


# --- resolve_skill_template ----------------------------------------------


@pytest.mark.asyncio
async def test_resolve_template_returns_manifest() -> None:
    """Resolution: expand the template, fetch the resulting SKILL.md,
    parse frontmatter, return a regular `SkillManifest`."""
    resolved_uri = "skill://docs/anvil/SKILL.md"
    responses = {
        ("srv", resolved_uri): ReadResourceResult(
            contents=[
                TextResourceContents(
                    uri=AnyUrl(resolved_uri),
                    mimeType="text/markdown",
                    text=_skill_md("anvil", description="Anvil docs"),
                )
            ]
        ),
    }
    agg = _make_aggregator(responses)
    template = SkillTemplateEntry(
        server_name="srv",
        url_template="skill://docs/{product}/SKILL.md",
        description="per-product",
    )

    # Frontmatter `name` mismatches URI final segment — the loader will
    # log a warning but still produce a manifest with the frontmatter
    # name. Here they match.
    manifest = await resolve_skill_template(agg, template, {"product": "anvil"})
    assert manifest is not None
    assert manifest.name == "anvil"
    assert manifest.uri == resolved_uri
    assert manifest.server_name == "srv"


@pytest.mark.asyncio
async def test_resolve_template_missing_variable() -> None:
    template = SkillTemplateEntry(
        server_name="srv",
        url_template="skill://docs/{product}/SKILL.md",
        description="x",
    )
    agg = _make_aggregator({})
    manifest = await resolve_skill_template(agg, template, {})
    assert manifest is None


@pytest.mark.asyncio
async def test_resolve_template_fetch_failure_returns_none() -> None:
    """If the server returns nothing for the resolved URI, resolution
    fails cleanly — no manifest, no exception."""
    template = SkillTemplateEntry(
        server_name="srv",
        url_template="skill://docs/{product}/SKILL.md",
        description="x",
    )
    agg = _make_aggregator({})  # no responses — every read raises
    manifest = await resolve_skill_template(agg, template, {"product": "missing"})
    assert manifest is None
