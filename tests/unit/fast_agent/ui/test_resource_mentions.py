from __future__ import annotations

import pytest
from mcp.types import ReadResourceResult, TextResourceContents
from pydantic import AnyUrl

from fast_agent.ui.prompt.resource_mentions import (
    ResourceMentionError,
    build_prompt_with_resources,
    parse_mentions,
    resolve_mentions,
)


class _ResourceAgentStub:
    async def get_resource(self, resource_uri: str, namespace: str | None = None):
        assert namespace is not None
        return ReadResourceResult(
            contents=[
                TextResourceContents(
                    uri=AnyUrl(resource_uri),
                    mimeType="text/plain",
                    text=f"{namespace}:{resource_uri}",
                )
            ]
        )


def test_parse_mentions_extracts_and_strips_resource_tokens() -> None:
    parsed = parse_mentions("Summarize ^demo:file:///tmp/report.md for me")

    assert parsed.cleaned_text == "Summarize for me"
    assert len(parsed.mentions) == 1
    mention = parsed.mentions[0]
    assert mention.server_name == "demo"
    assert mention.resource_uri == "file:///tmp/report.md"


def test_parse_mentions_deduplicates_mentions() -> None:
    parsed = parse_mentions("Use ^demo:file:///tmp/a and ^demo:file:///tmp/a")

    assert len(parsed.mentions) == 1


def test_parse_mentions_renders_template_values() -> None:
    parsed = parse_mentions("Inspect ^demo:file:///repo/{branch}/{path}{branch=main,path=README.md}")

    assert len(parsed.mentions) == 1
    assert parsed.mentions[0].resource_uri == "file:///repo/main/README.md"


def test_parse_mentions_preserves_slashes_for_simple_template_values() -> None:
    parsed = parse_mentions("Inspect ^demo:file:///repo/{path}{path=src/main.py}")

    assert len(parsed.mentions) == 1
    assert parsed.mentions[0].resource_uri == "file:///repo/src/main.py"


def test_parse_mentions_renders_rfc6570_path_expression_values() -> None:
    parsed = parse_mentions(
        "Inspect ^githubcopilot:repo://{owner}/{repo}/contents{/path*}{owner=evalstate,repo=fast-agent,path=plan/hot-mcp-auth.md}"
    )

    assert len(parsed.mentions) == 1
    assert (
        parsed.mentions[0].resource_uri
        == "repo://evalstate/fast-agent/contents/plan/hot-mcp-auth.md"
    )


def test_parse_mentions_records_template_warning_on_missing_args() -> None:
    parsed = parse_mentions("Inspect ^demo:file:///repo/{branch}")

    assert parsed.mentions == []
    assert parsed.warnings


@pytest.mark.asyncio
async def test_resolve_mentions_builds_embedded_resources() -> None:
    parsed = parse_mentions("Read ^demo:file:///tmp/notes.txt")

    resolved = await resolve_mentions(_ResourceAgentStub(), parsed)
    prompt = build_prompt_with_resources(parsed.text, resolved)

    assert prompt.role == "user"
    assert len(prompt.content) == 2


@pytest.mark.asyncio
async def test_resolve_mentions_raises_on_resource_errors() -> None:
    class _FailingAgent:
        async def get_resource(self, *_args, **_kwargs):
            raise ValueError("boom")

    parsed = parse_mentions("Read ^demo:file:///tmp/notes.txt")

    with pytest.raises(ResourceMentionError):
        await resolve_mentions(_FailingAgent(), parsed)
