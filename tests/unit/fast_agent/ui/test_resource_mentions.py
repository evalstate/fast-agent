from __future__ import annotations

import base64

import pytest
from mcp.types import (
    EmbeddedResource,
    ImageContent,
    ReadResourceResult,
    ResourceLink,
    TextResourceContents,
)
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


def test_parse_mentions_normalizes_local_file_paths(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = tmp_path / "report.pdf"
    report.write_bytes(b"%PDF-1.4")
    monkeypatch.chdir(tmp_path)

    parsed = parse_mentions("Summarize ^file:./report.pdf")

    assert len(parsed.mentions) == 1
    assert parsed.mentions[0].server_name == "file"
    assert parsed.mentions[0].resource_uri == str(report.resolve())


def test_parse_mentions_normalizes_local_file_paths_from_explicit_cwd(tmp_path) -> None:
    working_dir = tmp_path / "shell-cwd"
    working_dir.mkdir()
    report = working_dir / "report.pdf"
    report.write_bytes(b"%PDF-1.4")

    parsed = parse_mentions("Summarize ^file:./report.pdf", cwd=working_dir)

    assert len(parsed.mentions) == 1
    assert parsed.mentions[0].server_name == "file"
    assert parsed.mentions[0].resource_uri == str(report.resolve())


def test_parse_mentions_normalizes_remote_urls() -> None:
    parsed = parse_mentions("Describe ^url:https://example.com/image.png?size=full")

    assert len(parsed.mentions) == 1
    assert parsed.mentions[0].server_name == "url"
    assert parsed.mentions[0].resource_uri == "https://example.com/image.png?size=full"


@pytest.mark.asyncio
async def test_resolve_mentions_builds_embedded_resources() -> None:
    parsed = parse_mentions("Read ^demo:file:///tmp/notes.txt")

    resolved = await resolve_mentions(_ResourceAgentStub(), parsed)
    prompt = build_prompt_with_resources(parsed.text, resolved)

    assert prompt.role == "user"
    assert len(prompt.content) == 2


@pytest.mark.asyncio
async def test_resolve_mentions_builds_local_file_resource_without_agent_support(tmp_path) -> None:
    notes = tmp_path / "notes.txt"
    notes.write_text("hello", encoding="utf-8")
    parsed = parse_mentions(f"Read ^file:{notes}")

    resolved = await resolve_mentions(object(), parsed)
    prompt = build_prompt_with_resources(parsed.text, resolved)

    assert any(isinstance(item, EmbeddedResource) for item in prompt.content)


@pytest.mark.asyncio
async def test_resolve_mentions_builds_local_image_content(tmp_path) -> None:
    image_path = tmp_path / "pixel.png"
    image_path.write_bytes(
        base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9s2nRwAAAABJRU5ErkJggg=="
        )
    )
    parsed = parse_mentions(f"^file:{image_path}")

    resolved = await resolve_mentions(object(), parsed)
    prompt = build_prompt_with_resources(parsed.text, resolved)

    assert len(prompt.content) == 2
    assert isinstance(prompt.content[1], ImageContent)


@pytest.mark.asyncio
async def test_resolve_mentions_builds_remote_url_resource_link_without_agent_support() -> None:
    parsed = parse_mentions("Describe ^url:https://example.com/image.png")

    resolved = await resolve_mentions(object(), parsed)
    prompt = build_prompt_with_resources(parsed.text, resolved)

    assert isinstance(prompt.content[1], ResourceLink)
    assert str(prompt.content[1].uri) == "https://example.com/image.png"
    assert prompt.content[1].mimeType == "image/png"


@pytest.mark.asyncio
async def test_resolve_mentions_infers_image_type_from_query_and_defaults_to_image() -> None:
    parsed = parse_mentions(
        "Describe ^url:https://pbs.twimg.com/media/HCaWzdDWYAArgCf?format=jpg&name=4096x4096"
    )

    resolved = await resolve_mentions(object(), parsed)
    prompt = build_prompt_with_resources(parsed.text, resolved)

    assert isinstance(prompt.content[1], ResourceLink)
    assert prompt.content[1].mimeType == "image/jpeg"


@pytest.mark.asyncio
async def test_resolve_mentions_keeps_unknown_remote_type_questionable() -> None:
    parsed = parse_mentions("Describe ^url:https://example.com/download")

    resolved = await resolve_mentions(object(), parsed)
    prompt = build_prompt_with_resources(parsed.text, resolved)

    assert isinstance(prompt.content[1], ResourceLink)
    assert prompt.content[1].mimeType == "application/octet-stream"


@pytest.mark.asyncio
async def test_resolve_mentions_raises_on_resource_errors() -> None:
    class _FailingAgent:
        async def get_resource(self, *_args, **_kwargs):
            raise ValueError("boom")

    parsed = parse_mentions("Read ^demo:file:///tmp/notes.txt")

    with pytest.raises(ResourceMentionError):
        await resolve_mentions(_FailingAgent(), parsed)
