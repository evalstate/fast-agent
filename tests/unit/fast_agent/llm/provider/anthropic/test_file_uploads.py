from __future__ import annotations

import base64
from types import SimpleNamespace

import pytest
from mcp.types import BlobResourceContents, EmbeddedResource
from pydantic import AnyUrl

from fast_agent.config import AnthropicSettings, Settings
from fast_agent.context import Context
from fast_agent.llm.provider.anthropic.llm_anthropic import AnthropicLLM
from fast_agent.llm.provider.anthropic.multipart_converter_anthropic import (
    ANTHROPIC_FILE_ID_META_KEY,
)
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended


def _make_llm(model: str = "claude-sonnet-4-5") -> AnthropicLLM:
    settings = Settings()
    settings.anthropic = AnthropicSettings(api_key="test-key")
    context = Context(config=settings)
    return AnthropicLLM(context=context, model=model, name="test-agent")


class _FakeFilesApi:
    def __init__(self) -> None:
        self.calls: list[tuple[str | None, bytes, str | None]] = []

    async def upload(self, *, file):
        if isinstance(file, tuple):
            if len(file) == 3:
                filename, data, mime_type = file
            elif len(file) == 2:
                filename, data = file
                mime_type = None
            else:
                raise AssertionError(f"Unexpected file tuple: {file}")
        else:
            filename, data, mime_type = None, file, None

        assert isinstance(data, bytes)
        self.calls.append((filename, data, mime_type))
        return SimpleNamespace(id=f"file_{len(self.calls)}")


class _FakeAnthropic:
    def __init__(self) -> None:
        self.beta = SimpleNamespace(files=_FakeFilesApi())


@pytest.mark.asyncio
async def test_prepare_anthropic_file_resources_uploads_office_documents() -> None:
    llm = _make_llm()
    anthropic = _FakeAnthropic()
    docx_bytes = b"PK\x03\x04docx"
    resource = BlobResourceContents(
        uri=AnyUrl("file:///tmp/report.docx"),
        mimeType="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        blob=base64.b64encode(docx_bytes).decode("ascii"),
    )
    message = PromptMessageExtended(
        role="user",
        content=[EmbeddedResource(type="resource", resource=resource)],
    )

    await llm._prepare_anthropic_file_resources(anthropic, [message])

    meta = dict(resource.meta or {})
    assert meta[ANTHROPIC_FILE_ID_META_KEY] == "file_1"
    assert anthropic.beta.files.calls == [("report.docx", docx_bytes, resource.mimeType)]


@pytest.mark.asyncio
async def test_prepare_anthropic_file_resources_caches_repeated_uploads() -> None:
    llm = _make_llm()
    anthropic = _FakeAnthropic()
    docx_bytes = b"PK\x03\x04docx"
    blob = base64.b64encode(docx_bytes).decode("ascii")

    first = BlobResourceContents(
        uri=AnyUrl("file:///tmp/report.docx"),
        mimeType="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        blob=blob,
    )
    second = BlobResourceContents(
        uri=AnyUrl("file:///tmp/report.docx"),
        mimeType="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        blob=blob,
    )
    messages = [
        PromptMessageExtended(role="user", content=[EmbeddedResource(type="resource", resource=first)]),
        PromptMessageExtended(
            role="user", content=[EmbeddedResource(type="resource", resource=second)]
        ),
    ]

    await llm._prepare_anthropic_file_resources(anthropic, messages)

    assert len(anthropic.beta.files.calls) == 1
    assert dict(first.meta or {})[ANTHROPIC_FILE_ID_META_KEY] == "file_1"
    assert dict(second.meta or {})[ANTHROPIC_FILE_ID_META_KEY] == "file_1"


@pytest.mark.asyncio
async def test_prepare_anthropic_file_resources_infers_document_mime_from_uri() -> None:
    llm = _make_llm()
    anthropic = _FakeAnthropic()
    docx_bytes = b"PK\x03\x04docx"
    resource = BlobResourceContents(
        uri=AnyUrl("file:///tmp/report.docx"),
        blob=base64.b64encode(docx_bytes).decode("ascii"),
    )
    message = PromptMessageExtended(
        role="user",
        content=[EmbeddedResource(type="resource", resource=resource)],
    )

    await llm._prepare_anthropic_file_resources(anthropic, [message])

    meta = dict(resource.meta or {})
    assert meta[ANTHROPIC_FILE_ID_META_KEY] == "file_1"
    assert anthropic.beta.files.calls == [
        (
            "report.docx",
            docx_bytes,
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
    ]
