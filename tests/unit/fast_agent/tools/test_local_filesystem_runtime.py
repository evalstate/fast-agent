import logging
from pathlib import Path

import pytest
from mcp.types import (
    BlobResourceContents,
    EmbeddedResource,
    ImageContent,
    ResourceLink,
    TextContent,
)

from fast_agent.llm.model_info import ModelInfo
from fast_agent.llm.provider_types import Provider
from fast_agent.tools.local_filesystem_runtime import LocalFilesystemRuntime


def _tool_by_name(runtime: LocalFilesystemRuntime, name: str):
    for tool in runtime.tools:
        if tool.name == name:
            return tool
    return None


def _model_info(*mime_types: str) -> ModelInfo:
    return ModelInfo(
        name="test-model",
        provider=Provider.GENERIC,
        context_window=None,
        max_output_tokens=None,
        tokenizes=list(mime_types),
        json_mode=None,
        reasoning=None,
    )


def test_read_text_file_tool_schema_matches_acp_signature() -> None:
    runtime = LocalFilesystemRuntime(logging.getLogger("local-filesystem-runtime-test"))

    tool = _tool_by_name(runtime, "read_text_file")
    assert tool is not None
    assert tool.name == "read_text_file"
    assert tool.inputSchema == {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute path to the file to read.",
            },
            "line": {
                "type": "integer",
                "description": "Optional line number to start reading from (1-based).",
                "minimum": 1,
            },
            "limit": {
                "type": "integer",
                "description": "Optional maximum number of lines to read.",
                "minimum": 1,
            },
        },
        "required": ["path"],
        "additionalProperties": False,
    }


def test_write_text_file_tool_schema_matches_acp_signature() -> None:
    runtime = LocalFilesystemRuntime(logging.getLogger("local-filesystem-runtime-test"))

    tool = _tool_by_name(runtime, "write_text_file")
    assert tool is not None
    assert tool.name == "write_text_file"
    assert tool.inputSchema == {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute path to the file to write.",
            },
            "content": {
                "type": "string",
                "description": "The text content to write to the file.",
            },
        },
        "required": ["path", "content"],
        "additionalProperties": False,
    }


def test_tools_property_respects_enable_flags() -> None:
    runtime = LocalFilesystemRuntime(
        logging.getLogger("local-filesystem-runtime-test"),
        enable_read=False,
        enable_write=True,
    )
    assert [tool.name for tool in runtime.tools] == ["write_text_file"]

    runtime.set_enabled_tools(enable_read=True, enable_write=False, enable_apply_patch=False)
    assert [tool.name for tool in runtime.tools] == ["read_text_file"]


def test_attach_media_auto_exposes_for_attachment_capable_model() -> None:
    runtime = LocalFilesystemRuntime(
        logging.getLogger("local-filesystem-runtime-test"),
        model_info=_model_info("image/png"),
    )

    assert _tool_by_name(runtime, "attach_media") is not None


def test_attach_media_auto_hidden_for_text_only_model() -> None:
    runtime = LocalFilesystemRuntime(
        logging.getLogger("local-filesystem-runtime-test"),
        model_info=_model_info("text/plain"),
    )

    assert _tool_by_name(runtime, "attach_media") is None


@pytest.mark.asyncio
async def test_attach_media_local_png_stages_image_content(tmp_path: Path) -> None:
    image_path = tmp_path / "pixel.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\nfake")
    runtime = LocalFilesystemRuntime(
        logging.getLogger("local-filesystem-runtime-test"),
        enable_attach_media="on",
        working_directory=tmp_path,
        model_info=_model_info("image/png"),
    )

    result = await runtime.attach_media({"source": "pixel.png"})

    assert result.isError is False
    assert result.content is not None
    assert isinstance(result.content[0], TextContent)
    assert len(result.content) == 1
    pending = runtime.consume_pending_media_attachments()
    assert len(pending) == 1
    assert isinstance(pending[0], ImageContent)
    assert pending[0].mimeType == "image/png"
    assert runtime.consume_pending_media_attachments() == []


@pytest.mark.asyncio
async def test_attach_media_local_pdf_stages_embedded_blob(tmp_path: Path) -> None:
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    runtime = LocalFilesystemRuntime(
        logging.getLogger("local-filesystem-runtime-test"),
        enable_attach_media="on",
        model_info=_model_info("application/pdf"),
    )

    result = await runtime.attach_media({"source": pdf_path.as_uri()})

    assert result.isError is False
    assert result.content is not None
    assert len(result.content) == 1
    pending = runtime.consume_pending_media_attachments()
    assert len(pending) == 1
    assert isinstance(pending[0], EmbeddedResource)
    assert isinstance(pending[0].resource, BlobResourceContents)
    assert pending[0].resource.mimeType == "application/pdf"


@pytest.mark.asyncio
async def test_attach_media_https_image_stages_resource_link() -> None:
    runtime = LocalFilesystemRuntime(
        logging.getLogger("local-filesystem-runtime-test"),
        enable_attach_media="on",
        model_info=_model_info("image/jpeg"),
    )

    result = await runtime.attach_media({"source": "https://example.com/photo.jpg"})

    assert result.isError is False
    assert result.content is not None
    assert len(result.content) == 1
    pending = runtime.consume_pending_media_attachments()
    assert len(pending) == 1
    assert isinstance(pending[0], ResourceLink)
    assert str(pending[0].uri) == "https://example.com/photo.jpg"
    assert pending[0].mimeType == "image/jpeg"


@pytest.mark.asyncio
async def test_attach_media_youtube_url_stages_video_resource_link() -> None:
    runtime = LocalFilesystemRuntime(
        logging.getLogger("local-filesystem-runtime-test"),
        enable_attach_media="on",
        model_info=_model_info("video/mp4"),
    )

    result = await runtime.attach_media(
        {"source": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"}
    )

    assert result.isError is False
    assert result.content is not None
    assert len(result.content) == 1
    pending = runtime.consume_pending_media_attachments()
    assert len(pending) == 1
    assert isinstance(pending[0], ResourceLink)
    assert pending[0].mimeType == "video/mp4"


@pytest.mark.asyncio
async def test_attach_media_rejects_google_remote_pdf_link() -> None:
    runtime = LocalFilesystemRuntime(
        logging.getLogger("local-filesystem-runtime-test"),
        enable_attach_media="on",
        model_info=ModelInfo(
            name="gemini-test",
            provider=Provider.GOOGLE,
            context_window=None,
            max_output_tokens=None,
            tokenizes=["application/pdf"],
            json_mode=None,
            reasoning=None,
        ),
    )

    result = await runtime.attach_media({"source": "https://example.com/report.pdf"})

    assert result.isError is True
    assert result.content is not None
    assert isinstance(result.content[0], TextContent)
    assert "remote PDF links" in result.content[0].text


@pytest.mark.asyncio
async def test_attach_media_rejects_unsupported_mime_for_model(tmp_path: Path) -> None:
    image_path = tmp_path / "pixel.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\nfake")
    runtime = LocalFilesystemRuntime(
        logging.getLogger("local-filesystem-runtime-test"),
        enable_attach_media="on",
        model_info=_model_info("application/pdf"),
    )

    result = await runtime.attach_media({"source": str(image_path)})

    assert result.isError is True
    assert result.content is not None
    assert isinstance(result.content[0], TextContent)
    assert "does not support embedded attachments with MIME type 'image/png'" in result.content[0].text


@pytest.mark.asyncio
async def test_attach_media_rejects_oversized_local_file(tmp_path: Path) -> None:
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    runtime = LocalFilesystemRuntime(
        logging.getLogger("local-filesystem-runtime-test"),
        enable_attach_media="on",
        attach_media_max_bytes=4,
        model_info=_model_info("application/pdf"),
    )

    result = await runtime.attach_media({"source": str(pdf_path)})

    assert result.isError is True
    assert result.content is not None
    assert isinstance(result.content[0], TextContent)
    assert "maximum inline attachment size" in result.content[0].text


@pytest.mark.asyncio
async def test_attach_media_rejects_internal_resource_uri() -> None:
    runtime = LocalFilesystemRuntime(
        logging.getLogger("local-filesystem-runtime-test"),
        enable_attach_media="on",
        model_info=_model_info("application/pdf"),
    )

    result = await runtime.attach_media({"source": "internal://fast-agent/example.pdf"})

    assert result.isError is True
    assert result.content is not None
    assert isinstance(result.content[0], TextContent)
    assert "use get_resource" in result.content[0].text


def test_set_enabled_tools_preserves_edit_file_flag_when_omitted() -> None:
    runtime = LocalFilesystemRuntime(
        logging.getLogger("local-filesystem-runtime-test"),
        enable_read=False,
        enable_write=False,
        enable_edit_file=True,
    )

    runtime.set_enabled_tools(enable_read=True, enable_write=False, enable_apply_patch=False)

    assert [tool.name for tool in runtime.tools] == ["read_text_file", "edit_file"]


@pytest.mark.asyncio
async def test_read_text_file_reads_full_file(tmp_path: Path) -> None:
    runtime = LocalFilesystemRuntime(logging.getLogger("local-filesystem-runtime-test"))
    test_file = tmp_path / "sample.txt"
    test_file.write_text("first\nsecond\nthird\n", encoding="utf-8")

    result = await runtime.read_text_file({"path": str(test_file)})

    assert result.isError is False
    assert result.content is not None
    assert isinstance(result.content[0], TextContent)
    assert result.content[0].text == "first\nsecond\nthird\n"


@pytest.mark.asyncio
async def test_read_text_file_supports_line_and_limit(tmp_path: Path) -> None:
    runtime = LocalFilesystemRuntime(logging.getLogger("local-filesystem-runtime-test"))
    test_file = tmp_path / "sample.txt"
    test_file.write_text("first\nsecond\nthird\nfourth\n", encoding="utf-8")

    result = await runtime.read_text_file(
        {"path": str(test_file), "line": 2, "limit": 2},
    )

    assert result.isError is False
    assert result.content is not None
    assert isinstance(result.content[0], TextContent)
    assert result.content[0].text == "second\nthird"


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["line", "limit"])
async def test_read_text_file_rejects_boolean_line_or_limit(
    tmp_path: Path,
    field: str,
) -> None:
    runtime = LocalFilesystemRuntime(logging.getLogger("local-filesystem-runtime-test"))
    test_file = tmp_path / "sample.txt"
    test_file.write_text("first\nsecond\nthird\n", encoding="utf-8")

    result = await runtime.read_text_file({"path": str(test_file), field: True})

    assert result.isError is True
    assert result.content is not None
    assert isinstance(result.content[0], TextContent)
    assert (
        result.content[0].text
        == f"Error: '{field}' argument must be an integer greater than or equal to 1"
    )


@pytest.mark.asyncio
async def test_read_text_file_resolves_relative_paths_from_working_directory(
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    nested_dir = project_dir / "nested"
    nested_dir.mkdir()
    test_file = nested_dir / "sample.txt"
    test_file.write_text("relative content", encoding="utf-8")

    runtime = LocalFilesystemRuntime(
        logging.getLogger("local-filesystem-runtime-test"),
        working_directory=project_dir,
    )

    result = await runtime.read_text_file({"path": "nested/sample.txt"})

    assert result.isError is False
    assert result.content is not None
    assert isinstance(result.content[0], TextContent)
    assert result.content[0].text == "relative content"


@pytest.mark.asyncio
async def test_write_text_file_writes_file_successfully(tmp_path: Path) -> None:
    runtime = LocalFilesystemRuntime(logging.getLogger("local-filesystem-runtime-test"))
    output_file = tmp_path / "output.txt"

    result = await runtime.write_text_file(
        {"path": str(output_file), "content": "hello world"},
    )

    assert result.isError is False
    assert output_file.read_text(encoding="utf-8") == "hello world"
    assert result.content is not None
    assert isinstance(result.content[0], TextContent)
    assert result.content[0].text == f"Successfully wrote 11 characters to {output_file}"


@pytest.mark.asyncio
async def test_write_text_file_creates_parent_directories(tmp_path: Path) -> None:
    runtime = LocalFilesystemRuntime(logging.getLogger("local-filesystem-runtime-test"))
    output_file = tmp_path / "nested" / "dir" / "output.txt"

    result = await runtime.write_text_file(
        {"path": str(output_file), "content": "nested"},
    )

    assert result.isError is False
    assert output_file.exists()
    assert output_file.read_text(encoding="utf-8") == "nested"


@pytest.mark.asyncio
async def test_write_text_file_overwrites_existing_content(tmp_path: Path) -> None:
    runtime = LocalFilesystemRuntime(logging.getLogger("local-filesystem-runtime-test"))
    output_file = tmp_path / "output.txt"
    output_file.write_text("old", encoding="utf-8")

    result = await runtime.write_text_file(
        {"path": str(output_file), "content": "new"},
    )

    assert result.isError is False
    assert output_file.read_text(encoding="utf-8") == "new"


@pytest.mark.asyncio
async def test_write_text_file_invalid_args_returns_error() -> None:
    runtime = LocalFilesystemRuntime(logging.getLogger("local-filesystem-runtime-test"))

    invalid_results = [
        await runtime.write_text_file(None),
        await runtime.write_text_file({"path": "file.txt"}),
        await runtime.write_text_file({"path": "", "content": "x"}),
        await runtime.write_text_file({"path": "file.txt", "content": 123}),
    ]

    for result in invalid_results:
        assert result.isError is True
        assert result.content is not None
        assert isinstance(result.content[0], TextContent)
        assert result.content[0].text.startswith("Error:")


@pytest.mark.asyncio
async def test_write_text_file_resolves_relative_paths_from_working_directory(
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()

    runtime = LocalFilesystemRuntime(
        logging.getLogger("local-filesystem-runtime-test"),
        working_directory=project_dir,
    )

    result = await runtime.write_text_file(
        {"path": "nested/output.txt", "content": "relative write"},
    )

    output_file = project_dir / "nested" / "output.txt"
    assert result.isError is False
    assert output_file.exists()
    assert output_file.read_text(encoding="utf-8") == "relative write"


def test_apply_patch_tool_schema_uses_input_field() -> None:
    runtime = LocalFilesystemRuntime(
        logging.getLogger("local-filesystem-runtime-test"),
        enable_apply_patch=True,
    )

    tool = _tool_by_name(runtime, "apply_patch")
    assert tool is not None
    assert tool.name == "apply_patch"
    assert tool.inputSchema == {
        "type": "object",
        "properties": {
            "input": {
                "type": "string",
                "description": (
                    "Patch text in apply_patch format beginning with "
                    "'*** Begin Patch' and ending with '*** End Patch'."
                ),
            }
        },
        "required": ["input"],
        "additionalProperties": False,
    }
    assert tool.meta is not None
    assert "fast-agent/openai.responses_custom_tool" in tool.meta
    tool_payload = tool.model_dump(mode="json", by_alias=True, exclude_none=True)
    assert "_meta" in tool_payload
    assert "fast-agent/openai.responses_custom_tool" in tool_payload["_meta"]


@pytest.mark.asyncio
async def test_apply_patch_updates_file_relative_to_working_directory(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    file_path = project_dir / "notes.txt"
    file_path.write_text("one\ntwo\n", encoding="utf-8")

    runtime = LocalFilesystemRuntime(
        logging.getLogger("local-filesystem-runtime-test"),
        working_directory=project_dir,
        enable_write=False,
        enable_apply_patch=True,
    )

    patch_text = (
        "*** Begin Patch\n"
        "*** Update File: notes.txt\n"
        "@@\n"
        "-one\n"
        "+ONE\n"
        " two\n"
        "*** End Patch\n"
    )
    result = await runtime.apply_patch({"input": patch_text})

    assert result.isError is False
    assert file_path.read_text(encoding="utf-8") == "ONE\ntwo\n"
    assert result.content is not None
    assert isinstance(result.content[0], TextContent)
    assert "Success. Updated the following files:" in result.content[0].text


@pytest.mark.asyncio
async def test_apply_patch_invalid_args_returns_error() -> None:
    runtime = LocalFilesystemRuntime(logging.getLogger("local-filesystem-runtime-test"))

    invalid_results = [
        await runtime.apply_patch(None),
        await runtime.apply_patch({}),
        await runtime.apply_patch({"input": 123}),
    ]

    for result in invalid_results:
        assert result.isError is True
        assert result.content is not None
        assert isinstance(result.content[0], TextContent)
        assert result.content[0].text.startswith("Error:")


def test_attach_media_tool_description_conditional() -> None:
    # Google/Gemini
    google_runtime = LocalFilesystemRuntime(
        logging.getLogger("local-filesystem-runtime-test"),
        enable_attach_media="on",
        model_info=ModelInfo(
            name="gemini-test",
            provider=Provider.GOOGLE,
            context_window=None,
            max_output_tokens=None,
            tokenizes=["image/png"],
            json_mode=None,
            reasoning=None,
        ),
    )
    google_tool = _tool_by_name(google_runtime, "attach_media")
    assert google_tool is not None
    assert "Gemini YouTube links" in google_tool.description

    # OpenAI (Non-Google)
    openai_runtime = LocalFilesystemRuntime(
        logging.getLogger("local-filesystem-runtime-test"),
        enable_attach_media="on",
        model_info=ModelInfo(
            name="gpt-4o",
            provider=Provider.OPENAI,
            context_window=None,
            max_output_tokens=None,
            tokenizes=["image/png"],
            json_mode=None,
            reasoning=None,
        ),
    )
    openai_tool = _tool_by_name(openai_runtime, "attach_media")
    assert openai_tool is not None
    assert "Gemini YouTube links" not in openai_tool.description

