from pathlib import Path

import pytest
from mcp_types import BlobResourceContents, EmbeddedResource, TextResourceContents

from fast_agent.mcp.mcp_content import MCPFile
from fast_agent.mcp.resource_utils import load_resource_content

# Extensions that Python's mimetypes table does not know, so the extension alone says
# application/octet-stream.
TEXT_FILES = {
    "main.go": "package main\n\nfunc main() {}\n",
    "App.tsx": "export default () => <div>café</div>;\n",
}


@pytest.mark.parametrize("filename", sorted(TEXT_FILES))
def test_text_file_with_unregistered_extension_is_attached_as_text(
    tmp_path: Path, filename: str
) -> None:
    path = tmp_path / filename
    path.write_text(TEXT_FILES[filename], encoding="utf-8")

    attachment = MCPFile(path)["content"]

    assert isinstance(attachment, EmbeddedResource)
    assert isinstance(attachment.resource, TextResourceContents)
    assert attachment.resource.text == TEXT_FILES[filename]
    assert attachment.resource.mime_type == "text/plain"


@pytest.mark.parametrize("filename", sorted(TEXT_FILES))
def test_prompt_resource_with_unregistered_extension_is_loaded_as_text(
    tmp_path: Path, filename: str
) -> None:
    (tmp_path / filename).write_text(TEXT_FILES[filename], encoding="utf-8")

    resource = load_resource_content(filename, [tmp_path / "prompt.txt"])

    assert not resource.is_binary
    assert resource.content == TEXT_FILES[filename]
    assert resource.mime_type == "text/plain"


@pytest.mark.parametrize("data", [b"\x00\x01\x02binary", b"\xff\xfe\xfa not utf-8"])
def test_binary_file_with_unregistered_extension_stays_binary(tmp_path: Path, data: bytes) -> None:
    path = tmp_path / "blob.data"
    path.write_bytes(data)

    attachment = MCPFile(path)["content"]
    resource = load_resource_content("blob.data", [tmp_path / "prompt.txt"])

    assert isinstance(attachment.resource, BlobResourceContents)
    assert attachment.resource.mime_type == "application/octet-stream"
    assert resource.is_binary
    assert resource.mime_type == "application/octet-stream"
