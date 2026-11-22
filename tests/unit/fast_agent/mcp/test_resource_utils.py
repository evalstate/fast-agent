import unittest
from urllib.parse import unquote, urlsplit

import pytest
from pydantic import AnyUrl

from fast_agent.mcp.resource_utils import create_embedded_resource, extract_title_from_uri


@pytest.mark.parametrize(
    "filename", ["report.txt", "report #1?.txt", "literal%2Fname.txt", "资料 notes.txt"]
)
@pytest.mark.parametrize("is_binary", [False, True])
def test_create_embedded_resource_preserves_filename(filename: str, is_binary: bool) -> None:
    embedded = create_embedded_resource(
        f"attachments/{filename}",
        "aGVsbG8=" if is_binary else "hello",
        "application/octet-stream" if is_binary else "text/plain",
        is_binary,
    )
    uri = str(embedded.resource.uri)
    parsed = urlsplit(uri)

    assert parsed.scheme == "resource"
    assert parsed.netloc == "fast-agent"
    assert unquote(parsed.path) == f"/{filename}"
    assert parsed.query == ""
    assert parsed.fragment == ""
    assert extract_title_from_uri(uri) == filename


@pytest.mark.parametrize(
    ("uri", "title"),
    [
        ("https://example.com/report%20%231%3F.txt?download=1#page", "report #1?.txt"),
        ("resource://fast-agent/literal%252Fname.txt", "literal%2Fname.txt"),
        ("resource://fast-agent/name%2Fpart.txt", "name/part.txt"),
    ],
)
def test_extract_title_from_uri_decodes_filename(uri: str, title: str) -> None:
    assert extract_title_from_uri(uri) == title


class TestUriTitleExtraction(unittest.TestCase):
    """Tests for URI title extraction."""

    def test_uri_extraction_edge_cases(self):
        from fast_agent.mcp.resource_utils import extract_title_from_uri

        test_cases = [
            ("https://example.com/path/file.txt", "file.txt"),
            ("https://example.com/path/", "path"),
            ("file:///C:/Users/name/document.pdf", "document.pdf"),
            ("file:///home/user/file.py", "file.py"),
        ]

        for uri, expected in test_cases:
            result = extract_title_from_uri(AnyUrl(uri))
            self.assertEqual(result, expected if expected else uri)
