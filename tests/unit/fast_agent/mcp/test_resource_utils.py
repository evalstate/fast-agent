import unittest

from pydantic import AnyUrl


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
