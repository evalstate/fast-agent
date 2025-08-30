import unittest

from pydantic import AnyUrl

from mcp_agent.mcp.resource_utils import normalize_uri


class TestUriNormalization(unittest.TestCase):
    """Tests for URI normalization functionality."""

    def test_already_valid_uri(self):
        """Test that already valid URIs are left unchanged."""
        uri = "https://example.com/path/file.txt"
        result = normalize_uri(uri)
        self.assertEqual(result, uri)

    def test_file_uri(self):
        """Test that file:// URIs are left unchanged."""
        uri = "file:///path/to/file.txt"
        result = normalize_uri(uri)
        self.assertEqual(result, uri)

    def test_simple_filename(self):
        """Test that simple filenames are converted to file:// URIs."""
        filename = "example.py"
        result = normalize_uri(filename)
        self.assertEqual(result, "file:///example.py")

    def test_relative_path(self):
        """Test that relative paths are converted to file:// URIs."""
        path = "path/to/file.txt"
        result = normalize_uri(path)
        self.assertEqual(result, "file:///path/to/file.txt")

    def test_absolute_path(self):
        """Test that absolute paths are converted to file:// URIs."""
        path = "/path/to/file.txt"
        result = normalize_uri(path)
        self.assertEqual(result, "file:///path/to/file.txt")

    def test_windows_path(self):
        """Test that Windows paths are normalized and converted."""
        path = "C:\\path\\to\\file.txt"
        result = normalize_uri(path)
        self.assertEqual(result, "file:///C:/path/to/file.txt")

    def test_empty_string(self):
        """Test handling of empty strings."""
        result = normalize_uri("")
        self.assertEqual(result, "")

    def test_normalize_uri(self):
        """Test that URIs are normalized correctly."""
        # Test different URI formats
        test_cases = [
            ("https://example.com/path/file.txt", "https://example.com/path/file.txt"),
            ("file:///path/to/file.txt", "file:///path/to/file.txt"),
            ("example.py", "file:///example.py"),
            ("path/to/file.txt", "file:///path/to/file.txt"),
            ("/path/to/file.txt", "file:///path/to/file.txt"),
            ("C:\\path\\to\\file.txt", "file:///C:/path/to/file.txt"),
            ("", ""),
        ]

        for input_uri, expected in test_cases:
            result = normalize_uri(input_uri)
            self.assertEqual(result, expected)

    def test_uri_extraction_edge_cases(self):
        """Test extraction of filenames from various URI formats."""
        from mcp_agent.mcp.resource_utils import extract_title_from_uri

        # Test different URI formats
        test_cases = [
            ("https://example.com/path/file.txt", "file.txt"),
            ("https://example.com/path/", "path"),
            ("file:///C:/Users/name/document.pdf", "document.pdf"),
            ("file:///home/user/file.py", "file.py"),
        ]

        for uri, expected in test_cases:
            result = extract_title_from_uri(AnyUrl(uri))
            self.assertEqual(result, expected if expected else uri)
