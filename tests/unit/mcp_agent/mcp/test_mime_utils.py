from mcp_agent.mcp import mime_utils


class TestMimeUtils:
    def test_guess_mime_type(self):
        """Test guessing MIME types from file extensions."""
        assert mime_utils.guess_mime_type("file.txt") == "text/plain"
        assert mime_utils.guess_mime_type("file.py") == "text/x-python"
        assert mime_utils.guess_mime_type("file.js") in [
            "application/javascript",
            "text/javascript",
        ]
        assert mime_utils.guess_mime_type("file.json") == "application/json"
        assert mime_utils.guess_mime_type("file.html") == "text/html"
        assert mime_utils.guess_mime_type("file.css") == "text/css"
        assert mime_utils.guess_mime_type("file.png") == "image/png"
        assert mime_utils.guess_mime_type("file.jpg") == "image/jpeg"
        assert mime_utils.guess_mime_type("file.jpeg") == "image/jpeg"

        # TODO: decide if this should default to text or not...
        assert mime_utils.guess_mime_type("file.unknown") == "application/octet-stream"
