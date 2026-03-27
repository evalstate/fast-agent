from fast_agent.mcp import mime_utils


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
        assert (
            mime_utils.guess_mime_type("file.docx")
            == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
        assert (
            mime_utils.guess_mime_type("file.xlsx")
            == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        assert (
            mime_utils.guess_mime_type("file.pptx")
            == "application/vnd.openxmlformats-officedocument.presentationml.presentation"
        )

        # TODO: decide if this should default to text or not...
        assert mime_utils.guess_mime_type("file.unknown") == "application/octet-stream"

    def test_is_document_mime_type(self):
        assert mime_utils.is_document_mime_type("application/pdf")
        assert mime_utils.is_document_mime_type(
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
        assert not mime_utils.is_document_mime_type("text/plain")
