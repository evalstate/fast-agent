from fast_agent.mcp import mime_utils


class TestMimeUtils:
    def test_additional_mime_types_are_registered(self):
        """Project-specific MIME registrations should remain active."""
        for mime_type, extension in mime_utils.ADDITIONAL_MIME_TYPES:
            assert mime_utils.guess_mime_type(f"file{extension}") == mime_type

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
        assert mime_utils.is_document_mime_type(" APPLICATION/X-PDF ")
        assert mime_utils.is_document_mime_type("pdf")
        assert mime_utils.is_document_mime_type(
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
        assert not mime_utils.is_document_mime_type("text/plain")

    def test_is_text_mime_type(self):
        assert mime_utils.is_text_mime_type("text/plain")
        assert mime_utils.is_text_mime_type(" TEXT/PLAIN ")
        assert mime_utils.is_text_mime_type("application/json")
        assert mime_utils.is_text_mime_type("JSON")
        assert mime_utils.is_text_mime_type("application/activity+json")
        assert not mime_utils.is_text_mime_type("")
        assert not mime_utils.is_text_mime_type("application/octet-stream")

    def test_is_image_mime_type_normalizes_common_forms(self):
        assert mime_utils.is_image_mime_type(" IMAGE/JPG ")
        assert mime_utils.is_image_mime_type("png")
        assert not mime_utils.is_image_mime_type("image/svg+xml")
