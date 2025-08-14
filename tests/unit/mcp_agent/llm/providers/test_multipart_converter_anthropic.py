import base64
import json
from typing import Any, Dict, List, Tuple

import pytest
from anthropic.types import MessageParam
from mcp.types import (
    BlobResourceContents,
    CallToolResult,
    ContentBlock,
    EmbeddedResource,
    ImageContent,
    PromptMessage,
    TextContent,
    TextResourceContents,
)
from pydantic_core import to_jsonable_python

from mcp_agent.llm.providers.multipart_converter_anthropic import AnthropicConverter
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart
from mcp_agent.mcp.resource_utils import normalize_uri

# --- Constants ---
# A dummy 1x1 red pixel PNG, base64 encoded
DUMMY_IMAGE_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/wcAAwAB/epv2AAAAABJRU5ErkJggg=="
# A dummy PDF, base64 encoded
DUMMY_PDF_B64 = base64.b64encode(b"%PDF-1.0\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj 2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj 3 0 obj<</Type/Page/MediaBox[0 0 3 3]>>endobj\nxref\n0 4\n0000000000 65535 f\n0000000010 00000 n\n0000000059 000000 n\n0000000103 000000 n\ntrailer<</Size 4/Root 1 0 R>>\nstartxref\n149\n%%EOF").decode("utf-8")
DUMMY_SVG_CONTENT = '<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100"></svg>'

# --- Helper Functions ---
def create_pdf_resource(pdf_base64: str, uri: str = "test://example.com/document.pdf") -> EmbeddedResource:
    """Creates an EmbeddedResource for a PDF blob."""
    pdf_resource_contents = BlobResourceContents(
        uri=uri,
        mimeType="application/pdf",
        blob=pdf_base64,
    )
    return EmbeddedResource(type="resource", resource=pdf_resource_contents)


def create_text_resource(text: str, filename_or_uri: str, mime_type: str = "text/plain") -> TextResourceContents:
    """Creates a TextResourceContents with normalized URI."""
    uri = normalize_uri(filename_or_uri)
    return TextResourceContents(uri=uri, mimeType=mime_type, text=text)


# --- Test Class ---
class TestAnthropicMultipartConverter:
    """
    Harmonized test suite for AnthropicConverter, covering conversions between
    MCP multipart messages and the Anthropic API format.
    """

    # --- Test MCP -> Anthropic Message Conversion ---
    @pytest.mark.parametrize(
        "role, mcp_content, expected_anthropic_content",
        [
            # Basic Text (User)
            ("user", [TextContent(type="text", text="Hello")], [{'type': 'text', 'text': 'Hello'}]),
            # Basic Text (Assistant)
            ("assistant", [TextContent(type="text", text="Hi there")], [{'type': 'text', 'text': 'Hi there'}]),
            # Basic Image (User)
            ("user", [ImageContent(type="image", mimeType="image/png", data=DUMMY_IMAGE_B64)], [{'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png', 'data': DUMMY_IMAGE_B64}}]),
            # Text + Image (User)
            ("user", [TextContent(type="text", text="Look:"), ImageContent(type="image", mimeType="image/jpeg", data=DUMMY_IMAGE_B64)], [{'type': 'text', 'text': 'Look:'}, {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/jpeg', 'data': DUMMY_IMAGE_B64}}]),
            # Embedded Text Resource (User)
            ("user", [EmbeddedResource(type="resource", resource=create_text_resource("doc content", "doc.txt"))], [{'type': 'document', 'title': 'doc.txt', 'source': {'type': 'text', 'media_type': 'text/plain', 'data': 'doc content'}}]),
            # Embedded PDF Resource from Blob (User)
            ("user", [create_pdf_resource(DUMMY_PDF_B64)], [{'type': 'document', 'title': 'document.pdf', 'source': {'type': 'base64', 'media_type': 'application/pdf', 'data': DUMMY_PDF_B64}}]),
            # In your parametrize list for test_convert_to_anthropic

            # Corrected Image URL test case
            ("user", [EmbeddedResource(type="resource", resource=BlobResourceContents(uri="https://example.com/image.jpg", mimeType="image/jpeg", blob="dummy_data"))], [{'type': 'image', 'source': {'type': 'url', 'url': 'https://example.com/image.jpg'}}]),

            # Corrected PDF URL test case
            ("user", [EmbeddedResource(type="resource", resource=BlobResourceContents(uri="https://example.com/doc.pdf", mimeType="application/pdf", blob="dummy_data"))], [{'type': 'document', 'title': 'doc.pdf', 'source': {'type': 'url', 'url': 'https://example.com/doc.pdf'}}]),
            # Unsupported Image Format (User) -> Fallback text
            ("user", [ImageContent(type="image", mimeType="image/bmp", data="bmpdata")], [{'type': 'text', 'text': "Image with unsupported format 'image/bmp' (7 bytes)"}]),
            # Unsupported Binary Resource (User) -> Fallback text
            ("user", 
                [EmbeddedResource(type="resource", resource=BlobResourceContents(uri="resource://data.bin", mimeType="application/octet-stream", blob="bindata"))], 
                [{'type': 'text', 'text': "Embedded Resource resource://data.bin with unsupported format application/octet-stream (7 characters)"}]
            ),
            # SVG Resource (User) -> Text block with code fence
            ("user", [EmbeddedResource(type="resource", resource=create_text_resource(DUMMY_SVG_CONTENT, "drawing.svg", "image/svg+xml"))], [{'type': 'text', 'text': f'```xml\n{DUMMY_SVG_CONTENT}\n```'}]),
            # Python code file (User) -> Document
            ("user", [EmbeddedResource(type="resource", resource=create_text_resource("print('hi')", "script.py", "text/x-python"))], [{'type': 'document', 'title': 'script.py', 'source': {'type': 'text', 'media_type': 'text/plain', 'data': "print('hi')"}}]),
            # Empty Content List (User & Assistant)
            ("user", [], []),
            ("assistant", [], []),
            # Assistant with non-text content -> Stripped
            ("assistant", [TextContent(type="text", text="I can help."), ImageContent(type="image", mimeType="image/png", data=DUMMY_IMAGE_B64)], [{'type': 'text', 'text': 'I can help.'}]),
            ("assistant", [TextContent(type="text", text="Okay."), create_pdf_resource(DUMMY_PDF_B64)], [{'type': 'text', 'text': 'Okay.'}]),
        ],
    )
    def test_convert_to_anthropic(self, role: str, mcp_content: List[ContentBlock], expected_anthropic_content: List[Dict[str, Any]], mocker):
        """Tests conversion from PromptMessageMultipart to Anthropic's MessageParam format."""
        mock_logger = mocker.patch("mcp_agent.llm.providers.multipart_converter_anthropic._logger")
        
        multipart_msg = PromptMessageMultipart(role=role, content=mcp_content)
        anthropic_msg = AnthropicConverter.convert_to_anthropic(multipart_msg)

        assert anthropic_msg["role"] == role
        
        actual_json = json.dumps(to_jsonable_python(anthropic_msg["content"]), sort_keys=True)
        expected_json = json.dumps(expected_anthropic_content, sort_keys=True)

        assert actual_json == expected_json
        
        if role == "assistant" and len(mcp_content) > len(expected_anthropic_content):
            mock_logger.warning.assert_called()

    # --- Test Tool Result -> Anthropic Message Conversion ---
    @pytest.mark.parametrize(
        "tool_results, expected_anthropic_content",
        [
            # Single text result
            (
                [("tool1", CallToolResult(content=[TextContent(type="text", text="Result text")]))],
                [{'type': 'tool_result', 'tool_use_id': 'tool1', 'content': [{'type': 'text', 'text': 'Result text'}], 'is_error': False}]
            ),
            # Single image result
            (
                [("tool1", CallToolResult(content=[ImageContent(type="image", mimeType="image/png", data=DUMMY_IMAGE_B64)]))],
                [{'type': 'tool_result', 'tool_use_id': 'tool1', 'content': [{'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png', 'data': DUMMY_IMAGE_B64}}], 'is_error': False}]
            ),
            # Error result
            (
                [("tool1", CallToolResult(content=[TextContent(type="text", text="Failed")], isError=True))],
                [{'type': 'tool_result', 'tool_use_id': 'tool1', 'content': [{'type': 'text', 'text': 'Failed'}], 'is_error': True}]
            ),
            # Empty content result
            (
                [("tool1", CallToolResult(content=[]))],
                [{'type': 'tool_result', 'tool_use_id': 'tool1', 'content': [{'type': 'text', 'text': '[No content in tool result]'}], 'is_error': False}]
            ),
            # PDF result -> Splits into tool_result and document blocks
            (
                [("tool1", CallToolResult(content=[create_pdf_resource(DUMMY_PDF_B64)]))],
                [
                    {'type': 'tool_result', 'tool_use_id': 'tool1', 'content': [{'type': 'text', 'text': '[No content in tool result]'}], 'is_error': False},
                    {'type': 'document', 'title': 'document.pdf', 'source': {'type': 'base64', 'media_type': 'application/pdf', 'data': DUMMY_PDF_B64}}
                ]
            ),
            # Text + PDF result -> Splits correctly
            (
                [("tool1", CallToolResult(content=[TextContent(type="text", text="See attached PDF."), create_pdf_resource(DUMMY_PDF_B64)]))],
                [
                    {'type': 'tool_result', 'tool_use_id': 'tool1', 'content': [{'type': 'text', 'text': 'See attached PDF.'}], 'is_error': False},
                    {'type': 'document', 'title': 'document.pdf', 'source': {'type': 'base64', 'media_type': 'application/pdf', 'data': DUMMY_PDF_B64}}
                ]
            ),
            # Multiple tool results
            (
                [
                    ("tool1", CallToolResult(content=[TextContent(type="text", text="First result")])),
                    ("tool2", CallToolResult(content=[ImageContent(type="image", mimeType="image/jpeg", data=DUMMY_IMAGE_B64)]))
                ],
                [
                    {'type': 'tool_result', 'tool_use_id': 'tool1', 'content': [{'type': 'text', 'text': 'First result'}], 'is_error': False},
                    {'type': 'tool_result', 'tool_use_id': 'tool2', 'content': [{'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/jpeg', 'data': DUMMY_IMAGE_B64}}], 'is_error': False}
                ]
            ),
        ]
    )
    def test_create_tool_results_message(self, tool_results: List[Tuple[str, CallToolResult]], expected_anthropic_content: List[Dict[str, Any]]):
        """Tests the creation of a user message from a list of tool results."""
        user_message = AnthropicConverter.create_tool_results_message(tool_results)
        
        assert user_message["role"] == "user"

        actual_json = json.dumps(to_jsonable_python(user_message["content"]), sort_keys=True)
        expected_json = json.dumps(expected_anthropic_content, sort_keys=True)

        assert actual_json == expected_json
        
    # --- Test Anthropic -> MCP Message Conversion ---
    @pytest.mark.parametrize(
        "anthropic_message, expected_mcp_multipart",
        [
            (MessageParam(role="assistant", content=[{"type": "text", "text": "Response text"}]), PromptMessageMultipart(role="assistant", content=[TextContent(type="text", text="Response text")])),
            (MessageParam(role="user", content=[{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": DUMMY_IMAGE_B64}}]), PromptMessageMultipart(role="user", content=[ImageContent(type="image", mimeType="image/jpeg", data=DUMMY_IMAGE_B64)])),
            (MessageParam(role="user", content=[{"type": "document", "name": "report.txt", "source": {"type": "text", "media_type": "text/plain", "data": "This is the report."}}]), PromptMessageMultipart(role="user", content=[EmbeddedResource(type="resource", resource=TextResourceContents(uri="file:///report.txt", mimeType="text/plain", text="This is the report."))])),
            (MessageParam(role="user", content=[{"type": "document", "name": "file.pdf", "source": {"type": "base64", "media_type": "application/pdf", "data": DUMMY_PDF_B64}}]), PromptMessageMultipart(role="user", content=[EmbeddedResource(type="resource", resource=BlobResourceContents(uri="file:///file.pdf", mimeType="application/pdf", blob=DUMMY_PDF_B64))])),
        ]
    )
    def test_convert_from_anthropic_to_multipart(self, anthropic_message: MessageParam, expected_mcp_multipart: PromptMessageMultipart):
        """Test conversion from Anthropic's MessageParam to PromptMessageMultipart."""
        result = AnthropicConverter.convert_from_anthropic_to_multipart(anthropic_message)
        assert result.role == expected_mcp_multipart.role
        assert result.content == expected_mcp_multipart.content

    def test_convert_from_anthropic_list_to_multipart(self):
        """Tests converting a list of Anthropic messages."""
        anthropic_messages = [MessageParam(role="user", content="Hello"), MessageParam(role="assistant", content="I am an assistant.")]
        mcp_messages = AnthropicConverter.convert_from_anthropic_list_to_multipart(anthropic_messages)
        assert len(mcp_messages) == 2
        assert mcp_messages[0].role == "user"
        assert mcp_messages[0].content[0].text == "Hello"
        assert mcp_messages[1].role == "assistant"
        assert mcp_messages[1].content[0].text == "I am an assistant."

    def test_convert_prompt_message_to_anthropic(self):
        """Tests conversion of a single-content PromptMessage to Anthropic format."""
        prompt_message = PromptMessage(role="user", content=ImageContent(type="image", data=DUMMY_IMAGE_B64, mimeType="image/jpeg"))
        anthropic_msg = AnthropicConverter.convert_prompt_message_to_anthropic(prompt_message)
        
        assert anthropic_msg["role"] == "user"
        assert len(anthropic_msg["content"]) == 1
        assert anthropic_msg["content"][0]["type"] == "image"
        assert anthropic_msg["content"][0]["source"]["type"] == "base64"
        assert anthropic_msg["content"][0]["source"]["media_type"] == "image/jpeg"