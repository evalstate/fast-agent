from base64 import b64encode
from pathlib import Path

from fastmcp import FastMCP
from fastmcp.prompts import Message
from mcp_types import BlobResourceContents, EmbeddedResource

app = FastMCP("Prompt and Resource Fixture")
_FIXTURE_DIR = Path(__file__).parent


def _pdf_resource() -> EmbeddedResource:
    return EmbeddedResource(
        resource=BlobResourceContents(
            uri="resource://fast-agent/sample.pdf",
            mime_type="application/pdf",
            blob=b64encode((_FIXTURE_DIR / "sample.pdf").read_bytes()).decode(),
        )
    )


@app.prompt(name="simple")
def simple(name: str) -> str:
    return f"Repeat the following text verbatim: {name}"


@app.prompt(name="with_attachment")
def with_attachment() -> list[Message]:
    return [
        Message("Good morning, how are you?"),
        Message(
            "Very well thank you, can I help you by summarising documents?",
            role="assistant",
        ),
        Message("Can you summarise this document please. Make sure to include the company name."),
        Message(_pdf_resource()),
    ]


@app.prompt(name="multiturn")
def multiturn() -> list[Message]:
    return [
        Message("l l M i n d s ET uk"),
        Message("llmindsetuk", role="assistant"),
        Message("fA st age NT"),
        Message("fastagent", role="assistant"),
        Message("m ORE training OK"),
        Message("moretrainingok", role="assistant"),
        Message("t ESt ca seOK"),
    ]


@app.resource("resource://fast-agent/sample.pdf", mime_type="application/pdf")
def sample_pdf() -> bytes:
    return (_FIXTURE_DIR / "sample.pdf").read_bytes()


@app.resource("resource://fast-agent/style.css", mime_type="text/css")
def stylesheet() -> str:
    return (_FIXTURE_DIR / "style.css").read_text()


if __name__ == "__main__":
    app.run(transport="stdio")
