#!/usr/bin/env python3
"""
Simple MCP server that responds to tool calls with text and image content.
"""

import base64
import logging
import sys
from pathlib import Path

from mcp.server.fastmcp import Context, FastMCP, Image
from mcp.types import BlobResourceContents, EmbeddedResource, ImageContent, TextContent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create the FastMCP server
app = FastMCP(name="ImageToolServer", debug=True)

# Global variable to store the image path
image_path = "image.png"


@app.tool(name="get_image", description="Returns the sample image with some descriptive text")
async def get_image(image_name: str = "default", ctx: Context = None) -> list[TextContent | ImageContent]:
    try:
        # Use the global image path
        return [
            TextContent(type="text", text="Here's your image:"),
            Image(path=image_path).to_image_content(),
        ]
    except Exception as e:
        logger.exception(f"Error processing image: {e}")
        return [TextContent(type="text", text=f"Error processing image: {str(e)}")]


@app.tool(
    name="get_pdf",
    description="Returns 'sample.pdf' - use when the User requests a sample PDF file",
)
async def get_pdf() -> list[TextContent | EmbeddedResource]:
    try:
        pdf_path = "sample.pdf"
        # Check if file exists
        if not Path(pdf_path).exists():
            return [TextContent(type="text", text=f"Error: PDF file '{pdf_path}' not found")]

        # Read the PDF file as binary data
        with open(pdf_path, "rb") as f:
            pdf_data = f.read()

        # Encode to base64
        b64_data = base64.b64encode(pdf_data).decode("ascii")

        # Create embedded resource
        return [
            TextContent(type="text", text="Here is the PDF"),
            EmbeddedResource(
                type="resource",
                resource=BlobResourceContents(
                    uri=f"file://{Path(pdf_path).absolute()}",
                    blob=b64_data,
                    mimeType="application/pdf",
                ),
            ),
        ]
    except Exception as e:
        logger.exception(f"Error processing PDF: {e}")
        return [TextContent(type="text", text=f"Error processing PDF: {str(e)}")]


if __name__ == "__main__":
    # Get image path from command line argument or use default
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        logger.info(f"Using image file: {image_path}")
    else:
        logger.info(f"No image path provided, using default: {image_path}")

    # Check if the specified image exists
    if not Path(image_path).exists():
        logger.warning(f"Image file '{image_path}' not found in the current directory")
        logger.warning("Please add an image file or specify a valid path as the first argument")

    # Run the server using stdio transport
    app.run(transport="stdio")
