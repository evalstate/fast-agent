import base64
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from mcp_types import (
    BlobResourceContents,
    EmbeddedResource,
    ImageContent,
    TextResourceContents,
)
from pydantic import AnyUrl, TypeAdapter

import fast_agent.mcp.mime_utils as mime_utils


@dataclass(frozen=True, slots=True)
class ResourceContent:
    content: str
    mime_type: str
    is_binary: bool


def find_resource_file(resource_path: str, prompt_files: list[Path]) -> Path | None:
    """Find a resource file relative to one of the prompt files"""
    for prompt_file in prompt_files:
        potential_path = prompt_file.parent / resource_path
        if potential_path.exists():
            return potential_path
    return None


def load_resource_content(resource_path: str, prompt_files: list[Path]) -> ResourceContent:
    """
    Load a resource's content and determine its mime type

    Args:
        resource_path: Path to the resource file
        prompt_files: List of prompt files (to find relative paths)

    Returns:
        Resource content with MIME metadata
        - content: String content for text files, base64-encoded string for binary files
        - mime_type: The MIME type of the resource
        - is_binary: Whether the content is binary (and base64-encoded)

    Raises:
        FileNotFoundError: If the resource cannot be found
    """
    # Try to locate the resource file
    resource_file = find_resource_file(resource_path, prompt_files)
    if resource_file is None:
        raise FileNotFoundError(f"Resource not found: {resource_path}")

    # Determine mime type
    mime_type = mime_utils.guess_mime_type(str(resource_file))
    is_binary = mime_utils.is_binary_content(mime_type)

    if is_binary:
        # For binary files, read as binary and base64 encode
        with resource_file.open("rb") as f:
            content = base64.b64encode(f.read()).decode("utf-8")
    else:
        # For text files, read as text
        with resource_file.open("r", encoding="utf-8") as f:
            content = f.read()

    return ResourceContent(content=content, mime_type=mime_type, is_binary=is_binary)


# Create a safe way to generate resource URIs that Pydantic accepts
_ANY_URL_ADAPTER = TypeAdapter(AnyUrl)


def to_any_url(value: str | AnyUrl) -> AnyUrl:
    """Normalize a URI string to AnyUrl (validates via pydantic)."""
    return _ANY_URL_ADAPTER.validate_python(value)


def create_resource_uri(path: str) -> str:
    """Create a resource URI from a path"""
    return f"resource://fast-agent/{Path(path).name}"


def create_embedded_resource(
    resource_path: str, content: str, mime_type: str, is_binary: bool = False
) -> EmbeddedResource:
    """Create an embedded resource content object"""
    # Format a valid resource URI string
    resource_uri_str = create_resource_uri(resource_path)

    if is_binary:
        return EmbeddedResource(
            type="resource",
            resource=BlobResourceContents(
                uri=resource_uri_str,
                mime_type=mime_type,
                blob=content,
            ),
        )
    return EmbeddedResource(
        type="resource",
        resource=TextResourceContents(
            uri=resource_uri_str,
            mime_type=mime_type,
            text=content,
        ),
    )


def create_image_content(data: str, mime_type: str) -> ImageContent:
    """Create an image content object from base64-encoded data"""
    return ImageContent(
        type="image",
        data=data,
        mime_type=mime_type,
    )


def extract_title_from_uri(uri: AnyUrl | str) -> str:
    """Extract a readable title from a URI."""
    uri_str = str(uri)
    path = urlsplit(uri_str).path
    return Path(path).name if path else uri_str
