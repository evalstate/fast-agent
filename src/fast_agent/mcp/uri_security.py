"""URI scheme checks used at remote-content trust boundaries."""

from urllib.parse import urlsplit

SANITIZED_INLINE_RESOURCE_URI = "urn:fast-agent:remote-mcp-inline"


def is_file_uri(uri: str) -> bool:
    """Return whether *uri* uses the file scheme, regardless of spelling."""
    scheme, separator, _ = uri.lstrip().partition(":")
    if separator and scheme.casefold() == "file":
        return True
    try:
        return urlsplit(uri).scheme.casefold() == "file"
    except ValueError:
        return False
