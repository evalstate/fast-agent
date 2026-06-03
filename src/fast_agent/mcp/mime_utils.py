# mime_utils.py

import mimetypes

from fast_agent.utils.text import strip_casefold

# Initialize mimetypes database
mimetypes.init()

# Extend with additional types that might be missing
ADDITIONAL_MIME_TYPES = (
    ("text/x-python", ".py"),
    ("image/webp", ".webp"),
    ("application/msword", ".doc"),
    ("application/vnd.ms-excel", ".xls"),
    ("application/vnd.ms-powerpoint", ".ppt"),
    (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".docx",
    ),
    (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".xlsx",
    ),
    (
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        ".pptx",
    ),
)
for mime_type, extension in ADDITIONAL_MIME_TYPES:
    mimetypes.add_type(mime_type, extension)

# Known text-based MIME types not starting with "text/"
TEXT_MIME_TYPES = {
    "application/json",
    "application/javascript",
    "application/xml",
    "application/ld+json",
    "application/xhtml+xml",
    "application/x-httpd-php",
    "application/x-sh",
    "application/ecmascript",
    "application/graphql",
    "application/x-www-form-urlencoded",
    "application/yaml",
    "application/toml",
    "application/x-python-code",
    "application/vnd.api+json",
}

# Common text-based MIME type patterns
TEXT_MIME_PATTERNS = ("+xml", "+json", "+yaml", "+text")

DOCUMENT_MIME_TYPES = (
    "application/pdf",
    "application/msword",
    "application/vnd.ms-excel",
    "application/vnd.ms-powerpoint",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
)


def guess_mime_type(file_path: str) -> str:
    """
    Guess the MIME type of a file based on its extension.
    """
    mime_type, _ = mimetypes.guess_type(file_path)
    return mime_type or "application/octet-stream"


def is_text_mime_type(mime_type: str) -> bool:
    """Determine if a MIME type represents text content."""
    normalized = normalize_mime_type(mime_type)
    return bool(
        normalized
        and (
            normalized.startswith("text/")
            or normalized in TEXT_MIME_TYPES
            or normalized.endswith(TEXT_MIME_PATTERNS)
        )
    )


def is_binary_content(mime_type: str) -> bool:
    """Check if content should be treated as binary."""
    return not is_text_mime_type(mime_type)


def is_image_mime_type(mime_type: str) -> bool:
    """Check if a MIME type represents an image."""
    normalized = normalize_mime_type(mime_type)
    return bool(normalized and normalized.startswith("image/") and normalized != "image/svg+xml")


def is_document_mime_type(mime_type: str) -> bool:
    """Check if a MIME type represents a document attachment."""
    normalized = normalize_mime_type(mime_type)
    return normalized in DOCUMENT_MIME_TYPES


# Common reference mapping and normalization helpers
_MIME_ALIASES = {
    # Friendly or non-standard labels
    "document/pdf": "application/pdf",
    "image/jpg": "image/jpeg",
    # Some providers sometimes return these variants
    "application/x-pdf": "application/pdf",
}


def normalize_mime_type(mime: str | None) -> str | None:
    """
    Normalize a MIME-like string to a canonical MIME type.

    - Lowercases and trims
    - Resolves common aliases (e.g. image/jpg -> image/jpeg, document/pdf -> application/pdf)
    - If input looks like a bare extension (e.g. "pdf", "png"), map via mimetypes
    - Returns None for falsy inputs
    """
    if not mime:
        return None

    m = strip_casefold(mime)

    # If it's an alias we know about
    if m in _MIME_ALIASES:
        return _MIME_ALIASES[m]

    # If it already looks like a full MIME type
    if "/" in m:
        # image/jpg -> image/jpeg etc.
        return _MIME_ALIASES.get(m, m)

    # Treat as a bare file extension (e.g. "pdf", "png")
    if not m.startswith("."):
        m = "." + m
    return mimetypes.types_map.get(m, None)
