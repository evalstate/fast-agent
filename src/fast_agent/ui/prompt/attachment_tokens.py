"""Helpers for inline attachment tokens."""

from __future__ import annotations

import os
import re
from pathlib import Path
from urllib.parse import quote, unquote, urlparse
from urllib.request import url2pathname

FILE_MENTION_SERVER = "file"
URL_MENTION_SERVER = "url"
_ATTACHMENT_TOKEN_RE = re.compile(r"(?P<prefix>^|\s)(?P<token>\^(?:file|url):[^\s]+)")
_ATTACHMENT_BODY_RE = r"\^(?:file|url):[^\s]+"


def normalize_local_attachment_reference(
    reference: str,
    *,
    cwd: Path | None = None,
) -> Path:
    """Normalize a ``^file:...`` payload into an absolute local path."""
    raw_value = reference.strip()
    if not raw_value:
        raise ValueError("Attachment path is empty")

    decoded_value = unquote(raw_value)
    path_value = os.path.expandvars(decoded_value)

    if path_value.lower().startswith("file://"):
        parsed = urlparse(path_value)
        if parsed.scheme.lower() != "file":
            raise ValueError(f"Unsupported attachment URI scheme: {parsed.scheme}")
        uri_path = parsed.path
        if parsed.netloc and parsed.netloc.lower() != "localhost":
            uri_path = f"//{parsed.netloc}{uri_path}"
        if not uri_path:
            raise ValueError("Attachment URI path is empty")
        resolved_path = Path(url2pathname(uri_path))
    else:
        resolved_path = Path(os.path.expanduser(path_value))

    if not resolved_path.is_absolute():
        resolved_path = (cwd or Path.cwd()) / resolved_path

    return resolved_path.resolve(strict=False)


def normalize_remote_attachment_reference(reference: str) -> str:
    """Normalize an HTTP(S) attachment reference into a remote URL."""
    raw_value = reference.strip()
    if not raw_value:
        raise ValueError("Attachment URL is empty")

    parsed = urlparse(raw_value)
    scheme = parsed.scheme.lower()
    if scheme not in ("http", "https"):
        raise ValueError(f"Unsupported attachment URI scheme: {parsed.scheme or '<missing>'}")
    if not parsed.netloc:
        raise ValueError("Attachment URL is missing host")
    return raw_value


def encode_local_attachment_reference(path_text: str) -> str:
    """Percent-encode a token path while keeping it compact and path-like."""
    normalized = path_text.replace("\\", "/")
    return quote(normalized, safe="/._~-:")


def build_local_attachment_token(path: str | Path) -> str:
    """Build a canonical ``^file:...`` token for a local path."""
    if not isinstance(path, Path):
        path = normalize_local_attachment_reference(path)
    normalized = path.resolve(strict=False)
    return f"^{FILE_MENTION_SERVER}:{encode_local_attachment_reference(normalized.as_posix())}"


def build_remote_attachment_token(url: str) -> str:
    """Build a canonical ``^url:...`` token for a remote URL."""
    normalized = normalize_remote_attachment_reference(url)
    return f"^{URL_MENTION_SERVER}:{quote(normalized, safe='/._~-:?&=#%')}"


def strip_local_attachment_tokens(text: str) -> str:
    """Remove inline attachment tokens while preserving other text."""
    stripped = re.sub(
        rf"(^|\n)[ \t]*{_ATTACHMENT_BODY_RE}[ \t]*(?:\n|$)",
        lambda match: match.group(1),
        text,
        flags=re.MULTILINE,
    )
    stripped = re.sub(
        rf"(?P<lead>[ \t]){_ATTACHMENT_BODY_RE}(?P<trail>[ \t])",
        r"\g<lead>",
        stripped,
    )
    stripped = re.sub(
        rf"(?P<lead>[ \t]+){_ATTACHMENT_BODY_RE}(?=$|\n)",
        "",
        stripped,
    )
    stripped = re.sub(
        rf"(?:(?<=^)|(?<=\s)){_ATTACHMENT_BODY_RE}(?P<trail>[ \t]+)",
        "",
        stripped,
        flags=re.MULTILINE,
    )
    stripped = re.sub(
        rf"(?:(?<=^)|(?<=\s)){_ATTACHMENT_BODY_RE}",
        "",
        stripped,
        flags=re.MULTILINE,
    )
    return stripped


def append_attachment_tokens(text: str, tokens: list[str]) -> str:
    """Append attachment tokens to existing draft text."""
    if not tokens:
        return text
    if not text:
        return " ".join(tokens)
    if text[-1].isspace():
        return f"{text}{' '.join(tokens)}"
    return f"{text} {' '.join(tokens)}"
