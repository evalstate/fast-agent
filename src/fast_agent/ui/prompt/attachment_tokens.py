"""Helpers for inline attachment tokens."""

from __future__ import annotations

import os
import re
from pathlib import Path
from urllib.parse import quote, unquote, urlparse

from fast_agent.io.path_uri import file_uri_to_path
from fast_agent.utils.text import starts_with_casefold, strip_casefold

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

    if starts_with_casefold(path_value, "file://"):
        parsed = urlparse(path_value)
        if strip_casefold(parsed.scheme) != "file":
            raise ValueError(f"Unsupported attachment URI scheme: {parsed.scheme}")
        if not parsed.path:
            raise ValueError("Attachment URI path is empty")
        resolved_path = file_uri_to_path(parsed)
    else:
        resolved_path = Path(path_value).expanduser()

    if not resolved_path.is_absolute():
        resolved_path = (cwd or Path.cwd()) / resolved_path

    return resolved_path.resolve(strict=False)


def normalize_remote_attachment_reference(reference: str) -> str:
    """Normalize an HTTP(S) attachment reference into a remote URL."""
    raw_value = reference.strip()
    if not raw_value:
        raise ValueError("Attachment URL is empty")

    parsed = urlparse(raw_value)
    scheme = strip_casefold(parsed.scheme)
    if scheme not in ("http", "https"):
        raise ValueError(f"Unsupported attachment URI scheme: {parsed.scheme or '<missing>'}")
    if not parsed.netloc:
        raise ValueError("Attachment URL is missing host")
    return raw_value


def is_remote_attachment_reference(reference: str) -> bool:
    """Return true when the reference is an HTTP(S) attachment URL."""
    scheme = strip_casefold(urlparse(reference.strip()).scheme)
    return scheme in ("http", "https")


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
    return re.sub(
        rf"(?:(?<=^)|(?<=\s)){_ATTACHMENT_BODY_RE}",
        "",
        stripped,
        flags=re.MULTILINE,
    )


def append_attachment_tokens(text: str, tokens: list[str]) -> str:
    """Append attachment tokens to existing draft text."""
    if not tokens:
        return text
    token_text = " ".join(tokens)
    if not text:
        return token_text
    separator = "" if text[-1].isspace() else " "
    return f"{text}{separator}{token_text}"
