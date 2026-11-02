"""Recognize image paths pasted/uploaded by a terminal, including remote Herdr."""

from __future__ import annotations

import os
import shlex
from pathlib import Path

from fast_agent.mcp.mime_utils import guess_mime_type, is_image_mime_type
from fast_agent.ui.prompt.attachment_tokens import (
    build_local_attachment_token,
    normalize_local_attachment_reference,
)


def _image_path(value: str) -> Path | None:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        value = value[1:-1]
    # Never reinterpret ordinary words, URLs, code or relative paths as uploads.
    if not (value.startswith(("file://", "~/", "~\\")) or Path(value).is_absolute()):
        return None
    try:
        path = normalize_local_attachment_reference(value)
        if is_image_mime_type(guess_mime_type(str(path))) and path.is_file():
            return path
    except (OSError, ValueError, RuntimeError):
        pass
    return None


def pasted_image_tokens(text: str) -> list[str]:
    """Convert only a whole paste consisting of existing image paths.

    Paths are local to the process running fast-agent: Herdr has already
    uploaded a remote clipboard image before pasting its remote path.
    Support raw paths with spaces, shell quoting/escaping, and newline lists.
    No clipboard access, file upload, or shell execution happens here.
    """
    value = text.strip()
    if not value or "\x00" in value:
        return []
    if path := _image_path(value):
        return [build_local_attachment_token(path)]
    candidates = [value.splitlines()]
    try:
        candidates.append(shlex.split(value, posix=os.name != "nt"))
    except ValueError:
        pass
    for parts in candidates:
        if not parts:
            continue
        paths = [_image_path(part.strip()) for part in parts]
        if all(path is not None for path in paths):
            return [build_local_attachment_token(path) for path in paths if path is not None]
    return []
