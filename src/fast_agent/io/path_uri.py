from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable
from urllib.request import url2pathname

if TYPE_CHECKING:
    from urllib.parse import ParseResult


def file_uri_to_path(
    parsed: ParseResult,
    *,
    pathname_decoder: Callable[[str], str] = url2pathname,
) -> Path:
    uri_path = parsed.path
    if parsed.netloc and parsed.netloc.lower() != "localhost":
        uri_path = f"//{parsed.netloc}{uri_path}"
    return Path(pathname_decoder(uri_path))
