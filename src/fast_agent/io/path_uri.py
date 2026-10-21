from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname

from fast_agent.utils.text import strip_casefold

if TYPE_CHECKING:
    from collections.abc import Callable
    from urllib.parse import ParseResult


def file_uri_to_path(
    parsed: ParseResult,
    *,
    pathname_decoder: Callable[[str], str] = url2pathname,
) -> Path:
    if strip_casefold(parsed.scheme) != "file":
        raise ValueError(f"Expected file URI, got {parsed.scheme or '<missing>'}")

    uri_path = parsed.path
    if parsed.netloc and strip_casefold(parsed.netloc) != "localhost":
        uri_path = f"//{parsed.netloc}{uri_path}"
        if pathname_decoder is url2pathname:
            return Path(unquote(uri_path))
    return Path(pathname_decoder(uri_path))


def local_file_url_path(file_url: str) -> Path | None:
    """Return the existing local file a ``file://`` URL names, or ``None``.

    Accepts RFC 8089 URLs (as built by ``Path.as_uri()``) and the unencoded
    ``f"file://{path}"`` form, including native Windows drive paths.
    """
    if not file_url.startswith("file://"):
        return None
    unencoded = Path(file_url.removeprefix("file://"))
    if unencoded.drive:
        candidates: tuple[Path, ...] = (unencoded,)
    else:
        candidates = (file_uri_to_path(urlparse(file_url)), unencoded)
    return next((path for path in candidates if path.is_file()), None)
