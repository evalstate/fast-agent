"""HTTP error formatting shared by MCP transports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from fast_agent.utils.text import strip_to_none

if TYPE_CHECKING:
    import httpx


@dataclass(frozen=True, slots=True)
class HttpErrorDetail:
    status_code: int | None
    detail: str


def format_http_error_detail(exc: httpx.HTTPStatusError) -> HttpErrorDetail:
    if exc.response is None:
        return HttpErrorDetail(status_code=None, detail=str(exc))

    status_code = exc.response.status_code
    reason = exc.response.reason_phrase or _response_text(exc.response)
    return HttpErrorDetail(
        status_code=status_code,
        detail=f"HTTP {status_code}: {reason or 'response'}",
    )


def _response_text(response: httpx.Response) -> str:
    try:
        return strip_to_none(response.text) or ""
    except Exception:
        return ""
