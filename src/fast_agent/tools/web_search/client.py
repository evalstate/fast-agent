"""Auth-independent, single-attempt HTTP transport for standalone search."""

from collections.abc import Mapping
from typing import Literal, Self

import httpx
from pydantic import ValidationError

from fast_agent.tools.web_search.models import SearchRequest, SearchResponse


class WebSearchError(Exception):
    """Bounded diagnostic without server bodies, request data, URLs, or credentials."""

    def __init__(
        self,
        kind: Literal["http", "transport", "response"],
        status_code: int | None = None,
    ) -> None:
        self.kind = kind
        self.status_code = status_code
        message = f"Web search {kind} error"
        if status_code is not None:
            message += f" (HTTP {status_code})"
        super().__init__(message)


class WebSearchClient:
    """POST to ``base_url/alpha/search`` without retries or redirects.

    Supply API headers explicitly; no provider credentials are discovered. An injected
    client remains caller-owned (including any transport retry configuration). The
    timeout applies per HTTP operation, not as an overall search deadline.
    """

    def __init__(
        self,
        *,
        base_url: str,
        headers: Mapping[str, str] | None = None,
        http_client: httpx.AsyncClient | None = None,
        timeout: float | httpx.Timeout = 60.0,
    ) -> None:
        url = httpx.URL(base_url)
        if url.scheme not in {"http", "https"} or not url.host:
            raise ValueError("base_url must be an absolute HTTP(S) URL")
        if url.query or url.fragment or url.userinfo:
            raise ValueError("base_url must not contain credentials, query, or fragment")
        self._url = base_url.rstrip("/") + "/alpha/search"
        self._headers = dict(headers or {})
        self._owns_client = http_client is None
        self._http = http_client if http_client is not None else httpx.AsyncClient()
        self._timeout = timeout
        self._closed = False

    async def search(self, request: SearchRequest) -> SearchResponse:
        if self._closed:
            raise RuntimeError("WebSearchClient is closed")
        try:
            async with self._http.stream(
                "POST",
                self._url,
                headers=self._headers,
                json=request.model_dump(mode="json", exclude_none=True),
                timeout=self._timeout,
                follow_redirects=False,
            ) as response:
                if not response.is_success:
                    raise WebSearchError("http", response.status_code)
                body = await response.aread()
        except httpx.HTTPError:
            raise WebSearchError("transport") from None
        try:
            return SearchResponse.model_validate_json(body)
        except ValidationError:
            raise WebSearchError("response") from None

    async def aclose(self) -> None:
        if self._owns_client:
            await self._http.aclose()
        self._closed = True

    async def __aenter__(self) -> Self:
        if self._closed:
            raise RuntimeError("WebSearchClient is closed")
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.aclose()
