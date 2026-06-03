from httpx import HTTPStatusError, Request, Response

from fast_agent.mcp.http_errors import format_http_error_detail


def test_format_http_error_detail_uses_stripped_response_text() -> None:
    request = Request("GET", "https://example.com/mcp")
    response = Response(599, content=b"  upstream unavailable  ", request=request)
    error = HTTPStatusError("failed", request=request, response=response)

    detail = format_http_error_detail(error)

    assert detail.status_code == 599
    assert detail.detail == "HTTP 599: upstream unavailable"


def test_format_http_error_detail_uses_response_fallback_for_blank_text() -> None:
    request = Request("GET", "https://example.com/mcp")
    response = Response(599, content=b"   ", request=request)
    error = HTTPStatusError("failed", request=request, response=response)

    detail = format_http_error_detail(error)

    assert detail.status_code == 599
    assert detail.detail == "HTTP 599: response"

