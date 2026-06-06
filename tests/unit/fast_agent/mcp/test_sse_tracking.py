from fast_agent.mcp.sse_tracking import _same_origin


def test_same_origin_normalizes_hostname_case() -> None:
    assert _same_origin("https://Example.com/sse", "https://example.com/messages")


def test_same_origin_normalizes_scheme_case() -> None:
    assert _same_origin("HTTPS://example.com:443/sse", "https://example.com/messages")


def test_same_origin_treats_default_ports_as_equivalent() -> None:
    assert _same_origin("https://example.com:443/sse", "https://example.com/messages")
    assert _same_origin("http://example.com/sse", "http://example.com:80/messages")


def test_same_origin_rejects_different_ports_and_hosts() -> None:
    assert not _same_origin("https://example.com/sse", "https://example.com:8443/messages")
    assert not _same_origin("https://example.com/sse", "https://other.example.com/messages")
