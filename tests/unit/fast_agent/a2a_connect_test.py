from fast_agent.a2a.connect import (
    normalize_a2a_transport,
    normalize_a2a_url,
    parse_a2a_connect_arguments,
)


def test_normalize_a2a_transport_aliases() -> None:
    assert normalize_a2a_transport("json-rpc") == "JSONRPC"
    assert normalize_a2a_transport("rest") == "HTTP+JSON"
    assert normalize_a2a_transport("grpc") is None
    assert normalize_a2a_transport("bogus") is None


def test_normalize_a2a_base_url() -> None:
    url, card_path, error = normalize_a2a_url("http://127.0.0.1:41241/")
    assert url == "http://127.0.0.1:41241"
    assert card_path is None
    assert error is None


def test_normalize_a2a_agent_card_url() -> None:
    url, card_path, error = normalize_a2a_url(
        "http://127.0.0.1:41241/.well-known/agent-card.json"
    )
    assert url == "http://127.0.0.1:41241"
    assert card_path == "/.well-known/agent-card.json"
    assert error is None


def test_parse_a2a_connect_arguments() -> None:
    request, error = parse_a2a_connect_arguments(
        'http://127.0.0.1:41241 --transport rest --name "remote docs" --card-path /card.json'
    )
    assert error is None
    assert request is not None
    assert request.url == "http://127.0.0.1:41241"
    assert request.transport == "HTTP+JSON"
    assert request.name == "remote_docs"
    assert request.relative_card_path == "/card.json"


def test_parse_a2a_connect_oauth_switches() -> None:
    request, error = parse_a2a_connect_arguments("http://127.0.0.1:41241 --oauth")
    assert error is None
    assert request is not None
    assert request.auth is not None
    assert request.auth.oauth is True

    request, error = parse_a2a_connect_arguments("http://127.0.0.1:41241 --no-oauth")
    assert error is None
    assert request is not None
    assert request.auth is not None
    assert request.auth.oauth is False


def test_parse_a2a_connect_rejects_endpointless_url() -> None:
    request, error = parse_a2a_connect_arguments("127.0.0.1:41241")
    assert request is None
    assert error == "A2A connect expects an http(s) base URL or agent-card URL"
