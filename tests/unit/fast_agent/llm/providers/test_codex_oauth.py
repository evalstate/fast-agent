import json
from pathlib import Path
from unittest.mock import Mock
from urllib.parse import parse_qs

import httpx
import pytest

from fast_agent.auth.credentials import OAuthCredential, StoredCredential, save_oauth_credential
from fast_agent.llm.provider.openai import codex_oauth
from fast_agent.llm.provider.openai.codex_oauth import CodexOAuthTokens


def test_resolve_codex_cli_auth_path_defaults_to_user_home(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("CODEX_AUTH_JSON_PATH", raising=False)
    monkeypatch.delenv("CODEX_HOME", raising=False)
    monkeypatch.setattr(codex_oauth.Path, "home", lambda: tmp_path)

    assert codex_oauth._resolve_codex_cli_auth_path() == tmp_path / ".codex" / "auth.json"


def test_explicit_auth_json_path_overrides_fast_agent_store(monkeypatch, tmp_path: Path) -> None:
    auth_path = tmp_path / "local-auth.json"
    auth_path.write_text(
        json.dumps(
            {
                "auth_mode": "oauth",
                "tokens": {
                    "access_token": "local-token",
                    "refresh_token": "local-refresh",
                    "token_type": "Bearer",
                },
            }
        )
    )
    monkeypatch.delenv("FAST_AGENT_AUTH_FILE", raising=False)
    monkeypatch.setenv("CODEX_AUTH_JSON_PATH", str(auth_path))
    monkeypatch.delenv("CODEX_HOME", raising=False)
    monkeypatch.setattr(
        codex_oauth,
        "load_oauth_credential",
        lambda provider: pytest.fail("auth.json must not access the fast-agent credential store"),
    )

    tokens, source = codex_oauth._load_codex_tokens_with_source()

    assert source == "auth.json"
    assert tokens is not None
    assert tokens.access_token == "local-token"


def test_default_auth_json_path_is_used_when_fast_agent_store_is_empty(
    monkeypatch, tmp_path: Path
) -> None:
    auth_path = tmp_path / ".codex" / "auth.json"
    auth_path.parent.mkdir(parents=True)
    auth_path.write_text(
        json.dumps(
            {
                "auth_mode": "oauth",
                "tokens": {
                    "access_token": "cli-token",
                    "refresh_token": "cli-refresh",
                    "token_type": "Bearer",
                },
            }
        )
    )
    monkeypatch.delenv("FAST_AGENT_AUTH_FILE", raising=False)
    monkeypatch.delenv("CODEX_AUTH_JSON_PATH", raising=False)
    monkeypatch.delenv("CODEX_HOME", raising=False)
    monkeypatch.setattr(codex_oauth.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(
        codex_oauth,
        "load_oauth_credential",
        lambda provider: pytest.fail("auth.json must not access the fast-agent credential store"),
    )

    tokens, source = codex_oauth._load_codex_tokens_with_source()

    assert source == "auth.json"
    assert tokens is not None
    assert tokens.access_token == "cli-token"


def test_codex_cli_credential_precedes_fast_agent_owned_credential(monkeypatch) -> None:
    monkeypatch.delenv("FAST_AGENT_AUTH_FILE", raising=False)
    monkeypatch.setattr(
        codex_oauth,
        "load_oauth_credential",
        lambda provider: StoredCredential(
            OAuthCredential(access_token="fast-agent-token"), "keyring"
        ),
    )
    monkeypatch.setattr(
        codex_oauth,
        "_load_codex_cli_tokens",
        lambda: CodexOAuthTokens(access_token="cli-token"),
    )

    tokens, source = codex_oauth._load_codex_tokens_with_source()

    assert source == "auth.json"
    assert tokens is not None
    assert tokens.access_token == "cli-token"


def test_fast_agent_auth_file_is_authoritative_over_codex_cli(monkeypatch, tmp_path: Path) -> None:
    portable_auth_path = tmp_path / "fast-agent-auth.json"
    cli_auth_path = tmp_path / "codex-auth.json"
    cli_auth_path.write_text(json.dumps({"tokens": {"access_token": "cli-token"}}))
    monkeypatch.setenv("FAST_AGENT_AUTH_FILE", str(portable_auth_path))
    monkeypatch.setenv("CODEX_AUTH_JSON_PATH", str(cli_auth_path))
    save_oauth_credential("codex", OAuthCredential(access_token="portable-token"))

    tokens, source = codex_oauth._load_codex_tokens_with_source()

    assert source == "file"
    assert tokens is not None
    assert tokens.access_token == "portable-token"


def test_fast_agent_auth_file_does_not_fall_back_to_codex_cli(monkeypatch, tmp_path: Path) -> None:
    portable_auth_path = tmp_path / "fast-agent-auth.json"
    cli_auth_path = tmp_path / "codex-auth.json"
    cli_auth_path.write_text(json.dumps({"tokens": {"access_token": "cli-token"}}))
    monkeypatch.setenv("FAST_AGENT_AUTH_FILE", str(portable_auth_path))
    monkeypatch.setenv("CODEX_AUTH_JSON_PATH", str(cli_auth_path))

    assert codex_oauth._load_codex_tokens_with_source() == (None, None)


def test_save_codex_tokens_prefers_fast_agent_auth_file(monkeypatch, tmp_path: Path) -> None:
    portable_auth_path = tmp_path / "fast-agent-auth.json"
    cli_auth_path = tmp_path / "codex-auth.json"
    cli_auth_path.write_text(json.dumps({"tokens": {"access_token": "cli-token"}}))
    monkeypatch.setenv("FAST_AGENT_AUTH_FILE", str(portable_auth_path))
    monkeypatch.setenv("CODEX_AUTH_JSON_PATH", str(cli_auth_path))

    codex_oauth.save_codex_tokens(CodexOAuthTokens(access_token="refreshed-portable-token"))

    portable_payload = json.loads(portable_auth_path.read_text())
    assert portable_payload["providers"]["codex"]["access_token"] == "refreshed-portable-token"
    cli_payload = json.loads(cli_auth_path.read_text())
    assert cli_payload["tokens"]["access_token"] == "cli-token"


def test_legacy_codex_keyring_credentials_print_reauthentication_warning(monkeypatch) -> None:
    messages: list[str] = []
    monkeypatch.delenv("FAST_AGENT_AUTH_FILE", raising=False)
    monkeypatch.setattr(codex_oauth, "_load_codex_cli_tokens", lambda: None)
    monkeypatch.setattr(codex_oauth, "load_oauth_credential", lambda provider: None)
    monkeypatch.setattr(codex_oauth, "_legacy_codex_keyring_credentials_present", lambda: True)
    monkeypatch.setattr(codex_oauth.console, "ensure_blocking_console", lambda: None)
    monkeypatch.setattr(
        codex_oauth.console.error_console,
        "print",
        lambda message, **kwargs: messages.append(message),
    )

    assert codex_oauth._load_codex_tokens_with_source() == (None, None)
    assert messages == [
        (
            "Legacy Codex credentials were found in the OS keyring but are no longer loaded. "
            "Run `fast-agent auth provider login codex` to authenticate again."
        )
    ]


@pytest.mark.parametrize("location", ["default", "codex_home", "explicit"])
def test_save_codex_tokens_never_modifies_codex_cli_auth_file(
    monkeypatch, tmp_path: Path, location: str
) -> None:
    monkeypatch.delenv("FAST_AGENT_AUTH_FILE", raising=False)
    monkeypatch.delenv("CODEX_AUTH_JSON_PATH", raising=False)
    monkeypatch.delenv("CODEX_HOME", raising=False)
    monkeypatch.setattr(codex_oauth.Path, "home", lambda: tmp_path)
    if location == "codex_home":
        monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex-profile"))
    elif location == "explicit":
        monkeypatch.setenv("CODEX_AUTH_JSON_PATH", str(tmp_path / "explicit-auth.json"))
    auth_path = codex_oauth._resolve_codex_cli_auth_path()
    auth_path.parent.mkdir(parents=True, exist_ok=True)
    original = json.dumps({"tokens": {"access_token": "cli-token"}})
    auth_path.write_text(original)
    saved: list[OAuthCredential] = []
    monkeypatch.setattr(codex_oauth, "load_oauth_credential", lambda provider: None)
    monkeypatch.setattr(
        codex_oauth,
        "save_oauth_credential",
        lambda provider, credential, source=None: saved.append(credential) or "keyring",
    )

    codex_oauth.save_codex_tokens(CodexOAuthTokens(access_token="refreshed-token"))

    assert auth_path.read_text() == original
    assert [credential.access_token for credential in saved] == ["refreshed-token"]


def test_clear_codex_tokens_preserves_cli_when_explicit_auth_file_is_authoritative(
    monkeypatch, tmp_path: Path
) -> None:
    portable_auth_path = tmp_path / "fast-agent-auth.json"
    cli_auth_path = tmp_path / "codex-profile" / "auth.json"
    cli_auth_path.parent.mkdir()
    original = json.dumps({"tokens": {"access_token": "cli-token"}})
    cli_auth_path.write_text(original)
    monkeypatch.setenv("FAST_AGENT_AUTH_FILE", str(portable_auth_path))
    monkeypatch.setenv("CODEX_HOME", str(cli_auth_path.parent))
    save_oauth_credential("codex", OAuthCredential(access_token="portable-token"))

    assert codex_oauth.clear_codex_tokens() is True
    assert cli_auth_path.read_text() == original
    assert codex_oauth._load_codex_tokens_with_source() == (None, None)


def test_fast_agent_auth_file_cannot_target_codex_cli_auth_file(
    monkeypatch, tmp_path: Path
) -> None:
    cli_auth_path = tmp_path / "codex-profile" / "auth.json"
    cli_auth_path.parent.mkdir()
    original = json.dumps({"tokens": {"access_token": "cli-token"}})
    cli_auth_path.write_text(original)
    monkeypatch.setenv("CODEX_HOME", str(cli_auth_path.parent))
    monkeypatch.setenv("FAST_AGENT_AUTH_FILE", str(cli_auth_path))

    with pytest.raises(codex_oauth.ProviderKeyError, match="read-only"):
        codex_oauth.save_codex_tokens(CodexOAuthTokens(access_token="replacement"))

    assert codex_oauth.clear_codex_tokens() is False
    assert cli_auth_path.read_text() == original


@pytest.mark.parametrize("location", ["default", "codex_home", "explicit"])
def test_clear_codex_tokens_never_modifies_codex_cli_auth_file(
    monkeypatch, tmp_path: Path, location: str
) -> None:
    monkeypatch.delenv("FAST_AGENT_AUTH_FILE", raising=False)
    monkeypatch.delenv("CODEX_AUTH_JSON_PATH", raising=False)
    monkeypatch.delenv("CODEX_HOME", raising=False)
    monkeypatch.setattr(codex_oauth.Path, "home", lambda: tmp_path)
    if location == "codex_home":
        monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex-profile"))
    elif location == "explicit":
        monkeypatch.setenv("CODEX_AUTH_JSON_PATH", str(tmp_path / "explicit-auth.json"))
    auth_path = codex_oauth._resolve_codex_cli_auth_path()
    auth_path.parent.mkdir(parents=True, exist_ok=True)
    original = json.dumps({"auth": {"access_token": "cli-token"}})
    auth_path.write_text(original)
    monkeypatch.setattr(codex_oauth, "delete_oauth_credential", lambda provider: False)

    assert codex_oauth.clear_codex_tokens() is False
    assert auth_path.read_text() == original


def test_tokens_from_response_rejects_bool_expires_in(monkeypatch) -> None:
    monkeypatch.setattr(codex_oauth.time, "time", lambda: 1000.0)

    bool_expiry = codex_oauth._tokens_from_response({"access_token": "token", "expires_in": True})
    valid_expiry = codex_oauth._tokens_from_response({"access_token": "token", "expires_in": 60})

    assert bool_expiry.expires_at is None
    assert valid_expiry.expires_at == 1060.0


def test_login_rejects_callback_without_oauth_state(monkeypatch) -> None:
    class CallbackWithoutState:
        def __init__(self, port: int) -> None:
            del port

        def start(self) -> None:
            pass

        def serve_once(self, timeout_seconds: int) -> tuple[str, None]:
            del timeout_seconds
            return "authorization-code", None

        def close(self) -> None:
            pass

    monkeypatch.setattr(codex_oauth, "_CallbackServer", CallbackWithoutState)
    monkeypatch.setattr(codex_oauth.console, "ensure_blocking_console", lambda: None)
    monkeypatch.setattr(codex_oauth.console.console, "print", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        codex_oauth,
        "exchange_code_for_tokens",
        lambda code, verifier: pytest.fail("token exchange must not run without OAuth state"),
    )
    monkeypatch.setattr(
        codex_oauth,
        "save_codex_tokens",
        lambda tokens: pytest.fail("tokens must not be saved without OAuth state"),
    )

    with pytest.raises(codex_oauth.ProviderKeyError, match="State parameter mismatch"):
        codex_oauth.login_codex_browser_oauth()


@pytest.fixture
def device_login(monkeypatch):

    clock = [0.0]
    requests: list[httpx.Request] = []
    responses: list[httpx.Response | Exception] = []
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        assert seconds > 0
        sleeps.append(seconds)
        clock[0] += seconds

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        response = responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    client_type = httpx.Client
    monkeypatch.setattr(
        codex_oauth.httpx,
        "Client",
        lambda **kwargs: client_type(transport=httpx.MockTransport(handle)),
    )
    monkeypatch.setattr(codex_oauth.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(codex_oauth.time, "sleep", sleep)
    monkeypatch.setattr(codex_oauth.console, "ensure_blocking_console", lambda: None)
    display = Mock()
    save = Mock()
    monkeypatch.setattr(codex_oauth.console.console, "print", display)
    monkeypatch.setattr(codex_oauth, "save_codex_tokens", save)
    monkeypatch.setattr(
        codex_oauth, "login_codex_browser_oauth", Mock(side_effect=AssertionError("no fallback"))
    )
    return responses, requests, sleeps, save, display


@pytest.mark.parametrize("alias", ["user_code", "usercode"])
def test_device_login_protocol(device_login, alias: str) -> None:

    responses, requests, sleeps, save, display = device_login
    responses.extend(
        [
            httpx.Response(
                200, json={"device_auth_id": "id", alias: "[bold]code", "interval": "2"}
            ),
            httpx.Response(403),
            httpx.Response(404),
            httpx.Response(
                200, json={"authorization_code": "auth-code", "code_verifier": "verifier"}
            ),
            httpx.Response(200, json={"access_token": "access", "refresh_token": "refresh"}),
        ]
    )
    tokens = codex_oauth.login_codex_oauth()
    assert tokens.access_token == "access"
    save.assert_called_once_with(tokens)
    assert sleeps == [2, 2]
    assert str(requests[0].url) == "https://auth.openai.com/api/accounts/deviceauth/usercode"
    assert json.loads(requests[0].content) == {"client_id": codex_oauth.CODEX_CLIENT_ID}
    for request in requests[1:4]:
        assert str(request.url) == "https://auth.openai.com/api/accounts/deviceauth/token"
        assert json.loads(request.content) == {"device_auth_id": "id", "user_code": "[bold]code"}

    exchange = parse_qs(requests[-1].content.decode())
    assert exchange["redirect_uri"] == ["https://auth.openai.com/deviceauth/callback"]
    assert exchange["code"] == ["auth-code"]
    assert exchange["code_verifier"] == ["verifier"]
    display.assert_any_call("[bold]code", markup=False)
    assert any("only if you started" in call.args[0] for call in display.call_args_list)


@pytest.mark.parametrize("interval,expected", [(None, [5, 1]), ("0", [1] * 6), ("999", [6])])
def test_device_timeout_has_bounded_nonzero_sleeps(device_login, interval, expected) -> None:

    responses, requests, sleeps, save, _ = device_login
    payload = {"device_auth_id": "id", "user_code": "code"}
    if interval is not None:
        payload["interval"] = interval
    responses.append(httpx.Response(200, json=payload))
    responses.extend(httpx.Response(403) for _ in expected)
    with pytest.raises(codex_oauth.ProviderKeyError, match="timed out"):
        codex_oauth.login_codex_device_oauth(6)
    assert sleeps == expected
    assert len(requests) == len(expected) + 1
    save.assert_not_called()


@pytest.mark.parametrize(
    "stage,status,payload",
    [
        ("request", 404, {}),
        ("request", 500, {}),
        ("request", 200, {"device_auth_id": "id"}),
        ("request", 200, {"device_auth_id": 123, "user_code": "code"}),
        ("request", 200, {"device_auth_id": "id", "user_code": "code", "interval": "bad"}),
        ("poll", 401, {"secret": "sensitive"}),
        ("poll", 429, {}),
        ("poll", 200, {"authorization_code": "sensitive"}),
        ("poll", 200, {"authorization_code": "sensitive", "code_verifier": ""}),
    ],
)
def test_device_failures_do_not_store(device_login, stage, status, payload) -> None:

    responses, _, _, save, _ = device_login
    if stage == "poll":
        responses.append(httpx.Response(200, json={"device_auth_id": "id", "user_code": "code"}))
    responses.append(httpx.Response(status, json=payload))
    with pytest.raises(codex_oauth.ProviderKeyError) as error:
        codex_oauth.login_codex_device_oauth()
    assert "fast-agent auth provider login codex --method browser" in str(error.value)
    assert "sensitive" not in str(error.value)
    save.assert_not_called()


@pytest.mark.parametrize("failure", ["network", "json", "exchange"])
def test_device_transport_and_exchange_failures(device_login, monkeypatch, failure) -> None:

    responses, _, _, save, _ = device_login
    if failure == "network":
        responses.append(httpx.ConnectError("sensitive"))
    elif failure == "json":
        responses.append(httpx.Response(200, content=b"not json sensitive"))
    else:
        responses.extend(
            [
                httpx.Response(200, json={"device_auth_id": "id", "user_code": "code"}),
                httpx.Response(
                    200, json={"authorization_code": "code", "code_verifier": "verifier"}
                ),
            ]
        )
        monkeypatch.setattr(
            codex_oauth,
            "exchange_code_for_tokens",
            Mock(side_effect=codex_oauth.ProviderKeyError("sensitive", "sensitive")),
        )
    with pytest.raises(codex_oauth.ProviderKeyError) as error:
        codex_oauth.login_codex_device_oauth()
    assert "--method browser" in str(error.value)
    assert "sensitive" not in str(error.value)
    save.assert_not_called()


@pytest.mark.parametrize(
    "method,timeout,expected",
    [("device", None, 900), ("browser", None, 300), ("device", 7, 7), ("browser", 8, 8)],
)
def test_login_dispatch(monkeypatch, method, timeout, expected) -> None:

    device = Mock()
    browser = Mock()
    monkeypatch.setattr(codex_oauth, "login_codex_device_oauth", device)
    monkeypatch.setattr(codex_oauth, "login_codex_browser_oauth", browser)
    result = codex_oauth.login_codex_oauth(timeout, method=method)
    selected, unused = (device, browser) if method == "device" else (browser, device)
    selected.assert_called_once_with(expected)
    assert result is selected.return_value
    unused.assert_not_called()


def test_browser_login_preserves_callback_and_redirect(monkeypatch) -> None:

    server = Mock()
    server.serve_once.return_value = ("code", "state")
    monkeypatch.setattr(codex_oauth, "_CallbackServer", Mock(return_value=server))
    monkeypatch.setattr(codex_oauth.secrets, "token_urlsafe", lambda size: "state")
    monkeypatch.setattr(codex_oauth.console, "ensure_blocking_console", lambda: None)
    monkeypatch.setattr(codex_oauth.console.console, "print", Mock())
    request = Mock(return_value=CodexOAuthTokens(access_token="token"))
    save = Mock()
    monkeypatch.setattr(codex_oauth, "_token_request", request)
    monkeypatch.setattr(codex_oauth, "save_codex_tokens", save)

    tokens = codex_oauth.login_codex_oauth(method="browser")

    server.start.assert_called_once()
    server.serve_once.assert_called_once_with(timeout_seconds=300)
    server.close.assert_called_once()
    assert request.call_args.args[0]["redirect_uri"] == codex_oauth.CODEX_REDIRECT_URI
    save.assert_called_once_with(tokens)


def test_device_deadline_is_capped_at_fifteen_minutes(device_login) -> None:

    responses, requests, sleeps, save, _ = device_login
    responses.extend(
        [
            httpx.Response(
                200, json={"device_auth_id": "id", "user_code": "code", "interval": "1000"}
            ),
            httpx.Response(404),
        ]
    )
    with pytest.raises(codex_oauth.ProviderKeyError, match="timed out"):
        codex_oauth.login_codex_device_oauth(1800)
    assert sleeps == [900]
    assert len(requests) == 2
    save.assert_not_called()


def test_device_expired_deadline_makes_no_requests(device_login) -> None:
    _, requests, sleeps, save, _ = device_login
    with pytest.raises(codex_oauth.ProviderKeyError, match="timed out"):
        codex_oauth.login_codex_device_oauth(0)
    assert requests == []
    assert sleeps == []
    save.assert_not_called()
