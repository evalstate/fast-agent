import json
from pathlib import Path

from fast_agent.llm.provider.openai import codex_oauth
from fast_agent.llm.provider.openai.codex_oauth import CodexOAuthTokens


class _KeyringStub:
    def __init__(self) -> None:
        self.passwords: dict[tuple[str, str], str] = {}
        self.deleted: list[tuple[str, str]] = []

    def get_password(self, service: str, username: str) -> str | None:
        return self.passwords.get((service, username))

    def set_password(self, service: str, username: str, password: str) -> None:
        self.passwords[(service, username)] = password

    def delete_password(self, service: str, username: str) -> None:
        self.deleted.append((service, username))
        self.passwords.pop((service, username), None)


def test_resolve_codex_cli_auth_path_defaults_to_user_home(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("CODEX_AUTH_JSON_PATH", raising=False)
    monkeypatch.delenv("CODEX_HOME", raising=False)
    monkeypatch.setattr(codex_oauth.Path, "home", lambda: tmp_path)

    assert codex_oauth._resolve_codex_cli_auth_path() == tmp_path / ".codex" / "auth.json"


def test_explicit_auth_json_path_overrides_keyring(monkeypatch, tmp_path: Path) -> None:
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
    monkeypatch.setenv("CODEX_AUTH_JSON_PATH", str(auth_path))
    monkeypatch.delenv("CODEX_HOME", raising=False)
    monkeypatch.setattr(
        codex_oauth,
        "_get_keyring_password",
        lambda: json.dumps({"access_token": "keyring-token", "token_type": "Bearer"}),
    )

    tokens, source = codex_oauth._load_codex_tokens_with_source()

    assert source == "auth.json"
    assert tokens is not None
    assert tokens.access_token == "local-token"


def test_default_auth_json_path_overrides_keyring(monkeypatch, tmp_path: Path) -> None:
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
    monkeypatch.delenv("CODEX_AUTH_JSON_PATH", raising=False)
    monkeypatch.delenv("CODEX_HOME", raising=False)
    monkeypatch.setattr(codex_oauth.Path, "home", lambda: tmp_path)

    def _unexpected_keyring_read() -> str | None:
        raise AssertionError("keyring should not be read when default auth.json has tokens")

    monkeypatch.setattr(codex_oauth, "_get_keyring_password", _unexpected_keyring_read)

    tokens, source = codex_oauth._load_codex_tokens_with_source()

    assert source == "auth.json"
    assert tokens is not None
    assert tokens.access_token == "cli-token"


def test_save_codex_tokens_writes_local_auth_file_without_keyring(
    monkeypatch, tmp_path: Path
) -> None:
    auth_path = tmp_path / ".codex" / "auth.json"
    auth_path.parent.mkdir(parents=True, exist_ok=True)
    auth_path.write_text(json.dumps({"OPENAI_API_KEY": "preserve-me"}))
    monkeypatch.setenv("CODEX_AUTH_JSON_PATH", str(auth_path))
    monkeypatch.delenv("CODEX_HOME", raising=False)

    def _unexpected_keyring_write(_: str) -> None:
        raise AssertionError("keyring should not be used when CODEX_AUTH_JSON_PATH is set")

    monkeypatch.setattr(codex_oauth, "_set_keyring_password", _unexpected_keyring_write)

    codex_oauth.save_codex_tokens(
        CodexOAuthTokens(
            access_token="saved-token",
            refresh_token="saved-refresh",
            expires_at=1234.5,
            scope="openid profile",
            token_type="Bearer",
        )
    )

    payload = json.loads(auth_path.read_text())
    assert payload["OPENAI_API_KEY"] == "preserve-me"
    assert payload["tokens"]["access_token"] == "saved-token"
    assert payload["tokens"]["refresh_token"] == "saved-refresh"
    assert payload["tokens"]["expires_at"] == 1234.5
    assert payload["tokens"]["scope"] == "openid profile"


def test_save_codex_tokens_updates_existing_default_auth_file_without_keyring(
    monkeypatch, tmp_path: Path
) -> None:
    auth_path = tmp_path / ".codex" / "auth.json"
    auth_path.parent.mkdir(parents=True)
    auth_path.write_text(json.dumps({"OPENAI_API_KEY": "preserve-me"}))
    monkeypatch.delenv("CODEX_AUTH_JSON_PATH", raising=False)
    monkeypatch.delenv("CODEX_HOME", raising=False)
    monkeypatch.setattr(codex_oauth.Path, "home", lambda: tmp_path)

    def _unexpected_keyring_write(_: str) -> None:
        raise AssertionError("keyring should not be used when default auth.json exists")

    monkeypatch.setattr(codex_oauth, "_set_keyring_password", _unexpected_keyring_write)

    codex_oauth.save_codex_tokens(
        CodexOAuthTokens(
            access_token="saved-token",
            refresh_token="saved-refresh",
            token_type="Bearer",
        )
    )

    payload = json.loads(auth_path.read_text())
    assert payload["OPENAI_API_KEY"] == "preserve-me"
    assert payload["tokens"]["access_token"] == "saved-token"
    assert payload["tokens"]["refresh_token"] == "saved-refresh"


def test_tokens_from_response_rejects_bool_expires_in(monkeypatch) -> None:
    monkeypatch.setattr(codex_oauth.time, "time", lambda: 1000.0)

    bool_expiry = codex_oauth._tokens_from_response({"access_token": "token", "expires_in": True})
    valid_expiry = codex_oauth._tokens_from_response({"access_token": "token", "expires_in": 60})

    assert bool_expiry.expires_at is None
    assert valid_expiry.expires_at == 1060.0


def test_load_chunked_payload_rejects_bool_part_count() -> None:
    keyring = _KeyringStub()
    keyring.set_password(
        codex_oauth.CODEX_KEYRING_SERVICE,
        codex_oauth.CODEX_TOKEN_META_KEY,
        json.dumps({"parts": True}),
    )
    keyring.set_password(
        codex_oauth.CODEX_KEYRING_SERVICE,
        f"{codex_oauth.CODEX_TOKEN_CHUNK_PREFIX}:0",
        "unexpected",
    )

    assert codex_oauth._load_chunked_payload(keyring) is None


def test_delete_chunked_payload_rejects_bool_part_count() -> None:
    keyring = _KeyringStub()
    keyring.set_password(
        codex_oauth.CODEX_KEYRING_SERVICE,
        codex_oauth.CODEX_TOKEN_META_KEY,
        json.dumps({"parts": True}),
    )

    codex_oauth._delete_chunked_payload(keyring)

    deleted_usernames = [username for _service, username in keyring.deleted]
    assert f"{codex_oauth.CODEX_TOKEN_CHUNK_PREFIX}:0" in deleted_usernames
    assert f"{codex_oauth.CODEX_TOKEN_CHUNK_PREFIX}:9" in deleted_usernames
