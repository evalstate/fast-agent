import json
from pathlib import Path

from fast_agent.llm.provider.openai import codex_oauth
from fast_agent.llm.provider.openai.codex_oauth import CodexOAuthTokens


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


def test_load_codex_tokens_reuses_parsed_tokens_for_unchanged_keyring_payload(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("CODEX_AUTH_JSON_PATH", raising=False)
    monkeypatch.delenv("CODEX_HOME", raising=False)
    monkeypatch.setattr(codex_oauth.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(codex_oauth, "_codex_token_cache", None)
    calls = 0

    def _load_keyring() -> str:
        nonlocal calls
        calls += 1
        return json.dumps({"access_token": "keyring-token", "token_type": "Bearer"})

    monkeypatch.setattr(codex_oauth, "_get_keyring_password", _load_keyring)

    first = codex_oauth.load_codex_tokens()
    second = codex_oauth.load_codex_tokens()

    assert first is not None
    assert second is not None
    assert first.access_token == "keyring-token"
    assert second.access_token == "keyring-token"
    assert first is second
    assert calls == 1


def test_load_codex_tokens_invalidates_cache_when_keyring_payload_changes(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("CODEX_AUTH_JSON_PATH", raising=False)
    monkeypatch.delenv("CODEX_HOME", raising=False)
    monkeypatch.setattr(codex_oauth.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(codex_oauth, "_codex_token_cache", None)
    keyring_payload = json.dumps({"access_token": "initial-token", "token_type": "Bearer"})

    def _load_keyring() -> str:
        return keyring_payload

    monkeypatch.setattr(codex_oauth, "_get_keyring_password", _load_keyring)

    first = codex_oauth.load_codex_tokens()
    keyring_payload = json.dumps({"access_token": "external-token", "token_type": "Bearer"})
    second = codex_oauth.load_codex_tokens()
    codex_oauth.clear_codex_token_cache()
    third = codex_oauth.load_codex_tokens()

    assert first is not None
    assert second is not None
    assert third is not None
    assert first.access_token == "initial-token"
    assert second.access_token == "initial-token"
    assert third.access_token == "external-token"


def test_load_codex_tokens_invalidates_cache_when_auth_json_changes(
    monkeypatch,
    tmp_path: Path,
) -> None:
    auth_path = tmp_path / "local-auth.json"
    monkeypatch.setenv("CODEX_AUTH_JSON_PATH", str(auth_path))
    monkeypatch.delenv("CODEX_HOME", raising=False)
    monkeypatch.setattr(codex_oauth, "_codex_token_cache", None)

    auth_path.write_text(
        json.dumps({"tokens": {"access_token": "initial-token", "token_type": "Bearer"}})
    )
    first = codex_oauth.load_codex_tokens()
    auth_path.write_text(
        json.dumps({"tokens": {"access_token": "external-token", "token_type": "Bearer"}})
    )
    second = codex_oauth.load_codex_tokens()

    assert first is not None
    assert second is not None
    assert first.access_token == "initial-token"
    assert second.access_token == "external-token"


def test_save_codex_tokens_writes_local_auth_file_without_keyring(monkeypatch, tmp_path: Path) -> None:
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
