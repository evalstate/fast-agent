import json
from pathlib import Path

import pytest

from fast_agent.auth.credentials import OAuthCredential, StoredCredential, save_oauth_credential
from fast_agent.llm.provider.openai import codex_oauth
from fast_agent.llm.provider.openai.codex_oauth import CodexOAuthTokens


def test_resolve_codex_cli_auth_path_defaults_to_user_home(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("CODEX_AUTH_JSON_PATH", raising=False)
    monkeypatch.delenv("CODEX_HOME", raising=False)
    monkeypatch.setattr(codex_oauth.Path, "home", lambda: tmp_path)

    assert codex_oauth._resolve_codex_cli_auth_path() == tmp_path / ".codex" / "auth.json"


def test_explicit_auth_json_path_is_fallback_when_fast_agent_store_is_empty(
    monkeypatch, tmp_path: Path
) -> None:
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
    monkeypatch.setattr(codex_oauth, "load_oauth_credential", lambda provider: None)

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
    monkeypatch.setattr(codex_oauth, "load_oauth_credential", lambda provider: None)

    tokens, source = codex_oauth._load_codex_tokens_with_source()

    assert source == "auth.json"
    assert tokens is not None
    assert tokens.access_token == "cli-token"


def test_fast_agent_owned_credential_precedes_codex_cli_credential(monkeypatch) -> None:
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

    assert source == "keyring"
    assert tokens is not None
    assert tokens.access_token == "fast-agent-token"


def test_fast_agent_auth_file_is_authoritative_over_codex_cli(
    monkeypatch, tmp_path: Path
) -> None:
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


def test_fast_agent_auth_file_does_not_fall_back_to_codex_cli(
    monkeypatch, tmp_path: Path
) -> None:
    portable_auth_path = tmp_path / "fast-agent-auth.json"
    cli_auth_path = tmp_path / "codex-auth.json"
    cli_auth_path.write_text(json.dumps({"tokens": {"access_token": "cli-token"}}))
    monkeypatch.setenv("FAST_AGENT_AUTH_FILE", str(portable_auth_path))
    monkeypatch.setenv("CODEX_AUTH_JSON_PATH", str(cli_auth_path))

    assert codex_oauth._load_codex_tokens_with_source() == (None, None)


def test_save_codex_tokens_prefers_fast_agent_auth_file(
    monkeypatch, tmp_path: Path
) -> None:
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
        "Legacy Codex credentials were found in the OS keyring but are no longer loaded. "
        "Run `fast-agent auth login codex` to authenticate again."
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
