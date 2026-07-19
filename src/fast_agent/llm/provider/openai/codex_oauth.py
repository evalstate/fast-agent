"""Codex OAuth helpers for ChatGPT/Codex tokens.

Implements the OAuth PKCE flow used by the Codex CLI, including keyring
storage and refresh. Access tokens are used as API keys when calling the
Codex responses endpoint.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import time
from contextlib import suppress
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse

import httpx
from pydantic import BaseModel

from fast_agent.auth.credentials import (
    OAuthCredential,
    configured_auth_path,
    credential_refresh_lock,
    delete_oauth_credential,
    load_oauth_credential,
    save_oauth_credential,
)
from fast_agent.core.exceptions import ProviderKeyError
from fast_agent.core.keyring_utils import maybe_print_keyring_access_notice
from fast_agent.core.logging.logger import get_logger
from fast_agent.ui import console
from fast_agent.utils.numeric import positive_int_or_none

logger = get_logger(__name__)


CODEX_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
CODEX_AUTHORIZE_URL = "https://auth.openai.com/oauth/authorize"
CODEX_TOKEN_URL = "https://auth.openai.com/oauth/token"
CODEX_REDIRECT_HOST = "localhost"
CODEX_REDIRECT_PORT = 1455
CODEX_REDIRECT_PATH = "/auth/callback"
CODEX_REDIRECT_URI = f"http://{CODEX_REDIRECT_HOST}:{CODEX_REDIRECT_PORT}{CODEX_REDIRECT_PATH}"
CODEX_SCOPE = "openid profile email offline_access"
CODEX_AUTH_CLAIM = "https://api.openai.com/auth"
LEGACY_CODEX_KEYRING_SERVICE = "fast-agent-codex"
LEGACY_CODEX_TOKEN_KEY = "oauth:tokens:openai-codex"
LEGACY_CODEX_TOKEN_META_KEY = f"{LEGACY_CODEX_TOKEN_KEY}:meta"
CODEX_CREDENTIAL_CONTAINER_KEYS = (
    "auth",
    "token",
    "tokens",
    "session",
    "credential",
    "credentials",
    "data",
)


class CodexOAuthTokens(BaseModel):
    access_token: str
    refresh_token: str | None = None
    expires_at: float | None = None
    scope: str | None = None
    token_type: str = "Bearer"

    def is_expired(self, margin_seconds: int = 60) -> bool:
        if self.expires_at is None:
            return False
        return time.time() >= (self.expires_at - margin_seconds)


@dataclass
class _CallbackResult:
    authorization_code: str | None = None
    state: str | None = None
    error: str | None = None


class _CallbackHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, result: _CallbackResult, **kwargs):
        self._result = result
        super().__init__(*args, **kwargs)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if (parsed.path.rstrip("/") or CODEX_REDIRECT_PATH) != CODEX_REDIRECT_PATH:
            self.send_response(404)
            self.end_headers()
            return

        params = parse_qs(parsed.query)
        if "code" in params:
            self._result.authorization_code = params["code"][0]
            self._result.state = params.get("state", [None])[0]
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(
                b"""
                <html><body>
                <h1>Authorization Successful</h1>
                <p>You can close this window.</p>
                <script>setTimeout(() => window.close(), 1000);</script>
                </body></html>
                """
            )
            return

        if "error" in params:
            self._result.error = params["error"][0]
            self.send_response(400)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(
                f"""
                <html><body>
                <h1>Authorization Failed</h1>
                <p>Error: {self._result.error}</p>
                </body></html>
                """.encode()
            )
            return

        self.send_response(404)
        self.end_headers()

    def log_message(self, format: str, *args: Any) -> None:  # silence default logging
        return


class _CallbackServer:
    def __init__(self, port: int) -> None:
        self._port = port
        self._result = _CallbackResult()
        self._server: HTTPServer | None = None

    def start(self) -> None:
        try:
            self._server = HTTPServer(
                ("127.0.0.1", self._port),
                lambda *args, **kwargs: _CallbackHandler(*args, result=self._result, **kwargs),
            )
            logger.info(
                "Codex OAuth callback server listening",
                data={"redirect_uri": CODEX_REDIRECT_URI},
            )
        except OSError as exc:
            raise OSError("Port 1455 unavailable") from exc

    def serve_once(self, timeout_seconds: int = 300) -> tuple[str, str | None]:
        if not self._server:
            raise RuntimeError("Callback server not started")
        self._server.timeout = 0.25
        end_time = time.time() + timeout_seconds
        while time.time() < end_time:
            self._server.handle_request()
            if self._result.authorization_code:
                return self._result.authorization_code, self._result.state
            if self._result.error:
                raise RuntimeError(f"OAuth error: {self._result.error}")
        raise TimeoutError("Timeout waiting for OAuth callback")

    def close(self) -> None:
        if self._server:
            with suppress(Exception):
                self._server.server_close()


def _base64url_decode(value: str) -> bytes:
    padded = value + "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(padded)


def _pkce_verifier() -> str:
    return secrets.token_urlsafe(64)


def _pkce_challenge(verifier: str) -> str:
    digest = hashlib.sha256(verifier.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("utf-8")


def _tokens_from_response(payload: dict[str, Any]) -> CodexOAuthTokens:
    expires_in = positive_int_or_none(payload.get("expires_in"))
    expires_at = None
    if expires_in is not None:
        expires_at = time.time() + expires_in
    return CodexOAuthTokens(
        access_token=payload["access_token"],
        refresh_token=payload.get("refresh_token"),
        expires_at=expires_at,
        scope=payload.get("scope"),
        token_type=payload.get("token_type", "Bearer"),
    )


def _token_request(payload: dict[str, Any]) -> CodexOAuthTokens:
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(CODEX_TOKEN_URL, data=payload)
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPError as exc:
        raise ProviderKeyError(
            "Codex OAuth request failed",
            "Unable to exchange tokens with auth.openai.com. Please retry the login flow.",
        ) from exc
    return _tokens_from_response(data)


def _default_codex_cli_auth_path() -> Path:
    return Path.home() / ".codex" / "auth.json"


def _resolve_codex_cli_auth_path() -> Path:
    explicit = str(os.environ.get("CODEX_AUTH_JSON_PATH") or "").strip()
    if explicit:
        return Path(explicit).expanduser()
    codex_home = str(os.environ.get("CODEX_HOME") or "").strip()
    if codex_home:
        return Path(codex_home).expanduser() / "auth.json"
    return _default_codex_cli_auth_path()


def _normalize_codex_cli_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    if "access_token" in payload:
        return {
            "access_token": payload.get("access_token"),
            "refresh_token": payload.get("refresh_token"),
            "expires_at": payload.get("expires_at"),
            "scope": payload.get("scope"),
            "token_type": payload.get("token_type") or "Bearer",
        }
    if "accessToken" in payload:
        expires_at = payload.get("expiresAt") or payload.get("expires_at")
        if isinstance(expires_at, (int, float)) and expires_at > 1_000_000_000_000:
            expires_at = expires_at / 1000.0
        return {
            "access_token": payload.get("accessToken"),
            "refresh_token": payload.get("refreshToken"),
            "expires_at": expires_at,
            "scope": payload.get("scope"),
            "token_type": payload.get("tokenType") or "Bearer",
        }
    return None


def _load_codex_cli_tokens() -> CodexOAuthTokens | None:
    auth_path = _resolve_codex_cli_auth_path()
    try:
        if not auth_path.exists():
            return None
    except OSError:
        return None
    try:
        payload = json.loads(auth_path.read_text())
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    candidates: list[dict[str, Any]] = [payload]
    for key in CODEX_CREDENTIAL_CONTAINER_KEYS:
        value = payload.get(key)
        if isinstance(value, dict):
            candidates.append(value)
    for candidate in candidates:
        normalized = _normalize_codex_cli_payload(candidate)
        if not normalized or not normalized.get("access_token"):
            continue
        try:
            return CodexOAuthTokens.model_validate(normalized)
        except Exception:
            continue
    return None


def _tokens_from_credential(credential: OAuthCredential) -> CodexOAuthTokens:
    return CodexOAuthTokens(
        access_token=credential.access_token,
        refresh_token=credential.refresh_token,
        expires_at=credential.expires_at,
        scope=credential.scope,
        token_type=credential.token_type,
    )


def _credential_from_tokens(tokens: CodexOAuthTokens) -> OAuthCredential:
    return OAuthCredential(
        access_token=tokens.access_token,
        refresh_token=tokens.refresh_token,
        expires_at=tokens.expires_at,
        scope=tokens.scope,
        token_type=tokens.token_type,
    )


def _legacy_codex_keyring_credentials_present() -> bool:
    try:
        maybe_print_keyring_access_notice(purpose="checking for legacy Codex OAuth tokens")
        import keyring

        return any(
            keyring.get_password(LEGACY_CODEX_KEYRING_SERVICE, key) is not None
            for key in (LEGACY_CODEX_TOKEN_KEY, LEGACY_CODEX_TOKEN_META_KEY)
        )
    except Exception:
        return False


def _warn_about_legacy_codex_keyring_credentials() -> None:
    if not _legacy_codex_keyring_credentials_present():
        return
    console.ensure_blocking_console()
    console.error_console.print(
        "Legacy Codex credentials were found in the OS keyring but are no longer loaded. "
        "Run `fast-agent auth login codex` to authenticate again.",
        style="bold yellow",
    )


def _load_codex_tokens_with_source() -> tuple[CodexOAuthTokens | None, str | None]:
    # FAST_AGENT_AUTH_FILE is an explicit, authoritative credential source. Do not
    # silently fall back to a developer's local Codex CLI account when it is set.
    if configured_auth_path() is not None:
        stored = load_oauth_credential("codex")
        return (_tokens_from_credential(stored.credential), stored.source) if stored else (None, None)

    stored = load_oauth_credential("codex")
    if stored:
        return _tokens_from_credential(stored.credential), stored.source

    # Codex CLI credentials are an external, read-only fallback. Fast-agent-owned
    # credentials take precedence so refreshed tokens can persist without modifying
    # the CLI's auth.json.
    tokens = _load_codex_cli_tokens()
    if tokens:
        return tokens, "auth.json"
    _warn_about_legacy_codex_keyring_credentials()
    return None, None


def load_codex_tokens() -> CodexOAuthTokens | None:
    tokens, source = _load_codex_tokens_with_source()
    if tokens and source == "auth.json":
        logger.info(
            "codex_cli_tokens",
            "Loaded Codex OAuth tokens from auth.json",
            data={"path": str(_resolve_codex_cli_auth_path())},
        )
    return tokens


def save_codex_tokens(tokens: CodexOAuthTokens) -> None:
    # Codex CLI auth files are read-only external sources. Persist login and refresh
    # results only in fast-agent's credential store.
    if configured_auth_path() is not None:
        save_oauth_credential("codex", _credential_from_tokens(tokens))
        return
    stored = load_oauth_credential("codex")
    save_oauth_credential(
        "codex",
        _credential_from_tokens(tokens),
        source=stored.source if stored else None,
    )


def clear_codex_tokens() -> bool:
    return delete_oauth_credential("codex")


def build_authorization_url(code_challenge: str, state: str) -> str:
    params = {
        "response_type": "code",
        "client_id": CODEX_CLIENT_ID,
        "redirect_uri": CODEX_REDIRECT_URI,
        "scope": CODEX_SCOPE,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        "state": state,
        "id_token_add_organizations": "true",
        "codex_cli_simplified_flow": "true",
        "originator": "fast-agent",
    }
    return f"{CODEX_AUTHORIZE_URL}?{urlencode(params)}"


def exchange_code_for_tokens(code: str, code_verifier: str) -> CodexOAuthTokens:
    payload = {
        "grant_type": "authorization_code",
        "client_id": CODEX_CLIENT_ID,
        "code": code,
        "code_verifier": code_verifier,
        "redirect_uri": CODEX_REDIRECT_URI,
    }
    return _token_request(payload)


def refresh_codex_tokens(refresh_token: str) -> CodexOAuthTokens:
    payload = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": CODEX_CLIENT_ID,
    }
    return _token_request(payload)


def get_codex_access_token(*, force_refresh: bool = False) -> str | None:
    tokens = load_codex_tokens()
    if not tokens:
        return None
    if force_refresh or tokens.is_expired():
        with credential_refresh_lock("codex"):
            tokens = load_codex_tokens()
            if not tokens:
                return None
            if force_refresh or tokens.is_expired():
                if not tokens.refresh_token:
                    raise ProviderKeyError(
                        "Codex OAuth token expired",
                        "The stored Codex OAuth token is expired and has no refresh token. "
                        "Run `fast-agent auth login codex` to reauthenticate.",
                    )
                refreshed = refresh_codex_tokens(tokens.refresh_token)
                if not refreshed.refresh_token:
                    refreshed.refresh_token = tokens.refresh_token
                save_codex_tokens(refreshed)
                tokens = refreshed
    return tokens.access_token


def get_codex_token_status() -> dict[str, Any]:
    tokens, source = _load_codex_tokens_with_source()
    if not tokens:
        return {"present": False, "expires_at": None, "expired": False, "source": None}
    expired = tokens.is_expired(margin_seconds=0)
    return {
        "present": True,
        "expires_at": tokens.expires_at,
        "expired": expired,
        "source": source,
    }


def parse_chatgpt_account_id(access_token: str) -> str | None:
    try:
        parts = access_token.split(".")
        if len(parts) < 2:
            return None
        payload = json.loads(_base64url_decode(parts[1]))
        auth_block = payload.get(CODEX_AUTH_CLAIM)
        if not isinstance(auth_block, dict):
            return None
        account_id = auth_block.get("chatgpt_account_id")
        return str(account_id) if account_id else None
    except Exception:
        return None


def login_codex_oauth(timeout_seconds: int = 300) -> CodexOAuthTokens:
    verifier = _pkce_verifier()
    challenge = _pkce_challenge(verifier)
    state = secrets.token_urlsafe(16)
    auth_url = build_authorization_url(challenge, state)

    server = _CallbackServer(CODEX_REDIRECT_PORT)
    code: str | None = None
    returned_state: str | None = None

    console.ensure_blocking_console()
    console.console.print("[bold]Open this link to authorize:[/bold]")
    console.ensure_blocking_console()
    console.console.print(f"[link={auth_url}]{auth_url}[/link]")

    try:
        server.start()
        try:
            code, returned_state = server.serve_once(timeout_seconds=timeout_seconds)
        except Exception as exc:
            logger.info("Codex OAuth callback failed, falling back to paste flow", exc_info=exc)
    except Exception as exc:
        logger.info("Codex OAuth callback server unavailable, using paste flow", exc_info=exc)
    finally:
        server.close()

    if not code:
        console.ensure_blocking_console()
        console.console.print(
            "Paste the full callback URL after completing the authorization in your browser:",
            style="bold",
        )
        console.ensure_blocking_console()
        pasted = console.console.input("Callback URL: ")
        parsed = urlparse(pasted)
        params = parse_qs(parsed.query)
        code = params.get("code", [None])[0]
        returned_state = params.get("state", [None])[0]

    if not code:
        raise ProviderKeyError(
            "Codex OAuth login failed",
            "Authorization code missing from callback URL.",
        )

    if returned_state and returned_state != state:
        raise ProviderKeyError(
            "Codex OAuth login failed",
            "State parameter mismatch. Please retry login.",
        )

    tokens = exchange_code_for_tokens(code, verifier)
    save_codex_tokens(tokens)
    return tokens
