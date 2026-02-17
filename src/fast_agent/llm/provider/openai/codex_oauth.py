"""Codex OAuth helpers for ChatGPT/Codex tokens.

Implements the OAuth PKCE flow used by the Codex CLI, including keyring
storage and refresh. Access tokens are used as API keys when calling the
Codex responses endpoint.
"""

from __future__ import annotations

import base64
import hashlib
import json
import secrets
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Protocol, cast
from urllib.parse import parse_qs, urlencode, urlparse

import httpx
from pydantic import BaseModel

from fast_agent.core.exceptions import ProviderKeyError
from fast_agent.core.keyring_utils import (
    get_keyring_status,
    maybe_print_keyring_access_notice,
)
from fast_agent.core.logging.logger import get_logger
from fast_agent.mcp.oauth_client import add_identity_to_index, remove_identity_from_index
from fast_agent.ui import console

logger = get_logger(__name__)


class _KeyringProtocol(Protocol):
    def get_password(self, service: str, username: str) -> str | None: ...
    def set_password(self, service: str, username: str, password: str) -> None: ...
    def delete_password(self, service: str, username: str) -> None: ...

CODEX_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
CODEX_AUTHORIZE_URL = "https://auth.openai.com/oauth/authorize"
CODEX_TOKEN_URL = "https://auth.openai.com/oauth/token"
CODEX_REDIRECT_HOST = "localhost"
CODEX_REDIRECT_PORT = 1455
CODEX_REDIRECT_PATH = "/auth/callback"
CODEX_REDIRECT_URI = (
    f"http://{CODEX_REDIRECT_HOST}:{CODEX_REDIRECT_PORT}{CODEX_REDIRECT_PATH}"
)
CODEX_SCOPE = "openid profile email offline_access"
CODEX_AUTH_CLAIM = "https://api.openai.com/auth"
CODEX_KEYRING_SERVICE = "fast-agent-codex"
CODEX_KEYRING_IDENTITY = "openai-codex"
CODEX_TOKEN_KEY = f"oauth:tokens:{CODEX_KEYRING_IDENTITY}"
CODEX_TOKEN_META_KEY = f"{CODEX_TOKEN_KEY}:meta"
CODEX_TOKEN_CHUNK_PREFIX = f"{CODEX_TOKEN_KEY}:chunk"
CODEX_KEYRING_MAX_PAYLOAD_BYTES = 512
CODEX_CLI_AUTH_PATH = Path("/root/.codex/auth.json")


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

    def do_GET(self) -> None:  # noqa: N802 - http.server signature
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
            try:
                self._server.server_close()
            except Exception:
                pass


def _base64url_decode(value: str) -> bytes:
    padded = value + "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(padded)


def _pkce_verifier() -> str:
    return secrets.token_urlsafe(64)


def _pkce_challenge(verifier: str) -> str:
    digest = hashlib.sha256(verifier.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("utf-8")


def _tokens_from_response(payload: dict[str, Any]) -> CodexOAuthTokens:
    expires_in = payload.get("expires_in")
    expires_at = None
    if isinstance(expires_in, int) and expires_in > 0:
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


def _payload_byte_length(payload: str) -> int:
    return len(payload.encode("utf-8"))


def _chunk_payload(payload: str, chunk_size: int) -> list[str]:
    return [payload[i : i + chunk_size] for i in range(0, len(payload), chunk_size)]


def _chunk_key(index: int) -> str:
    return f"{CODEX_TOKEN_CHUNK_PREFIX}:{index}"


def _safe_delete(keyring_module: _KeyringProtocol, username: str) -> None:
    try:
        keyring_module.delete_password(CODEX_KEYRING_SERVICE, username)
    except Exception:
        return


def _load_chunked_payload(keyring_module: _KeyringProtocol) -> str | None:
    meta = keyring_module.get_password(CODEX_KEYRING_SERVICE, CODEX_TOKEN_META_KEY)
    if not meta:
        return None
    try:
        payload = json.loads(meta)
    except Exception:
        return None
    parts = payload.get("parts") if isinstance(payload, dict) else None
    if not isinstance(parts, int) or parts <= 0:
        return None
    chunks: list[str] = []
    for index in range(parts):
        chunk = keyring_module.get_password(CODEX_KEYRING_SERVICE, _chunk_key(index))
        if chunk is None:
            return None
        chunks.append(chunk)
    return "".join(chunks)


def _store_chunked_payload(keyring_module: _KeyringProtocol, payload: str) -> None:
    chunks = _chunk_payload(payload, CODEX_KEYRING_MAX_PAYLOAD_BYTES)
    for index, chunk in enumerate(chunks):
        keyring_module.set_password(CODEX_KEYRING_SERVICE, _chunk_key(index), chunk)
    meta_payload = json.dumps(
        {
            "version": 1,
            "parts": len(chunks),
            "size": _payload_byte_length(payload),
        }
    )
    keyring_module.set_password(CODEX_KEYRING_SERVICE, CODEX_TOKEN_META_KEY, meta_payload)


def _delete_chunked_payload(keyring_module: _KeyringProtocol) -> None:
    meta = keyring_module.get_password(CODEX_KEYRING_SERVICE, CODEX_TOKEN_META_KEY)
    parts: int | None = None
    if meta:
        try:
            payload = json.loads(meta)
            if isinstance(payload, dict):
                parts = payload.get("parts")
        except Exception:
            parts = None
    if isinstance(parts, int) and parts > 0:
        for index in range(parts):
            _safe_delete(keyring_module, _chunk_key(index))
    else:
        for index in range(10):
            _safe_delete(keyring_module, _chunk_key(index))
    _safe_delete(keyring_module, CODEX_TOKEN_META_KEY)


def _keyring_payload_present() -> bool:
    try:
        maybe_print_keyring_access_notice(purpose="checking Codex OAuth tokens")
        import keyring

        keyring_module = cast("_KeyringProtocol", keyring)
        if keyring_module.get_password(CODEX_KEYRING_SERVICE, CODEX_TOKEN_KEY) is not None:
            return True
        if keyring_module.get_password(CODEX_KEYRING_SERVICE, CODEX_TOKEN_META_KEY) is not None:
            return True
        return False
    except Exception:
        return False


def _get_keyring_password() -> str | None:
    try:
        maybe_print_keyring_access_notice(purpose="loading Codex OAuth tokens")
        import keyring

        keyring_module = cast("_KeyringProtocol", keyring)
        payload = keyring_module.get_password(CODEX_KEYRING_SERVICE, CODEX_TOKEN_KEY)
        if payload:
            return payload
        return _load_chunked_payload(keyring_module)
    except Exception:
        return None


def _set_keyring_password(payload: str) -> None:
    maybe_print_keyring_access_notice(purpose="saving Codex OAuth tokens")
    import keyring

    keyring_module = cast("_KeyringProtocol", keyring)
    try:
        _safe_delete(keyring_module, CODEX_TOKEN_KEY)
        _delete_chunked_payload(keyring_module)
        if _payload_byte_length(payload) <= CODEX_KEYRING_MAX_PAYLOAD_BYTES:
            keyring_module.set_password(CODEX_KEYRING_SERVICE, CODEX_TOKEN_KEY, payload)
        else:
            _store_chunked_payload(keyring_module, payload)
        add_identity_to_index(CODEX_KEYRING_SERVICE, CODEX_KEYRING_IDENTITY)
    except Exception as exc:
        status = get_keyring_status()
        if not status.available:
            backend_note = "No usable keyring backend was detected."
        elif not status.writable:
            backend_note = (
                f"Keyring backend '{status.name}' is present but not writable."
            )
        else:
            backend_note = f"Keyring backend '{status.name}' failed to store tokens."
        raise ProviderKeyError(
            "Keyring unavailable",
            "Codex OAuth tokens could not be saved to the keyring. "
            f"{backend_note} "
            "On Windows, ensure Credential Manager is available in the current session. "
            "You can also set PYTHON_KEYRING_BACKEND to a file-based backend "
            "(e.g., keyrings.alt.file.PlaintextKeyring).",
        ) from exc


def _delete_keyring_password() -> None:
    maybe_print_keyring_access_notice(purpose="clearing Codex OAuth tokens")
    import keyring

    keyring_module = cast("_KeyringProtocol", keyring)
    _safe_delete(keyring_module, CODEX_TOKEN_KEY)
    _delete_chunked_payload(keyring_module)
    remove_identity_from_index(CODEX_KEYRING_SERVICE, CODEX_KEYRING_IDENTITY)


def keyring_available() -> bool:
    return get_keyring_status().writable


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
    try:
        if not CODEX_CLI_AUTH_PATH.exists():
            return None
    except OSError:
        return None
    try:
        payload = json.loads(CODEX_CLI_AUTH_PATH.read_text())
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    candidates: list[dict[str, Any]] = [payload]
    for key in ("auth", "token", "tokens", "session", "credential", "credentials", "data"):
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


def load_codex_tokens() -> CodexOAuthTokens | None:
    payload = _get_keyring_password()
    if payload:
        try:
            return CodexOAuthTokens.model_validate_json(payload)
        except Exception:
            return None
    tokens = _load_codex_cli_tokens()
    if tokens:
        logger.info(
            "codex_cli_tokens",
            "Loaded Codex OAuth tokens from auth.json",
            data={"path": str(CODEX_CLI_AUTH_PATH)},
        )
    return tokens


def save_codex_tokens(tokens: CodexOAuthTokens) -> None:
    _set_keyring_password(tokens.model_dump_json())


def clear_codex_tokens() -> bool:
    if not _keyring_payload_present():
        return False
    try:
        _delete_keyring_password()
        return True
    except Exception:
        return False


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


def get_codex_access_token() -> str | None:
    tokens = load_codex_tokens()
    if not tokens:
        return None
    if tokens.is_expired():
        if not tokens.refresh_token:
            raise ProviderKeyError(
                "Codex OAuth token expired",
                "The stored Codex OAuth token is expired and has no refresh token. "
                "Run `fast-agent auth codex-login` to reauthenticate.",
            )
        refreshed = refresh_codex_tokens(tokens.refresh_token)
        if not refreshed.refresh_token:
            refreshed.refresh_token = tokens.refresh_token
        save_codex_tokens(refreshed)
        tokens = refreshed
    return tokens.access_token


def get_codex_token_status() -> dict[str, Any]:
    tokens = load_codex_tokens()
    if not tokens:
        return {"present": False, "expires_at": None, "expired": False}
    expired = tokens.is_expired(margin_seconds=0)
    return {"present": True, "expires_at": tokens.expires_at, "expired": expired}


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
    status = get_keyring_status()
    if not status.writable:
        if not status.available:
            backend_note = "No usable keyring backend was detected."
        else:
            backend_note = f"Keyring backend '{status.name}' is not writable."
        raise ProviderKeyError(
            "Keyring unavailable",
            "Codex OAuth requires a writable keyring backend. "
            f"{backend_note} "
            "On Windows, ensure Credential Manager is available in the current session. "
            "You can also set PYTHON_KEYRING_BACKEND to a file-based backend "
            "(e.g., keyrings.alt.file.PlaintextKeyring).",
        )

    verifier = _pkce_verifier()
    challenge = _pkce_challenge(verifier)
    state = secrets.token_urlsafe(16)
    auth_url = build_authorization_url(challenge, state)

    server = _CallbackServer(CODEX_REDIRECT_PORT)
    code: str | None = None
    returned_state: str | None = None

    console.console.print("[bold]Open this link to authorize:[/bold]")
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
        console.console.print(
            "Paste the full callback URL after completing the authorization in your browser:",
            style="bold",
        )
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
