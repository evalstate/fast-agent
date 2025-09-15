"""
OAuth v2.1 integration helpers for MCP client transports.

Provides token storage (in-memory and OS keyring), a local callback server
with paste-URL fallback, and a builder for OAuthClientProvider that can be
passed to SSE/HTTP transports as the `auth` parameter.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import TYPE_CHECKING, Any, Callable
from urllib.parse import parse_qs, urlparse

from mcp.client.auth import OAuthClientProvider, TokenStorage
from mcp.shared.auth import (
    OAuthClientInformationFull,
    OAuthClientMetadata,
    OAuthToken,
)
from pydantic import AnyUrl

from fast_agent.core.logging.logger import get_logger
from fast_agent.ui import console

if TYPE_CHECKING:
    from fast_agent.config import MCPServerSettings

logger = get_logger(__name__)


class InMemoryTokenStorage(TokenStorage):
    """Non-persistent token storage (process memory only)."""

    def __init__(self) -> None:
        self._tokens: OAuthToken | None = None
        self._client_info: OAuthClientInformationFull | None = None

    async def get_tokens(self) -> OAuthToken | None:
        return self._tokens

    async def set_tokens(self, tokens: OAuthToken) -> None:
        self._tokens = tokens

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        return self._client_info

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        self._client_info = client_info


@dataclass
class _CallbackResult:
    authorization_code: str | None = None
    state: str | None = None
    error: str | None = None


class _CallbackHandler(BaseHTTPRequestHandler):
    """HTTP handler to capture OAuth callback query params."""

    def __init__(self, *args, result: _CallbackResult, expected_path: str, **kwargs):
        self._result = result
        self._expected_path = expected_path.rstrip("/") or "/callback"
        super().__init__(*args, **kwargs)

    def do_GET(self) -> None:  # noqa: N802 - http.server signature
        parsed = urlparse(self.path)

        # Only accept the configured callback path
        if (parsed.path.rstrip("/") or "/callback") != self._expected_path:
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
        elif "error" in params:
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
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format: str, *args: Any) -> None:  # silence default logging
        return


class _CallbackServer:
    """Simple background HTTP server to receive a single OAuth callback."""

    def __init__(self, port: int, path: str) -> None:
        self._port = port
        self._path = path.rstrip("/") or "/callback"
        self._result = _CallbackResult()
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None

    def _make_handler(self) -> Callable[..., BaseHTTPRequestHandler]:
        result = self._result
        expected_path = self._path

        def handler(*args, **kwargs):
            return _CallbackHandler(*args, result=result, expected_path=expected_path, **kwargs)

        return handler

    def start(self) -> None:
        self._server = HTTPServer(("localhost", self._port), self._make_handler())
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        logger.info(f"OAuth callback server listening on http://localhost:{self._port}{self._path}")

    def stop(self) -> None:
        if self._server:
            try:
                self._server.shutdown()
                self._server.server_close()
            except Exception:
                pass
        if self._thread:
            self._thread.join(timeout=1)

    def wait(self, timeout_seconds: int = 300) -> tuple[str, str | None]:
        start = time.time()
        while time.time() - start < timeout_seconds:
            if self._result.authorization_code:
                return self._result.authorization_code, self._result.state
            if self._result.error:
                raise RuntimeError(f"OAuth error: {self._result.error}")
            time.sleep(0.1)
        raise TimeoutError("Timeout waiting for OAuth callback")


def _derive_base_server_url(url: str | None) -> str | None:
    """Derive the base server URL for OAuth discovery from an MCP endpoint URL.

    - Strips a trailing "/mcp" or "/sse" path segment
    - Ignores query and fragment parts entirely
    """
    if not url:
        return None
    try:
        from urllib.parse import urlparse, urlunparse

        parsed = urlparse(url)
        # Normalize path without trailing slash
        path = parsed.path or ""
        path = path[:-1] if path.endswith("/") else path
        # Remove one trailing segment if it is mcp or sse
        for suffix in ("/mcp", "/sse"):
            if path.endswith(suffix):
                path = path[: -len(suffix)]
                break
        # Ensure path is at least '/'
        if not path:
            path = "/"
        # Rebuild URL without query/fragment
        clean = parsed._replace(path=path, params="", query="", fragment="")
        base = urlunparse(clean)
        # Drop trailing slash except for root
        if base.endswith("/") and base.count("/") > 2:
            base = base[:-1]
        return base
    except Exception:
        return url


def compute_server_identity(server_config: MCPServerSettings) -> str:
    """Compute a stable identity for token storage.

    Prefer the normalized base server URL; fall back to configured name, then 'default'.
    """
    base = _derive_base_server_url(server_config.url)
    if base:
        return base
    if server_config.name:
        return server_config.name
    return "default"


def keyring_has_token(server_config: MCPServerSettings) -> bool:
    """Check if keyring has a token stored for this server."""
    try:
        import keyring

        identity = compute_server_identity(server_config)
        token_key = f"oauth:tokens:{identity}"
        return keyring.get_password("fast-agent-mcp", token_key) is not None
    except Exception:
        return False


async def _print_authorization_link(auth_url: str, warn_if_no_keyring: bool = False) -> None:
    """Emit a clickable authorization link using rich console markup.

    If warn_if_no_keyring is True and the OS keyring backend is unavailable,
    print a warning to indicate tokens won't be persisted.
    """
    console.console.print("[bold]Open this link to authorize:[/bold]", markup=True)
    console.console.print(f"[link={auth_url}]{auth_url}[/link]")
    if warn_if_no_keyring:
        try:
            import keyring  # type: ignore

            backend = keyring.get_keyring()
            try:
                from keyring.backends.fail import Keyring as FailKeyring  # type: ignore

                if isinstance(backend, FailKeyring):
                    console.console.print(
                        "[yellow]Warning:[/yellow] Keyring backend not available — tokens will not be persisted."
                    )
            except Exception:
                # If we cannot detect the fail backend, do nothing
                pass
        except Exception:
            console.console.print(
                "[yellow]Warning:[/yellow] Keyring backend not available — tokens will not be persisted."
            )
    logger.info("OAuth authorization URL emitted to console")


class KeyringTokenStorage(TokenStorage):
    """Token storage backed by the OS keychain using 'keyring'."""

    def __init__(self, service_name: str, server_identity: str) -> None:
        self._service = service_name
        self._identity = server_identity

    @property
    def _token_key(self) -> str:
        return f"oauth:tokens:{self._identity}"

    @property
    def _client_key(self) -> str:
        return f"oauth:client_info:{self._identity}"

    async def get_tokens(self) -> OAuthToken | None:
        try:
            import keyring

            payload = keyring.get_password(self._service, self._token_key)
            if not payload:
                return None
            return OAuthToken.model_validate_json(payload)
        except Exception:
            return None

    async def set_tokens(self, tokens: OAuthToken) -> None:
        try:
            import keyring

            keyring.set_password(self._service, self._token_key, tokens.model_dump_json())
            # Update index
            add_identity_to_index(self._service, self._identity)
        except Exception:
            pass

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        try:
            import keyring

            payload = keyring.get_password(self._service, self._client_key)
            if not payload:
                return None
            return OAuthClientInformationFull.model_validate_json(payload)
        except Exception:
            return None

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        try:
            import keyring

            keyring.set_password(self._service, self._client_key, client_info.model_dump_json())
        except Exception:
            pass


# --- Keyring index helpers (to enable cross-platform token enumeration) ---


def _index_username() -> str:
    return "oauth:index"


def _read_index(service: str) -> set[str]:
    try:
        import json

        import keyring

        raw = keyring.get_password(service, _index_username())
        if not raw:
            return set()
        data = json.loads(raw)
        if isinstance(data, list):
            return set([str(x) for x in data])
        return set()
    except Exception:
        return set()


def _write_index(service: str, identities: set[str]) -> None:
    try:
        import json

        import keyring

        payload = json.dumps(sorted(list(identities)))
        keyring.set_password(service, _index_username(), payload)
    except Exception:
        pass


def add_identity_to_index(service: str, identity: str) -> None:
    identities = _read_index(service)
    if identity not in identities:
        identities.add(identity)
        _write_index(service, identities)


def remove_identity_from_index(service: str, identity: str) -> None:
    identities = _read_index(service)
    if identity in identities:
        identities.remove(identity)
        _write_index(service, identities)


def list_keyring_tokens(service: str = "fast-agent-mcp") -> list[str]:
    """List identities with stored tokens in keyring (using our index).

    Returns only identities that currently have a corresponding token entry.
    """
    try:
        import keyring

        identities = _read_index(service)
        present: list[str] = []
        for ident in sorted(identities):
            tok_key = f"oauth:tokens:{ident}"
            if keyring.get_password(service, tok_key):
                present.append(ident)
        return present
    except Exception:
        return []


def clear_keyring_token(identity: str, service: str = "fast-agent-mcp") -> bool:
    """Remove token+client info for identity and update the index.

    Returns True if anything was removed.
    """
    removed = False
    try:
        import keyring

        tok_key = f"oauth:tokens:{identity}"
        cli_key = f"oauth:client_info:{identity}"
        try:
            keyring.delete_password(service, tok_key)
            removed = True
        except Exception:
            pass
        try:
            keyring.delete_password(service, cli_key)
            removed = True or removed
        except Exception:
            pass
        if removed:
            remove_identity_from_index(service, identity)
    except Exception:
        return False
    return removed


def build_oauth_provider(server_config: MCPServerSettings) -> OAuthClientProvider | None:
    """
    Build an OAuthClientProvider for the given server config if applicable.

    Returns None for unsupported transports, or when disabled via config.
    """
    # Only for SSE/HTTP transports
    if server_config.transport not in ("sse", "http"):
        return None

    # Determine if OAuth should be enabled. Default to True if no auth block provided
    enable_oauth = True
    redirect_port = 3030
    redirect_path = "/callback"
    scope_value: str | None = None
    persist_mode: str = "keyring"

    if server_config.auth is not None:
        try:
            enable_oauth = getattr(server_config.auth, "oauth", True)
            redirect_port = getattr(server_config.auth, "redirect_port", 3030)
            redirect_path = getattr(server_config.auth, "redirect_path", "/callback")
            scope_field = getattr(server_config.auth, "scope", None)
            persist_mode = getattr(server_config.auth, "persist", "keyring")
            if isinstance(scope_field, list):
                scope_value = " ".join(scope_field)
            elif isinstance(scope_field, str):
                scope_value = scope_field
        except Exception:
            logger.debug("Malformed auth configuration; using defaults.")

    if not enable_oauth:
        return None

    base_url = _derive_base_server_url(server_config.url)
    if not base_url:
        # No usable URL -> cannot build provider
        return None

    # Construct client metadata with minimal defaults
    redirect_uri = f"http://localhost:{redirect_port}{redirect_path}"
    metadata_kwargs: dict[str, Any] = {
        "client_name": "fast-agent",
        "redirect_uris": [AnyUrl(redirect_uri)],
        "grant_types": ["authorization_code", "refresh_token"],
        "response_types": ["code"],
    }
    if scope_value:
        metadata_kwargs["scope"] = scope_value

    client_metadata = OAuthClientMetadata.model_validate(metadata_kwargs)

    # Local callback server handler
    async def _redirect_handler(authorization_url: str) -> None:
        # Warn if persisting to keyring but no backend is available
        await _print_authorization_link(
            authorization_url,
            warn_if_no_keyring=(persist_mode == "keyring"),
        )

    async def _callback_handler() -> tuple[str, str | None]:
        # Try local HTTP capture first
        try:
            server = _CallbackServer(port=redirect_port, path=redirect_path)
            server.start()
            try:
                code, state = server.wait(timeout_seconds=300)
                return code, state
            finally:
                server.stop()
        except Exception as e:
            # Fallback to paste-URL flow
            logger.info(f"OAuth local callback server unavailable, fallback to paste flow: {e}")
            try:
                import sys

                print("Paste the full callback URL after authorization:", file=sys.stderr)
                callback_url = input("Callback URL: ").strip()
            except Exception as ee:
                raise RuntimeError(f"Failed to read callback URL from user: {ee}")

            params = parse_qs(urlparse(callback_url).query)
            code = params.get("code", [None])[0]
            state = params.get("state", [None])[0]
            if not code:
                raise RuntimeError("Callback URL missing authorization code")
            return code, state

    # Choose storage
    storage: TokenStorage
    if persist_mode == "keyring":
        identity = compute_server_identity(server_config)
        # Update index on write via storage methods; creation here doesn't modify index yet.
        storage = KeyringTokenStorage(service_name="fast-agent-mcp", server_identity=identity)
    else:
        storage = InMemoryTokenStorage()

    provider = OAuthClientProvider(
        server_url=base_url,
        client_metadata=client_metadata,
        storage=storage,
        redirect_handler=_redirect_handler,
        callback_handler=_callback_handler,
    )

    return provider
