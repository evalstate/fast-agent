"""
OAuth v2.1 integration helpers for MCP client transports.

Provides token storage (in-memory and OS keyring), a local callback server
with paste-URL fallback, and a builder for OAuthClientProvider that can be
passed to SSE/HTTP transports as the `auth` parameter.
"""

from __future__ import annotations

import asyncio
import os
import socket
import sys
import threading
import time
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import TYPE_CHECKING, Any, ClassVar, Literal
from urllib.parse import parse_qs, urlparse, urlunparse

from mcp.client.auth import OAuthClientProvider as _BaseOAuthClientProvider
from mcp.client.auth import TokenStorage
from mcp.client.auth.utils import (
    build_oauth_authorization_server_metadata_discovery_urls,
    build_protected_resource_metadata_discovery_urls,
    create_client_info_from_metadata_url,
    create_client_registration_request,
    create_oauth_metadata_request,
    extract_field_from_www_auth,
    extract_resource_metadata_from_www_auth,
    extract_scope_from_www_auth,
    get_client_metadata_scopes,
    handle_auth_metadata_response,
    handle_protected_resource_response,
    handle_registration_response,
    should_use_client_metadata_url,
)
from mcp.client.streamable_http import MCP_PROTOCOL_VERSION
from mcp.shared.auth import (
    OAuthClientInformationFull,
    OAuthClientMetadata,
    OAuthToken,
)
from pydantic import AnyUrl
from rich.text import Text

from fast_agent.core.keyring_utils import maybe_print_keyring_access_notice
from fast_agent.core.logging.logger import get_logger
from fast_agent.ui import console
from fast_agent.utils.text import strip_to_none
from fast_agent.utils.transports import uses_mcp_remote_transport

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    import httpx

    from fast_agent.config import MCPServerSettings

logger = get_logger(__name__)

DEFAULT_CLIENT_METADATA_URL = "https://fast-agent.ai/oauth/client.json"

OAuthEventType = Literal[
    "authorization_url",
    "wait_start",
    "wait_end",
    "callback_received",
    "oauth_error",
]


@dataclass(frozen=True, slots=True)
class OAuthEvent:
    """Lifecycle event emitted by runtime OAuth integration."""

    event_type: OAuthEventType
    server_name: str
    url: str | None = None
    message: str | None = None
    is_timeout: bool = False
    occurred_at: float = field(default_factory=time.time)


OAuthEventHandler = Callable[[OAuthEvent], Awaitable[None]]


class OAuthCallbackTimeoutError(TimeoutError):
    """Raised when OAuth authorization callback does not arrive in time."""


class OAuthFlowCancelledError(RuntimeError):
    """Raised when an in-flight OAuth flow is cancelled by the caller."""


@dataclass(frozen=True, slots=True)
class _OAuthCallbackContext:
    event_handler: OAuthEventHandler | None
    server_name: str
    emit_console_output: bool
    abort_event: threading.Event | None
    selected_redirect_port: int
    redirect_path: str
    allow_paste_fallback: bool


@dataclass(frozen=True, slots=True)
class _OAuthProviderSettings:
    enabled: bool
    redirect_port: int
    redirect_path: str
    scope: str | None
    persist_mode: str
    client_metadata_url: str | None


async def _emit_oauth_event(
    event_handler: OAuthEventHandler | None,
    event: OAuthEvent,
) -> None:
    """Emit OAuth lifecycle events without allowing callback failures to break auth flow."""
    if event_handler is None:
        return

    try:
        await event_handler(event)
    except Exception:
        logger.debug(
            "OAuth event callback failed",
            event_type=event.event_type,
            server_name=event.server_name,
            exc_info=True,
        )


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

    def do_GET(self) -> None:
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
    """Simple background HTTP server to receive a single OAuth callback.

    Uses 127.0.0.1 (loopback IP) instead of localhost for RFC 8252 compliance.
    Per RFC 8252 Section 7.3, authorization servers MUST allow any port for
    loopback IP redirect URIs, enabling dynamic port allocation.
    """

    # Fallback ports to try if preferred port is unavailable
    FALLBACK_PORTS: ClassVar[list[int]] = [3030, 3031, 3032, 8080, 0]  # 0 = ephemeral port

    def __init__(
        self,
        port: int,
        path: str,
        *,
        fallback_ports: list[int] | None = None,
    ) -> None:
        self._preferred_port = port
        self._path = path.rstrip("/") or "/callback"
        self._fallback_ports = list(
            self.FALLBACK_PORTS if fallback_ports is None else fallback_ports
        )
        self._result = _CallbackResult()
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._actual_port: int | None = None

    @property
    def actual_port(self) -> int | None:
        """Return the actual port the server bound to (may differ from preferred)."""
        return self._actual_port

    def _make_handler(self) -> Callable[..., BaseHTTPRequestHandler]:
        result = self._result
        expected_path = self._path

        def handler(*args, **kwargs):
            return _CallbackHandler(*args, result=result, expected_path=expected_path, **kwargs)

        return handler

    def _try_bind(self, port: int) -> HTTPServer | None:
        """Try to bind to the given port. Returns server if successful, None otherwise."""
        try:
            # Use 127.0.0.1 (loopback IP) for RFC 8252 compliance
            return HTTPServer(("127.0.0.1", port), self._make_handler())
        except OSError as e:
            # EADDRINUSE (98 on Linux, 48 on macOS) or similar
            logger.debug(f"Port {port} unavailable: {e}")
            return None

    def start(self) -> None:
        """Start the callback server, trying fallback ports if preferred is unavailable."""
        # Build list of ports to try: preferred first, then fallbacks
        ports_to_try = [self._preferred_port]
        for p in self._fallback_ports:
            if p not in ports_to_try:
                ports_to_try.append(p)

        for port in ports_to_try:
            server = self._try_bind(port)
            if server is not None:
                self._server = server
                # Get actual port (important when using ephemeral port 0)
                self._actual_port = self._server.server_address[1]
                self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
                self._thread.start()
                logger.info(
                    f"OAuth callback server listening on http://127.0.0.1:{self._actual_port}{self._path}"
                )
                if self._actual_port != self._preferred_port:
                    logger.info(
                        f"Note: Using port {self._actual_port} instead of preferred port {self._preferred_port}"
                    )
                return

        raise OSError(
            f"Could not bind to any port. Tried: {ports_to_try}. All ports may be in use."
        )

    def stop(self) -> None:
        if self._server:
            try:
                self._server.shutdown()
                self._server.server_close()
            except Exception:
                pass
        if self._thread:
            self._thread.join(timeout=1)

    def wait(
        self,
        timeout_seconds: int = 300,
        abort_event: threading.Event | None = None,
    ) -> tuple[str, str | None]:
        start = time.time()
        while time.time() - start < timeout_seconds:
            if abort_event is not None and abort_event.is_set():
                raise OAuthFlowCancelledError("OAuth callback wait cancelled")
            if self._result.authorization_code:
                return self._result.authorization_code, self._result.state
            if self._result.error:
                raise RuntimeError(f"OAuth error: {self._result.error}")
            time.sleep(0.1)
        raise TimeoutError("Timeout waiting for OAuth callback")

    def get_redirect_uri(self) -> str:
        """Return the actual redirect URI based on bound port."""
        if self._actual_port is None:
            raise RuntimeError("Server not started; cannot determine redirect URI")
        return f"http://127.0.0.1:{self._actual_port}{self._path}"


def _select_preferred_redirect_port(preferred_port: int) -> int:
    """Pick a redirect port likely to be bindable for this OAuth attempt.

    The MCP OAuth client currently uses the first redirect URI in metadata for the
    authorization and token exchange. We therefore probe ports ahead of time and
    choose a concrete primary port to keep redirect URI and callback listener aligned.
    """

    ports_to_try = [preferred_port]
    for port in _CallbackServer.FALLBACK_PORTS:
        if port not in ports_to_try:
            ports_to_try.append(port)

    for port in ports_to_try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("127.0.0.1", port))
            return int(sock.getsockname()[1])
        except OSError:
            continue
        finally:
            with suppress(OSError):
                sock.close()

    raise OSError(
        f"Could not reserve any redirect port. Tried: {ports_to_try}. All ports may be in use."
    )


def _normalize_oauth_server_url(url: str | None) -> str | None:
    """Normalize an MCP endpoint URL for OAuth discovery and resource validation.

    Preserves the MCP endpoint path (for example ``/mcp`` or ``/sse``) while
    removing query/fragment parts so OAuth discovery operates on the canonical
    protected resource URL.
    """
    if not url:
        return None
    try:
        parsed = urlparse(url)
        path = (parsed.path or "").rstrip("/")
        clean = parsed._replace(path=path or "/", params="", query="", fragment="")
        normalized = urlunparse(clean)
        if normalized.endswith("/") and normalized.count("/") > 2:
            normalized = normalized[:-1]
        return normalized
    except Exception:
        return url


def _derive_base_server_url(url: str | None) -> str | None:
    """Derive the token-storage identity URL from an MCP endpoint URL.

    - Strips a trailing "/mcp" or "/sse" path segment
    - Ignores query and fragment parts entirely
    """
    normalized_url = _normalize_oauth_server_url(url)
    if not normalized_url:
        return None
    try:
        parsed = urlparse(normalized_url)
        path = parsed.path or ""
        for suffix in ("/mcp", "/sse"):
            if path.endswith(suffix):
                path = path[: -len(suffix)]
                break
        if not path:
            path = "/"
        clean = parsed._replace(path=path, params="", query="", fragment="")
        base = urlunparse(clean)
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


def _build_prm_discovery_urls(
    *,
    www_auth_resource_metadata_url: str | None,
    server_url: str,
    discovery_server_url: str,
) -> list[str]:
    """Build PRM discovery URLs without dropping endpoint-scoped lookups.

    Order matters here:
    1. Explicit ``resource_metadata`` from ``WWW-Authenticate``
    2. Endpoint-path well-known URL for the concrete MCP resource
    3. Parent-path well-known URL for deployments that publish at the base resource
    4. Root well-known URL
    """

    ordered_urls: list[str] = []
    seen: set[str] = set()
    root_urls: list[str] = []

    def _record(url: str) -> None:
        if url not in seen:
            ordered_urls.append(url)
            seen.add(url)

    if www_auth_resource_metadata_url:
        _record(www_auth_resource_metadata_url)

    for candidate_server_url in (server_url, discovery_server_url):
        for url in build_protected_resource_metadata_discovery_urls(
            None,
            candidate_server_url,
        ):
            if url.endswith("/.well-known/oauth-protected-resource"):
                if url not in seen and url not in root_urls:
                    root_urls.append(url)
                continue
            _record(url)

    for url in root_urls:
        _record(url)

    return ordered_urls


class _ProtectedResourceDiscoveryOAuthClientProvider(_BaseOAuthClientProvider):
    """Preserve endpoint validation while probing both endpoint and parent PRM URLs."""

    def __init__(self, *args: Any, discovery_server_url: str, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._discovery_server_url = discovery_server_url

    def _prm_discovery_urls_from_response(self, response: httpx.Response) -> list[str]:
        return _build_prm_discovery_urls(
            www_auth_resource_metadata_url=extract_resource_metadata_from_www_auth(response),
            server_url=self.context.server_url,
            discovery_server_url=self._discovery_server_url,
        )

    async def _store_client_info(self, client_information: OAuthClientInformationFull) -> None:
        self.context.client_info = client_information
        await self.context.storage.set_client_info(client_information)

    async def _handle_prm_discovery_response(
        self,
        discovery_response: httpx.Response,
        *,
        url: str,
    ) -> bool:
        prm = await handle_protected_resource_response(discovery_response)
        if not prm:
            logger.debug(f"Protected resource metadata discovery failed: {url}")
            return False

        await self._validate_resource_match(prm)
        self.context.protected_resource_metadata = prm
        self.context.auth_server_url = str(prm.authorization_servers[0])
        return True

    async def _handle_asm_discovery_response(
        self,
        oauth_metadata_response: httpx.Response,
        *,
        url: str,
    ) -> Literal["stop", "found", "continue"]:
        ok, asm = await handle_auth_metadata_response(oauth_metadata_response)
        if not ok:
            return "stop"
        if asm:
            self.context.oauth_metadata = asm
            return "found"
        logger.debug(f"OAuth metadata discovery failed: {url}")
        return "continue"

    def _update_client_metadata_scope(self, response: httpx.Response) -> None:
        self.context.client_metadata.scope = get_client_metadata_scopes(
            extract_scope_from_www_auth(response),
            self.context.protected_resource_metadata,
            self.context.oauth_metadata,
        )

    async def _client_registration_request_if_needed(self) -> httpx.Request | None:
        if self.context.client_info:
            return None

        client_metadata_url = self.context.client_metadata_url
        if (
            should_use_client_metadata_url(
                self.context.oauth_metadata,
                client_metadata_url,
            )
            and client_metadata_url is not None
        ):
            logger.debug(f"Using URL-based client ID (CIMD): {client_metadata_url}")
            await self._store_client_info(
                create_client_info_from_metadata_url(
                    client_metadata_url,
                    redirect_uris=self.context.client_metadata.redirect_uris,
                )
            )
            return None

        return create_client_registration_request(
            self.context.oauth_metadata,
            self.context.client_metadata,
            self.context.get_authorization_base_url(self.context.server_url),
        )

    async def _refresh_request_if_needed(self) -> httpx.Request | None:
        if self.context.is_token_valid() or not self.context.can_refresh_token():
            return None
        return await self._refresh_token()  # pragma: no cover

    async def _handle_refresh_attempt(self, refresh_response: httpx.Response) -> None:
        if not await self._handle_refresh_response(refresh_response):  # pragma: no cover
            self._initialized = False

    @staticmethod
    def _requires_scope_step_up(response: httpx.Response) -> bool:
        return (
            response.status_code == 403
            and extract_field_from_www_auth(response, "error") == "insufficient_scope"
        )

    async def _unauthorized_auth_flow(
        self,
        request: httpx.Request,
        response: httpx.Response,
    ) -> AsyncGenerator[httpx.Request, httpx.Response]:
        try:
            for url in self._prm_discovery_urls_from_response(response):
                discovery_request = create_oauth_metadata_request(url)
                discovery_response = yield discovery_request

                if await self._handle_prm_discovery_response(
                    discovery_response,
                    url=url,
                ):
                    break

            asm_discovery_urls = build_oauth_authorization_server_metadata_discovery_urls(
                self.context.auth_server_url,
                self._discovery_server_url,
            )

            for url in asm_discovery_urls:  # pragma: no cover
                oauth_metadata_request = create_oauth_metadata_request(url)
                oauth_metadata_response = yield oauth_metadata_request

                result = await self._handle_asm_discovery_response(
                    oauth_metadata_response,
                    url=url,
                )
                if result in {"stop", "found"}:
                    break

            self._update_client_metadata_scope(response)
            registration_request = await self._client_registration_request_if_needed()
            if registration_request is not None:
                registration_response = yield registration_request
                client_information = await handle_registration_response(registration_response)
                await self._store_client_info(client_information)

            token_response = yield await self._perform_authorization()
            await self._handle_token_response(token_response)
        except Exception:  # pragma: no cover
            logger.exception("OAuth flow error")
            raise

        self._add_auth_header(request)
        yield request

    async def _scope_step_up_auth_flow(
        self,
        request: httpx.Request,
        response: httpx.Response,
    ) -> AsyncGenerator[httpx.Request, httpx.Response]:
        try:
            self._update_client_metadata_scope(response)

            token_response = yield await self._perform_authorization()
            await self._handle_token_response(token_response)

            self._add_auth_header(request)
            yield request
        except Exception:  # pragma: no cover
            logger.exception("OAuth step-up error")
            raise

    async def async_auth_flow(
        self,
        request: httpx.Request,
    ) -> AsyncGenerator[httpx.Request, httpx.Response]:
        """Mirror SDK auth flow but scope PRM discovery to the base protected resource."""
        async with self.context.lock:
            if not self._initialized:
                await self._initialize()  # pragma: no cover

            self.context.protocol_version = request.headers.get(MCP_PROTOCOL_VERSION)

            refresh_request = await self._refresh_request_if_needed()
            if refresh_request is not None:
                refresh_response = yield refresh_request  # pragma: no cover
                await self._handle_refresh_attempt(refresh_response)

            if self.context.is_token_valid():
                self._add_auth_header(request)

            response = yield request

            if response.status_code == 401:
                followup_flow = self._unauthorized_auth_flow(request, response)
            elif self._requires_scope_step_up(response):
                followup_flow = self._scope_step_up_auth_flow(request, response)
            else:
                followup_flow = None

            if followup_flow is not None:
                try:
                    next_request = await anext(followup_flow)
                except StopAsyncIteration:
                    return

                while True:
                    followup_response = yield next_request
                    try:
                        next_request = await followup_flow.asend(followup_response)
                    except StopAsyncIteration:
                        break


OAuthClientProvider = _ProtectedResourceDiscoveryOAuthClientProvider


def keyring_token_present(identity: str, service: str = "fast-agent-mcp") -> bool:
    """Return True when a stored OAuth token exists for the given identity."""
    try:
        maybe_print_keyring_access_notice(purpose="checking stored MCP OAuth tokens")
        import keyring

        token_key = f"oauth:tokens:{identity}"
        return keyring.get_password(service, token_key) is not None
    except Exception:
        return False


def keyring_has_token(server_config: MCPServerSettings) -> bool:
    """Check if keyring has a token stored for this server."""
    return keyring_token_present(compute_server_identity(server_config))


async def _print_authorization_link(auth_url: str, warn_if_no_keyring: bool = False) -> None:
    """Emit a clickable authorization link using rich console markup.

    If warn_if_no_keyring is True and the OS keyring backend is unavailable,
    print a warning to indicate tokens won't be persisted.
    """
    _safe_console_print(
        "[bold]Open this link to authorize:[/bold]",
        fallback="Open this link to authorize:",
    )
    _safe_console_print(
        f"[link={auth_url}]{auth_url}[/link]",
        fallback=auth_url,
    )
    if warn_if_no_keyring:
        from fast_agent.core.keyring_utils import get_keyring_status

        status = get_keyring_status()
        if not status.writable:
            backend_note = (
                "Keyring backend not available"
                if not status.available
                else f"Keyring backend '{status.name}' not writable"
            )
            warning = Text("Warning:", style="yellow")
            warning.append(f" {backend_note} — tokens will not be persisted.")
            _safe_console_print(
                warning,
                fallback=f"Warning: {backend_note} — tokens will not be persisted.",
            )
    logger.info("OAuth authorization URL emitted to console")


def _safe_stderr_write(text: str) -> None:
    line = text if text.endswith("\n") else f"{text}\n"
    try:
        sys.stderr.write(line)
        sys.stderr.flush()
        return
    except Exception:
        pass

    try:
        fd = os.open("/dev/tty", os.O_WRONLY | os.O_NOCTTY)
    except Exception:
        return

    with suppress(Exception):
        os.set_blocking(fd, True)

    try:
        with os.fdopen(fd, "w", buffering=1, encoding="utf-8", errors="replace") as tty:
            tty.write(line)
            tty.flush()
    except Exception:
        with suppress(OSError):
            os.close(fd)


def _safe_console_print(
    message: object,
    *,
    markup: bool = True,
    fallback: str | None = None,
) -> None:
    for _ in range(2):
        try:
            console.ensure_blocking_console()
            console.console.print(message, markup=markup)
            return
        except BlockingIOError:
            continue
        except Exception:
            break

    _safe_stderr_write(fallback if fallback is not None else str(message))


def _read_callback_url_with_abort(
    prompt: str,
    abort_event: threading.Event | None,
    *,
    poll_seconds: float = 0.2,
) -> str:
    """Read a callback URL from stdin while allowing cooperative cancellation."""
    import select

    _safe_stderr_write(prompt)

    while True:
        if abort_event is not None and abort_event.is_set():
            raise OAuthFlowCancelledError("OAuth callback input cancelled")

        ready, _, _ = select.select([sys.stdin], [], [], poll_seconds)
        if not ready:
            continue

        line = sys.stdin.readline()
        if line == "":
            raise RuntimeError("No callback URL received (stdin closed)")
        return line


async def _emit_oauth_error(
    context: _OAuthCallbackContext,
    message: str,
    *,
    is_timeout: bool = False,
) -> None:
    await _emit_oauth_event(
        context.event_handler,
        OAuthEvent(
            event_type="oauth_error",
            server_name=context.server_name,
            message=message,
            is_timeout=is_timeout,
        ),
    )


async def _emit_oauth_callback_received(context: _OAuthCallbackContext) -> None:
    await _emit_oauth_event(
        context.event_handler,
        OAuthEvent(
            event_type="callback_received",
            server_name=context.server_name,
            message="OAuth callback received. Completing token exchange…",
        ),
    )


async def _emit_oauth_wait_start(
    context: _OAuthCallbackContext,
    wait_start_message: str,
) -> None:
    await _emit_oauth_event(
        context.event_handler,
        OAuthEvent(
            event_type="wait_start",
            server_name=context.server_name,
            message=wait_start_message,
        ),
    )
    if context.emit_console_output:
        _safe_console_print(wait_start_message, markup=False)
        _safe_console_print(
            "[dim]Press Ctrl+C to cancel and return to prompt.[/dim]",
            fallback="Press Ctrl+C to cancel and return to prompt.",
        )


async def _emit_oauth_wait_end(context: _OAuthCallbackContext) -> None:
    await _emit_oauth_event(
        context.event_handler,
        OAuthEvent(
            event_type="wait_end",
            server_name=context.server_name,
            message="OAuth callback wait ended.",
        ),
    )


def _callback_server_uri(server: _CallbackServer, context: _OAuthCallbackContext) -> str:
    try:
        return server.get_redirect_uri()
    except Exception:
        return f"http://127.0.0.1:{context.selected_redirect_port}{context.redirect_path}"


async def _capture_local_oauth_callback(
    context: _OAuthCallbackContext,
) -> tuple[str, str | None]:
    # MCP python-sdk currently uses the first redirect URI from client metadata
    # for both authorization and token exchange. Bind only the selected primary
    # redirect port so the callback listener matches that fixed redirect URI.
    server = _CallbackServer(
        port=context.selected_redirect_port,
        path=context.redirect_path,
        fallback_ports=[],
    )
    server.start()

    try:
        callback_uri = _callback_server_uri(server, context)
        await _emit_oauth_wait_start(
            context,
            f"Waiting for OAuth callback at {callback_uri} (startup timer paused)…",
        )

        try:
            code, state = await asyncio.to_thread(
                server.wait,
                timeout_seconds=300,
                abort_event=context.abort_event,
            )
            await _emit_oauth_callback_received(context)
            return code, state
        except OAuthFlowCancelledError as exc:
            await _emit_oauth_error(context, "OAuth authorization cancelled.")
            raise OAuthFlowCancelledError("OAuth authorization cancelled") from exc
        except TimeoutError as exc:
            timeout_message = "OAuth authorization was not completed in time."
            await _emit_oauth_error(context, timeout_message, is_timeout=True)
            raise OAuthCallbackTimeoutError(timeout_message) from exc
        finally:
            await _emit_oauth_wait_end(context)
    finally:
        server.stop()


def _extract_oauth_callback_params(callback_url: str) -> tuple[str, str | None]:
    params = parse_qs(urlparse(callback_url).query)
    code = params.get("code", [None])[0]
    if not code:
        raise RuntimeError("Callback URL missing authorization code")
    return code, params.get("state", [None])[0]


async def _capture_pasted_oauth_callback(
    context: _OAuthCallbackContext,
) -> tuple[str, str | None]:
    await _emit_oauth_wait_start(
        context,
        "Waiting for pasted OAuth callback URL (startup timer paused)…",
    )

    if context.abort_event is not None and context.abort_event.is_set():
        raise OAuthFlowCancelledError("OAuth authorization cancelled")

    try:
        if context.emit_console_output:
            _safe_stderr_write("Paste the full callback URL after authorization:")
        callback_url = (
            await asyncio.to_thread(
                _read_callback_url_with_abort,
                "Callback URL:",
                context.abort_event,
            )
        ).strip()
    except OAuthFlowCancelledError:
        await _emit_oauth_error(context, "OAuth authorization cancelled.")
        raise
    except Exception as exc:
        message = f"Failed to read callback URL from user: {exc}"
        await _emit_oauth_error(context, message)
        raise RuntimeError(message) from exc
    finally:
        await _emit_oauth_wait_end(context)

    try:
        code, state = _extract_oauth_callback_params(callback_url)
    except RuntimeError as exc:
        await _emit_oauth_error(context, str(exc))
        raise RuntimeError(str(exc)) from None

    await _emit_oauth_callback_received(context)
    return code, state


async def _handle_oauth_callback(
    context: _OAuthCallbackContext,
) -> tuple[str, str | None]:
    try:
        return await _capture_local_oauth_callback(context)
    except (OAuthCallbackTimeoutError, OAuthFlowCancelledError):
        raise
    except Exception as exc:
        if context.abort_event is not None and context.abort_event.is_set():
            raise OAuthFlowCancelledError("OAuth authorization cancelled") from exc

        if not context.allow_paste_fallback:
            message = (
                "OAuth local callback server unavailable and paste fallback is disabled "
                "for this connection mode."
            )
            await _emit_oauth_error(context, message)
            raise RuntimeError(message) from exc

        logger.info(f"OAuth local callback server unavailable, fallback to paste flow: {exc}")
        await _emit_oauth_error(
            context,
            f"OAuth local callback server unavailable, using paste URL fallback: {exc}",
        )
        return await _capture_pasted_oauth_callback(context)


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
            maybe_print_keyring_access_notice(purpose="loading MCP OAuth tokens")
            import keyring

            payload = keyring.get_password(self._service, self._token_key)
            if not payload:
                return None
            return OAuthToken.model_validate_json(payload)
        except Exception:
            return None

    async def set_tokens(self, tokens: OAuthToken) -> None:
        try:
            maybe_print_keyring_access_notice(purpose="saving MCP OAuth tokens")
            import keyring

            keyring.set_password(self._service, self._token_key, tokens.model_dump_json())
            # Update index
            add_identity_to_index(self._service, self._identity)
        except Exception:
            pass

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        try:
            maybe_print_keyring_access_notice(purpose="loading MCP OAuth client info")
            import keyring

            payload = keyring.get_password(self._service, self._client_key)
            if not payload:
                return None
            return OAuthClientInformationFull.model_validate_json(payload)
        except Exception:
            return None

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        try:
            maybe_print_keyring_access_notice(purpose="saving MCP OAuth client info")
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

        maybe_print_keyring_access_notice(purpose="reading MCP OAuth token index")
        import keyring

        raw = keyring.get_password(service, _index_username())
        if not raw:
            return set()
        data = json.loads(raw)
        if isinstance(data, list):
            return {str(x) for x in data}
        return set()
    except Exception:
        return set()


def _write_index(service: str, identities: set[str]) -> None:
    try:
        import json

        maybe_print_keyring_access_notice(purpose="updating MCP OAuth token index")
        import keyring

        payload = json.dumps(sorted(identities))
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
        maybe_print_keyring_access_notice(purpose="listing stored MCP OAuth tokens")
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
        maybe_print_keyring_access_notice(purpose="clearing stored MCP OAuth tokens")
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
            removed = True
        except Exception:
            pass
        if removed:
            remove_identity_from_index(service, identity)
    except Exception:
        return False
    return removed


def _default_client_metadata_url() -> str | None:
    # Use a default CIMD URL so OAuth can avoid dynamic client registration
    # on providers that don't expose registration endpoints.
    env_client_metadata_url = os.environ.get("FAST_AGENT_OAUTH_CLIENT_METADATA_URL")
    if env_client_metadata_url is None:
        return DEFAULT_CLIENT_METADATA_URL
    return strip_to_none(env_client_metadata_url)


def _configured_oauth_scope(scope: str | list[str] | None) -> str | None:
    if isinstance(scope, list):
        return " ".join(scope)
    return scope


def _oauth_provider_settings(server_config: MCPServerSettings) -> _OAuthProviderSettings:
    auth_config = server_config.auth
    client_metadata_url = _default_client_metadata_url()
    if auth_config is None:
        return _OAuthProviderSettings(
            enabled=True,
            redirect_port=3030,
            redirect_path="/callback",
            scope=None,
            persist_mode="keyring",
            client_metadata_url=client_metadata_url,
        )

    if auth_config.client_metadata_url is not None:
        client_metadata_url = auth_config.client_metadata_url

    return _OAuthProviderSettings(
        enabled=auth_config.oauth,
        redirect_port=auth_config.redirect_port,
        redirect_path=auth_config.redirect_path,
        scope=_configured_oauth_scope(auth_config.scope),
        persist_mode=auth_config.persist,
        client_metadata_url=client_metadata_url,
    )


def _select_oauth_redirect_port(redirect_port: int) -> int:
    try:
        return _select_preferred_redirect_port(redirect_port)
    except OSError:
        # Defer bind failures to callback handling where we can provide richer
        # OAuth diagnostics for the active connection mode.
        return redirect_port


def _oauth_redirect_uris(
    *,
    selected_redirect_port: int,
    configured_redirect_port: int,
    redirect_path: str,
) -> list[AnyUrl]:
    # Use 127.0.0.1 (loopback IP) for RFC 8252 compliance. Per RFC 8252
    # Section 7.3, authorization servers MUST allow any port for loopback IP
    # redirect URIs. Register fallback ports for servers that do not fully
    # implement RFC 8252 dynamic port matching.
    ports_for_registration = [selected_redirect_port]
    if configured_redirect_port not in ports_for_registration:
        ports_for_registration.append(configured_redirect_port)
    for port in _CallbackServer.FALLBACK_PORTS:
        if port != 0 and port not in ports_for_registration:
            ports_for_registration.append(port)
    return [AnyUrl(f"http://127.0.0.1:{port}{redirect_path}") for port in ports_for_registration]


def _oauth_client_metadata(
    settings: _OAuthProviderSettings,
    *,
    selected_redirect_port: int,
) -> OAuthClientMetadata:
    metadata_kwargs: dict[str, Any] = {
        "client_name": "fast-agent",
        "redirect_uris": _oauth_redirect_uris(
            selected_redirect_port=selected_redirect_port,
            configured_redirect_port=settings.redirect_port,
            redirect_path=settings.redirect_path,
        ),
        "grant_types": ["authorization_code", "refresh_token"],
        "response_types": ["code"],
    }
    if settings.scope:
        metadata_kwargs["scope"] = settings.scope
    return OAuthClientMetadata.model_validate(metadata_kwargs)


def _oauth_token_storage(
    server_config: MCPServerSettings,
    settings: _OAuthProviderSettings,
) -> TokenStorage:
    if settings.persist_mode == "keyring":
        identity = compute_server_identity(server_config)
        # Update index on write via storage methods; creation here doesn't modify index yet.
        return KeyringTokenStorage(service_name="fast-agent-mcp", server_identity=identity)
    return InMemoryTokenStorage()


def build_oauth_provider(
    server_config: MCPServerSettings,
    *,
    event_handler: OAuthEventHandler | None = None,
    emit_console_output: bool = True,
    abort_event: threading.Event | None = None,
    allow_paste_fallback: bool = True,
) -> OAuthClientProvider | None:
    """
    Build an OAuthClientProvider for the given server config if applicable.

    Returns None for unsupported transports, or when disabled via config.
    """
    if not uses_mcp_remote_transport(server_config.transport):
        return None

    settings = _oauth_provider_settings(server_config)
    if not settings.enabled:
        return None

    oauth_server_url = _normalize_oauth_server_url(server_config.url)
    if not oauth_server_url:
        # No usable URL -> cannot build provider
        return None

    server_name = server_config.name or "default"
    selected_redirect_port = _select_oauth_redirect_port(settings.redirect_port)
    client_metadata = _oauth_client_metadata(
        settings,
        selected_redirect_port=selected_redirect_port,
    )

    # Local callback server handler
    async def _redirect_handler(authorization_url: str) -> None:
        await _emit_oauth_event(
            event_handler,
            OAuthEvent(
                event_type="authorization_url",
                server_name=server_name,
                url=authorization_url,
                message="Open this link to authorize",
            ),
        )

        if emit_console_output:
            # Warn if persisting to keyring but no backend is available
            await _print_authorization_link(
                authorization_url,
                warn_if_no_keyring=(settings.persist_mode == "keyring"),
            )

    async def _callback_handler() -> tuple[str, str | None]:
        return await _handle_oauth_callback(
            _OAuthCallbackContext(
                event_handler=event_handler,
                server_name=server_name,
                emit_console_output=emit_console_output,
                abort_event=abort_event,
                selected_redirect_port=selected_redirect_port,
                redirect_path=settings.redirect_path,
                allow_paste_fallback=allow_paste_fallback,
            )
        )

    discovery_server_url = _derive_base_server_url(server_config.url) or oauth_server_url

    return OAuthClientProvider(
        # Keep the concrete MCP endpoint URL for validation and OAuth resource
        # selection, but scope path-based PRM discovery to the parent protected
        # resource URL so `/api/mcp` can still discover metadata published at `/api`.
        # Token storage identity is normalized separately via
        # compute_server_identity().
        server_url=oauth_server_url,
        discovery_server_url=discovery_server_url,
        client_metadata=client_metadata,
        storage=_oauth_token_storage(server_config, settings),
        redirect_handler=_redirect_handler,
        callback_handler=_callback_handler,
        client_metadata_url=settings.client_metadata_url,
    )
