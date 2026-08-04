"""MCP client construction and startup authentication policy."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal

import httpx2
from httpx2 import HTTPStatusError
from mcp.client.sse import sse_client
from mcp.client.stdio import StdioServerParameters, get_default_environment
from mcp.client.streamable_http import streamable_http_client
from mcp_types import JSONRPCMessage, JSONRPCRequest
from pydantic import TypeAdapter, ValidationError

from fast_agent.core.exceptions import walk_exception_chain
from fast_agent.core.logging.logger import get_logger
from fast_agent.home import build_child_environment
from fast_agent.mcp.client_connection import MCPClientConnection
from fast_agent.mcp.hf_auth import add_forwarded_hf_auth_header
from fast_agent.mcp.logger_textio import get_stderr_handler
from fast_agent.mcp.oauth_client import OAuthEventHandler, build_oauth_provider
from fast_agent.mcp.stdio_tracking_simple import tracking_stdio_client
from fast_agent.mcp.transport_tracking import ChannelEvent, ChannelName
from fast_agent.utils.count_display import format_count
from fast_agent.utils.text import strip_casefold
from fast_agent.utils.transports import uses_mcp_remote_transport

if TYPE_CHECKING:
    import threading
    from collections.abc import AsyncIterator
    from pathlib import Path

    from mcp.client import Transport
    from mcp.client.auth import OAuthClientProvider

    from fast_agent.config import MCPServerSettings
    from fast_agent.mcp.client_callback_runtime import MCPClientCallbackRuntime
    from fast_agent.mcp.transport_tracking import TransportChannelMetrics

logger = get_logger(__name__)
type OAuthMode = Literal["disabled", "auto", "force"]
type EffectiveMCPAuthMode = Literal[
    "not_applicable",
    "provider_managed",
    "forwarded",
    "bearer",
    "oauth",
    "auto",
    "none",
]
HttpResponseHandler = Callable[[httpx2.Response], Awaitable[None]]
_JSONRPC_MESSAGE_ADAPTER = TypeAdapter(JSONRPCMessage)


@dataclass(frozen=True, slots=True)
class PreparedHttpAuth:
    headers: dict[str, str]
    oauth_provider: OAuthClientProvider | None
    user_auth_keys: set[str]

    def __iter__(self):
        yield self.headers
        yield self.oauth_provider
        yield self.user_auth_keys


@dataclass(frozen=True, slots=True)
class MCPClientHooks:
    """Optional product hooks around an SDK-owned client lifecycle."""

    active_home: str | Path | None = None
    no_home: bool = False
    stderr_line_handler: Callable[[str], None] | None = None
    http_response_handler: HttpResponseHandler | None = None
    oauth_event_handler: OAuthEventHandler | None = None
    oauth_abort_event: threading.Event | None = None
    allow_oauth_paste_fallback: bool = True
    transport_metrics: TransportChannelMetrics | None = None
    suppress_sse_errors: Callable[[], None] | None = None
    suppress_http_errors: Callable[[], None] | None = None
    suppress_oauth_errors: Callable[[], None] | None = None


@dataclass(slots=True)
class AuthChallengeSignal:
    challenged: bool = False


def resolve_oauth_mode(
    server_config: MCPServerSettings,
    *,
    trigger_oauth: bool | None,
) -> OAuthMode:
    if not uses_mcp_remote_transport(server_config.transport):
        return "disabled"
    if trigger_oauth is False:
        return "disabled"
    auth_config = server_config.auth
    if auth_config is not None and auth_config.forward == "huggingface":
        return "disabled"
    if auth_config is not None and auth_config.oauth is False:
        return "disabled"
    if trigger_oauth is True:
        return "force"
    if auth_config is not None and auth_config.oauth:
        return "force"
    return "auto"


def resolve_effective_mcp_auth_mode(
    server_config: MCPServerSettings,
) -> EffectiveMCPAuthMode:
    """Project the runtime authentication policy for status and diagnostics."""
    if server_config.management == "provider":
        return "provider_managed"
    if not uses_mcp_remote_transport(server_config.transport):
        return "not_applicable"
    if server_config.auth is not None and server_config.auth.forward == "huggingface":
        if _has_user_auth_headers(server_config):
            return "bearer"
        return "forwarded"

    oauth_mode = resolve_oauth_mode(server_config, trigger_oauth=None)
    if oauth_mode == "force":
        return "oauth"
    if _has_user_auth_headers(server_config):
        return "bearer"
    if oauth_mode == "auto":
        return "auto"
    return "none"


def is_http_auth_challenge(
    error: object,
    *,
    response_challenged: bool = False,
) -> bool:
    """Classify an HTTP auth challenge without discarding structured causes."""
    exceptions = list(walk_exception_chain(error)) if isinstance(error, BaseException) else []
    if any(
        isinstance(exc, HTTPStatusError)
        and exc.response is not None
        and exc.response.status_code == 401
        for exc in exceptions
    ):
        return True
    if response_challenged:
        return True

    if exceptions:
        return any(_text_signals_auth_challenge(value) for value in exceptions)
    elif isinstance(error, list):
        values = list(error)
    else:
        values = [error]
    return any(_text_signals_auth_challenge(value) for value in values)


def _text_signals_auth_challenge(value: object) -> bool:
    normalized = " ".join(strip_casefold("" if value is None else str(value)).split())
    return any(
        marker in normalized
        for marker in (
            "http error: 401",
            "401 unauthorized",
            "401 client error: unauthorized",
            "www-authenticate",
        )
    )


def _has_user_auth_headers(config: MCPServerSettings) -> bool:
    return any(
        strip_casefold(key) in {"authorization", "x-hf-authorization"}
        for key in (config.headers or {})
    )


@asynccontextmanager
async def open_request_scoped_client(
    *,
    server_name: str,
    config: MCPServerSettings,
    callback_runtime: MCPClientCallbackRuntime,
    trigger_oauth: bool | None = None,
    hooks: MCPClientHooks | None = None,
) -> AsyncIterator[MCPClientConnection]:
    """Open a fresh client, escalating an automatic startup once at most."""
    oauth_mode = resolve_oauth_mode(config, trigger_oauth=trigger_oauth)
    response_signal = AuthChallengeSignal()
    original_response_handler = hooks.http_response_handler if hooks is not None else None

    async def capture_response(response: httpx2.Response) -> None:
        if response.status_code == 401:
            response_signal.challenged = True
        if original_response_handler is not None:
            await original_response_handler(response)

    request_hooks = replace(
        hooks or MCPClientHooks(),
        http_response_handler=capture_response,
    )
    oauth_attempts = (True,) if oauth_mode == "force" else (False,)
    if oauth_mode == "auto" and not _has_user_auth_headers(config):
        oauth_attempts = (False, True)

    for attempt, oauth_active in enumerate(oauth_attempts):
        connection = create_client_connection(
            server_name=server_name,
            config=config,
            callback_runtime=callback_runtime,
            oauth_mode=oauth_mode,
            oauth_active=oauth_active,
            cache=False,
            hooks=request_hooks,
        )
        try:
            await connection.__aenter__()
        except Exception as exc:
            can_escalate = (
                oauth_mode == "auto"
                and attempt == 0
                and not _has_user_auth_headers(config)
                and is_http_auth_challenge(
                    exc,
                    response_challenged=response_signal.challenged,
                )
            )
            if not can_escalate:
                raise
            logger.info(
                "%s: Received authentication challenge; retrying with OAuth enabled",
                server_name,
            )
            continue

        try:
            yield connection
        except BaseException as exc:
            suppressed = await connection.__aexit__(type(exc), exc, exc.__traceback__)
            if not suppressed:
                raise
        else:
            await connection.__aexit__(None, None, None)
        return


def create_client_connection(
    *,
    server_name: str,
    config: MCPServerSettings,
    callback_runtime: MCPClientCallbackRuntime,
    oauth_mode: OAuthMode,
    oauth_active: bool,
    cache: bool = True,
    hooks: MCPClientHooks | None = None,
) -> MCPClientConnection:
    """The sole production constructor for MCP transports and client connections."""
    transport = _create_transport(
        server_name=server_name,
        config=config,
        oauth_mode=oauth_mode,
        oauth_active=oauth_active,
        hooks=hooks or MCPClientHooks(),
    )
    return MCPClientConnection(
        transport,
        callback_runtime,
        read_timeout_seconds=config.read_timeout_seconds,
        cache=cache,
        protocol_mode=config.protocol_mode,
    )


def _create_transport(
    *,
    server_name: str,
    config: MCPServerSettings,
    oauth_mode: OAuthMode,
    oauth_active: bool,
    hooks: MCPClientHooks,
) -> Transport:
    if config.transport == "stdio":
        if not config.command:
            raise ValueError(
                f"Server '{server_name}' uses stdio transport but no command is specified"
            )
        params = StdioServerParameters(
            command=config.command,
            args=config.args or [],
            env=build_child_environment(
                active_home=hooks.active_home,
                no_home=hooks.no_home,
                base=get_default_environment(),
                overrides=config.env,
            ),
            cwd=config.cwd,
        )
        errlog = get_stderr_handler(server_name, on_line=hooks.stderr_line_handler)
        channel_hook = _transport_metrics_hook(server_name, hooks.transport_metrics)
        return tracking_stdio_client(params, channel_hook=channel_hook, errlog=errlog)

    if config.transport not in {"sse", "http"}:
        raise ValueError(f"Unsupported transport: {config.transport}")
    if not config.url:
        raise ValueError(
            f"Server '{server_name}' uses {config.transport} transport but no url is specified"
        )

    suppress_errors = (
        hooks.suppress_sse_errors if config.transport == "sse" else hooks.suppress_http_errors
    )
    if suppress_errors is not None:
        suppress_errors()
    if hooks.suppress_oauth_errors is not None:
        hooks.suppress_oauth_errors()

    prepared_auth = _prepare_headers_and_auth(
        config,
        trigger_oauth=oauth_active,
        oauth_mode=oauth_mode if oauth_active else "disabled",
        oauth_event_handler=hooks.oauth_event_handler,
        emit_oauth_console_output=hooks.oauth_event_handler is None,
        oauth_abort_event=hooks.oauth_abort_event,
        allow_oauth_paste_fallback=hooks.allow_oauth_paste_fallback,
    )
    if prepared_auth.user_auth_keys and prepared_auth.oauth_provider is None:
        logger.debug(
            _format_user_auth_skip_oauth_message(server_name, prepared_auth.user_auth_keys),
            user_auth_headers=sorted(prepared_auth.user_auth_keys),
        )

    if config.transport == "sse":
        return sse_client(
            config.url,
            prepared_auth.headers,
            sse_read_timeout=config.read_transport_sse_timeout_seconds,
            auth=prepared_auth.oauth_provider,
        )

    http_client = httpx2.AsyncClient(
        headers=prepared_auth.headers,
        auth=prepared_auth.oauth_provider,
        timeout=_http_timeout(config),
        follow_redirects=True,
        event_hooks=_http_diagnostic_hooks(server_name, hooks),
    )
    return _managed_http_transport_context(
        http_client,
        streamable_http_client(config.url, http_client=http_client),
    )


def _transport_metrics_hook(
    server_name: str,
    metrics: TransportChannelMetrics | None,
) -> Callable[[ChannelEvent], None] | None:
    if metrics is None:
        return None

    def record(event: ChannelEvent) -> None:
        try:
            metrics.record_event(event)
        except Exception:
            logger.debug("%s: transport metrics hook failed", server_name, exc_info=True)

    return record


def _http_post_channel(
    response: httpx2.Response,
    message: JSONRPCMessage | None = None,
) -> ChannelName:
    if (
        isinstance(message, JSONRPCRequest)
        and strip_casefold(message.method or "") == "subscriptions/listen"
    ):
        return "listen"
    content_type = strip_casefold(response.headers.get("content-type", ""))
    return "post-sse" if content_type.startswith("text/event-stream") else "post-json"


def _http_request_message(request: httpx2.Request) -> JSONRPCMessage | None:
    if request.method != "POST" or not request.content:
        return None
    return _JSONRPC_MESSAGE_ADAPTER.validate_json(request.content)


def _http_diagnostic_hooks(
    server_name: str,
    hooks: MCPClientHooks,
) -> dict[str, list[Callable]] | None:
    metrics = hooks.transport_metrics
    response_handlers: list[Callable] = []
    if hooks.http_response_handler is not None:
        response_handlers.append(hooks.http_response_handler)

    if metrics is None:
        return {"response": response_handlers} if response_handlers else None

    async def capture_request(request: httpx2.Request) -> None:
        try:
            if request.method == "GET":
                metrics.record_event(
                    ChannelEvent(
                        channel="get",
                        event_type="connect",
                        detail=(
                            f"Last-Event-ID {request.headers['last-event-id']}"
                            if "last-event-id" in request.headers
                            else None
                        ),
                    )
                )
                if "last-event-id" in request.headers:
                    metrics.record_event(
                        ChannelEvent(
                            channel="resumption",
                            event_type="connect",
                            detail=request.headers["last-event-id"],
                        )
                    )
        except Exception:
            logger.debug("%s: HTTP diagnostics hook failed", server_name, exc_info=True)

    async def capture_response(response: httpx2.Response) -> None:
        try:
            if response.is_redirect:
                return
            request = response.request
            if request.method == "GET":
                event_type = "error" if response.status_code >= 400 else "connect"
                metrics.record_event(
                    ChannelEvent(
                        channel="get",
                        event_type=event_type,
                        status_code=response.status_code,
                        detail=f"HTTP {response.status_code}",
                    )
                )
            elif request.method == "POST":
                try:
                    message = _http_request_message(request)
                except (ValidationError, ValueError):
                    message = None
                channel = _http_post_channel(response, message)
                if message is not None:
                    metrics.record_event(
                        ChannelEvent(
                            channel=channel,
                            event_type="message",
                            message=message,
                        )
                    )
                if response.status_code >= 400:
                    metrics.record_event(
                        ChannelEvent(
                            channel=channel,
                            event_type="error",
                            status_code=response.status_code,
                            detail=f"HTTP {response.status_code}",
                        )
                    )
        except Exception:
            logger.debug("%s: HTTP diagnostics hook failed", server_name, exc_info=True)

    response_handlers.append(capture_response)
    return {
        "request": [capture_request],
        "response": response_handlers,
    }


def _prepare_headers_and_auth(
    server_config: MCPServerSettings,
    *,
    trigger_oauth: bool | None = None,
    oauth_mode: OAuthMode | None = None,
    oauth_event_handler: OAuthEventHandler | None = None,
    emit_oauth_console_output: bool = True,
    oauth_abort_event: threading.Event | None = None,
    allow_oauth_paste_fallback: bool = True,
) -> PreparedHttpAuth:
    headers: dict[str, str] = dict(server_config.headers or {})
    auth_header_keys = {"authorization", "x-hf-authorization"}
    user_auth_keys = {key for key in headers if strip_casefold(key) in auth_header_keys}

    if (
        server_config.auth is not None
        and server_config.auth.forward == "huggingface"
        and server_config.url
        and not user_auth_keys
    ):
        headers = add_forwarded_hf_auth_header(server_config.url, headers) or {}
        user_auth_keys = {key for key in headers if strip_casefold(key) in auth_header_keys}

    if server_config.auth is not None and server_config.auth.forward == "huggingface":
        return PreparedHttpAuth(headers, None, user_auth_keys)

    force_oauth = oauth_mode == "force" or (oauth_mode is None and trigger_oauth is True)
    auto_oauth = oauth_mode == "auto"
    if (
        not (force_oauth or auto_oauth)
        or not uses_mcp_remote_transport(server_config.transport)
        or (auto_oauth and user_auth_keys)
    ):
        return PreparedHttpAuth(headers, None, user_auth_keys)

    oauth_provider = build_oauth_provider(
        server_config,
        event_handler=oauth_event_handler,
        emit_console_output=emit_oauth_console_output,
        abort_event=oauth_abort_event,
        allow_paste_fallback=allow_oauth_paste_fallback,
    )
    if oauth_provider is not None:
        for header_name in (
            "Authorization",
            "authorization",
            "X-HF-Authorization",
            "x-hf-authorization",
        ):
            headers.pop(header_name, None)
    return PreparedHttpAuth(headers, oauth_provider, user_auth_keys)


def _format_user_auth_skip_oauth_message(server_name: str, user_auth_keys: set[str]) -> str:
    return (
        f"{server_name}: Using user-specified "
        f"{format_count(len(user_auth_keys), 'auth header')}; skipping OAuth provider."
    )


def _http_timeout(config: MCPServerSettings) -> httpx2.Timeout | None:
    if config.http_timeout_seconds is None and config.http_read_timeout_seconds is None:
        return None
    return httpx2.Timeout(
        config.http_timeout_seconds or 30,
        read=config.http_read_timeout_seconds or 300,
    )


@asynccontextmanager
async def _managed_http_transport_context(
    http_client: httpx2.AsyncClient,
    transport_context: AbstractAsyncContextManager,
):
    async with http_client, transport_context as streams:
        yield streams
