"""
Manages the lifecycle of multiple MCP server connections.
"""

from __future__ import annotations

import asyncio
import threading
import time
import traceback
from collections import deque
from contextlib import suppress
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn, Protocol, runtime_checkable
from urllib.parse import urlsplit

import httpx2
from anyio import CancelScope, Event, Lock
from httpx2 import HTTPStatusError
from mcp.client.subscriptions import SubscriptionLost
from mcp.shared.exceptions import MCPError
from mcp_types import JSONRPCNotification, SubscriptionFilter

from fast_agent.context_dependent import ContextDependent
from fast_agent.core.exceptions import ServerInitializationError
from fast_agent.core.logging.logger import get_logger
from fast_agent.event_progress import ProgressAction
from fast_agent.mcp.client_callback_runtime import MCPClientCallbackRuntime
from fast_agent.mcp.client_gateway import (
    MCPClientHooks,
    create_client_connection,
)
from fast_agent.mcp.client_gateway import (
    is_http_auth_challenge as _is_http_auth_challenge_error,
)
from fast_agent.mcp.client_gateway import (
    resolve_oauth_mode as _resolve_oauth_mode,
)
from fast_agent.mcp.oauth_client import (
    OAuthEvent,
    OAuthEventHandler,
    OAuthFlowCancelledError,
)
from fast_agent.mcp.transport_tracking import (
    ChannelEvent,
    ChannelName,
    EventType,
    TransportChannelMetrics,
)
from fast_agent.utils.commandline import join_commandline
from fast_agent.utils.count_display import format_count
from fast_agent.utils.text import strip_casefold
from fast_agent.utils.transports import is_mcp_client_transport

if TYPE_CHECKING:
    from collections.abc import Callable

    from mcp_types import Implementation, ServerCapabilities

    from fast_agent.config import MCPServerSettings
    from fast_agent.context import Context
    from fast_agent.mcp.client_connection import MCPClientConnection
    from fast_agent.mcp.client_gateway import OAuthMode
    from fast_agent.mcp_server_registry import ServerRegistry

logger = get_logger(__name__)
STDIO_STDERR_BUFFER_LINES = 12
PARTIAL_SUBSCRIPTION_REFRESH_SECONDS = 5.0


@runtime_checkable
class PingableMCPClient(Protocol):
    """High-level client capability used by the optional legacy keepalive loop."""

    async def ping(self, read_timeout_seconds: float | None = None) -> object: ...


def _pingable_client(client: object | None) -> PingableMCPClient | None:
    return client if isinstance(client, PingableMCPClient) else None


def _format_ping_shutdown_error(missed: int, exc: Exception) -> str:
    return f"Ping failed {format_count(missed, 'time')}; last error: {exc}"


def _format_lifecycle_exception_group_errors(exception_group: BaseExceptionGroup) -> list[str]:
    messages: list[str] = []
    for subexc in exception_group.exceptions:
        if isinstance(subexc, BaseExceptionGroup):
            messages.extend(_format_lifecycle_exception_group_errors(subexc))
            continue
        if isinstance(subexc, HTTPStatusError):
            messages.append(
                f"HTTP Error: {subexc.response.status_code} {subexc.response.reason_phrase} for URL: {subexc.request.url}"
            )
            continue

        messages.append(f"{type(subexc).__name__}: {subexc}")
        if subexc.__cause__ is not None:
            messages.append(f"Caused by: {type(subexc.__cause__).__name__}: {subexc.__cause__}")
    return messages


class ServerConnection:
    """
    Represents an attached local MCP client runtime and its product status.
    """

    def __init__(
        self,
        server_name: str,
        server_config: MCPServerSettings,
        client_connection_factory: Callable[[], MCPClientConnection],
        callback_runtime: MCPClientCallbackRuntime,
    ) -> None:
        self.server_name = server_name
        self.server_config = server_config
        self.client: MCPClientConnection | None = None
        self._callback_runtime = callback_runtime
        self._client_connection_factory = client_connection_factory
        # Signal that the runtime is fully up and negotiated.
        self._initialized_event = Event()

        # Signal we want to shut down
        self._shutdown_event = Event()
        self._lifecycle_complete_event = Event()

        # Track error state
        self._error_occurred = False
        self._error_message = None
        self._lifecycle_error: Exception | None = None

        # Server instructions from initialization
        self.server_instructions: str | None = None
        self.server_capabilities: ServerCapabilities | None = None
        self.server_implementation: Implementation | None = None
        self.protocol_version: str | None = None
        self.protocol_era: str | None = None
        self.supported_protocol_versions: tuple[str, ...] = ()
        self.negotiation: str | None = None
        self.subscription_state: str | None = None
        self.server_instructions_available: bool = False
        self.server_instructions_enabled: bool = (
            server_config.include_instructions if server_config else True
        )
        self.session_id: str | None = None
        self.transport_metrics: TransportChannelMetrics | None = None
        self._ping_ok_count = 0
        self._ping_fail_count = 0
        self._ping_consecutive_failures = 0
        self._ping_last_ok_at: datetime | None = None
        self._ping_last_fail_at: datetime | None = None
        self._ping_last_error: str | None = None
        self._ping_history: deque[tuple[datetime, str]] = deque(maxlen=200)
        self._oauth_wait_active = False
        self._oauth_wait_started_at: float | None = None
        self._oauth_wait_accumulated_seconds = 0.0
        self._oauth_callback_timed_out = False
        self._auth_challenge_received = False
        self._oauth_abort_event = threading.Event()
        self._stdio_stderr_lines: deque[str] = deque(maxlen=STDIO_STDERR_BUFFER_LINES)
        self._lifecycle_cancel_scope: CancelScope | None = None

    def is_healthy(self) -> bool:
        """Check if the server connection is healthy and ready to use."""
        return self.client is not None and not self._error_occurred

    def request_shutdown(self) -> None:
        """
        Request the server to shut down. Signals the server lifecycle task to exit.
        """
        self._oauth_abort_event.set()
        self._shutdown_event.set()

    def cancel_lifecycle(self) -> None:
        """Request shutdown and cancel the lifecycle task if it is still blocked."""
        self.request_shutdown()
        if self._lifecycle_cancel_scope is not None:
            self._lifecycle_cancel_scope.cancel()

    def shutdown_lifecycle(self) -> None:
        """Shut down gracefully once initialized, or cancel blocked startup."""
        self.request_shutdown()
        if not self.is_initialized():
            self.cancel_lifecycle()

    async def wait_for_shutdown_request(self) -> None:
        """
        Wait until the shutdown event is set.
        """
        await self._shutdown_event.wait()

    async def wait_for_lifecycle_completion(self) -> None:
        """Wait until the lifecycle task has released all runtime resources."""
        await self._lifecycle_complete_event.wait()

    async def initialize_client(self) -> None:
        """
        Capture negotiated peer metadata from the entered SDK client.
        Must be called within an async context.
        """
        assert self.client, "Client must be entered before initialization"
        self.protocol_version = self.client.protocol_version
        discover_result = self.client.discover_result
        self.protocol_era = "modern" if discover_result is not None else "legacy"
        if discover_result is not None:
            self.negotiation = "discover"
        else:
            self.negotiation = "initialize"
        self.supported_protocol_versions = (
            tuple(discover_result.supported_versions) if discover_result is not None else ()
        )
        self.server_capabilities = self.client.server_capabilities
        self.server_implementation = self.client.server_info

        raw_instructions = self.client.instructions
        self.server_instructions_available = bool(raw_instructions)

        # Store instructions if provided by the server and enabled in config
        if self.server_config.include_instructions:
            self.server_instructions = raw_instructions
            if self.server_instructions:
                logger.debug(
                    f"{self.server_name}: Received server instructions",
                    data={"instructions": self.server_instructions},
                )
        else:
            self.server_instructions = None
            if self.server_instructions_available:
                logger.debug(
                    f"{self.server_name}: Server instructions disabled by configuration",
                    data={"instructions": raw_instructions},
                )
            else:
                logger.debug(f"{self.server_name}: No server instructions provided")

        # If there's an init hook, run it

        # The runtime is ready for use.
        self._initialized_event.set()

    async def wait_for_initialized(self) -> None:
        """
        Wait until the client runtime is fully initialized.
        """
        await self._initialized_event.wait()

    def is_initialized(self) -> bool:
        """Return True once initialization (success or failure) has completed."""
        return self._initialized_event.is_set()

    def mark_oauth_wait_start(self, now: float | None = None) -> None:
        if self._oauth_wait_active:
            return
        self._oauth_wait_active = True
        self._oauth_wait_started_at = now if now is not None else time.monotonic()

    def mark_oauth_wait_end(self, now: float | None = None) -> None:
        if not self._oauth_wait_active:
            return
        end_time = now if now is not None else time.monotonic()
        if self._oauth_wait_started_at is not None:
            self._oauth_wait_accumulated_seconds += max(
                0.0,
                end_time - self._oauth_wait_started_at,
            )
        self._oauth_wait_started_at = None
        self._oauth_wait_active = False

    def oauth_wait_accumulated_seconds(self, now: float | None = None) -> float:
        total = self._oauth_wait_accumulated_seconds
        if self._oauth_wait_active and self._oauth_wait_started_at is not None:
            current_time = now if now is not None else time.monotonic()
            total += max(0.0, current_time - self._oauth_wait_started_at)
        return total

    def record_stdio_stderr(self, line: str) -> None:
        text = line.strip()
        if text:
            self._stdio_stderr_lines.append(text)

    def recent_stdio_stderr_lines(self) -> tuple[str, ...]:
        return tuple(self._stdio_stderr_lines)

    async def capture_http_response(self, response: httpx2.Response) -> None:
        """Capture public HTTP response metadata before SDK normalization."""
        if response.status_code == 401:
            self._auth_challenge_received = True
        if session_id := response.headers.get("mcp-session-id"):
            self.session_id = session_id

    def record_ping_event(self, state: str) -> None:
        if self.transport_metrics is None:
            return
        self._ping_history.append((datetime.now(timezone.utc), state))

    def build_ping_activity_buckets(self, bucket_seconds: int, bucket_count: int) -> list[str]:
        try:
            seconds = int(bucket_seconds)
        except (TypeError, ValueError):
            seconds = 30
        if seconds <= 0:
            seconds = 30

        try:
            count = int(bucket_count)
        except (TypeError, ValueError):
            count = 20
        if count <= 0:
            count = 20

        if not self._ping_history:
            return ["none"] * count

        priority = {"error": 2, "ping": 1, "none": 0}
        history_map: dict[int, str] = {}
        for timestamp, state in self._ping_history:
            bucket = int(timestamp.timestamp() // seconds)
            existing = history_map.get(bucket)
            if existing is None or priority.get(state, 0) >= priority.get(existing, 0):
                history_map[bucket] = state

        current_bucket = int(datetime.now(timezone.utc).timestamp() // seconds)
        buckets: list[str] = []
        for offset in range(count - 1, -1, -1):
            bucket_index = current_bucket - offset
            buckets.append(history_map.get(bucket_index, "none"))

        return buckets


async def _run_ping_loop(server_conn: ServerConnection) -> None:
    interval = server_conn.server_config.ping_interval_seconds
    if not interval or interval <= 0:
        return

    max_missed = server_conn.server_config.max_missed_pings
    missed = 0
    read_timeout = server_conn.server_config.read_timeout_seconds
    if read_timeout is None:
        read_timeout = float(interval)

    while not server_conn._shutdown_event.is_set():
        await asyncio.sleep(interval)
        if server_conn._shutdown_event.is_set():
            break
        client = _pingable_client(server_conn.client)
        if client is None:
            return
        try:
            from fast_agent.human_input.elicitation_state import elicitation_state

            if elicitation_state.is_active(server_conn.server_name):
                continue
        except Exception:
            pass
        try:
            await client.ping(read_timeout_seconds=read_timeout)
            missed = 0
            server_conn._ping_ok_count += 1
            server_conn._ping_consecutive_failures = 0
            server_conn._ping_last_ok_at = datetime.now(timezone.utc)
            server_conn._ping_last_error = None
            server_conn.record_ping_event("ping")
        except Exception as exc:
            missed += 1
            server_conn._ping_fail_count += 1
            server_conn._ping_consecutive_failures = missed
            server_conn._ping_last_fail_at = datetime.now(timezone.utc)
            server_conn._ping_last_error = str(exc)
            server_conn.record_ping_event("error")
            logger.warning(f"{server_conn.server_name}: Ping failed ({missed}/{max_missed}): {exc}")
            if missed >= max_missed:
                server_conn._error_occurred = True
                server_conn._error_message = _format_ping_shutdown_error(missed, exc)
                server_conn.request_shutdown()
                break


def _format_stdio_startup_error(server_conn: ServerConnection, exc: OSError) -> str:
    config = server_conn.server_config
    command_parts = [config.command] if config.command else []
    command_parts.extend(config.args or [])
    command_display = (
        join_commandline(command_parts, syntax="posix") if command_parts else "<unspecified>"
    )

    lines = [f"Failed to start stdio MCP server command: {command_display}."]

    cwd = config.cwd
    if isinstance(exc, FileNotFoundError):
        if cwd and not Path(cwd).exists():
            lines.append(f"Working directory not found: {cwd}")
        elif config.command:
            if "/" in config.command or "\\" in config.command:
                lines.append(f"Command path not found: {config.command}")
            else:
                lines.append(f"Executable not found on PATH: {config.command}")
        else:
            lines.append("Executable not found.")
    elif isinstance(exc, PermissionError):
        lines.append("Permission denied while starting the stdio server process.")
    else:  # pragma: no cover - kept defensive for future callers
        lines.append(f"{type(exc).__name__}: {exc}")

    if cwd:
        lines.append(f"cwd: {cwd}")

    return "\n".join(lines)


def _append_stdio_stderr_details(server_conn: ServerConnection, details: str) -> str:
    if server_conn.server_config.transport != "stdio":
        return details

    stderr_lines = server_conn.recent_stdio_stderr_lines()
    if not stderr_lines:
        return details

    stderr_block = "\n".join(f"  {line}" for line in stderr_lines)
    suffix = f"Recent stderr from stdio server:\n{stderr_block}"
    if not details:
        return suffix
    return f"{details}\n\n{suffix}"


def _is_stdio_startup_error(server_conn: ServerConnection, error_text: str) -> bool:
    return server_conn.server_config.transport == "stdio" and error_text.startswith(
        "Failed to start stdio MCP server command:"
    )


async def _server_lifecycle_task(server_conn: ServerConnection) -> None:
    """
    Manage the lifecycle of a single server connection.
    Runs inside the MCPConnectionManager's shared TaskGroup.

    IMPORTANT: This function must NEVER raise an exception, as it runs in a shared
    task group. Any exceptions must be caught and handled gracefully, with errors
    recorded in server_conn._error_occurred and _error_message.
    """
    with CancelScope() as cancel_scope:
        server_conn._lifecycle_cancel_scope = cancel_scope
        try:
            if not server_conn._shutdown_event.is_set():
                await _run_server_lifecycle(server_conn)
        finally:
            server_conn._lifecycle_cancel_scope = None
            server_conn._lifecycle_complete_event.set()


async def _run_server_lifecycle(server_conn: ServerConnection) -> None:
    """Run the server lifecycle inside the connection-owned cancellation scope."""
    try:
        connection = server_conn._client_connection_factory()
        server_conn.client = connection
        try:
            async with connection:
                await server_conn.initialize_client()
                await _wait_for_shutdown_with_optional_ping(server_conn)
        except Exception as client_exit_exc:
            if not _handle_shutdown_cleanup_error(
                server_conn,
                client_exit_exc,
                cleanup_scope="client",
            ):
                raise
        finally:
            server_conn.client = None

    except HTTPStatusError as http_exc:
        _record_http_lifecycle_error(server_conn, http_exc)
        # No raise - let get_server handle it with a friendly message

    except Exception as exc:
        if server_conn._shutdown_event.is_set() and _is_oauth_cancelled_message(str(exc)):
            _record_oauth_cancelled_shutdown(server_conn, exc)
            return

        _record_lifecycle_error(server_conn, exc)
        # No raise - allow graceful exit


async def _wait_for_shutdown_with_optional_ping(server_conn: ServerConnection) -> None:
    subscription_task: asyncio.Task[None] | None = None
    if _subscription_filter(server_conn) is not None:
        subscription_task = asyncio.create_task(_run_subscription_loop(server_conn))
    elif server_conn.protocol_era == "modern":
        server_conn.subscription_state = "disabled"
    if not _ping_loop_enabled(server_conn):
        ping_task = None
    else:
        ping_task = asyncio.create_task(_run_ping_loop(server_conn))
    try:
        await server_conn.wait_for_shutdown_request()
    finally:
        if ping_task is not None and not ping_task.done():
            ping_task.cancel()
        if subscription_task is not None and not subscription_task.done():
            subscription_task.cancel()
        if ping_task is not None:
            with suppress(asyncio.CancelledError):
                await ping_task
        if subscription_task is not None:
            with suppress(asyncio.CancelledError):
                await subscription_task


@dataclass(frozen=True, slots=True)
class _ModernSubscriptionFilter:
    tools_list_changed: bool
    prompts_list_changed: bool
    resources_list_changed: bool
    resource_subscriptions: tuple[str, ...]
    resource_subscription_capable: bool


def _subscription_filter(server_conn: ServerConnection) -> _ModernSubscriptionFilter | None:
    client = server_conn.client
    if client is None:
        return None
    discover_result = client.discover_result
    if discover_result is None:
        return None
    capabilities = discover_result.capabilities
    tools_list_changed = bool(capabilities.tools and capabilities.tools.list_changed is True)
    prompts_list_changed = bool(capabilities.prompts and capabilities.prompts.list_changed is True)
    resources = discover_result.capabilities.resources
    resources_list_changed = bool(resources and resources.list_changed is True)
    resource_subscription_capable = bool(resources and resources.subscribe is True)
    resource_subscriptions = (
        server_conn._callback_runtime.subscription_resource_uris()
        if resource_subscription_capable
        else ()
    )
    if not (
        tools_list_changed
        or prompts_list_changed
        or resources_list_changed
        or resource_subscription_capable
    ):
        return None
    return _ModernSubscriptionFilter(
        tools_list_changed=tools_list_changed,
        prompts_list_changed=prompts_list_changed,
        resources_list_changed=resources_list_changed,
        resource_subscriptions=resource_subscriptions,
        resource_subscription_capable=resource_subscription_capable,
    )


def _subscription_filter_fully_honored(
    requested: _ModernSubscriptionFilter,
    honored: SubscriptionFilter,
) -> bool:
    if requested.tools_list_changed and honored.tools_list_changed is not True:
        return False
    if requested.prompts_list_changed and honored.prompts_list_changed is not True:
        return False
    if requested.resources_list_changed and honored.resources_list_changed is not True:
        return False
    return set(requested.resource_subscriptions).issubset(honored.resource_subscriptions or ())


async def _run_subscription_loop(server_conn: ServerConnection) -> None:
    client = server_conn.client
    if client is None:
        server_conn.subscription_state = "unsupported"
        return
    delay = 0.25
    while not server_conn._shutdown_event.is_set():
        subscription_filter = _subscription_filter(server_conn)
        if subscription_filter is None:
            server_conn.subscription_state = "disabled"
            return
        opened = False
        planned_rotation = False
        try:
            server_conn.subscription_state = "connecting"
            subscription_context = client.listen(
                tools_list_changed=subscription_filter.tools_list_changed,
                prompts_list_changed=subscription_filter.prompts_list_changed,
                resources_list_changed=subscription_filter.resources_list_changed,
                resource_subscriptions=subscription_filter.resource_subscriptions,
            )
            async with subscription_context as subscription:
                await server_conn._callback_runtime.wait_until_subscription_ready()
                refreshed_uris = await server_conn._callback_runtime.refresh_subscription_state()
                if (
                    subscription_filter.resource_subscription_capable
                    and refreshed_uris != subscription_filter.resource_subscriptions
                ):
                    server_conn.subscription_state = "rotating"
                    planned_rotation = True
                    continue
                fully_honored = _subscription_filter_fully_honored(
                    subscription_filter,
                    subscription.honored,
                )
                server_conn.subscription_state = "open" if fully_honored else "partial"
                opened = True
                _record_listen_transport_event(server_conn, "connect")
                delay = 0.25
                while True:
                    try:
                        if fully_honored:
                            event = await anext(subscription)
                        else:
                            async with asyncio.timeout(PARTIAL_SUBSCRIPTION_REFRESH_SECONDS):
                                event = await anext(subscription)
                    except TimeoutError:
                        refreshed_uris = (
                            await server_conn._callback_runtime.refresh_subscription_state()
                        )
                        if (
                            subscription_filter.resource_subscription_capable
                            and refreshed_uris != subscription_filter.resource_subscriptions
                        ):
                            server_conn.subscription_state = "rotating"
                            planned_rotation = True
                            break
                        continue
                    except StopAsyncIteration:
                        break
                    _record_listen_transport_event(
                        server_conn,
                        "message",
                        detail=type(event).__name__,
                    )
                    await server_conn._callback_runtime.handle_subscription_event(event)
                    if (
                        subscription_filter.resource_subscription_capable
                        and server_conn._callback_runtime.subscription_resource_uris()
                        != subscription_filter.resource_subscriptions
                    ):
                        server_conn.subscription_state = "rotating"
                        planned_rotation = True
                        break
            server_conn.subscription_state = "closed"
        except SubscriptionLost as exc:
            server_conn.subscription_state = "error"
            _record_listen_transport_event(server_conn, "error", detail=str(exc))
        except MCPError as exc:
            if exc.code == -32601:
                server_conn.subscription_state = "unsupported"
                _record_listen_transport_event(server_conn, "unsupported")
                return
            server_conn.subscription_state = "error"
            _record_listen_transport_event(server_conn, "error", detail=str(exc))
            if strip_casefold(exc.message).startswith("subscription limit reached"):
                # Treat server subscription capacity as terminal for now; we may revisit
                # this retry policy when servers expose more actionable recovery signals.
                return
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            server_conn.subscription_state = "error"
            _record_listen_transport_event(server_conn, "error", detail=str(exc))
            logger.debug(
                "%s: subscription stream failed",
                server_conn.server_name,
                exc_info=True,
            )
        finally:
            if opened:
                _record_listen_transport_event(server_conn, "disconnect")
        if server_conn._shutdown_event.is_set():
            return
        if planned_rotation:
            continue
        await asyncio.sleep(delay)
        delay = min(delay * 2, 5)


def _record_listen_transport_event(
    server_conn: ServerConnection,
    event_type: EventType,
    *,
    detail: str | None = None,
) -> None:
    if server_conn.server_config.transport != "http" or server_conn.transport_metrics is None:
        return
    server_conn.transport_metrics.record_event(
        ChannelEvent(
            channel="listen",
            event_type=event_type,
            detail=detail,
        )
    )


def _transport_notification_handler(
    config: MCPServerSettings,
    metrics: TransportChannelMetrics,
) -> Callable[[str], None]:
    channel: ChannelName
    if config.transport == "stdio":
        channel = "stdio"
    elif config.transport == "sse":
        channel = "get"
    else:
        channel = "post-sse"

    def record(method: str) -> None:
        metrics.record_event(
            ChannelEvent(
                channel=channel,
                event_type="message",
                message=JSONRPCNotification(
                    jsonrpc="2.0",
                    method=method,
                ),
            )
        )

    return record


def _ping_loop_enabled(server_conn: ServerConnection) -> bool:
    interval = server_conn.server_config.ping_interval_seconds
    return server_conn.protocol_era == "legacy" and bool(interval and interval > 0)


def _handle_shutdown_cleanup_error(
    server_conn: ServerConnection,
    exc: Exception,
    *,
    cleanup_scope: str,
) -> bool:
    if not server_conn._shutdown_event.is_set():
        return False

    logger.debug(
        f"{server_conn.server_name}: Exception during {cleanup_scope} cleanup "
        f"(expected during shutdown): {exc}"
    )
    if not server_conn._initialized_event.is_set():
        server_conn._error_occurred = True
        server_conn._error_message = "Shutdown requested before initialization"
        server_conn._initialized_event.set()
    return True


def _record_http_lifecycle_error(
    server_conn: ServerConnection,
    http_exc: HTTPStatusError,
) -> None:
    logger.error(
        f"{server_conn.server_name}: Lifecycle task encountered HTTP error: {http_exc}",
        exc_info=True,
        data={
            "progress_action": ProgressAction.FATAL_ERROR,
            "server_name": server_conn.server_name,
        },
    )
    server_conn._error_occurred = True
    server_conn._lifecycle_error = http_exc
    server_conn._error_message = (
        f"HTTP Error: {http_exc.response.status_code} "
        f"{http_exc.response.reason_phrase} for URL: {http_exc.request.url}"
    )
    server_conn._initialized_event.set()


def _record_oauth_cancelled_shutdown(
    server_conn: ServerConnection,
    exc: Exception,
) -> None:
    logger.debug(f"{server_conn.server_name}: OAuth authorization cancelled during shutdown")
    server_conn._error_occurred = True
    server_conn._lifecycle_error = exc
    server_conn._error_message = str(exc)
    server_conn._initialized_event.set()


def _record_lifecycle_error(server_conn: ServerConnection, exc: Exception) -> None:
    logger.error(
        f"{server_conn.server_name}: Lifecycle task encountered an error: {exc}",
        exc_info=True,
        data={
            "progress_action": ProgressAction.FATAL_ERROR,
            "server_name": server_conn.server_name,
        },
    )
    server_conn._error_occurred = True
    server_conn._lifecycle_error = exc
    server_conn._error_message = _lifecycle_error_message(server_conn, exc)
    server_conn._initialized_event.set()


def _lifecycle_error_message(
    server_conn: ServerConnection,
    exc: Exception,
) -> str | list[str]:
    if isinstance(exc, BaseExceptionGroup):
        error_messages = _format_lifecycle_exception_group_errors(exc)
        return error_messages or [f"{type(exc).__name__}: {exc}"]
    if server_conn.server_config.transport == "stdio" and isinstance(
        exc,
        (FileNotFoundError, PermissionError),
    ):
        return _format_stdio_startup_error(server_conn, exc)
    return traceback.format_exception(exc)


def _is_oauth_timeout_message(message: str | None) -> bool:
    if not message:
        return False
    normalized = " ".join(strip_casefold(str(message)).split())

    oauth_timeout_phrases = (
        "oauth authorization timed out",
        "oauth authorization was not completed in time",
        "oauth callback timeout",
        "oauth callback timed out",
        "oauth flow timed out",
    )
    return any(phrase in normalized for phrase in oauth_timeout_phrases)


def _is_oauth_registration_404_message(message: str | None) -> bool:
    if not message:
        return False
    normalized = strip_casefold(message)
    return "oauth" in normalized and "registration failed: 404" in normalized


def _format_oauth_registration_404_details(error_text: str, server_url: str | None) -> str:
    details = (
        "OAuth client registration failed with HTTP 404.\n"
        "The server likely does not support dynamic client registration for this client.\n"
        "Try one of these options:\n"
        "- Configure a Client ID Metadata URL (CIMD): auth.client_metadata_url or --client-metadata-url\n"
        "- Use direct bearer authentication with --auth <token>\n"
    )
    if _server_url_host_matches(server_url, "githubcopilot.com"):
        details += "GitHub Copilot MCP commonly expects token auth for external hosts. Try --auth $GITHUB_TOKEN.\n"
    details += f"\nOriginal error:\n{error_text}"
    return details


def _server_url_host_matches(server_url: str | None, expected_host: str) -> bool:
    if not server_url:
        return False
    try:
        hostname = urlsplit(server_url).hostname
    except ValueError:
        return False
    if hostname is None:
        return False
    return strip_casefold(hostname) == expected_host


def _is_oauth_cancelled_message(message: str | None) -> bool:
    if not message:
        return False
    normalized = strip_casefold(message)
    return "oauth" in normalized and "cancel" in normalized


async def _wait_for_initialized_with_startup_budget(
    server_conn: ServerConnection,
    startup_timeout_seconds: float | None,
    *,
    poll_interval_seconds: float = 0.1,
) -> None:
    """Wait for server initialization while excluding OAuth wait windows from startup timeout."""

    if startup_timeout_seconds is None:
        await server_conn.wait_for_initialized()
        return

    if startup_timeout_seconds <= 0:
        raise ValueError("startup_timeout_seconds must be > 0 when provided")

    startup_clock = time.monotonic()
    interval = max(0.01, poll_interval_seconds)

    while not server_conn.is_initialized():
        now = time.monotonic()
        wall_elapsed = now - startup_clock
        oauth_wait_elapsed = server_conn.oauth_wait_accumulated_seconds(now=now)
        machine_elapsed = max(0.0, wall_elapsed - oauth_wait_elapsed)

        if machine_elapsed >= startup_timeout_seconds:
            raise TimeoutError("MCP server startup timed out (non-OAuth budget exhausted)")

        await asyncio.sleep(interval)

    await server_conn.wait_for_initialized()


class MCPConnectionManager(ContextDependent):
    """
    Manages the lifecycle of multiple MCP server connections.
    Integrates with the application context system for proper resource management.
    """

    def __init__(self, server_registry: "ServerRegistry", context: "Context | None" = None) -> None:
        super().__init__(context=context)
        self.server_registry = server_registry
        self.running_servers: dict[str, ServerConnection] = {}
        self._lock = Lock()
        self._task_group: asyncio.TaskGroup | None = None
        self._task_group_active = False
        self._mcp_sse_filter_added = False
        self._mcp_streamable_http_filter_added = False
        self._mcp_oauth_cancel_filter_added = False
        self._oauth_required_servers: set[str] = set()
        self._server_oauth_mode: dict[str, OAuthMode] = {}
        self._server_oauth_active: dict[str, bool] = {}

    async def __aenter__(self):
        self._task_group = asyncio.TaskGroup()
        await self._task_group.__aenter__()
        self._task_group_active = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Ensure clean shutdown of all connections before exiting."""
        try:
            await self.disconnect_all()
        finally:
            if self._task_group_active:
                assert self._task_group is not None
                try:
                    await self._task_group.__aexit__(exc_type, exc_val, exc_tb)
                finally:
                    self._task_group_active = False
                    self._task_group = None

    def _suppress_mcp_sse_errors(self) -> None:
        """Suppress MCP library's 'Error in sse_reader' messages."""
        if self._mcp_sse_filter_added:
            return

        import logging

        class MCPSSEErrorFilter(logging.Filter):
            def filter(self, record):
                return not (
                    record.name == "mcp.client.sse" and "Error in sse_reader" in record.getMessage()
                )

        mcp_sse_logger = logging.getLogger("mcp.client.sse")
        mcp_sse_logger.addFilter(MCPSSEErrorFilter())
        self._mcp_sse_filter_added = True

    def _suppress_mcp_streamable_http_errors(self) -> None:
        """Suppress noisy MCP streamable_http post-writer tracebacks for transient network loss."""
        if self._mcp_streamable_http_filter_added:
            return

        import logging

        class MCPStreamableHTTPErrorFilter(logging.Filter):
            def filter(self, record):
                message = record.getMessage()
                return not (
                    record.name == "mcp.client.streamable_http"
                    and "Error in post_writer" in message
                )

        mcp_http_logger = logging.getLogger("mcp.client.streamable_http")
        mcp_http_logger.addFilter(MCPStreamableHTTPErrorFilter())
        self._mcp_streamable_http_filter_added = True

    def _suppress_mcp_oauth_cancel_errors(self) -> None:
        """Suppress noisy OAuth flow tracebacks from MCP OAuth internals."""
        if self._mcp_oauth_cancel_filter_added:
            return

        import logging

        class MCPOAuthCancellationFilter(logging.Filter):
            def filter(self, record):
                if record.name != "mcp.client.auth.oauth2":
                    return True
                if "OAuth flow error" not in record.getMessage():
                    return True
                exc_info = getattr(record, "exc_info", None)
                if not exc_info:
                    return True
                try:
                    _exc_type, exc_value, _exc_tb = exc_info
                except Exception:
                    return True

                # User-cancelled OAuth flows are expected.
                if isinstance(exc_value, OAuthFlowCancelledError):
                    return False

                # Avoid traceback spam in normal operation. Keep full OAuth tracebacks
                # visible when debug logging is enabled.
                return logging.getLogger().isEnabledFor(logging.DEBUG)

        oauth_logger = logging.getLogger("mcp.client.auth.oauth2")
        oauth_logger.addFilter(MCPOAuthCancellationFilter())
        self._mcp_oauth_cancel_filter_added = True

    def _build_oauth_event_handler(
        self,
        server_conn: ServerConnection,
        user_event_handler: OAuthEventHandler | None,
    ) -> OAuthEventHandler:
        async def handle_event(event: OAuthEvent) -> None:
            if event.event_type == "wait_start":
                server_conn.mark_oauth_wait_start()
            elif event.event_type == "wait_end":
                server_conn.mark_oauth_wait_end()
            elif event.event_type == "oauth_error":
                if event.is_timeout or _is_oauth_timeout_message(event.message):
                    server_conn._oauth_callback_timed_out = True

            if user_event_handler is None:
                return

            try:
                await user_event_handler(event)
            except Exception:
                logger.debug(
                    f"{server_conn.server_name}: OAuth event callback failed",
                    event_type=event.event_type,
                    exc_info=True,
                )

        return handle_event

    async def launch_server(
        self,
        server_name: str,
        *,
        server_config: MCPServerSettings | None = None,
        callback_runtime: MCPClientCallbackRuntime,
        startup_timeout_seconds: float | None = None,
        trigger_oauth: bool | None = None,
        oauth_event_handler: OAuthEventHandler | None = None,
        allow_oauth_paste_fallback: bool = True,
    ) -> ServerConnection:
        """
        Connect to a server and return a RunningServer instance that will persist
        until explicitly disconnected.
        """
        if startup_timeout_seconds is not None and startup_timeout_seconds <= 0:
            raise ValueError("startup_timeout_seconds must be > 0 when provided")

        await self._ensure_task_group(server_name)

        config = server_config or self.server_registry.get_server_config(server_name)
        if not config:
            raise ValueError(f"Server '{server_name}' not found in registry.")

        logger.debug(f"{server_name}: Found server configuration=", data=config.model_dump())
        oauth_mode = _resolve_oauth_mode(config, trigger_oauth=trigger_oauth)
        oauth_active = oauth_mode == "force" or (
            oauth_mode == "auto" and server_name in self._oauth_required_servers
        )

        transport_metrics = self._launch_transport_metrics(config)
        connection_callback_runtime = replace(
            callback_runtime,
            transport_notification_handler=(
                _transport_notification_handler(config, transport_metrics)
                if transport_metrics is not None
                else None
            ),
        )
        server_conn_holder: list[ServerConnection] = []

        server_conn = ServerConnection(
            server_name=server_name,
            server_config=config,
            client_connection_factory=self._client_connection_factory(
                server_conn_holder,
                server_name=server_name,
                config=config,
                oauth_mode=oauth_mode,
                oauth_active=oauth_active,
                oauth_event_handler=oauth_event_handler,
                allow_oauth_paste_fallback=allow_oauth_paste_fallback,
                transport_metrics=transport_metrics,
            ),
            callback_runtime=connection_callback_runtime,
        )
        server_conn_holder.append(server_conn)

        if transport_metrics is not None:
            server_conn.transport_metrics = transport_metrics

        async with self._lock:
            # Check if already running
            if server_name in self.running_servers:
                existing = self.running_servers[server_name]
                if existing.server_config != config:
                    raise ValueError(
                        f"MCP server '{server_name}' is already starting with different settings"
                    )
                return existing

            self.running_servers[server_name] = server_conn
            self._server_oauth_mode[server_name] = oauth_mode
            self._server_oauth_active[server_name] = oauth_active
            assert self._task_group is not None
            self._task_group.create_task(_server_lifecycle_task(server_conn))

        logger.info(f"{server_name}: Attached MCP client runtime is ready")
        return server_conn

    async def _ensure_task_group(self, server_name: str) -> None:
        if self._task_group_active:
            return
        self._task_group = asyncio.TaskGroup()
        await self._task_group.__aenter__()
        self._task_group_active = True
        logger.info(f"Auto-created task group for server: {server_name}")

    def _launch_transport_metrics(
        self,
        config: MCPServerSettings,
    ) -> TransportChannelMetrics | None:
        if not is_mcp_client_transport(config.transport):
            return None

        timeline_steps = 20
        timeline_seconds = 30
        try:
            ctx = self.context
        except RuntimeError:
            ctx = None

        config_obj = ctx.config if ctx is not None else None
        mcp_config = config_obj.mcp if config_obj is not None else None
        diagnostics = mcp_config.diagnostics if mcp_config is not None else None
        if diagnostics is not None:
            if not diagnostics.enabled:
                return None
            timeline_steps = diagnostics.timeline.steps
            timeline_seconds = diagnostics.timeline.step_seconds

        return TransportChannelMetrics(
            bucket_seconds=timeline_seconds,
            bucket_count=timeline_steps,
        )

    def _client_connection_factory(
        self,
        server_conn_holder: list[ServerConnection],
        *,
        server_name: str,
        config: MCPServerSettings,
        oauth_mode: OAuthMode,
        oauth_active: bool,
        oauth_event_handler: OAuthEventHandler | None,
        allow_oauth_paste_fallback: bool,
        transport_metrics: TransportChannelMetrics | None,
    ) -> Callable[[], MCPClientConnection]:
        def client_connection_factory() -> MCPClientConnection:
            server_conn = server_conn_holder[0]
            hooks = MCPClientHooks(
                active_home=self.server_registry.active_home,
                no_home=self.server_registry.no_home,
                stderr_line_handler=server_conn.record_stdio_stderr,
                http_response_handler=server_conn.capture_http_response,
                oauth_event_handler=self._build_oauth_event_handler(
                    server_conn, oauth_event_handler
                ),
                oauth_abort_event=server_conn._oauth_abort_event,
                allow_oauth_paste_fallback=allow_oauth_paste_fallback,
                transport_metrics=transport_metrics,
                suppress_sse_errors=self._suppress_mcp_sse_errors,
                suppress_http_errors=self._suppress_mcp_streamable_http_errors,
                suppress_oauth_errors=self._suppress_mcp_oauth_cancel_errors,
            )
            return create_client_connection(
                server_name=server_name,
                config=config,
                callback_runtime=server_conn._callback_runtime,
                oauth_mode=oauth_mode,
                oauth_active=oauth_active,
                hooks=hooks,
            )

        return client_connection_factory

    async def _launch_and_wait_for_server(
        self,
        *,
        server_name: str,
        server_config: MCPServerSettings | None = None,
        callback_runtime: MCPClientCallbackRuntime,
        startup_timeout_seconds: float | None,
        trigger_oauth: bool | None,
        oauth_event_handler: OAuthEventHandler | None,
        allow_oauth_paste_fallback: bool,
        timeout_action: str,
    ) -> ServerConnection:
        """Launch a server connection and wait for initialization to complete."""
        server_conn = await self.launch_server(
            server_name=server_name,
            server_config=server_config,
            callback_runtime=callback_runtime,
            startup_timeout_seconds=startup_timeout_seconds,
            trigger_oauth=trigger_oauth,
            oauth_event_handler=oauth_event_handler,
            allow_oauth_paste_fallback=allow_oauth_paste_fallback,
        )

        try:
            await _wait_for_initialized_with_startup_budget(server_conn, startup_timeout_seconds)
        except asyncio.CancelledError:
            await self._clear_running_server_state(server_name, server_conn)
            raise
        except TimeoutError as exc:
            await self._clear_running_server_state(server_name, server_conn)
            raise ServerInitializationError(
                (
                    f"MCP Server: '{server_name}': {timeout_action} timed out after "
                    f"{startup_timeout_seconds:.1f}s (non-OAuth startup budget)"
                ),
                _append_stdio_stderr_details(
                    server_conn,
                    "Try increasing --timeout or verify server/network startup.",
                ),
                server_name=server_name,
            ) from exc

        return server_conn

    async def _clear_running_server_state(
        self,
        server_name: str,
        server_conn: ServerConnection,
    ) -> None:
        async with self._lock:
            server_conn.shutdown_lifecycle()
            await server_conn.wait_for_lifecycle_completion()
            current = self.running_servers.get(server_name)
            if current is server_conn:
                self.running_servers.pop(server_name, None)
                self._server_oauth_mode.pop(server_name, None)
                self._server_oauth_active.pop(server_name, None)

    async def _retry_server_with_oauth(
        self,
        *,
        server_name: str,
        server_conn: ServerConnection,
        server_config: MCPServerSettings | None = None,
        callback_runtime: MCPClientCallbackRuntime,
        startup_timeout_seconds: float | None,
        oauth_event_handler: OAuthEventHandler | None,
        allow_oauth_paste_fallback: bool,
        timeout_action: str,
    ) -> ServerConnection:
        logger.info(
            "%s: Received authentication challenge; retrying with OAuth enabled",
            server_name,
        )
        self._oauth_required_servers.add(server_name)
        await self._clear_running_server_state(server_name, server_conn)
        return await self._launch_and_wait_for_server(
            server_name=server_name,
            server_config=server_config,
            callback_runtime=callback_runtime,
            startup_timeout_seconds=startup_timeout_seconds,
            trigger_oauth=True,
            oauth_event_handler=oauth_event_handler,
            allow_oauth_paste_fallback=allow_oauth_paste_fallback,
            timeout_action=timeout_action,
        )

    def should_retry_server_with_oauth(self, server_name: str, error: object) -> bool:
        server_conn = self.running_servers.get(server_name)
        return (
            self._server_oauth_mode.get(server_name) == "auto"
            and not self._server_oauth_active.get(server_name, False)
            and (
                _is_http_auth_challenge_error(
                    error,
                    response_challenged=bool(server_conn and server_conn._auth_challenge_received),
                )
            )
        )

    async def get_server(
        self,
        server_name: str,
        *,
        server_config: MCPServerSettings | None = None,
        callback_runtime: MCPClientCallbackRuntime,
        startup_timeout_seconds: float | None = None,
        trigger_oauth: bool | None = None,
        oauth_event_handler: OAuthEventHandler | None = None,
        allow_oauth_paste_fallback: bool = True,
    ) -> ServerConnection:
        """
        Get a running server instance, launching it if needed.
        """
        if running_server := await self._healthy_running_server(server_name, server_config):
            return running_server

        server_conn = await self._launch_and_wait_for_server(
            server_name=server_name,
            server_config=server_config,
            callback_runtime=callback_runtime,
            startup_timeout_seconds=startup_timeout_seconds,
            trigger_oauth=trigger_oauth,
            oauth_event_handler=oauth_event_handler,
            allow_oauth_paste_fallback=allow_oauth_paste_fallback,
            timeout_action="Startup",
        )

        return await self._healthy_or_retry_server(
            server_name=server_name,
            server_conn=server_conn,
            server_config=server_config,
            callback_runtime=callback_runtime,
            startup_timeout_seconds=startup_timeout_seconds,
            oauth_event_handler=oauth_event_handler,
            allow_oauth_paste_fallback=allow_oauth_paste_fallback,
        )

    async def _healthy_running_server(
        self,
        server_name: str,
        server_config: MCPServerSettings | None,
    ) -> ServerConnection | None:
        async with self._lock:
            server_conn = self.running_servers.get(server_name)
            if server_conn is None:
                return None
            if server_conn.is_healthy() and (
                server_config is None or server_conn.server_config == server_config
            ):
                return server_conn
            logger.info(f"{server_name}: Server exists but is unhealthy, recreating...")
            server_conn.shutdown_lifecycle()
            await server_conn.wait_for_lifecycle_completion()
            self.running_servers.pop(server_name, None)
            self._server_oauth_mode.pop(server_name, None)
            self._server_oauth_active.pop(server_name, None)
            return None

    async def _healthy_or_retry_server(
        self,
        *,
        server_name: str,
        server_conn: ServerConnection,
        server_config: MCPServerSettings | None,
        callback_runtime: MCPClientCallbackRuntime,
        startup_timeout_seconds: float | None,
        oauth_event_handler: OAuthEventHandler | None,
        allow_oauth_paste_fallback: bool,
    ) -> ServerConnection:
        if server_conn.is_healthy():
            return server_conn

        if self.should_retry_server_with_oauth(
            server_name,
            server_conn._lifecycle_error or server_conn._error_message,
        ):
            retried_conn = await self._retry_server_with_oauth(
                server_name=server_name,
                server_conn=server_conn,
                server_config=server_config,
                callback_runtime=callback_runtime,
                startup_timeout_seconds=startup_timeout_seconds,
                oauth_event_handler=oauth_event_handler,
                allow_oauth_paste_fallback=allow_oauth_paste_fallback,
                timeout_action="Startup",
            )
            if retried_conn.is_healthy():
                return retried_conn
            server_conn = retried_conn

        await self._clear_running_server_state(server_name, server_conn)
        return self._raise_server_initialization_error(server_name, server_conn)

    def _raise_server_initialization_error(
        self,
        server_name: str,
        server_conn: ServerConnection,
    ) -> NoReturn:
        error_msg = server_conn._error_message or "Unknown error"
        formatted_error = self._server_initialization_error_text(error_msg)

        if server_conn._oauth_callback_timed_out or _is_oauth_timeout_message(formatted_error):
            raise ServerInitializationError(
                f"MCP Server: '{server_name}': OAuth authorization timed out.",
                "Authorization was not completed in time; retry /mcp connect.",
                server_name=server_name,
            ) from server_conn._lifecycle_error

        if _is_oauth_registration_404_message(formatted_error):
            raise ServerInitializationError(
                f"MCP Server: '{server_name}': OAuth client registration failed.",
                _format_oauth_registration_404_details(
                    formatted_error,
                    server_conn.server_config.url,
                ),
                server_name=server_name,
            ) from server_conn._lifecycle_error

        if _is_stdio_startup_error(server_conn, formatted_error):
            raise ServerInitializationError(
                f"MCP Server: '{server_name}': Failed to start stdio server.",
                _append_stdio_stderr_details(server_conn, formatted_error),
                server_name=server_name,
            ) from server_conn._lifecycle_error

        origin = self.server_registry.get_server_origin(server_name)
        remediation = {
            "central": " Check the server's fast-agent.yaml configuration.",
            "card": " Check the server's AgentCard configuration.",
        }.get(origin, "")
        raise ServerInitializationError(
            f"MCP Server: '{server_name}': Failed to initialize - see details.{remediation}",
            _append_stdio_stderr_details(server_conn, formatted_error),
            server_name=server_name,
        ) from server_conn._lifecycle_error

    @staticmethod
    def _server_initialization_error_text(error_msg: str | list[str]) -> str:
        if isinstance(error_msg, list):
            return "\n".join(str(line) for line in error_msg)
        return str(error_msg)

    async def get_server_capabilities(self, server_name: str) -> ServerCapabilities | None:
        """Get the capabilities of a specific server."""
        config = self.server_registry.get_server_config(server_name)
        if config is None:
            return None
        server_conn = await self.get_server(
            server_name,
            callback_runtime=MCPClientCallbackRuntime(
                server_name=server_name,
                server_config=config,
                context=self.context,
            ),
        )
        return server_conn.server_capabilities if server_conn else None

    async def disconnect_server(self, server_name: str) -> None:
        """
        Disconnect a specific server if it's running under this connection manager.
        """
        logger.info(f"{server_name}: Detaching MCP client runtime...")

        async with self._lock:
            server_conn = self.running_servers.get(server_name)
            if server_conn:
                server_conn.shutdown_lifecycle()
                await server_conn.wait_for_lifecycle_completion()
                self.running_servers.pop(server_name, None)
                self._server_oauth_mode.pop(server_name, None)
                self._server_oauth_active.pop(server_name, None)
                logger.info(f"{server_name}: Attached runtime shut down.")
            else:
                logger.info(f"{server_name}: No attached runtime found. Skipping shutdown")

    async def reconnect_server(
        self,
        server_name: str,
        *,
        server_config: MCPServerSettings | None = None,
        callback_runtime: MCPClientCallbackRuntime,
        startup_timeout_seconds: float | None = None,
        trigger_oauth: bool | None = None,
        oauth_event_handler: OAuthEventHandler | None = None,
        allow_oauth_paste_fallback: bool = True,
    ) -> "ServerConnection":
        """
        Replace a server runtime after a transport or legacy-session failure.

        Modern MCP has no durable protocol session; reconnecting replaces local
        client and transport resources.

        Args:
            server_name: Name of the server to reconnect

        Returns:
            The new ServerConnection instance
        """
        logger.info(f"{server_name}: Initiating reconnection...")

        # First, disconnect the existing connection
        await self.disconnect_server(server_name)

        server_conn = await self._launch_and_wait_for_server(
            server_name=server_name,
            server_config=server_config,
            callback_runtime=callback_runtime,
            startup_timeout_seconds=startup_timeout_seconds,
            trigger_oauth=trigger_oauth,
            oauth_event_handler=oauth_event_handler,
            allow_oauth_paste_fallback=allow_oauth_paste_fallback,
            timeout_action="Reconnect",
        )

        # Check if the reconnection was successful
        if not server_conn.is_healthy():
            if self.should_retry_server_with_oauth(
                server_name,
                server_conn._lifecycle_error or server_conn._error_message,
            ):
                server_conn = await self._retry_server_with_oauth(
                    server_name=server_name,
                    server_conn=server_conn,
                    server_config=server_config,
                    callback_runtime=callback_runtime,
                    startup_timeout_seconds=startup_timeout_seconds,
                    oauth_event_handler=oauth_event_handler,
                    allow_oauth_paste_fallback=allow_oauth_paste_fallback,
                    timeout_action="Reconnect",
                )
                if server_conn.is_healthy():
                    logger.info(f"{server_name}: Reconnection successful")
                    return server_conn

            await self._clear_running_server_state(server_name, server_conn)
            error_msg = server_conn._error_message or "Unknown error during reconnection"

            if isinstance(error_msg, list):
                oauth_error_text = "\n".join(str(line) for line in error_msg)
            else:
                oauth_error_text = str(error_msg)

            if server_conn._oauth_callback_timed_out or _is_oauth_timeout_message(oauth_error_text):
                raise ServerInitializationError(
                    f"MCP Server: '{server_name}': OAuth authorization timed out during reconnect.",
                    "Authorization was not completed in time; retry /mcp connect.",
                    server_name=server_name,
                ) from server_conn._lifecycle_error
            if isinstance(error_msg, list):
                formatted_error = "\n".join(error_msg)
            else:
                formatted_error = str(error_msg)

            if _is_oauth_registration_404_message(formatted_error):
                raise ServerInitializationError(
                    f"MCP Server: '{server_name}': OAuth client registration failed during reconnect.",
                    _format_oauth_registration_404_details(
                        formatted_error, server_conn.server_config.url
                    ),
                    server_name=server_name,
                ) from server_conn._lifecycle_error

            if _is_stdio_startup_error(server_conn, formatted_error):
                raise ServerInitializationError(
                    f"MCP Server: '{server_name}': Failed to start stdio server during reconnect.",
                    _append_stdio_stderr_details(server_conn, formatted_error),
                    server_name=server_name,
                ) from server_conn._lifecycle_error

            raise ServerInitializationError(
                f"MCP Server: '{server_name}': Failed to reconnect - see details.",
                _append_stdio_stderr_details(server_conn, formatted_error),
                server_name=server_name,
            ) from server_conn._lifecycle_error

        logger.info(f"{server_name}: Reconnection successful")
        return server_conn

    async def disconnect_all(self) -> bool:
        """Disconnect all servers that are running under this connection manager."""
        async with self._lock:
            if not self.running_servers:
                return False

            servers_to_shutdown = list(self.running_servers.items())
            for name, conn in servers_to_shutdown:
                logger.info(f"{name}: Requesting shutdown...")
                conn.shutdown_lifecycle()
            await asyncio.gather(
                *(conn.wait_for_lifecycle_completion() for _, conn in servers_to_shutdown)
            )
            self.running_servers.clear()
            self._server_oauth_mode.clear()
            self._server_oauth_active.clear()
        return True
