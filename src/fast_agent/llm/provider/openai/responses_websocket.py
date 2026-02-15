from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol, cast
from urllib.parse import urlparse, urlunparse

import aiohttp
from aiohttp import WSMsgType

RESPONSES_WEBSOCKET_BETA_HEADER = "responses_websockets=2026-02-06"
RESPONSES_WEBSOCKET_BETA_HEADER_NAME = "OpenAI-Beta"
RESPONSES_CREATE_EVENT_TYPE = "response.create"
TERMINAL_RESPONSE_EVENT_TYPES = {
    "response.completed",
    "response.done",
    "response.incomplete",
}

_STREAM_START_EVENT_TYPES = {
    "response.output_item.added",
    "response.function_call_arguments.delta",
    "response.reasoning_summary_text.delta",
    "response.reasoning_summary.delta",
    "response.reasoning.delta",
    "response.reasoning_text.delta",
    "response.output_text.delta",
    "response.text.delta",
}


class ResponsesWebSocketError(RuntimeError):
    """Raised for WebSocket transport failures.

    Attributes:
        stream_started: Whether any meaningful streaming output/tool event was observed.
    """

    def __init__(self, message: str, *, stream_started: bool = False) -> None:
        super().__init__(message)
        self.stream_started = stream_started


class _AttrObjectView:
    """Tiny adapter that exposes dictionary keys as attributes recursively."""

    __slots__ = ("_data",)

    def __init__(self, data: Mapping[str, Any]) -> None:
        self._data = {key: _to_attr_object(value) for key, value in data.items()}

    def __getattr__(self, key: str) -> Any:
        if key in self._data:
            return self._data[key]
        raise AttributeError(key)

    def model_dump(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return _to_plain_data(self._data)

    def __repr__(self) -> str:
        return f"_AttrObjectView({self._data!r})"


def _to_plain_data(value: Any) -> Any:
    if isinstance(value, _AttrObjectView):
        return {key: _to_plain_data(item) for key, item in value._data.items()}
    if isinstance(value, list):
        return [_to_plain_data(item) for item in value]
    return value


def _to_attr_object(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _AttrObjectView(value)
    if isinstance(value, list):
        return [_to_attr_object(item) for item in value]
    return value


def _stream_event_started(event_type: str | None) -> bool:
    if not event_type:
        return False
    if event_type in _STREAM_START_EVENT_TYPES:
        return True
    if event_type.startswith("response.output_text"):
        return True
    if event_type.startswith("response.text"):
        return True
    return False


def resolve_responses_ws_url(base_url: str) -> str:
    """Build a WebSocket URL matching the Responses endpoint path."""

    parsed = urlparse(base_url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"Invalid base URL for websocket transport: '{base_url}'")

    path = (parsed.path or "").rstrip("/")
    if not path.endswith("/responses"):
        path = f"{path}/responses" if path else "/responses"

    if parsed.scheme == "https":
        scheme = "wss"
    elif parsed.scheme == "http":
        scheme = "ws"
    else:
        scheme = parsed.scheme

    return urlunparse(
        (
            scheme,
            parsed.netloc,
            path,
            "",
            parsed.query,
            parsed.fragment,
        )
    )


def build_ws_headers(
    *,
    api_key: str,
    default_headers: Mapping[str, str] | None = None,
    extra_headers: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build headers for Responses websocket requests."""

    headers = dict(default_headers or {})
    headers.setdefault("Authorization", f"Bearer {api_key}")
    headers[RESPONSES_WEBSOCKET_BETA_HEADER_NAME] = RESPONSES_WEBSOCKET_BETA_HEADER
    if extra_headers:
        headers.update(extra_headers)
    return headers


class WebSocketLike(Protocol):
    closed: bool

    async def receive(self, timeout: float | None = None) -> Any: ...

    async def send_str(self, payload: str, compress: int | None = None) -> Any: ...

    async def close(self) -> Any: ...

    def exception(self) -> BaseException | None: ...


class ClientSessionLike(Protocol):
    closed: bool

    async def close(self) -> Any: ...


@dataclass
class ManagedWebSocketConnection:
    """A websocket + owning HTTP session."""

    session: ClientSessionLike
    websocket: WebSocketLike
    busy: bool = False
    last_used_monotonic: float = 0.0


async def connect_websocket(
    *,
    url: str,
    headers: Mapping[str, str],
    timeout_seconds: float | None = None,
) -> ManagedWebSocketConnection:
    timeout = aiohttp.ClientTimeout(total=timeout_seconds) if timeout_seconds else None
    session = aiohttp.ClientSession(timeout=timeout)
    try:
        websocket = await session.ws_connect(url, headers=dict(headers), autoping=True)
    except Exception:
        await session.close()
        raise
    return ManagedWebSocketConnection(
        session=cast("ClientSessionLike", session),
        websocket=cast("WebSocketLike", websocket),
    )


async def close_websocket_connection(connection: ManagedWebSocketConnection) -> None:
    if not connection.websocket.closed:
        try:
            await connection.websocket.close()
        except Exception:
            pass
    if not connection.session.closed:
        await connection.session.close()


async def send_response_create(
    websocket: WebSocketLike,
    arguments: Mapping[str, Any],
) -> None:
    payload = dict(arguments)
    payload.setdefault("stream", True)
    payload["type"] = RESPONSES_CREATE_EVENT_TYPE
    await websocket.send_str(json.dumps(payload))


class WebSocketResponsesStream:
    """Adapter exposing websocket payloads through the Responses stream interface."""

    def __init__(self, websocket: WebSocketLike) -> None:
        self._websocket = websocket
        self._stream_started = False
        self._saw_terminal_event = False
        self._stop_after_next = False
        self._final_response: Any | None = None
        self._events_seen = 0
        self._last_frame_preview: str | None = None

    @property
    def stream_started(self) -> bool:
        return self._stream_started

    def __aiter__(self) -> WebSocketResponsesStream:
        return self

    async def __anext__(self) -> Any:
        if self._stop_after_next:
            raise StopAsyncIteration

        while True:
            message = await self._websocket.receive()

            if message.type in {WSMsgType.CLOSE, WSMsgType.CLOSING, WSMsgType.CLOSED}:
                if self._saw_terminal_event:
                    raise StopAsyncIteration
                close_code = getattr(message, "data", None) or getattr(
                    self._websocket, "close_code", None
                )
                close_reason = getattr(message, "extra", None)
                diagnostics = []
                if close_code is not None:
                    diagnostics.append(f"close_code={close_code}")
                if close_reason:
                    diagnostics.append(f"reason={close_reason}")
                diagnostics.append(f"events_seen={self._events_seen}")
                if self._last_frame_preview:
                    diagnostics.append(f"last_frame={self._last_frame_preview}")
                if close_code == 1008:
                    diagnostics.append(
                        "hint=policy_violation (account/feature may not permit Responses websocket beta)"
                    )
                detail = "; ".join(diagnostics)
                raise ResponsesWebSocketError(
                    "WebSocket stream closed before completion event"
                    + (f" ({detail})" if detail else ""),
                    stream_started=self._stream_started,
                )

            if message.type == WSMsgType.ERROR:
                ws_error = self._websocket.exception()
                detail = str(ws_error) if ws_error else "unknown websocket error"
                raise ResponsesWebSocketError(
                    f"WebSocket transport error: {detail}",
                    stream_started=self._stream_started,
                )

            if message.type not in {WSMsgType.TEXT, WSMsgType.BINARY}:
                continue

            raw_data: str
            if message.type == WSMsgType.BINARY:
                raw_data = message.data.decode("utf-8", errors="replace")
            else:
                raw_data = str(message.data)

            try:
                payload = json.loads(raw_data)
            except json.JSONDecodeError as exc:
                self._last_frame_preview = _preview_text(raw_data)
                raise ResponsesWebSocketError(
                    "Received non-JSON websocket message"
                    + (f" ({self._last_frame_preview})" if self._last_frame_preview else ""),
                    stream_started=self._stream_started,
                ) from exc

            if not isinstance(payload, dict):
                self._last_frame_preview = _preview_text(raw_data)
                raise ResponsesWebSocketError(
                    "Received unexpected websocket payload"
                    + (f" ({self._last_frame_preview})" if self._last_frame_preview else ""),
                    stream_started=self._stream_started,
                )

            self._last_frame_preview = _preview_text(raw_data)

            event = _to_attr_object(payload)
            event_type = payload.get("type")
            if isinstance(event_type, str) and _stream_event_started(event_type):
                self._stream_started = True

            if "response" in payload:
                self._final_response = _to_attr_object(payload["response"])

            if event_type in {"error", "response.failed"}:
                error_message = self._extract_error_message(payload)
                raise ResponsesWebSocketError(
                    error_message,
                    stream_started=self._stream_started,
                )

            if event_type in TERMINAL_RESPONSE_EVENT_TYPES:
                self._saw_terminal_event = True
                self._stop_after_next = True

            self._events_seen += 1
            return event

    async def get_final_response(self) -> Any:
        if self._final_response is None:
            raise ResponsesWebSocketError(
                "WebSocket stream did not provide a final response payload.",
                stream_started=self._stream_started,
            )
        return self._final_response

    @staticmethod
    def _extract_error_message(payload: Mapping[str, Any]) -> str:
        message = payload.get("message")
        if isinstance(message, str) and message.strip():
            return message

        error = payload.get("error")
        if isinstance(error, str) and error.strip():
            return error
        if isinstance(error, Mapping):
            error_message = error.get("message")
            if isinstance(error_message, str) and error_message.strip():
                return error_message

        return "WebSocket Responses request failed."


def _preview_text(raw_data: str, *, limit: int = 240) -> str:
    compact = " ".join(raw_data.split())
    if len(compact) <= limit:
        return compact
    return f"{compact[:limit]}..."


class WebSocketConnectionManager:
    """Maintain one reusable websocket, with temporary sockets for concurrent calls."""

    def __init__(
        self,
        *,
        idle_timeout_seconds: float = 300.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._mutex = asyncio.Lock()
        self._reusable_connection: ManagedWebSocketConnection | None = None
        self._idle_timeout_seconds = idle_timeout_seconds
        self._clock = clock

    async def acquire(
        self,
        create_connection: Callable[[], Awaitable[ManagedWebSocketConnection]],
    ) -> tuple[ManagedWebSocketConnection, bool]:
        async with self._mutex:
            await self._expire_idle_locked()
            reusable = self._reusable_connection
            if reusable and self._is_open(reusable) and not reusable.busy:
                reusable.busy = True
                return reusable, True

            if reusable and reusable.busy and self._is_open(reusable):
                temp = await create_connection()
                temp.busy = True
                return temp, False

            if reusable:
                await close_websocket_connection(reusable)
                self._reusable_connection = None

            fresh = await create_connection()
            fresh.busy = True
            self._reusable_connection = fresh
            return fresh, True

    async def release(
        self,
        connection: ManagedWebSocketConnection,
        *,
        reusable: bool,
        keep: bool,
    ) -> None:
        async with self._mutex:
            if reusable and self._reusable_connection is connection:
                if keep and self._is_open(connection):
                    connection.busy = False
                    connection.last_used_monotonic = self._clock()
                    return
                await close_websocket_connection(connection)
                self._reusable_connection = None
                return

            await close_websocket_connection(connection)

    async def close(self) -> None:
        async with self._mutex:
            reusable = self._reusable_connection
            self._reusable_connection = None
            if reusable:
                await close_websocket_connection(reusable)

    async def _expire_idle_locked(self) -> None:
        reusable = self._reusable_connection
        if not reusable:
            return
        if reusable.busy:
            return
        if self._idle_timeout_seconds <= 0:
            return
        elapsed = self._clock() - reusable.last_used_monotonic
        if elapsed < self._idle_timeout_seconds:
            return
        await close_websocket_connection(reusable)
        self._reusable_connection = None

    @staticmethod
    def _is_open(connection: ManagedWebSocketConnection) -> bool:
        return not connection.session.closed and not connection.websocket.closed
