"""Advisory MCP definition digests (Definition Digests SEP draft).

Servers put an opaque ``digest`` on cacheable results (``server/discover``, ``tools/list``);
clients echo the digests they hold in request ``_meta`` and the server may reject stale calls
(error ``data.staleDigests``) or serve them and flag the stale methods in result ``_meta``.
The pinned SDK drops unknown result fields, so digests are read from raw transport messages.
"""

from collections.abc import Mapping
from contextlib import asynccontextmanager
from types import TracebackType
from typing import Any, Self

from mcp.client import Transport
from mcp.shared._context_streams import ContextReceiveStream
from mcp.shared._stream_protocols import ReadStream, WriteStream
from mcp.shared.exceptions import MCPError
from mcp.shared.message import SessionMessage
from mcp_types import JSONRPCError, JSONRPCRequest, JSONRPCResponse, RequestId

KNOWN_DIGESTS_META = "io.modelcontextprotocol/knownDigests"
STALE_DIGESTS_META = "io.modelcontextprotocol/staleDigests"
# Application code used by the Hugging Face reference server; the SEP has not allocated one.
DIGEST_MISMATCH = -32987

SERVER_DISCOVER = "server/discover"
TOOLS_LIST = "tools/list"
DIGESTED_METHODS = frozenset({SERVER_DISCOVER, TOOLS_LIST})

Digests = dict[str, str]
"""Digests keyed by the method that produced them."""


def covers(digests: Mapping[str, str], *, instructions: bool) -> bool:
    """Whether server digests cover every definition we cache (instructions only if rendered)."""
    return TOOLS_LIST in digests and (not instructions or SERVER_DISCOVER in digests)


def known_digests(digests: Mapping[str, str], *, instructions: bool) -> Digests:
    """Digests to echo on calls; the discovery digest only when we render instructions."""
    return {
        method: digest
        for method, digest in digests.items()
        if method in DIGESTED_METHODS and (instructions or method != SERVER_DISCOVER)
    }


def _stale_methods(value: object) -> frozenset[str]:
    named = value if isinstance(value, list) else []
    return frozenset(method for method in named if method in DIGESTED_METHODS)


def rejected_digests(error: BaseException) -> frozenset[str] | None:
    """Methods a digest-mismatch rejection names, or None for any other error."""
    if not isinstance(error, MCPError) or error.code != DIGEST_MISMATCH:
        return None
    data = error.data
    stale = _stale_methods(data.get("staleDigests") if isinstance(data, Mapping) else None)
    # A mismatch naming nothing we hold still means our view is stale: refresh everything.
    return stale or DIGESTED_METHODS


def signalled_digests(meta: Mapping[str, Any] | None) -> frozenset[str]:
    """Methods a served result flags as stale (refresh when convenient; not an error)."""
    return _stale_methods((meta or {}).get(STALE_DIGESTS_META))


class DigestTracker:
    """Latest digest per method, read from raw responses to requests this client sent.

    A digest describes the whole collection, not a page, so the latest value per method is
    what a caller wants; a response without one clears it.
    """

    def __init__(self) -> None:
        self.latest: Digests = {}
        self._pending: dict[RequestId, str] = {}

    def sent(self, message: SessionMessage) -> None:
        request = message.message
        if isinstance(request, JSONRPCRequest) and request.method in DIGESTED_METHODS:
            self._pending[request.id] = request.method

    def received(self, item: SessionMessage | Exception) -> None:
        if not isinstance(item, SessionMessage):
            return
        response = item.message
        if not isinstance(response, (JSONRPCResponse, JSONRPCError)) or response.id is None:
            return
        method = self._pending.pop(response.id, None)
        if method is None or isinstance(response, JSONRPCError):
            return
        digest = response.result.get("digest")
        if isinstance(digest, str) and digest:
            self.latest[method] = digest
        else:
            self.latest.pop(method, None)


class _ObservedReadStream:
    def __init__(self, inner: ReadStream[SessionMessage | Exception], tracker: DigestTracker):
        self._inner = inner
        self._tracker = tracker

    @property
    def last_context(self):
        # The SDK dispatcher reads the sender's context from context-aware streams.
        return self._inner.last_context if isinstance(self._inner, ContextReceiveStream) else None

    async def receive(self) -> SessionMessage | Exception:
        item = await self._inner.receive()
        self._tracker.received(item)
        return item

    async def aclose(self) -> None:
        await self._inner.aclose()

    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> SessionMessage | Exception:
        item = await self._inner.__anext__()
        self._tracker.received(item)
        return item

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool | None:
        await self.aclose()
        return None


class _ObservedWriteStream:
    def __init__(self, inner: WriteStream[SessionMessage], tracker: DigestTracker):
        self._inner = inner
        self._tracker = tracker

    async def send(self, item: SessionMessage, /) -> None:
        self._tracker.sent(item)
        await self._inner.send(item)

    async def aclose(self) -> None:
        await self._inner.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool | None:
        await self.aclose()
        return None


def digest_tracking_transport(inner: Transport, tracker: DigestTracker) -> Transport:
    """Wrap a transport so the tracker sees every request and response."""

    @asynccontextmanager
    async def observed():
        async with inner as (read_stream, write_stream):
            yield (
                _ObservedReadStream(read_stream, tracker),
                _ObservedWriteStream(write_stream, tracker),
            )

    return observed()
