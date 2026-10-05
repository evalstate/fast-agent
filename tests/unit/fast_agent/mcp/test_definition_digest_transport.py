"""Digests survive the SDK: read from raw results, echoed in request _meta, mismatches raised."""

from contextlib import asynccontextmanager
from typing import Any

import anyio
import pytest
from mcp.shared.exceptions import MCPError
from mcp.shared.message import SessionMessage
from mcp_types import JSONRPCError, JSONRPCRequest, JSONRPCResponse, RequestParamsMeta

from fast_agent.mcp.client_callback_runtime import MCPClientCallbackRuntime
from fast_agent.mcp.client_connection import MCPClientConnection
from fast_agent.mcp.definition_digests import (
    DIGEST_MISMATCH,
    KNOWN_DIGESTS_META,
    SERVER_DISCOVER,
    TOOLS_LIST,
    rejected_digests,
)

TOOLS = [{"name": "search", "inputSchema": {"type": "object"}}]
CACHE_HINTS = {"resultType": "complete", "ttlMs": 60_000, "cacheScope": "private"}


class DigestServer:
    """Speaks the SEP like the Hugging Face reference server: digests on discovery and
    tools/list, known digests checked on tools/call before anything runs."""

    def __init__(self) -> None:
        self.digests = {SERVER_DISCOVER: "sha256:d1", TOOLS_LIST: "sha256:t1"}
        self.calls: list[dict[str, Any]] = []

    def result(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if method == SERVER_DISCOVER:
            return {
                "supportedVersions": ["2026-07-28"],
                "capabilities": {"tools": {}},
                "instructions": "Use search.",
                "digest": self.digests[SERVER_DISCOVER],
                **CACHE_HINTS,
            }
        if method == TOOLS_LIST:
            return {"tools": TOOLS, "digest": self.digests[TOOLS_LIST], **CACHE_HINTS}
        if method == "tools/call":
            known = params.get("_meta", {}).get(KNOWN_DIGESTS_META, {})
            if stale := [m for m, d in known.items() if self.digests.get(m) != d]:
                raise MCPError(DIGEST_MISMATCH, "Definitions changed", {"staleDigests": stale})
            self.calls.append(params)
            return {"content": [{"type": "text", "text": "ran"}], "resultType": "complete"}
        raise MCPError(-32601, f"Method not found: {method}")

    @asynccontextmanager
    async def transport(self):
        to_server, server_in = anyio.create_memory_object_stream[SessionMessage](16)
        server_out, to_client = anyio.create_memory_object_stream[SessionMessage | Exception](16)

        async def serve() -> None:
            async for item in server_in:
                request = item.message
                if not isinstance(request, JSONRPCRequest):
                    continue
                try:
                    reply = JSONRPCResponse(
                        jsonrpc="2.0",
                        id=request.id,
                        result=self.result(request.method, request.params or {}),
                    )
                except MCPError as exc:
                    reply = JSONRPCError(jsonrpc="2.0", id=request.id, error=exc.error)
                await server_out.send(SessionMessage(reply))

        async with anyio.create_task_group() as tg:
            tg.start_soon(serve)
            yield to_client, to_server
            tg.cancel_scope.cancel()


@pytest.mark.asyncio
async def test_digests_are_tracked_echoed_and_rejected_through_the_sdk():
    server = DigestServer()
    async with MCPClientConnection(
        server.transport(), MCPClientCallbackRuntime(server_name="hf", server_config=None)
    ) as client:
        assert client.digests.latest == {SERVER_DISCOVER: "sha256:d1"}
        await client.list_tools(cache_mode="refresh")
        assert client.digests.latest == {SERVER_DISCOVER: "sha256:d1", TOOLS_LIST: "sha256:t1"}

        known: RequestParamsMeta = {KNOWN_DIGESTS_META: dict(client.digests.latest)}
        assert not (await client.call_tool("search", {}, meta=known)).is_error
        assert server.calls[0]["_meta"][KNOWN_DIGESTS_META] == client.digests.latest

        server.digests[TOOLS_LIST] = "sha256:t2"
        with pytest.raises(MCPError) as rejected:
            await client.call_tool("search", {}, meta=known)
        assert rejected_digests(rejected.value) == {TOOLS_LIST}
        assert len(server.calls) == 1  # rejected before running

        # Only a fresh listing moves the tracked digest.
        assert client.digests.latest[TOOLS_LIST] == "sha256:t1"
        await client.list_tools(cache_mode="refresh")
        assert client.digests.latest[TOOLS_LIST] == "sha256:t2"
