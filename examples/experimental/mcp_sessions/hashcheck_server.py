"""Hashcheck MCP server — per-session hash KV store.

A server that hashes strings and stores them in a per-session key-value store.
A reconnecting client presenting the same session cookie can verify strings
against previously stored hashes — demonstrating session persistence across
connection boundaries.

This is the strongest demo of why application-level sessions matter beyond
transport-level ``Mcp-Session-Id``: the client can disconnect, reconnect
with the same session cookie, and the server-side state is still available.

Tools:
  - ``hashcheck_store(key, text)``  — hash ``text`` with SHA-256, store under ``key``.
  - ``hashcheck_verify(key, text)`` — hash ``text``, compare against stored hash for ``key``.
  - ``hashcheck_list()``            — list all stored keys and their hashes.
  - ``hashcheck_delete(key)``       — remove a stored hash entry.
"""

from __future__ import annotations

import argparse
import hashlib
from typing import Any

import mcp.types as types
from _session_base import (
    SESSION_REQUIRED_ERROR_CODE,
    SessionStore,
    add_transport_args,
    cookie_meta,
    register_session_handlers,
    run_server,
    session_cookie_from_meta,
    session_id_from_cookie,
)
from mcp.server.fastmcp import Context, FastMCP

HASH_ALGORITHM = "sha256"


class HashKVStore:
    """Per-session key -> hash storage."""

    def __init__(self) -> None:
        self._stores: dict[str, dict[str, str]] = {}

    def _hash(self, text: str) -> str:
        return hashlib.new(HASH_ALGORITHM, text.encode("utf-8")).hexdigest()

    def store(self, session_id: str, key: str, text: str) -> str:
        """Hash text and store under key. Returns the hex digest."""
        if session_id not in self._stores:
            self._stores[session_id] = {}
        digest = self._hash(text)
        self._stores[session_id][key] = digest
        return digest

    def verify(self, session_id: str, key: str, text: str) -> dict[str, Any]:
        """Verify text against stored hash.

        Returns dict with ``match`` bool, ``stored_hash``, ``provided_hash``,
        and ``found`` (whether the key exists).
        """
        kv = self._stores.get(session_id, {})
        stored = kv.get(key)
        provided = self._hash(text)
        if stored is None:
            return {
                "found": False,
                "match": False,
                "key": key,
                "stored_hash": None,
                "provided_hash": provided,
            }
        return {
            "found": True,
            "match": stored == provided,
            "key": key,
            "stored_hash": stored,
            "provided_hash": provided,
        }

    def list_keys(self, session_id: str) -> dict[str, str]:
        """Return all key -> hash entries for a session."""
        return dict(self._stores.get(session_id, {}))

    def delete(self, session_id: str, key: str) -> bool:
        kv = self._stores.get(session_id, {})
        return kv.pop(key, None) is not None

    def count(self, session_id: str) -> int:
        return len(self._stores.get(session_id, {}))


def _require_session(
    ctx: Context, store: SessionStore
) -> tuple[dict[str, Any], str]:
    """Extract and validate the session cookie, or raise McpError."""
    cookie = session_cookie_from_meta(ctx.request_context.meta)
    session_id = session_id_from_cookie(cookie)
    if not session_id or store.get(session_id) is None:
        raise types.McpError(
            code=SESSION_REQUIRED_ERROR_CODE,
            message="Session required. Send session/create before using hashcheck.",
        )
    return cookie, session_id  # type: ignore[return-value]


def build_server() -> FastMCP:
    mcp = FastMCP("hashcheck", log_level="WARNING")
    sessions = SessionStore()
    hashes = HashKVStore()
    register_session_handlers(mcp._mcp_server, sessions)

    @mcp.tool(name="hashcheck_store")
    async def hashcheck_store(
        ctx: Context, key: str, text: str
    ) -> types.CallToolResult:
        """Hash a text string and store the digest under the given key."""
        cookie, session_id = _require_session(ctx, sessions)
        record = sessions.get(session_id)
        if record:
            record.tool_calls += 1
        digest = hashes.store(session_id, key, text)
        return types.CallToolResult(
            content=[
                types.TextContent(
                    type="text",
                    text=f"Stored {HASH_ALGORITHM}('{key}') = {digest[:16]}...",
                )
            ],
            structuredContent={
                "action": "store",
                "key": key,
                "algorithm": HASH_ALGORITHM,
                "hash": digest,
                "session_id": session_id,
            },
            _meta=cookie_meta(cookie),
        )

    @mcp.tool(name="hashcheck_verify")
    async def hashcheck_verify(
        ctx: Context, key: str, text: str
    ) -> types.CallToolResult:
        """Verify a text string against a previously stored hash."""
        cookie, session_id = _require_session(ctx, sessions)
        record = sessions.get(session_id)
        if record:
            record.tool_calls += 1
        result = hashes.verify(session_id, key, text)

        if not result["found"]:
            msg = f"Key '{key}' not found in session store."
        elif result["match"]:
            msg = f"MATCH - key '{key}' verified successfully."
        else:
            msg = f"MISMATCH - key '{key}' hash does not match."

        return types.CallToolResult(
            content=[types.TextContent(type="text", text=msg)],
            structuredContent={**result, "algorithm": HASH_ALGORITHM},
            _meta=cookie_meta(cookie),
        )

    @mcp.tool(name="hashcheck_list")
    async def hashcheck_list(ctx: Context) -> types.CallToolResult:
        """List all stored keys and their hashes."""
        cookie, session_id = _require_session(ctx, sessions)
        record = sessions.get(session_id)
        if record:
            record.tool_calls += 1
        entries = hashes.list_keys(session_id)
        if not entries:
            text = "(no hashes stored)"
        else:
            lines = [f"  {k}: {v[:16]}..." for k, v in entries.items()]
            text = f"Hash store ({len(entries)} entries):\n" + "\n".join(lines)
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=text)],
            structuredContent={
                "action": "list",
                "session_id": session_id,
                "entries": entries,
                "count": len(entries),
            },
            _meta=cookie_meta(cookie),
        )

    @mcp.tool(name="hashcheck_delete")
    async def hashcheck_delete(
        ctx: Context, key: str
    ) -> types.CallToolResult:
        """Delete a stored hash entry."""
        cookie, session_id = _require_session(ctx, sessions)
        record = sessions.get(session_id)
        if record:
            record.tool_calls += 1
        deleted = hashes.delete(session_id, key)
        msg = (
            f"Deleted key '{key}'."
            if deleted
            else f"Key '{key}' not found."
        )
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=msg)],
            structuredContent={
                "action": "delete",
                "key": key,
                "deleted": deleted,
                "session_id": session_id,
            },
            _meta=cookie_meta(cookie),
        )

    return mcp


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hashcheck MCP demo server (per-session hash KV store)"
    )
    add_transport_args(parser)
    args = parser.parse_args()
    run_server(build_server(), args)


if __name__ == "__main__":
    main()
