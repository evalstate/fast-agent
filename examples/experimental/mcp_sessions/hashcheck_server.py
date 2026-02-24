"""Hashcheck MCP server — cookie-carried hash KV store.

A server that hashes strings and stores key/value hashes in the session cookie
payload itself (``_meta["mcp/session"].data``). A reconnecting client that
presents the same cookie can verify strings against previously stored hashes
without requiring server-side in-memory session state.

This demo emphasizes a permissive, client-first model: state is transferred
from client to server on each call via cookie metadata.

Tools:
  - ``hashcheck_store(key, text)``  — hash ``text`` with SHA-256, store under ``key``.
  - ``hashcheck_verify(key, text)`` — hash ``text``, compare against stored hash for ``key``.
  - ``hashcheck_list()``            — list all stored keys and their hashes.
  - ``hashcheck_delete(key)``       — remove a stored hash entry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import secrets
from typing import Any

import mcp.types as types
from _session_base import (
    SessionStore,
    add_transport_args,
    cookie_meta,
    register_session_handlers,
    run_server,
    session_cookie_from_meta,
)
from mcp.server.fastmcp import Context, FastMCP

HASH_ALGORITHM = "sha256"
HASHES_COOKIE_KEY = "hashes"


class HashKVStore:
    """Cookie-backed key -> hash storage."""

    def _hash(self, text: str) -> str:
        return hashlib.new(HASH_ALGORITHM, text.encode("utf-8")).hexdigest()

    def store(self, kv: dict[str, str], key: str, text: str) -> str:
        """Hash text and store under key. Returns the hex digest."""
        digest = self._hash(text)
        kv[key] = digest
        return digest

    def verify(self, kv: dict[str, str], key: str, text: str) -> dict[str, Any]:
        """Verify text against stored hash.

        Returns dict with ``match`` bool, ``stored_hash``, ``provided_hash``,
        and ``found`` (whether the key exists).
        """
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

    @staticmethod
    def list_keys(kv: dict[str, str]) -> dict[str, str]:
        """Return all key -> hash entries for the cookie state."""
        return dict(kv)

    @staticmethod
    def delete(kv: dict[str, str], key: str) -> bool:
        return kv.pop(key, None) is not None

    @staticmethod
    def count(kv: dict[str, str]) -> int:
        return len(kv)


def _new_session_id() -> str:
    return f"sess-{secrets.token_hex(6)}"


def _parse_hashes(data: dict[str, Any]) -> dict[str, str]:
    raw = data.get(HASHES_COOKIE_KEY)
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if not isinstance(parsed, dict):
        return {}

    hashes: dict[str, str] = {}
    for key, value in parsed.items():
        if isinstance(key, str) and isinstance(value, str):
            hashes[key] = value
    return hashes


def _cookie_state_from_request(ctx: Context) -> tuple[dict[str, Any], str, dict[str, str]]:
    """Get or create a cookie state snapshot from request metadata."""
    source_cookie = session_cookie_from_meta(ctx.request_context.meta)
    cookie = dict(source_cookie) if isinstance(source_cookie, dict) else {}

    raw_id = cookie.get("id")
    session_id = raw_id if isinstance(raw_id, str) and raw_id else _new_session_id()

    raw_data = cookie.get("data")
    data = dict(raw_data) if isinstance(raw_data, dict) else {}
    hashes = _parse_hashes(data)

    return cookie, session_id, hashes


def _cookie_with_hashes(
    cookie: dict[str, Any],
    *,
    session_id: str,
    hashes: dict[str, str],
) -> dict[str, Any]:
    data = cookie.get("data")
    normalized_data = dict(data) if isinstance(data, dict) else {}
    normalized_data[HASHES_COOKIE_KEY] = json.dumps(hashes, ensure_ascii=False, sort_keys=True)
    return {
        "id": session_id,
        "data": normalized_data,
    }


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
        cookie, session_id, kv = _cookie_state_from_request(ctx)
        digest = hashes.store(kv, key, text)
        updated_cookie = _cookie_with_hashes(cookie, session_id=session_id, hashes=kv)
        return types.CallToolResult(
            content=[
                types.TextContent(
                    type="text",
                    text=f"Stored {HASH_ALGORITHM}('{key}') = {digest[:16]}...",
                )
            ],
            _meta=cookie_meta(updated_cookie),
        )

    @mcp.tool(name="hashcheck_verify")
    async def hashcheck_verify(
        ctx: Context, key: str, text: str
    ) -> types.CallToolResult:
        """Verify a text string against a previously stored hash."""
        cookie, session_id, kv = _cookie_state_from_request(ctx)
        result = hashes.verify(kv, key, text)

        if not result["found"]:
            msg = f"Key '{key}' not found in session store."
        elif result["match"]:
            msg = f"MATCH - key '{key}' verified successfully."
        else:
            msg = f"MISMATCH - key '{key}' hash does not match."

        return types.CallToolResult(
            content=[types.TextContent(type="text", text=msg)],
            _meta=cookie_meta(_cookie_with_hashes(cookie, session_id=session_id, hashes=kv)),
        )

    @mcp.tool(name="hashcheck_list")
    async def hashcheck_list(ctx: Context) -> types.CallToolResult:
        """List all stored keys and their hashes."""
        cookie, session_id, kv = _cookie_state_from_request(ctx)
        entries = hashes.list_keys(kv)
        if not entries:
            text = "(no hashes stored)"
        else:
            lines = [f"  {k}: {v[:16]}..." for k, v in entries.items()]
            text = f"Hash store ({len(entries)} entries):\n" + "\n".join(lines)
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=text)],
            _meta=cookie_meta(_cookie_with_hashes(cookie, session_id=session_id, hashes=kv)),
        )

    @mcp.tool(name="hashcheck_delete")
    async def hashcheck_delete(
        ctx: Context, key: str
    ) -> types.CallToolResult:
        """Delete a stored hash entry."""
        cookie, session_id, kv = _cookie_state_from_request(ctx)
        deleted = hashes.delete(kv, key)
        msg = (
            f"Deleted key '{key}'."
            if deleted
            else f"Key '{key}' not found."
        )
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=msg)],
            _meta=cookie_meta(_cookie_with_hashes(cookie, session_id=session_id, hashes=kv)),
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
