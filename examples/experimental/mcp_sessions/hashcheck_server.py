"""Hashcheck MCP server — per-session hash set (draft sessions).

This variant drops arbitrary keys and keeps a per-session set of SHA-256 hashes.
"""

from __future__ import annotations

import argparse
import hashlib

import mcp.types as types
from _session_base import (
    SessionStore,
    add_transport_args,
    register_session_handlers,
    require_session,
    run_server,
    session_meta,
)
from mcp.server.fastmcp import Context, FastMCP

HASH_ALGORITHM = "sha256"


class HashStore:
    """Per-session set of digests."""

    def __init__(self) -> None:
        self._store: dict[str, set[str]] = {}

    @staticmethod
    def digest(text: str) -> str:
        return hashlib.new(HASH_ALGORITHM, text.encode("utf-8")).hexdigest()

    def add(self, session_id: str, digest: str) -> int:
        bucket = self._store.setdefault(session_id, set())
        bucket.add(digest)
        return len(bucket)

    def contains(self, session_id: str, digest: str) -> bool:
        bucket = self._store.get(session_id)
        if bucket is None:
            return False
        return digest in bucket

    def remove(self, session_id: str, digest: str) -> bool:
        bucket = self._store.get(session_id)
        if bucket is None:
            return False
        if digest not in bucket:
            return False
        bucket.remove(digest)
        return True

    def list(self, session_id: str) -> list[str]:
        bucket = self._store.get(session_id)
        if bucket is None:
            return []
        return sorted(bucket)



def build_server() -> FastMCP:
    mcp = FastMCP("hashcheck", log_level="WARNING")
    sessions = SessionStore(include_state=False)
    hashes = HashStore()
    register_session_handlers(mcp._mcp_server, sessions)

    @mcp.tool(name="hashcheck_store")
    async def hashcheck_store(ctx: Context, text: str) -> types.CallToolResult:
        record = require_session(ctx, sessions)
        sessions.touch(record)
        digest = hashes.digest(text)
        count = hashes.add(record.session_id, digest)
        return types.CallToolResult(
            content=[
                types.TextContent(
                    type="text",
                    text=(
                        f"Stored {HASH_ALGORITHM} digest {digest[:16]}... "
                        f"({count} total)"
                    ),
                )
            ],
            _meta=session_meta(record, sessions),
        )

    @mcp.tool(name="hashcheck_verify")
    async def hashcheck_verify(ctx: Context, text: str) -> types.CallToolResult:
        record = require_session(ctx, sessions)
        sessions.touch(record)
        digest = hashes.digest(text)
        matched = hashes.contains(record.session_id, digest)
        message = "MATCH" if matched else "NOT FOUND"
        return types.CallToolResult(
            content=[
                types.TextContent(
                    type="text",
                    text=f"{message} for {digest[:16]}...",
                )
            ],
            _meta=session_meta(record, sessions),
        )

    @mcp.tool(name="hashcheck_list")
    async def hashcheck_list(ctx: Context) -> types.CallToolResult:
        record = require_session(ctx, sessions)
        sessions.touch(record)
        entries = hashes.list(record.session_id)
        if not entries:
            text = "(no hashes stored)"
        else:
            lines = [f"  {idx}. {digest[:16]}..." for idx, digest in enumerate(entries, 1)]
            text = f"Hash store ({len(entries)} entries):\n" + "\n".join(lines)
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=text)],
            _meta=session_meta(record, sessions),
        )

    @mcp.tool(name="hashcheck_delete")
    async def hashcheck_delete(ctx: Context, text: str) -> types.CallToolResult:
        record = require_session(ctx, sessions)
        sessions.touch(record)
        digest = hashes.digest(text)
        deleted = hashes.remove(record.session_id, digest)
        message = "Deleted" if deleted else "Not found"
        return types.CallToolResult(
            content=[
                types.TextContent(
                    type="text",
                    text=f"{message}: {digest[:16]}...",
                )
            ],
            _meta=session_meta(record, sessions),
        )

    return mcp



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hashcheck MCP demo server (per-session hash store)"
    )
    add_transport_args(parser)
    args = parser.parse_args()
    run_server(build_server(), args)


if __name__ == "__main__":
    main()
