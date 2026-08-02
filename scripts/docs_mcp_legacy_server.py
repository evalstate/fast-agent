#!/usr/bin/env python3
"""Deterministic legacy Streamable HTTP MCP server for documentation recordings."""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING, Any

import uvicorn
from starlette.applications import Starlette
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

if TYPE_CHECKING:
    from starlette.requests import Request

SESSION_ID = "docs-legacy-session"
PROTOCOL_VERSION = "2025-11-25"


def _result(request_id: object, result: dict[str, Any]) -> JSONResponse:
    return JSONResponse(
        {"jsonrpc": "2.0", "id": request_id, "result": result},
        headers={"MCP-Session-Id": SESSION_ID},
    )


async def health(_request: Request) -> Response:
    return Response("ready", media_type="text/plain")


async def mcp(request: Request) -> Response:
    if request.method == "GET":
        return Response(status_code=405, headers={"Allow": "POST, DELETE"})
    if request.method == "DELETE":
        return Response(status_code=200)

    payload = await request.json()
    if not isinstance(payload, dict):
        return JSONResponse({"error": "expected JSON-RPC object"}, status_code=400)

    method = payload.get("method")
    request_id = payload.get("id")
    if method == "initialize":
        return _result(
            request_id,
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {
                    "name": "Docs Legacy Remote",
                    "version": "1.0.0",
                },
                "instructions": "Deterministic legacy MCP documentation fixture.",
            },
        )
    if isinstance(method, str) and method.startswith("notifications/"):
        return Response(status_code=202, headers={"MCP-Session-Id": SESSION_ID})
    if method == "ping":
        return _result(request_id, {})
    if method == "tools/list":
        return _result(request_id, {"tools": []})

    return JSONResponse(
        {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"},
        },
        headers={"MCP-Session-Id": SESSION_ID},
    )


app = Starlette(
    routes=[
        Route("/healthz", health, methods=["GET"]),
        Route("/mcp", mcp, methods=["GET", "POST", "DELETE"]),
    ]
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
