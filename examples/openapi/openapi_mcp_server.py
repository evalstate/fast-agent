"""OpenAPI MCP server that exposes the provided specification and an API invocation tool."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict

import httpx
import yaml
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field


class ApiCallRequest(BaseModel):
    """Structured request payload for the OpenAPI invocation tool."""

    method: str = Field(..., description="HTTP method to use, e.g. GET or POST.")
    path: str = Field(..., description="Endpoint path, such as /pets or pets.")
    query: Dict[str, Any] | None = Field(
        default=None, description="Optional query string parameters, keyed by name."
    )
    body: Any | None = Field(default=None, description="Optional JSON request body.")
    headers: Dict[str, str] | None = Field(
        default=None, description="Optional HTTP headers to include with the request."
    )
    timeout: float | None = Field(
        default=30.0, description="Optional request timeout (seconds); defaults to 30."
    )


def load_spec(spec_path: Path) -> tuple[str, str | None]:
    """Return the raw spec text and the first server URL, if any."""
    if not spec_path.exists():
        raise FileNotFoundError(f"OpenAPI specification not found: {spec_path}")

    spec_text = spec_path.read_text()
    try:
        data = yaml.safe_load(spec_text) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"Failed to parse OpenAPI specification: {exc}") from exc

    base_url: str | None = None
    servers = data.get("servers", [])
    for entry in servers:
        if isinstance(entry, dict) and "url" in entry:
            base_url = entry["url"]
            break

    return spec_text, base_url


def build_server(spec_text: str, base_url: str | None, server_name: str) -> FastMCP:
    instructions = (
        "Here is the API specification you have access to:\n\n"
        f"{spec_text}\n\n"
        "When issuing requests, do not return more than 50 results in a single query."
    )

    mcp = FastMCP(server_name, instructions=instructions, log_level="INFO")

    @mcp.tool(
        name="call_openapi_endpoint",
        description=(
            "Invoke an API endpoint from the supplied OpenAPI document. "
            "Provide the HTTP method (GET, POST, etc.), the endpoint path (e.g. /pets), "
            "and optional query parameters, JSON body, or headers."
        ),
    )
    async def call_openapi_endpoint(request: ApiCallRequest) -> Dict[str, Any]:
        if not base_url:
            raise RuntimeError("The OpenAPI specification does not define a server URL to call.")

        path = request.path if request.path.startswith("/") else f"/{request.path}"
        timeout = request.timeout if request.timeout is not None else 30.0

        async with httpx.AsyncClient(base_url=base_url, timeout=timeout) as client:
            response = await client.request(
                request.method.upper(),
                path,
                params=request.query,
                json=request.body,
                headers=request.headers,
            )

        try:
            payload = response.json()
        except ValueError:
            payload = None

        result: Dict[str, Any] = {
            "status_code": response.status_code,
            "headers": dict(response.headers),
        }

        if isinstance(payload, list):
            limited_payload = payload[:50]
            result["json"] = limited_payload
            if len(payload) > 50:
                result["truncated"] = True
                result["truncated_count"] = len(payload) - 50
                result["note"] = (
                    "Response contained more than 50 items; returning the first 50 to comply with instructions."
                )
        elif payload is not None:
            result["json"] = payload
        else:
            result["text"] = response.text

        return result

    return mcp


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an MCP server backed by an OpenAPI specification.")
    parser.add_argument("spec", help="Path to the OpenAPI specification (YAML or JSON).")
    parser.add_argument(
        "--name",
        default="OpenAPI MCP Server",
        help="Optional server name to present to clients (default: OpenAPI MCP Server).",
    )
    args = parser.parse_args()

    spec_path = Path(args.spec)
    spec_text, base_url = load_spec(spec_path)

    logging.basicConfig(level=logging.INFO)

    server = build_server(spec_text, base_url, args.name)
    server.run("stdio")


if __name__ == "__main__":
    main()
