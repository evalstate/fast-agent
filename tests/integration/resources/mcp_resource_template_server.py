#!/usr/bin/env python3
"""MCP resource server exposing a resource template and completion handler."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP
from mcp.types import Completion, ResourceTemplateReference

mcp = FastMCP("Smart Resource Template Test Server")

_ITEMS = {
    "alpha": "item:alpha",
    "beta": "item:beta",
    "gamma": "item:gamma",
}
_TEMPLATE_URI = "resource://smart/items/{item_id}"


@mcp.resource(
    _TEMPLATE_URI,
    name="smart_item",
    description="Template-backed item resource",
    mime_type="text/plain",
)
def smart_item(item_id: str) -> str:
    return _ITEMS.get(item_id, f"item:{item_id}")


@mcp.resource(
    "resource://smart/static",
    name="smart_static",
    description="Static resource",
    mime_type="text/plain",
)
def smart_static() -> str:
    return "static"


@mcp.completion()
async def complete_resource_template_argument(ref, argument, context):
    del context
    if not isinstance(ref, ResourceTemplateReference):
        return Completion(values=[])
    if ref.uri != _TEMPLATE_URI or argument.name != "item_id":
        return Completion(values=[])

    prefix = argument.value or ""
    values = [name for name in sorted(_ITEMS.keys()) if name.startswith(prefix)]
    return Completion(values=values, total=len(_ITEMS), hasMore=False)


if __name__ == "__main__":
    mcp.run(transport="stdio")
