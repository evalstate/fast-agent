#!/usr/bin/env python3
"""Tiny Skills-over-MCP demo server.

Run from this directory with:
    uv run skill_server.py
"""

from __future__ import annotations

import json

from fastmcp import FastMCP
from mcp.types import Completion, ResourceTemplateReference

mcp = FastMCP("Skills-over-MCP Demo Server")

DOC_PRODUCTS = ("alpha", "beta")
TEMPLATE_URI = "skill://docs/{product}/SKILL.md"

INDEX = {
    "$schema": "https://schemas.agentskills.io/discovery/0.2.0/schema.json",
    "skills": [
        {
            "type": "skill-md",
            "url": "skill://pirate/SKILL.md",
        },
        {
            "type": "mcp-resource-template",
            "url": TEMPLATE_URI,
            "description": "Product-specific documentation skills.",
        },
    ],
}


@mcp.resource("skill://index.json", mime_type="application/json")
def skill_index() -> str:
    """Skills-over-MCP discovery index."""
    return json.dumps(INDEX)


@mcp.resource("skill://pirate/SKILL.md", mime_type="text/markdown")
def pirate_skill() -> str:
    return """---
name: pirate
description: Answer in an exaggerated pirate style.
---

When using this skill, rewrite the answer in pirate dialect. Use nautical
phrases, but keep the answer useful and concise.
"""


@mcp.resource("skill://pirate/references/example.md", mime_type="text/markdown")
def pirate_example() -> str:
    return """# Pirate style example

Plain: The server is ready.
Pirate: Arrr, the server be ready to sail.
"""


@mcp.resource(TEMPLATE_URI, mime_type="text/markdown")
def product_docs_skill(product: str) -> str:
    if product == "alpha":
        guidance = "Prefer Alpha examples about onboarding, setup, and first-run UX."
    elif product == "beta":
        guidance = "Prefer Beta examples about reporting, exports, and audit trails."
    else:
        guidance = f"Use general documentation examples for {product}."

    return f"""---
name: {product}
description: Product documentation guidance for {product}.
---

When using this skill, answer as a product documentation specialist for
`{product}`. {guidance}
"""


@mcp._mcp_server.completion()
async def complete_resource_template_argument(ref, argument, context):
    """Suggest values for skill://docs/{product}/SKILL.md."""
    del context

    if not isinstance(ref, ResourceTemplateReference):
        return Completion(values=[])
    if ref.uri != TEMPLATE_URI or argument.name != "product":
        return Completion(values=[])

    prefix = argument.value or ""
    values = [product for product in DOC_PRODUCTS if product.startswith(prefix)]
    return Completion(values=values, total=len(DOC_PRODUCTS), hasMore=False)


if __name__ == "__main__":
    mcp.run(transport="stdio")
